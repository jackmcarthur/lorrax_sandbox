"""Manual absorption ε₂(ω) at η=0.05 eV from BGW (vmtxel) and LORRAX (dipole.h5).

Computes:
  ε₂(ω) = (8π² / V_cell n_k) Σ_S^N |⟨0|d_x|S⟩|² L(ω - E_S, η)
with ⟨0|d_x|S⟩ = Σ_cvk A^S_cvk · d^x_cvk.

For BGW: d_cvk comes from vmtxel binary (velocity-gauge ⟨c|p_x + i[r,V_NL]|v⟩,
since absorption.inp says use_momentum); converted to position gauge via
d_r = i d_v / ΔE.

For LORRAX: A^S, E_S from in-process solve_bse_sharded; d_cvk from LORRAX
dipole.h5 sliced + valence-axis flipped to match LORRAX convention.

Sums are truncated at N_max ∈ {100, 400}.
"""
from __future__ import annotations
import os, sys, struct
import numpy as np
import h5py

import jax
import jax.numpy as jnp

from bse.bse_io import load_bse_data_from_restart_sharded, _find_restart_file
from bse.bse_lanczos import solve_bse_sharded
from bse.bse_ring_comm import create_mesh_2d
from bse.absorption_common import lorentzian_broaden, slice_dipole_to_bse_window


# ─────────────────────────────────────────────────────────────────────
#  BGW readers
# ─────────────────────────────────────────────────────────────────────
def read_bgw_vmtxel(path):
    """Return (vmtxel (nk,nc,nv) c128, header (nk,nc,nv,ns,opr))."""
    raw = open(path, 'rb').read()
    n_hdr = struct.unpack('<i', raw[:4])[0]
    hdr = struct.unpack(f'<{n_hdr // 4}i', raw[4:4 + n_hdr])
    nk, nc, nv, ns, opr = hdr
    ofs = 4 + n_hdr + 4
    n_data = struct.unpack('<i', raw[ofs:ofs + 4])[0]
    data = np.frombuffer(raw[ofs + 4:ofs + 4 + n_data], dtype=np.complex128)
    assert data.size == nk * nc * nv * ns
    # bse_index: flat = is + (iv-1 + (ic-1 + (ik-1)*nc)*nv)*nspin
    # k slow, c, v, then spin fast. Reshape (k, c, v, s).
    arr = data.reshape(nk, nc, nv, ns)
    return arr[..., 0], hdr   # spin=1 only


def read_bgw_eigvecs(path):
    """Return (eigvals_eV (N,), A (N, nk, nc, nv) complex)."""
    with h5py.File(path, 'r') as f:
        eig = np.asarray(f['exciton_data/eigenvalues'][:])
        ev = f['exciton_data/eigenvectors'][:]   # (1, N, nk, nc, nv, 1, 2)
        A = ev[0, :, :, :, :, 0, 0] + 1j * ev[0, :, :, :, :, 0, 1]
        nc = int(f['exciton_header/params/nc'][()])
        nv = int(f['exciton_header/params/nv'][()])
    return eig, A, nc, nv


def read_qe_volume_bohr3(qe_xml_path):
    """Volume in bohr^3 from QE .save/data-file-schema.xml."""
    import xml.etree.ElementTree as ET
    root = ET.parse(qe_xml_path).getroot()
    cell = root.find('.//atomic_structure/cell')
    a1 = np.array([float(x) for x in cell.find('a1').text.split()])
    a2 = np.array([float(x) for x in cell.find('a2').text.split()])
    a3 = np.array([float(x) for x in cell.find('a3').text.split()])
    return abs(np.dot(a1, np.cross(a2, a3)))   # bohr^3


# ─────────────────────────────────────────────────────────────────────
#  Absorption kernels
# ─────────────────────────────────────────────────────────────────────
def eps2_from_dipole(eigvals_Ry, projections, omegas_Ry, V_cell, eta_Ry,
                     n_k, n_spin=1, n_spinor=2, n_max=None):
    """ε₂(ω) = (8π²/V) · 1/(n_k n_spin n_spinor) · Σ_S |⟨0|d|S⟩|² L(ω-E_S, η)."""
    f = np.abs(projections) ** 2
    if n_max is not None:
        f = f[:n_max]
        eigvals_Ry = eigvals_Ry[:n_max]
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    return pref * lorentzian_broaden(omegas_Ry, eigvals_Ry, f, eta_Ry)


def main():
    bgw_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8"
    lorrax_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul"

    ryd2ev = 13.6056980659
    eta_eV = 0.05
    eta_Ry = eta_eV / ryd2ev

    # Volume from QE: Si 4x4x4 has lattice 5.43 Å conventional → 2-atom primitive
    qe_xml = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/scf/silicon.save/data-file-schema.xml"
    V_cell = read_qe_volume_bohr3(qe_xml)
    print(f"V_cell = {V_cell:.4f} bohr^3", flush=True)

    omegas_eV = np.linspace(2.0, 7.0, 5001)   # fine grid for sharp peaks; spectrum lives ≳ 3 eV
    omegas_Ry = omegas_eV / ryd2ev

    # ─── BGW side ───────────────────────────────────────────────────
    print("Reading BGW eigenvectors.h5 + vmtxel ...", flush=True)
    eig_bgw_eV, A_bgw, nc, nv = read_bgw_eigvecs(os.path.join(bgw_dir, "eigenvectors.h5"))
    vmtxel, hdr = read_bgw_vmtxel(os.path.join(bgw_dir, "vmtxel"))
    nk, nc_h, nv_h, ns, opr = hdr
    print(f"  BGW: {len(eig_bgw_eV)} eigvals, A shape {A_bgw.shape}, vmtxel {vmtxel.shape}, opr={opr}")

    # vmtxel from use_momentum is ⟨c|v̂_x|v⟩ (Ry). Convert to position-gauge dipole:
    #   d_r = ⟨c|x̂|v⟩ = i ⟨c|v̂|v⟩ / (E_c - E_v)    (in atomic / Ry units)
    # ΔE_cv: read from eigenvalues_noeh.dat? Easier — use the BGW eqp.dat or just
    # use the QP energies stored elsewhere. Since the spectrum's broadening is
    # large compared to the small ΔE corrections from gauge, we use a simpler
    # formula: |⟨0|v̂|S⟩|² / E_S² δ(ω - E_S) = |⟨0|x̂|S⟩|² δ(ω - E_S) on shell.
    # ε₂(ω) = (8π²/V) Σ_S |⟨0|v|S⟩|² / E_S² · L(ω-E_S, η).
    #
    # Compute matrix elements in velocity gauge then divide by E_S² in the sum.
    proj_bgw_v = np.einsum("Nkcv,kcv->N", A_bgw, vmtxel, optimize=True)
    f_bgw = (np.abs(proj_bgw_v) ** 2) / (eig_bgw_eV / ryd2ev) ** 2  # |M|²/E_S²

    pref = 16.0 * np.pi ** 2 / (V_cell * nk * 1 * 2)   # n_spin=1, n_spinor=2
    eps2_bgw_100 = pref * lorentzian_broaden(
        omegas_Ry, eig_bgw_eV[:100] / ryd2ev, f_bgw[:100], eta_Ry)
    eps2_bgw_400 = pref * lorentzian_broaden(
        omegas_Ry, eig_bgw_eV[:400] / ryd2ev, f_bgw[:400], eta_Ry)
    print(f"  BGW Σ|d|² over first 100 = {f_bgw[:100].sum():.4e}")
    print(f"  BGW Σ|d|² over first 400 = {f_bgw[:400].sum():.4e}")
    print(f"  BGW lowest 5 eigvals (eV): {eig_bgw_eV[:5]}")

    # ─── LORRAX side ────────────────────────────────────────────────
    # Run solve_bse_sharded in-process to get 100 eigenvalues + eigenvectors.
    print("\nRunning LORRAX BSE Lanczos (n_eig=100) ...", flush=True)
    os.chdir(lorrax_dir)
    restart = _find_restart_file("cohsex_bse.in")
    mesh_xy = create_mesh_2d()
    data = load_bse_data_from_restart_sharded(
        restart, n_val=8, n_cond=8, mesh_xy=mesh_xy,
        input_file="cohsex_bse.in", n_occ=None,
    )
    data["matvec_kind"] = "ring"

    # Apply BGW's eqp.dat corrections by ADJUSTING the existing eps_v/eps_c
    # in-place instead of re-slicing — this preserves the LORRAX internal
    # band layout (nv_pad=2, nc_pad=8 in the (Kramers,σ) representation).
    # The eqp shifts are read into a (nk, nbnd) table; for each LORRAX band
    # axis we apply the shift to the matching DFT band index.
    eqp_file = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/eqp.dat"
    eps_v_old = np.asarray(jax.device_get(data["eps_v"]))   # (nk, nv_pad)
    eps_c_old = np.asarray(jax.device_get(data["eps_c"]))   # (nk, nc_pad)
    nv_pad = eps_v_old.shape[1]
    nc_pad = eps_c_old.shape[1]
    print(f"  LORRAX original eps_v shape: {eps_v_old.shape}; eps_c: {eps_c_old.shape}")
    enk_full = np.asarray(h5py.File(restart, 'r')['enk_full'][:])
    from bse.bse_io import apply_eqp_corrections
    enk_full_eqp = apply_eqp_corrections(enk_full, eqp_file, input_file="cohsex_bse.in")
    # Find LORRAX band slicing: match eps_v_old to enk_full columns.
    # eps_v_old[k, i] should equal enk_full[k, j] for some mapping i↔j.
    # Use the first k-point to identify column indices.
    col_v = np.array([np.argmin(np.abs(enk_full[0] - eps_v_old[0, i])) for i in range(nv_pad)])
    col_c = np.array([np.argmin(np.abs(enk_full[0] - eps_c_old[0, i])) for i in range(nc_pad)])
    print(f"  LORRAX val band indices in enk_full: {col_v}")
    print(f"  LORRAX cond band indices in enk_full: {col_c}")
    eps_v_new = enk_full_eqp[:, col_v]
    eps_c_new = enk_full_eqp[:, col_c]
    data["eps_v"] = jnp.asarray(eps_v_new)
    data["eps_c"] = jnp.asarray(eps_c_new)
    print(f"  Applied BGW eqp.dat. mean Δval={np.mean(eps_v_new-eps_v_old)*ryd2ev:.3f} eV, "
          f"mean Δcond={np.mean(eps_c_new-eps_c_old)*ryd2ev:.3f} eV", flush=True)

    eig_lor_Ry, vec_lor, _ = solve_bse_sharded(
        data, mesh_xy, n_eig=100, max_iter=400, n_reorth=400,
        include_W=True, block_size=1, atol=1e-10, solver_kind="lanczos",
    )
    eig_lor_Ry = np.asarray(jax.device_get(eig_lor_Ry))
    vec_lor = np.asarray(jax.device_get(vec_lor))   # (N, 1, nc_pad, nv_pad, nk)
    print(f"  LORRAX eigvecs shape: {vec_lor.shape}")
    eig_lor_eV = eig_lor_Ry * ryd2ev
    print(f"  LORRAX lowest 5 eigvals (eV): {eig_lor_eV[:5]}")

    # LORRAX dipole.h5 — already velocity gauge ⟨c|v̂|v⟩ with full 60-band axis
    with h5py.File(os.path.join(lorrax_dir, "dipole.h5"), "r") as f:
        dipole_cart = np.asarray(f["dipole_cart"][:], dtype=np.complex128)  # (3, nk, nb, nb)
        deltaE = np.asarray(f["deltaE"][:], dtype=np.float64)               # (nk, nb, nb)
    nk_lor = vec_lor.shape[-1]
    nc_lor = vec_lor.shape[2]
    nv_lor = vec_lor.shape[3]
    print(f"  LORRAX nc={nc_lor} nv={nv_lor} nk={nk_lor}; dipole shape={dipole_cart.shape}")

    # Need n_occ for LORRAX. cohsex_bse.in has fermi_reference=midgap; for SOC
    # Si nval=8 → 8 occupied SP bands, n_occ=8.
    n_occ_lor = 8
    # slice_dipole_to_bse_window returns (3, nk, nc=n_cond, nv=n_val) with
    # v in low→high band order, position-gauge.
    d_alpha, _ = slice_dipole_to_bse_window(
        dipole_cart, deltaE, n_occ=n_occ_lor, n_val=nv_lor, n_cond=nc_lor)
    print(f"  d_alpha sliced shape: {d_alpha.shape}")

    # vec_lor has axis order (N, 1, nc, nv, nk) — squeeze and transpose to (N, nk, nc, nv).
    A_lor = vec_lor[:, 0]                                # (N, nc, nv, nk)
    A_lor = np.transpose(A_lor, (0, 3, 1, 2))            # (N, nk, nc, nv)

    # Polarisation x = (1, 0, 0): take alpha=0 component of d_alpha.
    d_x_lor = d_alpha[0]                                 # (nk, nc, nv)

    proj_lor = np.einsum("Nkcv,kcv->N", A_lor, d_x_lor, optimize=True)
    f_lor = np.abs(proj_lor) ** 2   # already position gauge → no /E_S²

    n_spinor_lor = 2
    pref_lor = 16.0 * np.pi ** 2 / (V_cell * nk_lor * 1 * n_spinor_lor)
    eps2_lor_100 = pref_lor * lorentzian_broaden(
        omegas_Ry, eig_lor_Ry[:100], f_lor[:100], eta_Ry)
    print(f"  LORRAX Σ|d|² over first 100 = {f_lor[:100].sum():.4e}")
    print(f"  LORRAX Σ|d|² (all 100) total = {f_lor.sum():.4e}")

    # ─── Save data + plot ───────────────────────────────────────────
    out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/compare"
    np.savez(os.path.join(out_dir, "absorption_data.npz"),
             omegas_eV=omegas_eV,
             eps2_bgw_100=eps2_bgw_100, eps2_bgw_400=eps2_bgw_400,
             eps2_lor_100=eps2_lor_100,
             eig_bgw_eV=eig_bgw_eV, eig_lor_eV=eig_lor_eV,
             f_bgw=f_bgw, f_lor=f_lor,
             V_cell=V_cell, nk=nk, eta_eV=eta_eV)
    print(f"\nSaved data to {out_dir}/absorption_data.npz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(omegas_eV, eps2_bgw_100, label="BGW (first 100 eigvals)", lw=1.0)
    ax[0].plot(omegas_eV, eps2_bgw_400, label="BGW (first 400 eigvals)", lw=1.0, ls="--")
    ax[0].set_ylabel(r"$\varepsilon_2$ (BGW, η=0.05 eV)")
    ax[0].legend(); ax[0].set_xlim(omegas_eV[0], omegas_eV[-1])
    # Normalise both spectra to peak 1.0 to compare structure (absolute scale
    # depends on dipole-gauge conventions that differ between BGW & LORRAX).
    norm_bgw = eps2_bgw_100.max()
    norm_lor = eps2_lor_100.max()
    ax[1].plot(omegas_eV, eps2_bgw_100 / norm_bgw, label=f"BGW (first 100, /max={norm_bgw:.2g})", lw=1.0)
    ax[1].plot(omegas_eV, eps2_lor_100 / norm_lor, label=f"LORRAX (first 100, /max={norm_lor:.2g})", lw=1.0)
    ax[1].set_ylabel(r"$\varepsilon_2$ (peak-normalised)")
    ax[1].set_xlabel(r"$\omega$ (eV)")
    ax[1].legend(); ax[1].set_xlim(omegas_eV[0], omegas_eV[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "absorption_compare.png"), dpi=110)
    print(f"Wrote {out_dir}/absorption_compare.png")


if __name__ == "__main__":
    main()
