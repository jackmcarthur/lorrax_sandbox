"""Isolate dipole correctness by projecting BGW eigenvectors with two
different dipole sources and comparing.

Branch A: BGW eigenvectors × BGW vmtxel  (BGW's own answer)
Branch B: BGW eigenvectors × LORRAX dipole.h5 (slice + valence-axis flip)

If both spectra match, LORRAX dipoles are gauge-equivalent to BGW vmtxel.
If they differ, the discrepancy is in LORRAX's dipole computation.
"""
from __future__ import annotations
import os, struct
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")
from bse.absorption_common import lorentzian_broaden, slice_dipole_to_bse_window


def read_bgw_vmtxel(path):
    raw = open(path, 'rb').read()
    n_hdr = struct.unpack('<i', raw[:4])[0]
    hdr = struct.unpack(f'<{n_hdr // 4}i', raw[4:4 + n_hdr])
    nk, nc, nv, ns, opr = hdr
    ofs = 4 + n_hdr + 4
    n_data = struct.unpack('<i', raw[ofs:ofs + 4])[0]
    data = np.frombuffer(raw[ofs + 4:ofs + 4 + n_data], dtype=np.complex128)
    return data.reshape(nk, nc, nv, ns)[..., 0], hdr


def read_bgw_eigvecs(path):
    with h5py.File(path, 'r') as f:
        eig = np.asarray(f['exciton_data/eigenvalues'][:])
        ev = f['exciton_data/eigenvectors'][:]   # (1, N, nk, nc, nv, 1, 2)
        A = ev[0, :, :, :, :, 0, 0] + 1j * ev[0, :, :, :, :, 0, 1]
    return eig, A


def read_qe_volume_bohr3(qe_xml_path):
    import xml.etree.ElementTree as ET
    root = ET.parse(qe_xml_path).getroot()
    cell = root.find('.//atomic_structure/cell')
    a1 = np.array([float(x) for x in cell.find('a1').text.split()])
    a2 = np.array([float(x) for x in cell.find('a2').text.split()])
    a3 = np.array([float(x) for x in cell.find('a3').text.split()])
    return abs(np.dot(a1, np.cross(a2, a3)))


def main():
    bgw_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8"
    lor_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul"
    qe_xml = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/scf/silicon.save/data-file-schema.xml"

    ryd2ev = 13.6056980659
    eta_eV = 0.05; eta_Ry = eta_eV / ryd2ev

    V_cell = read_qe_volume_bohr3(qe_xml)

    eig_eV, A = read_bgw_eigvecs(f"{bgw_dir}/eigenvectors.h5")
    eig_Ry = eig_eV / ryd2ev
    print(f"BGW eigvecs A shape: {A.shape}   (N, nk, nc, nv) — v=0 is highest valence")
    nk, nc, nv = A.shape[1:]
    n_occ = 8

    vmtxel, hdr = read_bgw_vmtxel(f"{bgw_dir}/vmtxel")
    print(f"BGW vmtxel shape: {vmtxel.shape}, opr={hdr[4]} (1=momentum)")

    # Stay in velocity gauge — simpler, avoids ΔE bookkeeping. BGW vmtxel is
    # ⟨c|v̂_x|v⟩ already; LORRAX dipole_cart in dipole.h5 is the same operator.
    d_vmtxel_v = vmtxel.copy()   # (nk, nc, nv), v-axis high→low (BGW)

    # ─── LORRAX dipole.h5 ───────────────────────────────────────────
    with h5py.File(f"{lor_dir}/dipole.h5", 'r') as f:
        dipole_cart = np.asarray(f['dipole_cart'][:], dtype=np.complex128)  # (3, nk, 60, 60)
    # Slice (m=c, n=v) like absorption_common.slice_dipole_to_bse_window but
    # WITHOUT the /ΔE conversion: stay in velocity gauge.
    val_lo = n_occ - 8; cond_hi = n_occ + 8
    d_lor_v = dipole_cart[0, :, n_occ:cond_hi, val_lo:n_occ]   # (nk, nc=8, nv=8)
    # BGW eigenvector valence axis is high→low; LORRAX is low→high. Flip.
    d_lor_v_flipped = d_lor_v[:, :, ::-1]

    # ─── Σ|d|² sanity-check: gauge-invariant on the SUMMED level ────
    sum_vmtxel = np.sum(np.abs(d_vmtxel_v) ** 2)
    sum_lor    = np.sum(np.abs(d_lor_v_flipped) ** 2)
    print(f"\nVelocity-gauge sum rule (gauge-invariant):")
    print(f"  Σ_cvk |⟨c|v|v⟩|² (BGW vmtxel):    {sum_vmtxel:.6e}  (Ry²)")
    print(f"  Σ_cvk |⟨c|v|v⟩|² (LORRAX dipole): {sum_lor:.6e}  (Ry²)")
    print(f"  ratio LORRAX / BGW = {sum_lor / sum_vmtxel:.4f}")

    # Per-element ratio histogram — if dipoles agree up to phase, |LOR|/|BGW| = 1.
    mag_b = np.abs(d_vmtxel_v)
    mag_l = np.abs(d_lor_v_flipped)
    mask = mag_b > 1e-3 * mag_b.max()
    ratio = (mag_l / mag_b)[mask]
    print(f"  per-element |LORRAX|/|BGW| ratio: median={np.median(ratio):.3f}, "
          f"mean={ratio.mean():.3f}, p10={np.percentile(ratio,10):.3f}, "
          f"p90={np.percentile(ratio,90):.3f}")
    # Per-element phase comparison: arg(LOR/BGW)
    phase_diff = np.angle((d_lor_v_flipped / d_vmtxel_v)[mask])
    print(f"  arg(LORRAX / BGW) std = {np.std(phase_diff):.3f} rad "
          f"(0 = perfect alignment, ~π/√3 ≈ 1.81 rad = uniform random U(1))")

    # ─── Project BGW eigvecs onto each dipole, build absorption ─────
    # Velocity-gauge ε₂(ω) = (8π²/V·n_k·n_spinor) Σ_S |⟨0|v̂|S⟩|² / E_S²  L(ω-E_S)
    proj_b = np.einsum("Nkcv,kcv->N", A, d_vmtxel_v, optimize=True)
    proj_l = np.einsum("Nkcv,kcv->N", A, d_lor_v_flipped, optimize=True)
    # weights = |M|²/E_S²  (Ry² / Ry² = dimensionless)
    f_b = (np.abs(proj_b) ** 2) / np.maximum(eig_Ry, 1e-3) ** 2
    f_l = (np.abs(proj_l) ** 2) / np.maximum(eig_Ry, 1e-3) ** 2

    print(f"\nVelocity-gauge oscillator strengths Σ_S |⟨0|v̂|S⟩|²/E_S² for first 100 BGW eigvals:")
    print(f"  ... BGW vmtxel: Σ = {f_b[:100].sum():.6e}")
    print(f"  ... LORRAX:     Σ = {f_l[:100].sum():.6e}")
    print(f"  ratio = {f_l[:100].sum() / f_b[:100].sum():.4f}")

    omegas_eV = np.linspace(2.0, 7.0, 5001)
    omegas_Ry = omegas_eV / ryd2ev
    pref = 16.0 * np.pi ** 2 / (V_cell * nk * 1 * 2)
    eps2_b_100 = pref * lorentzian_broaden(omegas_Ry, eig_Ry[:100], f_b[:100], eta_Ry)
    eps2_l_100 = pref * lorentzian_broaden(omegas_Ry, eig_Ry[:100], f_l[:100], eta_Ry)
    eps2_b_400 = pref * lorentzian_broaden(omegas_Ry, eig_Ry[:400], f_b[:400], eta_Ry)
    eps2_l_400 = pref * lorentzian_broaden(omegas_Ry, eig_Ry[:400], f_l[:400], eta_Ry)

    out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/compare"
    np.savez(f"{out_dir}/dipole_isolate.npz",
             omegas_eV=omegas_eV, eig_eV=eig_eV,
             f_b=f_b, f_l=f_l,
             eps2_b_100=eps2_b_100, eps2_l_100=eps2_l_100,
             eps2_b_400=eps2_b_400, eps2_l_400=eps2_l_400)

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(omegas_eV, eps2_b_100, label="BGW vmtxel (100)", lw=1.0)
    ax[0].plot(omegas_eV, eps2_l_100, label="LORRAX dipole.h5 (100)", lw=1.0, ls="--")
    ax[0].set_ylabel(r"$\varepsilon_2$ (η=0.05 eV) — first 100 BGW eigvals")
    ax[0].set_title("Same BGW eigenvectors, different dipole source")
    ax[0].legend(); ax[0].set_xlim(omegas_eV[0], omegas_eV[-1])
    ax[1].plot(omegas_eV, eps2_b_400, label="BGW vmtxel (400)", lw=1.0)
    ax[1].plot(omegas_eV, eps2_l_400, label="LORRAX dipole.h5 (400)", lw=1.0, ls="--")
    ax[1].set_ylabel(r"$\varepsilon_2$ (η=0.05 eV) — first 400 BGW eigvals")
    ax[1].set_xlabel(r"$\omega$ (eV)")
    ax[1].legend(); ax[1].set_xlim(omegas_eV[0], omegas_eV[-1])
    plt.tight_layout()
    plt.savefig(f"{out_dir}/dipole_isolate.png", dpi=110)
    print(f"\nWrote {out_dir}/dipole_isolate.png")


if __name__ == "__main__":
    main()
