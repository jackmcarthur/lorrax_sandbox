#!/usr/bin/env python3
"""
Direct diagonalization of H_DFT for Si 4x4x4 — correct k-point mapping.

Key fix: SymMaps.unfolded_kpts has 64 entries (full BZ), but WFN.h5
has 8 (IBZ). We must map each IBZ k-point to the correct unfolded
index so that build_kpoint_operators uses the right k-vector for the
kinetic energy and VNL projectors.

Usage:
    PYTHONPATH=".../lorrax_bse/src:$SITE:$SANDBOX/sources" \
    JAX_ENABLE_X64=1 HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u run_direct_diag_v2.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import h5py
import jax.numpy as jnp

from file_io import WFNReader
from common import symmetry_maps, Meta
import psp.dft_operators as dft_ops
from psp.get_DFT_mtxels import load_pseudopotentials, build_atom_pp_assignments
from psp.build_projectors_qe import build_local_ionic_potential_on_G_total
import psp.vnl_ops as vnl_ops
from psp.charge_density import (
    build_density_from_ibz,
    build_core_density,
    compute_grad_rho_sq,
    build_V_xc,
)
from common.load_wfns import load_kpoint_fftbox


def build_ibz_to_unfolded_map(wfn_path, sym):
    """Map each IBZ k-point (WFN.h5 ordering) to the unfolded index."""
    with h5py.File(wfn_path, 'r') as f:
        rk_wfn = f['mf_header/kpoints/rk'][:]
    unfolded = np.asarray(sym.unfolded_kpts)
    nk_ibz = rk_wfn.shape[0]
    mapping = np.zeros(nk_ibz, dtype=int)
    for i in range(nk_ibz):
        k = rk_wfn[i] % 1.0
        k[np.abs(k) < 1e-10] = 0.0
        k[np.abs(k - 1.0) < 1e-10] = 0.0
        u = unfolded % 1.0
        u[np.abs(u) < 1e-10] = 0.0
        u[np.abs(u - 1.0) < 1e-10] = 0.0
        dists = np.linalg.norm(u - k[None, :], axis=1)
        mapping[i] = np.argmin(dists)
        assert dists[mapping[i]] < 1e-6, (
            f"IBZ k={rk_wfn[i]} not found in unfolded, min dist={dists[mapping[i]]}")
    return mapping


def build_hartree(rho_r, wfn, meta):
    """V_H = 8π ρ(G)/|G|² in Rydberg units."""
    nx, ny, nz = meta.fft_grid
    N = nx * ny * nz
    vol = float(wfn.cell_volume)

    rho_G = jnp.fft.fftn(rho_r)

    gx = np.fft.fftfreq(nx, d=1.0/nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0/ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0/nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)
    bvec = np.asarray(wfn.bvec, dtype=float)
    B = float(wfn.blat) * bvec.T
    G_cart = np.einsum('...i,ij->...j', G_crys, B)
    G2 = jnp.asarray(np.sum(G_cart**2, axis=-1), dtype=jnp.float64)

    G2_safe = jnp.where(G2 > 0, G2, 1.0)
    V_H_G = jnp.where(G2 > 0, 8.0 * jnp.pi * rho_G / (G2_safe * N), 0.0)
    V_H_r = jnp.real(jnp.fft.ifftn(V_H_G)) * N

    return V_H_r


def main():
    print("=" * 60)
    print("Direct diag — Si 4x4x4 — correct k-mapping")
    print("=" * 60)

    wfn_path = os.path.join(os.path.dirname(__file__), "WFN.h5")
    pseudo_dir = os.path.dirname(__file__)

    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    pseudos = load_pseudopotentials(pseudo_dir)

    n_occ = 8
    nb = int(wfn.nbands)
    n_show = min(nb, 16)

    meta = Meta.from_system(wfn, sym, n_occ, nb - n_occ, nb, 0,
                            bispinor=False)
    nk_ibz = int(wfn.nkpts)

    ibz_map = build_ibz_to_unfolded_map(wfn_path, sym)
    print(f"  IBZ -> unfolded map: {ibz_map}")

    with h5py.File(wfn_path, 'r') as f:
        rk_wfn = f['mf_header/kpoints/rk'][:]

    print("\nBuilding potentials...")
    rho_val_r = build_density_from_ibz(wfn, sym, meta, n_occ)
    rho_core_r, rhog_core = build_core_density(wfn, pseudos, meta)
    rho_total_r = rho_val_r + rho_core_r

    grad_rho_sq = compute_grad_rho_sq(rho_total_r, wfn, rhog_core=rhog_core)
    V_xc_r = build_V_xc(rho_total_r, wfn, grad_rho_sq=grad_rho_sq,
                         rhog_core=rhog_core, xc_func='pbe')
    V_H_r = build_hartree(rho_val_r, wfn, meta)

    print(f"  V_xc: [{float(jnp.min(V_xc_r)):.4f}, {float(jnp.max(V_xc_r)):.4f}] Ry")
    print(f"  V_H:  [{float(jnp.min(V_H_r)):.4f}, {float(jnp.max(V_H_r)):.4f}] Ry")

    atom_pos = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types, pseudos)
    species_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        entry = species_tmp.setdefault(id(ap.pseudo), {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_groups = [
        (entry["pseudo"], np.asarray(entry["positions"], dtype=float))
        for entry in species_tmp.values()
    ]
    vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=wfn.nspinor)
    V_loc_r = jnp.asarray(
        build_local_ionic_potential_on_G_total(
            assignments=[
                {"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)}
                for ap in assignments
            ],
            species_groups=species_groups,
            fft_grid=meta.fft_grid,
            bdot=np.asarray(wfn.bdot, dtype=float),
            cell_volume=float(wfn.cell_volume),
            bvec=np.asarray(wfn.bvec, dtype=float),
            blat=float(wfn.blat),
            truncation_2d=False,
        ),
        dtype=jnp.float64,
    )
    V_full_r = dft_ops.build_V_scf(V_loc_r, V_H_r, V_xc_r)

    energies_ref = np.asarray(wfn.energies)

    print(f"\nDiagonalizing H ({nb}x{nb}) for {nk_ibz} IBZ k-points...")
    print()

    all_diag = []
    all_rq = []

    for i_ibz in range(nk_ibz):
        i_unf = ibz_map[i_ibz]
        kvec = rk_wfn[i_ibz]
        print(f"--- IBZ k={i_ibz}, unfolded={i_unf}: "
              f"({kvec[0]:.4f}, {kvec[1]:.4f}, {kvec[2]:.4f}) ---")

        H_k = dft_ops.setup_H_k(
            i_unf,
            V_full_r,
            vnl_setup,
            wfn,
            sym,
            meta,
            V_loc_r=V_loc_r,
        )
        print(f"  nG={H_k.nG}, T_diag[0]={float(H_k.T_diag[0]):.6f}")

        psi_box = load_kpoint_fftbox(wfn, sym, meta, i_ibz, nb)

        H = dft_ops.build_matrix_k(
            psi_box, H_k.T_diag, H_k.V_scf,
            H_k.Gx, H_k.Gy, H_k.Gz,
            H_k.vnl_Z, H_k.vnl_E, H_k.mask,
        )
        H_np = np.asarray(H)

        herm_err = np.max(np.abs(H_np - H_np.conj().T))
        if herm_err > 1e-10:
            print(f"  WARNING: H not Hermitian, err={herm_err:.2e}")

        eigvals = np.sort(np.linalg.eigvalsh(H_np))
        rq = np.real(np.diag(H_np))

        ref = energies_ref[0, i_ibz, :nb]

        all_diag.append((eigvals, ref))
        all_rq.append((rq, ref))

        print(f"  {'band':>5s}  {'RQ':>10s}  {'diag':>10s}  {'ref':>10s}  "
              f"{'Δ_RQ(mRy)':>12s}  {'Δ_diag(mRy)':>12s}")
        for ib in range(min(n_show, nb)):
            d_rq = (rq[ib] - ref[ib]) * 1000
            d_diag = (eigvals[ib] - ref[ib]) * 1000
            marker = " *" if ib < n_occ else ""
            print(f"  {ib:5d}  {rq[ib]:10.6f}  {eigvals[ib]:10.6f}  "
                  f"{ref[ib]:10.6f}  {d_rq:12.3f}  {d_diag:12.3f}{marker}")
        print()

    print("=" * 60)
    print("Summary: Rayleigh quotients <ψ_QE|H_me|ψ_QE>")
    print("=" * 60)
    all_rq_err = np.concatenate([rq[:n_occ] - ref[:n_occ] for rq, ref in all_rq])
    offset_rq = np.mean(all_rq_err)
    mae_rq = np.mean(np.abs(all_rq_err))
    mae_rq_no_off = np.mean(np.abs(all_rq_err - offset_rq))
    print(f"  Occ bands:")
    print(f"    MAE           = {mae_rq*1000:.3f} mRy = {mae_rq*13605.698:.1f} meV")
    print(f"    Offset        = {offset_rq*1000:.3f} mRy = {offset_rq*13605.698:.1f} meV")
    print(f"    MAE-no-offset = {mae_rq_no_off*1000:.3f} mRy = {mae_rq_no_off*13605.698:.1f} meV")
    print(f"    max|Δ|        = {np.max(np.abs(all_rq_err))*1000:.3f} mRy")

    print(f"\n  Band gap (RQ):")
    for i_ibz in range(nk_ibz):
        rq, ref = all_rq[i_ibz]
        gap_rq = (rq[n_occ] - rq[n_occ-1]) * 13605.698
        gap_ref = (ref[n_occ] - ref[n_occ-1]) * 13605.698
        print(f"    k={i_ibz}: ref={gap_ref:.1f} meV, RQ={gap_rq:.1f} meV, "
              f"Δ={gap_rq-gap_ref:.1f} meV")

    print(f"\n{'':=<60}")
    print("Summary: diagonalized eigenvalues")
    print(f"{'':=<60}")
    all_diag_err = np.concatenate([ev[:n_occ] - ref[:n_occ] for ev, ref in all_diag])
    offset_d = np.mean(all_diag_err)
    mae_d = np.mean(np.abs(all_diag_err))
    mae_d_no_off = np.mean(np.abs(all_diag_err - offset_d))
    print(f"  Occ bands:")
    print(f"    MAE           = {mae_d*1000:.3f} mRy = {mae_d*13605.698:.1f} meV")
    print(f"    Offset        = {offset_d*1000:.3f} mRy = {offset_d*13605.698:.1f} meV")
    print(f"    MAE-no-offset = {mae_d_no_off*1000:.3f} mRy = {mae_d_no_off*13605.698:.1f} meV")
    print(f"    max|Δ|        = {np.max(np.abs(all_diag_err))*1000:.3f} mRy")


if __name__ == "__main__":
    main()
