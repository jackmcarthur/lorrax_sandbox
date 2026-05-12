#!/usr/bin/env python3
"""
Direct diagonalization of H_DFT for Si 4x4x4.

Builds the full DFT Hamiltonian matrix <m|H|n> in the NSCF wavefunction
basis (T + V_loc + V_NL + V_H + V_xc) and diagonalizes.  Compares the
resulting eigenvalues to the NSCF reference from WFN.h5.

This bypasses the Davidson solver entirely to isolate H construction
accuracy from eigensolver issues.

Usage:
    PYTHONPATH=".../lorrax_bse/src:$SITE:$SANDBOX/sources" \
    JAX_ENABLE_X64=1 HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u run_direct_diag.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
import numpy as np
import jax
import jax.numpy as jnp

from file_io import WFNReader
from common import symmetry_maps, Meta
import psp.dft_operators as dft_ops
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.charge_density import (
    build_density_from_ibz,
    build_core_density,
    compute_grad_rho_sq,
    build_V_xc,
)
from common.load_wfns import load_kpoint_fftbox


def main():
    print("=" * 60)
    print("Direct diagonalization — Si 4x4x4")
    print("=" * 60)

    wfn_path = os.path.join(os.path.dirname(__file__), "WFN.h5")
    pseudo_dir = os.path.dirname(__file__)

    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    pseudos = load_pseudopotentials(pseudo_dir)

    nspinor = int(wfn.nspinor)
    n_occ = 8
    nb = int(wfn.nbands)
    n_show = min(nb, 20)

    meta = Meta.from_system(wfn, sym, n_occ, nb - n_occ, nb, 0,
                            bispinor=False)
    nk = int(wfn.nkpts)

    print(f"  k-points (irk): {nk}")
    print(f"  n_occ: {n_occ}, nbands: {nb}, nspinor: {nspinor}")
    print(f"  FFT grid: {meta.fft_grid}")
    print()

    print("Building valence density...")
    t0 = time.perf_counter()
    rho_val_r = build_density_from_ibz(wfn, sym, meta, n_occ)
    print(f"  {time.perf_counter()-t0:.1f} s")

    print("Building core density (NLCC)...")
    t0 = time.perf_counter()
    rho_core_r, rhog_core = build_core_density(wfn, pseudos, meta)
    print(f"  {time.perf_counter()-t0:.1f} s")

    rho_total_r = rho_val_r + rho_core_r

    print("Building V_xc (PBE)...")
    t0 = time.perf_counter()
    grad_rho_sq = compute_grad_rho_sq(rho_total_r, wfn, rhog_core=rhog_core)
    V_xc_r = build_V_xc(rho_total_r, wfn, grad_rho_sq=grad_rho_sq,
                         rhog_core=rhog_core, xc_func='pbe')
    print(f"  {time.perf_counter()-t0:.1f} s")
    print(f"  V_xc range: [{float(jnp.min(V_xc_r)):.4f}, "
          f"{float(jnp.max(V_xc_r)):.4f}] Ry")

    print("Building V_H...")
    t0 = time.perf_counter()
    V_H_r = _build_hartree(rho_val_r, wfn, meta)
    print(f"  {time.perf_counter()-t0:.1f} s")
    print(f"  V_H range: [{float(jnp.min(V_H_r)):.4f}, "
          f"{float(jnp.max(V_H_r)):.4f}] Ry")

    print("Building operator setup (T + V_loc + V_NL)...")
    t0 = time.perf_counter()
    setup = dft_ops.build_operator_setup(wfn, sym, meta, pseudos)
    print(f"  {time.perf_counter()-t0:.1f} s")

    V_total_r = setup.V_r + V_H_r + V_xc_r
    print(f"  V_loc:         [{float(jnp.min(setup.V_r)):.4f}, "
          f"{float(jnp.max(setup.V_r)):.4f}] Ry")
    print(f"  V_loc+V_H+V_xc: [{float(jnp.min(V_total_r)):.4f}, "
          f"{float(jnp.max(V_total_r)):.4f}] Ry")

    energies_ref = np.asarray(wfn.energies)

    print(f"\nDiagonalizing H ({nb}x{nb}) for {nk} k-points...")
    print()

    results = {}

    for ik in range(nk):
        kvec = sym.unfolded_kpts[ik]
        print(f"--- k-point {ik}/{nk}: ({kvec[0]:.4f}, {kvec[1]:.4f}, {kvec[2]:.4f}) ---")

        kops = dft_ops.build_kpoint_operators(ik, setup, wfn, sym, meta)
        psi_box = load_kpoint_fftbox(wfn, sym, meta, ik, nb)

        H_no_xc = dft_ops.build_matrix_k(
            psi_box, kops.T_diag, kops.V_r,
            kops.Gx, kops.Gy, kops.Gz,
            kops.vnl_Z, kops.vnl_E,
        )
        eigvals_no_xc = np.sort(np.linalg.eigvalsh(np.asarray(H_no_xc)))

        V_loc_xc = setup.V_r + V_xc_r
        H_with_xc = dft_ops.build_matrix_k(
            psi_box, kops.T_diag, V_loc_xc,
            kops.Gx, kops.Gy, kops.Gz,
            kops.vnl_Z, kops.vnl_E,
        )
        eigvals_xc = np.sort(np.linalg.eigvalsh(np.asarray(H_with_xc)))

        H_full = dft_ops.build_matrix_k(
            psi_box, kops.T_diag, V_total_r,
            kops.Gx, kops.Gy, kops.Gz,
            kops.vnl_Z, kops.vnl_E,
        )
        eigvals_full = np.sort(np.linalg.eigvalsh(np.asarray(H_full)))

        ref = energies_ref[0, ik, :nb]

        for name, ev in [('no_xc', eigvals_no_xc),
                         ('with_xc', eigvals_xc),
                         ('full', eigvals_full)]:
            results.setdefault(name, []).append((ev, ref))

        print(f"  {'band':>5s}  {'no_xc':>10s}  {'w/V_xc':>10s}  "
              f"{'full':>10s}  {'ref':>10s}  {'Δ_full(mRy)':>12s}")
        for ib in range(min(n_show, nb)):
            d_full = (eigvals_full[ib] - ref[ib]) * 1000
            marker = " *" if ib < n_occ else ""
            print(f"  {ib:5d}  {eigvals_no_xc[ib]:10.6f}  {eigvals_xc[ib]:10.6f}  "
                  f"{eigvals_full[ib]:10.6f}  {ref[ib]:10.6f}  {d_full:12.3f}{marker}")
        print()

    print("=" * 60)
    print("Summary (occupied bands)")
    print("=" * 60)

    for name in ['no_xc', 'with_xc', 'full']:
        all_errs = []
        for ev, ref in results[name]:
            all_errs.extend(ev[:n_occ] - ref[:n_occ])
        all_errs = np.array(all_errs)
        offset = np.mean(all_errs)
        mae = np.mean(np.abs(all_errs))
        mae_no_off = np.mean(np.abs(all_errs - offset))
        max_err = np.max(np.abs(all_errs))
        print(f"\n  {name}:")
        print(f"    MAE           = {mae*1000:.3f} mRy = {mae*13605.698:.1f} meV")
        print(f"    Offset        = {offset*1000:.3f} mRy = {offset*13605.698:.1f} meV")
        print(f"    MAE-no-offset = {mae_no_off*1000:.3f} mRy = {mae_no_off*13605.698:.1f} meV")
        print(f"    max|Δ|        = {max_err*1000:.3f} mRy = {max_err*13605.698:.1f} meV")

    print(f"\n{'':=<60}")
    print("Band gap comparison (band n_occ vs n_occ-1)")
    print(f"{'':=<60}")
    for ik in range(nk):
        ref = results['full'][ik][1]
        full_ev = results['full'][ik][0]
        gap_ref = (ref[n_occ] - ref[n_occ-1]) * 13605.698
        gap_full = (full_ev[n_occ] - full_ev[n_occ-1]) * 13605.698
        gap_err = gap_full - gap_ref
        print(f"  k={ik}: ref={gap_ref:.1f} meV, full={gap_full:.1f} meV, Δ={gap_err:.1f} meV")


def _build_hartree(rho_r, wfn, meta):
    """Simple V_H = 8π ρ(G)/|G|² in Rydberg units."""
    nx, ny, nz = meta.fft_grid
    N = nx * ny * nz

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

    vol = float(wfn.cell_volume)
    G2_safe = jnp.where(G2 > 0, G2, 1.0)
    V_H_G = jnp.where(G2 > 0, 8.0 * jnp.pi * rho_G / (G2_safe * N), 0.0)
    V_H_r = jnp.real(jnp.fft.ifftn(V_H_G)) * N

    print(f"  V_H integral: {float(jnp.sum(V_H_r))*vol/N:.4f}")

    return V_H_r


if __name__ == "__main__":
    main()
