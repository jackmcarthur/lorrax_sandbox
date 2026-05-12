#!/usr/bin/env python3
"""
Davidson eigensolver driver for Si 4x4x4.

Loads WFN.h5, builds the DFT Hamiltonian (T + V_loc + V_NL) for each
irreducible k-point, runs the Davidson solver, and compares eigenvalues
against the NSCF reference.

NOTE: V_xc is NOT yet included in H.  The eigenvalues will NOT match
the NSCF reference until V_xc is added.  This script scaffolds the
full pipeline so V_xc can be plugged in.

Usage (Perlmutter, 4 GPUs):
    SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
    SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \\
        --env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \\
        --env=JAX_ENABLE_X64=1 --env=HDF5_USE_FILE_LOCKING=FALSE"
    srun --gres=gpu:4 -N 1 -n 1 $SHIFTER python3 -u run_davidson.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

from file_io import WFNReader
from common import symmetry_maps, Meta
import psp.dft_operators as dft_ops
import psp.vnl_ops as vnl_ops
from psp.get_DFT_mtxels import (
    load_pseudopotentials, generate_gvectors_k,
)
from psp.davidson import davidson_k


def main():
    print("=" * 60)
    print("Davidson eigensolver — Si 4x4x4")
    print("=" * 60)

    wfn_path = os.path.join(os.path.dirname(__file__), "WFN.h5")
    pseudo_dir = os.path.dirname(__file__)

    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    pseudos = load_pseudopotentials(pseudo_dir)

    nelec = int(wfn.nelec)
    n_occ = nelec // 2   # bands to converge (occupied only for now)
    # Request a few extra bands above the Fermi level for testing
    n_tgt = n_occ + 4
    nspinor = int(wfn.nspinor)

    # Use nbands from WFN for meta (just needs to be >= n_tgt)
    nb_meta = max(int(wfn.nbands), n_tgt)
    meta = Meta.from_system(wfn, sym, n_occ, n_tgt - n_occ, nb_meta, 0,
                            bispinor=False)

    nk = int(wfn.nkpts)   # irreducible k-points only (sym-reduced)
    devices = jax.devices()
    n_dev = len(devices)

    print(f"  k-points (irk): {nk}")
    print(f"  n_occ: {n_occ}, n_tgt: {n_tgt}, nspinor: {nspinor}")
    print(f"  FFT grid: {meta.fft_grid}")
    print(f"  Devices: {n_dev}")
    print()

    # -- Build Hamiltonian setup (k-independent) -------------------------
    print("Building operator setup...")
    t0 = time.perf_counter()
    setup = dft_ops.build_operator_setup(wfn, sym, meta, pseudos)
    # Also build vnl_ops setup for the fast VNL
    vsetup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=nspinor)
    dt_setup = time.perf_counter() - t0
    print(f"  Setup: {dt_setup:.1f} s")

    # -- Reference eigenvalues from NSCF ---------------------------------
    energies_ref = np.asarray(wfn.energies)  # (nspin, nk_irk, nbands)

    # -- Run Davidson for each irreducible k-point -----------------------
    print(f"\nRunning Davidson for {nk} k-points (n_tgt={n_tgt})...")
    print("NOTE: V_xc not yet included — eigenvalues will NOT match NSCF.\n")

    all_eigvals = []
    all_errors = []

    for ik in range(nk):
        print(f"--- k-point {ik}/{nk} ---")
        kvec = sym.unfolded_kpts[ik]
        print(f"  k = ({kvec[0]:.4f}, {kvec[1]:.4f}, {kvec[2]:.4f})")

        # Build per-k operators
        kops = dft_ops.build_kpoint_operators(ik, setup, wfn, sym, meta)

        # Black-box H|psi>: takes (m, nspinor, nG) FFT-box, returns sparse-G
        # For Davidson we need sparse-G -> sparse-G.
        # The fused apply_H_k takes FFT-box input.  We need a wrapper that
        # scatters to box, applies H, and returns sparse-G.
        Gx, Gy, Gz = kops.Gx, kops.Gy, kops.Gz
        nx, ny, nz = kops.fft_grid

        def make_apply_H(kops_local, Gx_l, Gy_l, Gz_l, nx_l, ny_l, nz_l):
            """Create a closure for apply_H at this k-point."""
            @jax.jit
            def apply_H(psi_G):
                # psi_G: (m, nspinor, nG) sparse-G
                # Scatter to FFT box for V_loc
                m, ns, nG_loc = psi_G.shape
                psi_box = jnp.zeros((m, ns, nx_l, ny_l, nz_l),
                                    dtype=jnp.complex128)
                psi_box = psi_box.at[:, :, Gx_l, Gy_l, Gz_l].add(psi_G)
                # Apply H (returns sparse-G)
                return dft_ops.apply_H_k(
                    psi_box, kops_local.T_diag, kops_local.V_r,
                    Gx_l, Gy_l, Gz_l,
                    kops_local.vnl_Z, kops_local.vnl_E,
                )
                # TODO: add V_xc contribution here when available:
                #   V_xc_G = _apply_vxc(psi_box, V_xc_r, Gx_l, Gy_l, Gz_l)
                #   return H_G + V_xc_G
            return apply_H

        apply_H = make_apply_H(kops, Gx, Gy, Gz, nx, ny, nz)

        # Initial guess: random with low-G bias (no NSCF eigenvectors)
        # Weight by 1/(1 + T_G) so low-kinetic-energy components dominate
        key = jax.random.PRNGKey(ik)
        X0_raw = jax.random.normal(key, (n_tgt, nspinor, kops.nG),
                                   dtype=jnp.float64)
        weight = 1.0 / (1.0 + kops.T_diag[None, None, :])
        X0 = (X0_raw * weight).astype(jnp.complex128)

        t0 = time.perf_counter()
        eigvals, eigvecs = davidson_k(
            apply_H=apply_H,
            T_diag=kops.T_diag,
            nG=kops.nG,
            nspinor=nspinor,
            n_tgt=n_tgt,
            block_size=min(16, n_tgt),
            max_iter=200,
            tol=1e-8,
            X0=X0,
            verbose=True,
        )
        dt = time.perf_counter() - t0
        print(f"  Time: {dt:.2f} s")

        # Compare with NSCF reference
        ref = energies_ref[0, ik, :n_tgt]
        print(f"\n  {'band':>5s}  {'Davidson':>12s}  {'NSCF ref':>12s}  {'diff':>12s}")
        for ib in range(n_tgt):
            diff = eigvals[ib] - ref[ib]
            marker = " *" if ib < n_occ else ""
            print(f"  {ib:5d}  {eigvals[ib]:12.6f}  {ref[ib]:12.6f}  "
                  f"{diff:12.6f}{marker}")

        all_eigvals.append(eigvals)
        all_errors.append(eigvals[:n_occ] - ref[:n_occ])
        print()

    # -- Summary ---------------------------------------------------------
    print("=" * 60)
    print("Summary (occupied bands only)")
    print("=" * 60)
    for ik in range(nk):
        max_err = np.max(np.abs(all_errors[ik]))
        print(f"  k={ik}: max|Davidson - NSCF| = {max_err:.6f} Ry")
    print()
    print("NOTE: errors are expected to be large because V_xc is missing.")
    print("Once V_xc is added, the Davidson eigenvalues should match NSCF")
    print("to ~1e-8 Ry (the convergence tolerance).")


if __name__ == "__main__":
    main()
