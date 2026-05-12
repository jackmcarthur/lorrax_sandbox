#!/usr/bin/env python3
"""
Full NSCF calculation: QE .save → Davidson at all k-points → WFN.h5.

Reads crystal structure and SCF density from a QE .save directory,
builds the DFT Hamiltonian, runs Davidson at each IBZ k-point, and
writes a complete WFN.h5 compatible with BerkeleyGW/LORRAX.

Usage:
    JAX_ENABLE_X64=1 python3 run_nscf.py --save silicon.save --nk 4 4 4 --nbands 12
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
import time
import numpy as np
import jax
import jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import (
    load_pseudopotentials,
    build_atom_pp_assignments,
    poisson_potential_from_rhoG,
)
from psp.build_projectors_qe import build_local_ionic_potential_on_G_total
from psp.charge_density import build_core_density, compute_grad_rho_sq, build_V_xc
from psp.dft_operators import build_V_scf, build_T_diag_from_kvec, setup_H_k_from_kvec
from psp.operator_checks import validate_operator_inputs
import psp.vnl_ops as vnl_ops
from psp.davidson import davidson_k
from psp.wfn_writer import write_wfn_h5


def build_apply_H(H_k):
    """Build a JIT'd sparse-G -> sparse-G H|psi> for one k-point."""
    nx, ny, nz = H_k.fft_grid
    nspinor = H_k.vnl_E.shape[0]
    Gx, Gy, Gz = H_k.Gx, H_k.Gy, H_k.Gz
    T_diag, V_scf = H_k.T_diag, H_k.V_scf
    vnl_Z, vnl_E, mask = H_k.vnl_Z, H_k.vnl_E, H_k.mask

    @jax.jit
    def apply_H(psi_G):
        psi_box = jnp.zeros((psi_G.shape[0], nspinor, nx, ny, nz), dtype=psi_G.dtype)
        psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G)
        mask_f = mask[None, None, :].astype(psi_G.dtype)
        psi_G_in = psi_box[:, :, Gx, Gy, Gz] * mask_f

        H_G = T_diag[None, None, :] * psi_G_in
        psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
        H_G = H_G + jnp.fft.fftn(
            psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
        )[:, :, Gx, Gy, Gz] * mask_f
        P = jnp.einsum('RG,vsG->Rsv', jnp.conj(vnl_Z), psi_G_in, optimize=True)
        D = jnp.einsum('stRQ,Qtv->Rsv', vnl_E, P, optimize=True)
        H_G = H_G + jnp.einsum('RG,Rsv->vsG', vnl_Z, D, optimize=True) * mask_f
        return H_G

    return apply_H


def main():
    parser = argparse.ArgumentParser(description="NSCF: QE .save → WFN.h5")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None, help="Pseudopotential directory")
    parser.add_argument("--nk", nargs=3, type=int, default=[4, 4, 4], help="k-grid")
    parser.add_argument("--nbands", type=int, default=12, help="Number of bands")
    parser.add_argument("--tol", type=float, default=1e-6, help="Davidson convergence")
    parser.add_argument("--max_iter", type=int, default=50, help="Max Davidson iterations")
    parser.add_argument("--sys_dim", type=int, default=3, help="System dimensionality")
    parser.add_argument("-o", "--output", default="WFN.h5", help="Output file")
    parser.add_argument("--nosym", action="store_true")
    parser.add_argument("--noinv", action="store_true")
    parser.add_argument("--no_t_rev", action="store_true")
    parser.add_argument("--force_symmorphic", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()

    print("=" * 60)
    print(f"NSCF: {args.save} → {args.output}")
    print(f"k-grid: {args.nk}, bands: {args.nbands}, tol: {args.tol}")
    print(f"Device: {jax.devices()[0]}")
    print("=" * 60)

    t0 = time.perf_counter()
    crystal = CrystalData.from_qe_save(args.save)
    pseudo_dir = args.pseudo_dir or os.path.dirname(os.path.abspath(args.save))
    pseudos = load_pseudopotentials(pseudo_dir)
    ctx = validate_operator_inputs(
        pseudos=pseudos, wfn=crystal, sys_dim=args.sys_dim, caller="run_nscf"
    )
    print(f"Load: {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    kpts, weights = crystal.build_kgrid(
        nk=args.nk, nosym=args.nosym, noinv=args.noinv,
        no_t_rev=args.no_t_rev, force_symmorphic=args.force_symmorphic,
    )
    nk = len(kpts)
    print(f"K-grid: {nk} IBZ k-points ({time.perf_counter()-t0:.3f}s)")

    t0 = time.perf_counter()
    atom_pos = jnp.asarray(crystal.atom_crys, dtype=jnp.float64)
    atom_types_j = jnp.asarray(crystal.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types_j, pseudos)
    sp_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        e = sp_tmp.setdefault(id(ap.pseudo), {"pseudo": ap.pseudo, "positions": []})
        e["positions"].append(np.asarray(ap.position, dtype=float))
    sp_payload = [
        (e["pseudo"], np.asarray(e["positions"], dtype=float))
        for e in sp_tmp.values()
    ]

    V_loc_r = jnp.asarray(build_local_ionic_potential_on_G_total(
        assignments=[
            {"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)}
            for ap in assignments
        ],
        species_groups=sp_payload,
        fft_grid=crystal.fft_grid,
        bdot=crystal.bdot,
        cell_volume=crystal.cell_volume,
        bvec=crystal.bvec,
        blat=crystal.blat,
        truncation_2d=ctx.truncation_2d,
    ), dtype=jnp.float64)

    rho_r, _ = crystal.load_charge_density()
    rho_val = jnp.asarray(rho_r, dtype=jnp.float64)
    V_H_r = jnp.real(poisson_potential_from_rhoG(
        jnp.fft.fftn(rho_val, norm='ortho'),
        jnp.asarray(crystal.bdot, dtype=jnp.float64),
        jnp.asarray(crystal.bvec, dtype=jnp.float64),
        crystal.blat,
        truncation_2d=ctx.truncation_2d,
    ))

    class _MetaLike:
        fft_grid = crystal.fft_grid
        nspinor = crystal.nspinor

    rho_core, rhog_core = build_core_density(crystal, pseudos, _MetaLike())
    rho_total = rho_val + rho_core
    V_xc_r = build_V_xc(
        rho_total,
        crystal,
        grad_rho_sq=compute_grad_rho_sq(rho_total, crystal, rhog_core=rhog_core),
        rhog_core=rhog_core,
        xc_func='pbe',
    )
    V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
    t_vscf = time.perf_counter() - t0
    print(f"V_scf: {t_vscf:.1f}s")

    t0 = time.perf_counter()
    q_max = 0.0
    B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
    ngkmax = 0
    for kpt in kpts:
        T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kpt, crystal)
        ngkmax = max(ngkmax, int(T_diag.shape[0]))
        Gk = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
        K_cart = (Gk.astype(float) + np.asarray(kpt, dtype=float)[None, :]) @ B
        q_max = max(q_max, float(np.sqrt(np.max(np.sum(K_cart**2, axis=1)))))
    q_max *= 1.01 if q_max > 0 else math.sqrt(float(crystal.ecutwfc))
    vnl_setup = vnl_ops.build_vnl_setup(
        crystal,
        pseudos=pseudos,
        nspinor=crystal.nspinor,
        q_max=q_max,
    )
    t_vnl_setup = time.perf_counter() - t0
    print(f"VNL setup: {t_vnl_setup:.1f}s")
    print(f"ngkmax: {ngkmax}")

    nbands = args.nbands
    all_eigenvalues = np.zeros((nk, nbands))
    all_gvecs = []
    all_coeffs = []

    print(f"\nDavidson ({nk} k-points, {nbands} bands, tol={args.tol}):")
    t_davidson_total = 0.0

    for ik, kpt in enumerate(kpts):
        t0 = time.perf_counter()

        H_k = setup_H_k_from_kvec(
            np.asarray(kpt, dtype=float),
            V_scf,
            vnl_setup,
            crystal,
            _MetaLike(),
            V_loc_r=V_loc_r,
            ngkmax=ngkmax,
        )
        apply_H = build_apply_H(H_k)

        eigvals, eigvecs = davidson_k(
            apply_H,
            h_diag=H_k.h_diag,
            T_diag=H_k.T_diag,
            nG=H_k.nG,
            nspinor=crystal.nspinor,
            n_tgt=nbands,
            m_max=max(4 * nbands, 8),
            max_iter=args.max_iter,
            tol=args.tol,
            verbose=False,
        )

        dt = time.perf_counter() - t0
        t_davidson_total += dt

        all_eigenvalues[ik] = eigvals
        Gk = np.stack([
            np.asarray(H_k.Gx)[:H_k.nG],
            np.asarray(H_k.Gy)[:H_k.nG],
            np.asarray(H_k.Gz)[:H_k.nG],
        ], axis=-1)
        all_gvecs.append(Gk)
        all_coeffs.append(np.asarray(eigvecs)[:, :, :H_k.nG])

        print(f"  k={ik:3d}/{nk}  nG={H_k.nG:4d}  "
              f"eig=[{eigvals[0]:.4f}, {eigvals[-1]:.4f}]  {dt:.2f}s")

    print(f"Davidson total: {t_davidson_total:.1f}s ({t_davidson_total/nk:.2f}s/k)")

    t0 = time.perf_counter()
    write_wfn_h5(
        args.output,
        crystal=crystal,
        kpoints=kpts,
        weights=weights,
        kgrid=tuple(args.nk),
        eigenvalues=all_eigenvalues,
        gvecs_per_k=all_gvecs,
        coeffs_per_k=all_coeffs,
    )
    t_write = time.perf_counter() - t0
    print(f"\nWrote {args.output}: {t_write:.2f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {t_total:.1f}s")
    print(f"  V_scf:    {t_vscf:.1f}s")
    print(f"  VNL:      {t_vnl_setup:.1f}s")
    print(f"  Davidson: {t_davidson_total:.1f}s")
    print(f"  Write:    {t_write:.2f}s")


if __name__ == "__main__":
    main()
