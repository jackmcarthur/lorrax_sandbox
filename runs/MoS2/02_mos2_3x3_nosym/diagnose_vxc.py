#!/usr/bin/env python3
"""Diagnostic: isolate MoS2 NSCF eigenvalue error (2.7 mRy at Gamma).

Tests V_xc with different configurations to identify the error source:
1. Full GGA + NLCC (current code) — baseline
2. LDA only (sigma=0, no GGA correction)
3. Full GGA but with zero NLCC core charge
4. Full GGA + NLCC, but compare V_xc array-level stats

Usage (on Perlmutter GPU node):
    module load lorrax
    lxrun python3 -u diagnose_vxc.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
import numpy as np
import jax
import jax.numpy as jnp

# ── LORRAX imports ──
from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import (
    build_V_scf, compute_ngkmax,
    setup_H_k_from_kvec, apply_H_k,
)
from psp.davidson import davidson_k, warmup_jit
import psp.vnl_ops as vnl_ops


def compute_V_H_and_V_xc_diagnostic(
    rho_val, rho_core, rhog_core, G_cart, bdot, bvec, blat,
    *, mode="full"
):
    """Compute V_H and V_xc with diagnostic modes.

    mode="full":      standard GGA + NLCC (baseline)
    mode="lda":       LDA only (sigma=0, no GGA divergence)
    mode="no_nlcc":   GGA but with zero core charge
    mode="no_gga_div": GGA sigma but zero divergence correction
    """
    from jax_xc_local.pbe import pbe_xc
    from psp.get_DFT_mtxels import poisson_potential_from_rhoG

    # ── V_H via Poisson ──
    rho_G_ortho = jnp.fft.fftn(rho_val, norm='ortho')
    V_H_r = jnp.real(poisson_potential_from_rhoG(
        rho_G_ortho, bdot, bvec, blat, truncation_2d=False))

    # ── V_xc ──
    if mode == "no_nlcc":
        rho_total = rho_val  # pretend no core charge
        rho_safe = jnp.maximum(rho_total, 1e-10)
        rho_G_total = jnp.fft.fftn(rho_total)
    else:
        rho_total = rho_val + rho_core
        rho_safe = jnp.maximum(rho_total, 1e-10)
        rho_core_gridded = jnp.real(jnp.fft.ifftn(rhog_core))
        rho_G_total = jnp.fft.fftn(rho_total - rho_core_gridded) + rhog_core

    if mode == "lda":
        sigma = jnp.zeros_like(rho_total)
    else:
        grad_rho_sq = jnp.zeros_like(rho_total)
        for i in range(3):
            drho = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
            grad_rho_sq = grad_rho_sq + drho ** 2
        sigma = jnp.maximum(grad_rho_sq, 0.0)

    # PBE functional derivatives
    def E_xc_lda(rho):
        return jnp.sum(rho * pbe_xc(rho, jnp.zeros_like(rho)))
    def E_xc_full(rho, sig):
        return jnp.sum(rho * pbe_xc(rho, sig))

    df_drho_lda = jax.grad(E_xc_lda)(rho_safe)
    df_drho_full = jax.grad(E_xc_full, argnums=0)(rho_safe, sigma)
    df_dsigma = jax.grad(E_xc_full, argnums=1)(rho_safe, sigma)

    gga_mask = (rho_total > 1e-6) & (sigma > 1e-10)
    df_drho = df_drho_lda + jnp.where(gga_mask, df_drho_full - df_drho_lda, 0.0)
    df_dsigma = jnp.where(gga_mask, df_dsigma, 0.0)

    # GGA divergence
    if mode in ("lda", "no_gga_div"):
        gga_corr = jnp.zeros_like(rho_total)
    else:
        gga_corr = jnp.zeros_like(rho_total)
        for i in range(3):
            drho_ri = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
            h_i_G = jnp.fft.fftn(df_dsigma * drho_ri)
            gga_corr = gga_corr + jnp.real(
                jnp.fft.ifftn(1j * G_cart[..., i] * h_i_G))

    V_xc_r = df_drho - 2.0 * gga_corr
    return V_H_r, V_xc_r


def diag_at_gamma(V_scf, vnl_setup, crystal, V_loc_r, nbnd=26):
    """Diagonalize at Gamma, return eigenvalues in Ry."""
    fft_grid = crystal.fft_grid
    _nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    nspinor = crystal.nspinor
    bdot = np.asarray(crystal.bdot, dtype=float)
    kpoint = np.array([0.0, 0.0, 0.0])

    ngkmax = compute_ngkmax(
        kpoint.reshape(1, 3), bdot, crystal.ecutwfc, fft_grid)
    warmup_jit(ngkmax, nspinor, nbnd)

    H_k = setup_H_k_from_kvec(kpoint, V_scf, vnl_setup, crystal, None,
                                V_loc_r=V_loc_r, ngkmax=ngkmax)

    def apply_H(psi_G, _H=H_k):
        mask_f = _H.mask[None, None, :].astype(psi_G.dtype)
        psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
        psi_box = psi_box.at[:, :, _H.Gx, _H.Gy, _H.Gz].add(psi_G * mask_f)
        return apply_H_k(psi_box, _H.T_diag, _H.V_scf,
                          _H.Gx, _H.Gy, _H.Gz,
                          _H.vnl_Z, _H.vnl_E, _H.mask)

    evals, _ = davidson_k(
        apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=nspinor,
        n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=1e-8)
    return np.asarray(evals)


def parse_qe_eigenvalues(nscf_out, ik=0):
    """Extract eigenvalues from QE nscf.out for k-point ik."""
    import re
    with open(nscf_out) as f:
        lines = f.readlines()

    evals = []
    in_kblock = False
    k_count = -1
    for i, line in enumerate(lines):
        if "k =" in line and "bands" in line:
            k_count += 1
            if k_count == ik:
                in_kblock = True
                continue
            elif k_count > ik:
                break
        if in_kblock and line.strip() and not line.strip().startswith("k"):
            try:
                vals = [float(x) for x in line.split()]
                evals.extend(vals)
            except ValueError:
                if evals:
                    break
    return np.array(evals)  # in eV


def parse_vxc_dat(vxc_path):
    """Parse QE/pw2bgw vxc.dat → dict of {(ik, iband): vxc_eV}."""
    data = {}
    ik = -1
    with open(vxc_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                # Header: kx ky kz nbnd ispin
                ik += 1
            elif len(parts) == 4:
                ispin, iband, re_vxc, im_vxc = parts
                data[(ik, int(iband) - 1)] = float(re_vxc)  # eV
    return data


def main():
    save_dir = "qe/nscf/MoS2.save"
    nscf_out = "qe/nscf/nscf.out"
    vxc_path = "qe/nscf/vxc.dat"

    print("=" * 70)
    print("MoS2 V_xc Diagnostic: isolating 2.7 mRy Gamma error")
    print("=" * 70)

    # Load crystal data
    crystal = CrystalData.from_qe_save(save_dir)
    pseudos = load_pseudopotentials(save_dir)
    fft_grid = crystal.fft_grid
    _nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    print(f"Crystal: {crystal.nat} atoms, fft={fft_grid}, "
          f"ecutwfc={crystal.ecutwfc:.1f}, nspinor={crystal.nspinor}")

    # Build ionic potentials
    V_loc_r, rho_core_r, rho_core_G = build_ionic_and_core(
        crystal, pseudos, fft_grid, truncation_2d=False)

    rho_r, _ = crystal.load_charge_density()
    rho_val = jnp.asarray(rho_r, dtype=jnp.float64)

    B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
    G_cart = build_G_cart(_nx, _ny, _nz, B)
    bdot_j = jnp.asarray(crystal.bdot, dtype=jnp.float64)
    bvec_j = jnp.asarray(crystal.bvec, dtype=jnp.float64)

    # ── Report core charge statistics ──
    rho_core_np = np.asarray(rho_core_r)
    vol = crystal.cell_volume
    N = _nx * _ny * _nz
    core_integral = float(np.sum(rho_core_np)) * vol / N
    print(f"Core density integral: {core_integral:.4f} e")
    print(f"Core/Total ratio: {core_integral / crystal.nelec:.1%}")
    print(f"|rho_core| max: {np.max(np.abs(rho_core_np)):.6f}")
    print()

    # VNL setup (same for all modes)
    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, pseudos=pseudos, nspinor=crystal.nspinor,
        q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)

    # QE reference eigenvalues at Gamma
    import h5py
    ref_wfn = "qe/nscf/WFN.h5"
    with h5py.File(ref_wfn, "r") as f:
        evals_qe = f["mf_header/kpoints/el"][0, 0, :]  # (nbnd,) Ry

    # QE V_xc matrix elements
    vxc_data = parse_vxc_dat(vxc_path)

    nbnd = 26  # occupied bands for MoS2

    # ── Test each mode ──
    modes = ["full", "lda", "no_nlcc", "no_gga_div"]
    results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print(f"{'='*50}")

        t0 = time.perf_counter()
        V_H_r, V_xc_r = compute_V_H_and_V_xc_diagnostic(
            rho_val, rho_core_r, rho_core_G, G_cart,
            bdot_j, bvec_j, crystal.blat,
            mode=mode)
        jax.block_until_ready(V_xc_r)

        # V_xc statistics
        V_xc_np = np.asarray(V_xc_r)
        print(f"  V_xc: min={V_xc_np.min():.4f} max={V_xc_np.max():.4f} "
              f"mean={V_xc_np.mean():.4f} Ry")

        if mode == "full":
            V_xc_full = V_xc_np.copy()
        else:
            diff = V_xc_np - V_xc_full
            print(f"  V_xc - V_xc_full: "
                  f"MAE={np.mean(np.abs(diff)):.6f} "
                  f"max={np.max(np.abs(diff)):.6f} Ry")

        V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
        jax.block_until_ready(V_scf)
        dt = time.perf_counter() - t0
        print(f"  Potentials built: {dt:.2f}s")

        # Diagonalize at Gamma
        t0 = time.perf_counter()
        evals = diag_at_gamma(V_scf, vnl_setup, crystal, V_loc_r, nbnd=nbnd)
        dt = time.perf_counter() - t0
        print(f"  Davidson: {dt:.2f}s")

        # Compare with QE
        evals_qe_occ = evals_qe[:nbnd]
        diff_ry = evals - evals_qe_occ
        mae_mry = np.mean(np.abs(diff_ry)) * 1000
        offset_mry = np.mean(diff_ry) * 1000
        mae_no_off = np.mean(np.abs(diff_ry - np.mean(diff_ry))) * 1000
        max_mry = np.max(np.abs(diff_ry)) * 1000

        print(f"\n  Eigenvalue comparison (Gamma, {nbnd} bands):")
        print(f"    MAE:          {mae_mry:.3f} mRy")
        print(f"    Offset:       {offset_mry:.3f} mRy")
        print(f"    MAE-no-off:   {mae_no_off:.3f} mRy")
        print(f"    Max error:    {max_mry:.3f} mRy")

        # Per-band errors (first 10)
        print(f"\n  Per-band errors (mRy):")
        for ib in range(min(10, nbnd)):
            e_qe = evals_qe_occ[ib] * 13605.7  # to meV
            e_lx = evals[ib] * 13605.7
            err = diff_ry[ib] * 1000
            print(f"    band {ib:2d}: QE={evals_qe_occ[ib]:.6f} "
                  f"LX={evals[ib]:.6f} Ry  err={err:+.3f} mRy")

        results[mode] = {
            "mae_mry": mae_mry,
            "offset_mry": offset_mry,
            "mae_no_off": mae_no_off,
            "max_mry": max_mry,
            "evals": evals,
        }

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<16s} {'MAE (mRy)':>12s} {'Offset':>12s} {'MAE-no-off':>12s} {'Max':>12s}")
    for mode, r in results.items():
        print(f"{mode:<16s} {r['mae_mry']:12.3f} {r['offset_mry']:12.3f} "
              f"{r['mae_no_off']:12.3f} {r['max_mry']:12.3f}")

    print("\nInterpretation:")
    full_mae = results["full"]["mae_mry"]
    lda_mae = results["lda"]["mae_mry"]
    no_nlcc_mae = results["no_nlcc"]["mae_mry"]
    no_div_mae = results["no_gga_div"]["mae_mry"]

    if lda_mae < full_mae * 0.3:
        print("  -> LDA much better than GGA: GGA gradient correction is the problem")
    elif lda_mae > full_mae * 1.5:
        print("  -> LDA worse than GGA: error is NOT in the GGA gradient")
    else:
        print("  -> LDA and GGA similar: error may be in V_loc/V_H or base V_xc")

    if no_nlcc_mae < full_mae * 0.3:
        print("  -> No-NLCC much better: core charge handling is the problem")
    if no_div_mae < full_mae * 0.3:
        print("  -> No-GGA-div much better: divergence term is the problem")


if __name__ == "__main__":
    main()
