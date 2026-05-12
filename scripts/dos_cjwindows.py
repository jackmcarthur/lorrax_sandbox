#!/usr/bin/env python3
"""Plot KPM DOS with CJ window indicator functions.

Builds H at one k-point (same setup as run_nscf), computes the KPM DOS
via solvers.dos.compute_dos, then reconstructs CJ window indicators
analytically from solvers.chebyshev.jackson_coefficients.

Run from a directory containing a .save/ and *.upf:
    lxrun python3 -u dos_cjwindows.py
    lxrun python3 -u dos_cjwindows.py --ik 0 --M-max 1500 --n-windows 40
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from file_io import CrystalData, WFNReader
from psp.pseudos import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc, build_V_scf
from psp.h_dft import setup_H_k_from_kvec, make_apply_H
from psp.gvec_utils import build_master_gvec_list, compute_ngkmax
import psp.vnl_ops as vnl_ops

from solvers.dos import compute_dos, dos_weighted_windows, compute_window_partition
from solvers.chebyshev import jackson_coefficients


# ── CJ window indicators (analytical, no matvec) ──────────────────────

def cj_window_indicators(E_grid, boundaries, center, half_width, M_max):
    """Compute I_j(E) = C_{b_{j+1}}(E) - C_{b_j}(E) for all windows.

    Each C_b is the Jackson-damped Chebyshev step function at boundary b.
    """
    g = jackson_coefficients(M_max - 1)  # (M_max,)
    ns = np.arange(M_max)
    e = np.clip((E_grid - center) / half_width, -0.9999, 0.9999)
    arccos_e = np.arccos(e)
    T = np.cos(ns[:, None] * arccos_e[None, :])  # (M_max, n_grid)

    steps = np.zeros((len(boundaries), len(E_grid)))
    for j, bnd in enumerate(boundaries):
        b = np.clip((bnd - center) / half_width, -1.0, 1.0)
        gamma = np.zeros(M_max)
        gamma[0] = 1.0 - np.arccos(b) / np.pi
        gamma[1:] = -2.0 / (np.pi * ns[1:]) * np.sin(ns[1:] * np.arccos(b))
        steps[j] = (gamma * g) @ T

    return steps[1:] - steps[:-1]  # (N_S, n_grid)


# ── H setup (follows run_nscf._build_potentials + _setup_kgrid) ───────

def build_apply_H_flat(save_dir, ik=0):
    """Build flat matvec (nspinor*ngkmax,) → same at k-point ik."""
    crystal = CrystalData.from_qe_save(save_dir)
    pseudos = load_pseudopotentials(".")
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    truncation_2d = crystal.assume_isolated == "2D"

    V_loc, rho_core, rho_core_G = build_ionic_and_core(
        crystal, pseudos, fft_grid, truncation_2d=truncation_2d)
    rho_val = jnp.asarray(crystal.load_charge_density()[0], dtype=jnp.float64)
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    G_cart = build_G_cart(nx, ny, nz,
                          float(crystal.blat) * np.asarray(crystal.bvec, dtype=float))
    V_H, V_xc = compute_V_H_and_V_xc(
        rho_val, rho_core, rho_core_G, G_cart,
        jnp.asarray(crystal.bdot, dtype=jnp.float64),
        jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat,
        truncation_2d=truncation_2d)
    V_scf = build_V_scf(V_loc, V_H, V_xc)
    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)

    wfn = WFNReader("WFN.h5")
    kpoints = wfn.kpoints
    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)
    ngkmax = compute_ngkmax(kpoints, bdot, crystal.ecutwfc, crystal.fft_grid)

    H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup,
                               crystal, None, V_loc_r=V_loc, ngkmax=ngkmax)
    apply_H = make_apply_H(H_k)

    def apply_H_flat(x):
        return apply_H(x.reshape(1, nspinor, ngkmax)).reshape(-1)

    return apply_H_flat, nspinor * ngkmax, wfn, ngkmax


# ── Plot ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot KPM DOS + CJ window indicators")
    p.add_argument("--ik", type=int, default=0)
    p.add_argument("--nbnd", type=int, default=None, help="Bands to mark as det")
    p.add_argument("--n-windows", type=int, default=40)
    p.add_argument("--M-max", type=int, default=1500)
    p.add_argument("--F", type=float, default=0.10)
    p.add_argument("--n-moments", type=int, default=500)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--E-fermi", type=float, default=0.0)
    p.add_argument("-o", "--output", default="dos_cjwindows.png")
    args = p.parse_args()

    # ── Find .save ──
    save_dirs = [d for d in os.listdir(".") if d.endswith(".save") and os.path.isdir(d)]
    if not save_dirs:
        raise SystemExit("No .save directory found in cwd.")

    # ── Build H, compute DOS ──
    t0 = time.perf_counter()
    apply_H, dim, wfn, ngkmax = build_apply_H_flat(save_dirs[0], ik=args.ik)
    print(f"H setup: {time.perf_counter()-t0:.1f}s  (dim={dim})")

    dos = compute_dos(apply_H, dim, n_moments=args.n_moments,
                      n_random=args.n_random, seed=0, verbose=True)

    # ── Eigenvalues ──
    nbnd = args.nbnd or wfn.nbands
    E_det = np.array(wfn.energies[0, args.ik, :nbnd])

    # ── Window boundaries ──
    B = 2.0 * dos.half_width
    cj_res = np.pi * B / args.M_max
    eps_cross = cj_res / args.F
    E_cross = args.E_fermi + eps_cross

    boundaries = dos_weighted_windows(
        dos.E_grid, dos.rho, E_cross, dos.E_max,
        galerkin_order=1, n_windows_target=args.n_windows)
    N_S = len(boundaries) - 1

    # Classify windows
    E_avail = E_det[E_det >= boundaries[0]]
    modes = []
    for j in range(N_S):
        n_in = int(np.sum((E_avail >= boundaries[j]) & (E_avail < boundaries[j + 1])))
        modes.append("stoch" if n_in >= 1 else "CJ")

    n_prot = int(np.sum(E_det < boundaries[0]))
    print(f"CJ resolution: {cj_res:.4f} Ry, ε_cross = {eps_cross:.4f} Ry")
    print(f"{N_S} windows  ({modes.count('stoch')} stoch, {modes.count('CJ')} CJ), "
          f"{n_prot} protected, {len(E_det)-n_prot} available")

    # ── CJ indicators ──
    E_win = np.linspace(dos.E_min, dos.E_max, 4000)
    print(f"Computing {N_S} CJ window indicators (M={args.M_max})...")
    ind = cj_window_indicators(E_win, boundaries, dos.center, dos.half_width, args.M_max)

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5),
                                    sharex=True, height_ratios=[2, 3])
    fig.subplots_adjust(hspace=0.06)

    # Top: KPM DOS
    ax1.fill_between(dos.E_grid, dos.rho, alpha=0.25, color="steelblue")
    ax1.plot(dos.E_grid, dos.rho, color="steelblue", lw=0.8,
             label=f"KPM DOS ({dos.n_moments} moments)")
    ax1.axvspan(dos.E_min - 5, boundaries[0], alpha=0.10, color="green",
                label=f"Protected (E < {boundaries[0]:.2f} Ry)")
    for e in E_det:
        ax1.axvline(e, color="green" if e < boundaries[0] else "darkorange",
                    lw=0.3, alpha=0.4)
    ax1.axvline(np.nan, color="green", lw=1, alpha=0.6,
                label=f"Det eigenvalues (protected, n={n_prot})")
    ax1.axvline(np.nan, color="darkorange", lw=1, alpha=0.6,
                label=f"Det eigenvalues (available, n={len(E_det)-n_prot})")
    for b in boundaries:
        ax1.axvline(b, color="gray", lw=0.3, alpha=0.3, ls="--")
    ax1.set_ylabel("DOS (states / Ry)")
    ax1.set_xlim(dos.E_min, dos.E_max)
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.set_title(f"KPM DOS + CJ Window Functions  (k={args.ik}, "
                  f"M={args.M_max}, {N_S} windows)", fontsize=11)

    # Bottom: window indicators
    cm_cj = plt.cm.Blues
    cm_st = plt.cm.Oranges
    for j in range(N_S):
        f = j / max(N_S - 1, 1)
        color = cm_st(0.35 + 0.5*f) if modes[j] == "stoch" else cm_cj(0.25 + 0.55*f)
        label = f"w{j+1} [{modes[j]}]" if (j < 6 or j == N_S - 1) else None
        ax2.plot(E_win, ind[j], color=color, lw=0.6, alpha=0.85, label=label)
        ax2.fill_between(E_win, ind[j], alpha=0.06, color=color)

    ax2.plot(E_win, np.sum(ind, axis=0), "k--", lw=1.2, alpha=0.5, label="Σ I_j → 1")
    ax2.axvspan(dos.E_min - 5, boundaries[0], alpha=0.10, color="green")
    for b in boundaries:
        ax2.axvline(b, color="gray", lw=0.25, alpha=0.25, ls="--")

    ax2.set_xlabel("Energy (Ry)")
    ax2.set_ylabel("Window indicator I_j(E)")
    ax2.set_ylim(-0.15, 1.4)
    ax2.axhline(0, color="gray", lw=0.4, alpha=0.3)
    ax2.axhline(1, color="gray", lw=0.4, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=6.5, ncol=2)

    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
