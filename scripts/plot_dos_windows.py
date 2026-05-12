#!/usr/bin/env python3
"""Plot DOS with CJ window indicator functions and protected region.

Two DOS modes:
  --dos-npz FILE : load KPM DOS from a pre-saved .npz (from ritz_pseudobands)
  (default)      : approximate DOS from WFN.h5 eigenvalues via Gaussian broadening

CJ window indicators I_j(E) are always reconstructed analytically from
boundary Chebyshev-Jackson coefficients — no matvecs needed.

Usage:
    python3 plot_dos_windows.py WFN.h5                   # eigenvalue DOS
    python3 plot_dos_windows.py WFN.h5 --dos-npz dos.npz # KPM DOS
    python3 plot_dos_windows.py WFN.h5 --ik 3 --M-max 2000 -o plot.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only solver-level imports — no psp, no file_io
from solvers.dos import dos_weighted_windows, compute_window_partition, DOSResult
from solvers.chebyshev import jackson_coefficients


# ═══════════════════════════════════════════════════════════════════════
#  CJ window indicator functions (analytical, no matvec)
# ═══════════════════════════════════════════════════════════════════════

def cumulative_step(E_grid: np.ndarray, boundary: float,
                    center: float, half_width: float,
                    M_max: int) -> np.ndarray:
    """CJ smoothed step C_b(E) ≈ Θ(b - E), reconstructed from Chebyshev sum."""
    g = jackson_coefficients(M_max - 1)
    b = np.clip((boundary - center) / half_width, -1.0, 1.0)
    e = np.clip((E_grid - center) / half_width, -0.9999, 0.9999)

    ns = np.arange(M_max)
    arccos_b = np.arccos(b)

    gamma = np.zeros(M_max)
    gamma[0] = 1.0 - arccos_b / np.pi
    gamma[1:] = -2.0 / (np.pi * ns[1:]) * np.sin(ns[1:] * arccos_b)

    arccos_e = np.arccos(e)
    T = np.cos(ns[:, None] * arccos_e[None, :])  # (M, n_grid)
    return (gamma * g) @ T


def window_indicators(E_grid, boundaries, center, half_width, M_max):
    """I_j(E) = C_{b_{j+1}}(E) - C_{b_j}(E) for all windows."""
    N = len(boundaries)
    steps = np.stack([cumulative_step(E_grid, b, center, half_width, M_max)
                      for b in boundaries])
    return steps[1:] - steps[:-1]  # (N_S, n_grid)


# ═══════════════════════════════════════════════════════════════════════
#  DOS from eigenvalues (Gaussian broadening)
# ═══════════════════════════════════════════════════════════════════════

def eigenvalue_dos(eigenvalues, E_grid, sigma=None):
    """Gaussian-broadened DOS from discrete eigenvalues.

    Parameters
    ----------
    eigenvalues : 1-D array of eigenvalues (all k-points, all bands)
    E_grid : energy grid
    sigma : broadening width (default: median level spacing × 2)
    """
    evals = np.sort(eigenvalues.ravel())
    if sigma is None:
        spacings = np.diff(evals)
        sigma = max(2.0 * np.median(spacings[spacings > 1e-6]), 0.05)

    rho = np.zeros_like(E_grid)
    for e in evals:
        rho += np.exp(-0.5 * ((E_grid - e) / sigma) ** 2)
    rho /= sigma * np.sqrt(2 * np.pi)
    return rho, sigma


# ═══════════════════════════════════════════════════════════════════════
#  DOS .npz I/O
# ═══════════════════════════════════════════════════════════════════════

def load_dos_npz(path):
    d = np.load(path)
    return DOSResult(
        E_grid=d["E_grid"], rho=d["rho"],
        E_min=float(d["E_min"]), E_max=float(d["E_max"]),
        center=float(d["center"]), half_width=float(d["half_width"]),
        mu_raw=d["mu_raw"], mu_damped=d["mu_damped"],
        n_moments=int(d["n_moments"]), n_random=int(d["n_random"]),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def make_plot(E_dos, rho_dos, dos_label,
              boundaries, E_det, center, half_width,
              M_max, ik, output):
    N_S = len(boundaries) - 1

    # Classify windows by det eigenstate availability
    E_avail = E_det[E_det >= boundaries[0]]
    modes = []
    for j in range(N_S):
        n_in = int(np.sum((E_avail >= boundaries[j]) & (E_avail < boundaries[j + 1])))
        modes.append("stoch" if n_in >= 1 else "CJ")

    # Window indicators on a 4k grid
    E_win = np.linspace(boundaries[0] - 1.0, boundaries[-1] + 1.0, 4000)
    print(f"Computing {N_S} CJ window indicator functions (M={M_max})...")
    ind = window_indicators(E_win, boundaries, center, half_width, M_max)

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5),
                                    sharex=True, height_ratios=[2, 3])
    fig.subplots_adjust(hspace=0.06)

    # ---- Top: DOS ----
    ax1.fill_between(E_dos, rho_dos, alpha=0.25, color="steelblue")
    ax1.plot(E_dos, rho_dos, color="steelblue", lw=0.8, label=dos_label)

    ax1.axvspan(E_dos[0] - 5, boundaries[0], alpha=0.10, color="green",
                label=f"Protected (E < {boundaries[0]:.2f} Ry)")

    for e in E_det:
        c = "green" if e < boundaries[0] else "darkorange"
        ax1.axvline(e, color=c, lw=0.3, alpha=0.4)
    n_prot = int(np.sum(E_det < boundaries[0]))
    n_avail = len(E_det) - n_prot
    ax1.axvline(np.nan, color="green", lw=1, alpha=0.6,
                label=f"Det eigenvalues (protected, n={n_prot})")
    ax1.axvline(np.nan, color="darkorange", lw=1, alpha=0.6,
                label=f"Det eigenvalues (available, n={n_avail})")

    for b in boundaries:
        ax1.axvline(b, color="gray", lw=0.3, alpha=0.3, ls="--")

    ax1.set_ylabel("DOS (arb. units)")
    ax1.set_xlim(E_dos[0], min(E_dos[-1], boundaries[-1] + 2))
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.set_title(f"DOS + CJ Window Functions  (k={ik}, "
                  f"M={M_max}, {N_S} windows)", fontsize=11)

    # ---- Bottom: window indicators ----
    cm_cj = plt.cm.Blues
    cm_st = plt.cm.Oranges

    for j in range(N_S):
        f = j / max(N_S - 1, 1)
        color = cm_st(0.35 + 0.5 * f) if modes[j] == "stoch" else cm_cj(0.25 + 0.55 * f)
        label = f"w{j+1} [{modes[j]}]" if (j < 6 or j == N_S - 1) else None
        ax2.plot(E_win, ind[j], color=color, lw=0.6, alpha=0.85, label=label)
        ax2.fill_between(E_win, ind[j], alpha=0.06, color=color)

    ax2.plot(E_win, np.sum(ind, axis=0), "k--", lw=1.2, alpha=0.5, label="Σ I_j → 1")
    ax2.axvspan(E_win[0] - 5, boundaries[0], alpha=0.10, color="green")

    for b in boundaries:
        ax2.axvline(b, color="gray", lw=0.25, alpha=0.25, ls="--")

    ax2.set_xlabel("Energy (Ry)")
    ax2.set_ylabel("Window indicator I_j(E)")
    ax2.set_ylim(-0.15, 1.4)
    ax2.axhline(0, color="gray", lw=0.4, alpha=0.3)
    ax2.axhline(1, color="gray", lw=0.4, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=6.5, ncol=2)

    plt.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved → {output}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Plot DOS + CJ window indicators")
    p.add_argument("wfn", nargs="?", default="WFN.h5", help="WFN.h5 file")
    p.add_argument("--ik", type=int, default=0, help="k-point index")
    p.add_argument("--nbnd", type=int, default=None, help="bands to show as det")
    p.add_argument("--n-windows", type=int, default=40)
    p.add_argument("--M-max", type=int, default=1500, help="Chebyshev order for CJ")
    p.add_argument("--F", type=float, default=0.10)
    p.add_argument("--E-fermi", type=float, default=0.0)
    p.add_argument("--E-max", type=float, default=None,
                   help="Spectrum upper bound (Ry). Needed for correct CJ "
                        "resolution when using eigenvalue DOS with few bands.")
    p.add_argument("--dos-npz", default=None, help="Pre-computed KPM DOS .npz")
    p.add_argument("-o", "--output", default="dos_windows.png")
    args = p.parse_args()

    import h5py

    # ── Load eigenvalues from WFN.h5 ──
    with h5py.File(args.wfn, "r") as f:
        nbnd_file = int(f["mf_header/kpoints/mnband"][()])
        nbnd = args.nbnd or nbnd_file
        E_det = np.array(f["mf_header/kpoints/el"][0, args.ik, :nbnd])
        E_all = np.array(f["mf_header/kpoints/el"][0, :, :nbnd_file])  # all k, all bands

    print(f"WFN: {nbnd} det bands, eigenvalue range "
          f"[{E_det.min():.4f}, {E_det.max():.4f}] Ry at k={args.ik}")

    # ── DOS ──
    if args.dos_npz and os.path.exists(args.dos_npz):
        dos = load_dos_npz(args.dos_npz)
        E_dos, rho_dos = dos.E_grid, dos.rho
        center, half_width = dos.center, dos.half_width
        dos_label = f"KPM DOS ({dos.n_moments} moments)"
        print(f"Loaded KPM DOS from {args.dos_npz}")
    else:
        # Broadened eigenvalue DOS — approximate but needs no matvec.
        # Use all eigenvalues across k-points for better statistics.
        E_min, E_max = float(E_all.min()) - 1.0, float(E_all.max()) + 1.0
        E_dos = np.linspace(E_min, E_max, 5000)
        rho_dos, sigma = eigenvalue_dos(E_all, E_dos)
        center = 0.5 * (E_min + E_max)
        half_width = 0.5 * (E_max - E_min)
        dos_label = f"Eigenvalue DOS (σ={sigma:.3f} Ry)"
        print(f"Eigenvalue DOS: [{E_min:.2f}, {E_max:.2f}] Ry, σ={sigma:.3f}")

    # ── Spectrum bounds for CJ calculation ──
    # The CJ resolution depends on the FULL Hilbert space bandwidth,
    # not just the eigenvalue range of the bands we have.
    # For ecutwfc=60 Ry, the bandwidth is ~62 Ry.
    E_max_spectrum = args.E_max if args.E_max is not None else center + half_width
    E_min_spectrum = center - half_width
    center_cj = 0.5 * (E_max_spectrum + E_min_spectrum)
    half_width_cj = 0.5 * (E_max_spectrum - E_min_spectrum)
    # Override center/half_width for CJ if --E-max provided
    if args.E_max is not None:
        center = center_cj
        half_width = half_width_cj

    B = 2.0 * half_width_cj
    cj_res = np.pi * B / args.M_max
    eps_cross = cj_res / args.F
    E_cross = args.E_fermi + eps_cross

    boundaries = dos_weighted_windows(
        E_dos, rho_dos, E_cross, E_dos[-1],
        galerkin_order=1, n_windows_target=args.n_windows)

    N_S = len(boundaries) - 1
    n_prot = int(np.sum(E_det < boundaries[0]))
    print(f"CJ resolution: {cj_res:.4f} Ry, ε_cross = {eps_cross:.4f} Ry")
    print(f"{N_S} windows, {n_prot} protected / {len(E_det)-n_prot} available")

    make_plot(E_dos, rho_dos, dos_label,
              boundaries, E_det, center, half_width,
              args.M_max, args.ik, args.output)


if __name__ == "__main__":
    main()
