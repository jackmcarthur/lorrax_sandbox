"""LORRAX Davidson vs BGW (full diag) absorption comparison — Gaussian broadening.

Builds the lowest-100-state absorption spectrum from each code's
``eigenvalues*.dat`` (i.e. broadens the per-state oscillator strengths
that BGW writes from full diag and LORRAX writes from Davidson). For
cubic Si the three Cartesian polarisations should agree, so we plot
each LORRAX b1/b2/b3 plus the average vs BGW b1.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bse.eigvals_to_eps2 import read_bgw_eigvals, compute_eps2

BGW = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/eigenvalues.dat"
LOR_b = ["eigenvalues_lorrax_davidson_b1.dat",
         "eigenvalues_lorrax_davidson_b2.dat",
         "eigenvalues_lorrax_davidson_b3.dat"]

omegas_eV = np.linspace(2.5, 5.5, 6001)
N_TRUNC = 100

def eps2_of(path, eta_eV):
    E, f, V, ns, nspinor, _ = read_bgw_eigvals(path)
    return compute_eps2(E, f, V, ns, nspinor,
                        omegas_eV=omegas_eV, eta_eV=eta_eV,
                        n_max=N_TRUNC, kernel="gaussian"), E[:N_TRUNC], f[:N_TRUNC]


def plot_panel(ax, eta_eV):
    e_bgw, E_b, f_b = eps2_of(BGW, eta_eV)
    e_lor = []
    for p in LOR_b:
        e, _, _ = eps2_of(p, eta_eV)
        e_lor.append(e)
    e_lor = np.array(e_lor)
    e_avg = e_lor.mean(0)

    ax.plot(omegas_eV, e_bgw,          label=f"BGW (first 100)         peak {e_bgw.max():.1f} @ {omegas_eV[e_bgw.argmax()]:.3f} eV",
            color="black", lw=1.4)
    for j, e in enumerate(e_lor):
        ax.plot(omegas_eV, e, lw=0.7, alpha=0.5,
                label=f"LORRAX b{j+1} (100)   peak {e.max():.1f} @ {omegas_eV[e.argmax()]:.3f} eV")
    ax.plot(omegas_eV, e_avg, label=f"LORRAX avg(b1,b2,b3)  peak {e_avg.max():.1f} @ {omegas_eV[e_avg.argmax()]:.3f} eV",
            color="C3", lw=1.2)
    ax.set_xlabel(r"$\omega$ (eV)")
    ax.set_ylabel(r"$\varepsilon_2(\omega)$")
    ax.set_title(f"Si 4×4×4, BSE 8×8, Gaussian σ = {eta_eV} eV, lowest 100 states")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(omegas_eV[0], omegas_eV[-1])
    return e_bgw, e_lor, e_avg


fig, axes = plt.subplots(2, 1, figsize=(11, 9))
for ax, sigma in zip(axes, (0.15, 0.05)):
    plot_panel(ax, sigma)
plt.tight_layout()
plt.savefig("davidson_vs_bgw_compare.png", dpi=120)
print("Wrote davidson_vs_bgw_compare.png")

# Print numerical summary
for sigma in (0.15, 0.05):
    e_bgw, e_lor, e_avg = plot_panel(plt.figure().gca(), sigma)
    print(f"\nσ = {sigma} eV:")
    print(f"  BGW                peak ε₂ = {e_bgw.max():.2f} at {omegas_eV[e_bgw.argmax()]:.3f} eV")
    for j, e in enumerate(e_lor):
        print(f"  LORRAX b{j+1}            peak ε₂ = {e.max():.2f} at {omegas_eV[e.argmax()]:.3f} eV  "
              f"(LORRAX/BGW = {e.max()/e_bgw.max():.3f})")
    print(f"  LORRAX avg(b)      peak ε₂ = {e_avg.max():.2f} at {omegas_eV[e_avg.argmax()]:.3f} eV  "
          f"(avg/BGW = {e_avg.max()/e_bgw.max():.3f})")

# Also print eigval comparison at top
E_bgw = read_bgw_eigvals(BGW)[0][:20]
E_lor = read_bgw_eigvals(LOR_b[0])[0][:20]
print("\nLowest 20 eigvals (eV)  BGW vs LORRAX Davidson  Δ (meV)")
for i in range(20):
    print(f"  S={i:2d}  {E_bgw[i]:.6f}   {E_lor[i]:.6f}   {(E_lor[i]-E_bgw[i])*1e3:+8.3f}")
