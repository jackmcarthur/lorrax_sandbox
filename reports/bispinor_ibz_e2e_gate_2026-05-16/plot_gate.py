"""Scatter plot Σ^B_A vs Σ^B_B per (k, n) for the bispinor IBZ E2E gate."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARENT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14"
A = np.load(f"{PARENT}/sigma_b_gate_A.npz")
B = np.load(f"{PARENT}/sigma_b_gate_B.npz")
RYD_TO_EV = float(A["ryd_to_ev"])

sa = A["sig_x_b"] * RYD_TO_EV
sb = B["sig_x_b"] * RYD_TO_EV
da = np.real(np.diagonal(sa, axis1=1, axis2=2))   # (nk, nb)
db = np.real(np.diagonal(sb, axis1=1, axis2=2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter: A vs B (1:1 line + 1 meV band)
amin = da.min(); amax = da.max()
ax1.plot([amin, amax], [amin, amax], 'k-', lw=0.5, label='y = x')
ax1.scatter(da.ravel(), db.ravel(), s=8, alpha=0.6)
ax1.set_xlabel(r'$\Sigma^B_A[k, n]$ (no IBZ)  [eV]')
ax1.set_ylabel(r'$\Sigma^B_B[k, n]$ (IBZ cascade)  [eV]')
ax1.set_title(r'$\Sigma^B$ scatter: run A vs run B per $(k, n)$')
ax1.legend()
ax1.grid(alpha=0.3)

# Histogram of diff in meV
diff = (db - da).ravel() * 1000   # meV
ax2.hist(diff, bins=40)
ax2.axvline(0, color='black', lw=0.5)
ax2.axvline(+1, color='red', lw=0.5, ls='--', label='±1 meV gate')
ax2.axvline(-1, color='red', lw=0.5, ls='--')
ax2.set_xlabel(r'$\Sigma^B_B - \Sigma^B_A$ per $(k, n)$  [meV]')
ax2.set_ylabel('count')
maxabs = np.abs(diff).max()
ax2.set_title(f'Per-$(k,n)$ diff   max |Δ| = {maxabs:.4f} meV')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/pscratch/sd/j/jackm/lorrax_sandbox/reports/bispinor_ibz_e2e_gate_2026-05-16/sigma_b_gate_scatter.png",
            dpi=150)
print("saved plot")
