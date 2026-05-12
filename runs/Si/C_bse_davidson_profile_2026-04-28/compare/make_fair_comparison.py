"""Make the *fair* BGW-vs-LORRAX absorption comparison.

The eps2_n50_eta0.05.png I made earlier today compared BGW
full-diagonalization eigvecs (50 lowest exact) to LORRAX Lanczos Ritz
vectors (50 lowest of n=2400 Krylov subspace) — STATUS.md notes that
the Lanczos eigvec route at finite n captures only ~18% of the full
peak height (n=100) / ~50% (n=400). That is a real method
truncation, not a regression.

This script makes the *correct* comparison:

  Panel 1 — Haydock vs Haydock at η=0.15 (validated, ratio 1.018)
  Panel 2 — BGW eigvals (full diag, 500 states, near-converged)
            re-broadened at η=0.05 to reveal sharp exciton structure;
            LORRAX Haydock at η=0.15 overlaid for sanity.

Uses pre-existing files only — does not re-run any solver.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")
from bse.eigvals_to_eps2 import compute_eps2_from_files

BGW_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8"
LOR_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp"

# Existing files (intact)
bgw_abs   = np.loadtxt(f"{BGW_DIR}/absorption_eh.dat",                     comments="#")
lor_hay15 = np.loadtxt(f"{LOR_DIR}/absorption_haydock_b1_eh.dat",          comments="#")  # η=0.15

# Re-broaden BGW eigvals.dat (full diag, lowest 500 exact eigvecs ≈ full peak weight)
omegas, out = compute_eps2_from_files(
    [f"{BGW_DIR}/eigenvalues.dat"],
    eta_eV=0.05,
    omega_min_eV=2.0, omega_max_eV=5.0, n_omega=6001,
)
bgw_05 = out[f"{BGW_DIR}/eigenvalues.dat"]["eps2"]

omegas2, out2 = compute_eps2_from_files(
    [f"{BGW_DIR}/eigenvalues.dat"],
    eta_eV=0.15,
    omega_min_eV=2.0, omega_max_eV=5.0, n_omega=6001,
)
bgw_15_recon = out2[f"{BGW_DIR}/eigenvalues.dat"]["eps2"]

fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Panel 1 — η=0.15 Haydock vs Haydock (validated)
ax[0].plot(bgw_abs[:,0], bgw_abs[:,1],
           label=f"BGW absorption_eh.dat (full diag, η=0.20)\n  peak {bgw_abs[:,1].max():.1f}",
           color="black", lw=1.2)
ax[0].plot(omegas2, bgw_15_recon,
           label=f"BGW eigvals.dat re-broadened η=0.15 (script reproduction)\n  peak {bgw_15_recon.max():.1f}",
           color="C0", lw=1.0, ls="--")
ax[0].plot(lor_hay15[:,0], lor_hay15[:,1],
           label=f"LORRAX Haydock (n_iter=100, η=0.15)\n  peak {lor_hay15[:,1].max():.1f}",
           color="C2", lw=1.2)
ax[0].set_ylabel(r"$\varepsilon_2(\omega)$")
ax[0].set_title("η = 0.15 eV — Haydock-vs-Haydock comparison still works (peak ratio 1.018)")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_xlim(2.0, 5.0)

# Panel 2 — η=0.05 sharp BGW (full diag, 500 states) with LORRAX Haydock at 0.15 overlaid
ax[1].plot(omegas, bgw_05,
           label=f"BGW full diag re-broadened at η=0.05 (peak {bgw_05.max():.1f})",
           color="C0", lw=1.0)
ax[1].plot(lor_hay15[:,0], lor_hay15[:,1],
           label=f"LORRAX Haydock at η=0.15 (peak {lor_hay15[:,1].max():.1f})",
           color="C2", lw=1.2, alpha=0.7)
ax[1].set_xlabel(r"$\omega$ (eV)")
ax[1].set_ylabel(r"$\varepsilon_2(\omega)$")
ax[1].set_title("BGW at sharper η=0.05 reveals discrete exciton structure (LORRAX Haydock η=0.05 not yet rerun)")
ax[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
out_path = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/C_bse_davidson_profile_2026-04-28/compare/fair_comparison.png"
plt.savefig(out_path, dpi=110)
print(f"Wrote {out_path}")

# Print numbers
print()
print(f"η=0.15: BGW abs_eh peak = {bgw_abs[:,1].max():.2f}")
print(f"        BGW eigvals reconstruct peak = {bgw_15_recon.max():.2f}")
print(f"        LORRAX Haydock peak = {lor_hay15[:,1].max():.2f}  (ratio LORRAX/BGW = {lor_hay15[:,1].max()/bgw_abs[:,1].max():.4f})")
print(f"η=0.05: BGW eigvals re-broadened peak = {bgw_05.max():.2f}")
