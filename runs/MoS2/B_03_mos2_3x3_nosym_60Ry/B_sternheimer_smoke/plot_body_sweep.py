"""Plot body-element vs head/wing convergence."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Body sweep (G_pert ≠ 0)
B = np.load('body_sweep_60Ry/sweep.npz')
N_b = B['N_list']
chi_full_b = B['chi_full']
chi_sos_b = B['chi_sos']
diag_idx = int(B['diag_idx'])
g0_idx   = int(B['g0_idx'])

# Head sweep (G_pert = 0)
H = np.load('cond_sweep_60Ry/sweep.npz')
N_h = H['N_list']
chi_ref_h = H['chi_ref']
chi_sos_h = H['chi_sos']
Gint_h    = H['Gint']
qG2_h     = H['qG_sq']
is_head = np.all(Gint_h == 0, axis=-1)
i_head_h = int(np.where(is_head)[0][0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axA, axB = axes

# ── (A) magnitude convergence |chi_N| / |chi_full| ──
# Head with V_pert=1, G'=0 (the original head)
axA.plot(N_h, np.abs(chi_sos_h[:, i_head_h]) / np.abs(chi_ref_h[i_head_h]),
         'o-', color='red', lw=2.4, ms=9,
         label=f"head G'=0, V_pert=1  (|χ_full|={np.abs(chi_ref_h[i_head_h]):.2e})")

# Body diagonal: G' = G_pert with V_pert = exp(i G_pert r)
axA.plot(N_b, np.abs(chi_sos_b[:, diag_idx]) / np.abs(chi_full_b[diag_idx]),
         's-', color='black', lw=2.4, ms=9,
         label=f"body diag G'=G_pert={tuple(B['G_pert'])},  V_pert=e^(iG_pert·r)\n"
               f"   |q+G_pert|²={float(B['qG_pert_sq']):.2f}, "
               f"|χ_full|={np.abs(chi_full_b[diag_idx]):.2e}")

# Body off-diagonal: G' = 0 with same perturbation (for comparison)
axA.plot(N_b, np.abs(chi_sos_b[:, g0_idx]) / np.abs(chi_full_b[g0_idx]),
         '^-', color='tab:blue', lw=1.8, ms=8, alpha=0.7,
         label=f"body off-diag G'=0, V_pert=e^(iG_pert·r)\n"
               f"   |χ_full|={np.abs(chi_full_b[g0_idx]):.2e}")

axA.axhline(1.0, color='k', lw=0.5)
axA.set_xlabel("N_cond")
axA.set_ylabel(r"$|\chi_N| \,/\, |\chi_\mathrm{full}|$")
axA.set_title("Magnitude convergence — head vs body diagonal vs body off-diag")
axA.set_xscale('log')
axA.legend(fontsize=9, loc='lower right')
axA.grid(alpha=0.3)
axA.set_ylim(0, 1.7)

# ── (B) Real and Imag parts separately for the body diagonal ──
axB.plot(N_b, chi_sos_b[:, diag_idx].real / chi_full_b[diag_idx].real,
         'o-', color='tab:green', lw=2.0, label="body diag Re / Re_full")
axB.plot(N_b, chi_sos_b[:, diag_idx].imag / chi_full_b[diag_idx].imag,
         's-', color='tab:purple', lw=2.0, label="body diag Im / Im_full")
axB.axhline(1.0, color='k', lw=0.5)
axB.set_xlabel("N_cond")
axB.set_ylabel("Re(χ_N)/Re(χ_full),  Im(χ_N)/Im(χ_full)")
axB.set_title(f"Body diagonal Re/Im components  (G_pert={tuple(B['G_pert'])}, "
              f"|q+G_pert|²={float(B['qG_pert_sq']):.2f})")
axB.set_xscale('log')
axB.legend(fontsize=9, loc='lower right')
axB.grid(alpha=0.3)

fig.suptitle("Cond-band convergence  q crys=(1/3,1/3,0)  60 Ry  —  body vs head",
             fontsize=11)
plt.tight_layout()
out = 'body_sweep_60Ry/body_vs_head_convergence.png'
plt.savefig(out, dpi=140)
print(f"  Saved {out}")

# ── Tabulate ──
print("\n══ Magnitude ratios at N=150 ══")
print(f"  Original head  G'=0, V_pert=1:                "
      f"|χ_N|/|χ_full| = {np.abs(chi_sos_h[-1, i_head_h]) / np.abs(chi_ref_h[i_head_h]):.4f}")
print(f"  Body diag      G'=G_pert, V_pert=exp(iG_pert·r):"
      f" |χ_N|/|χ_full| = {np.abs(chi_sos_b[-1, diag_idx]) / np.abs(chi_full_b[diag_idx]):.4f}")
print(f"  Body off-diag  G'=0, V_pert=exp(iG_pert·r):     "
      f"|χ_N|/|χ_full| = {np.abs(chi_sos_b[-1, g0_idx]) / np.abs(chi_full_b[g0_idx]):.4f}")
print()
print("══ Re / Im of body diagonal ══")
for i, N in enumerate(N_b):
    print(f"  N={N:>4d}  Re(χ_N)/Re(χ_full) = "
          f"{chi_sos_b[i, diag_idx].real/chi_full_b[diag_idx].real:.4f}  "
          f"Im(χ_N)/Im(χ_full) = "
          f"{chi_sos_b[i, diag_idx].imag/chi_full_b[diag_idx].imag:.4f}")
