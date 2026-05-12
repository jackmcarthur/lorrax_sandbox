"""Plot χ_{G'0}(q=4) head + wings convergence vs n_cond_bands.

Reads cond_sweep/sweep.npz produced by sweep_cond_band_convergence.py.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('cond_sweep/sweep.npz')
N_list = d['N_list']                      # (nN,)
chi_ref = d['chi_ref']                    # (ng,)
chi_sos = d['chi_sos']                    # (nN, ng)
qG_sq   = d['qG_sq']                      # (ng,) cartesian |q+G'|²
Gint    = d['Gint']                       # (ng, 3)
ng      = int(d['ng_out'])

# Identify G'=0 (the actual head) — smallest |q+G'|² with G'=0
is_head = np.all(Gint == 0, axis=-1)        # (ng,) bool — exactly G'=0
i_head = int(np.where(is_head)[0][0])
print(f"  G'=0 head at index {i_head}, |q|² = {qG_sq[i_head]:.4f}")

# Group G' by unique |q+G'|² (with tiny tolerance for round-off)
def _groupby(qsq, atol=1e-3):
    groups = []
    used = np.zeros(len(qsq), dtype=bool)
    for i, q in enumerate(qsq):
        if used[i]: continue
        idx = np.where(np.abs(qsq - q) < atol)[0]
        groups.append(idx)
        used[idx] = True
    return groups

groups = _groupby(qG_sq)
print(f"  {len(groups)} unique |q+G'|² groups out of {ng} G'")

# Group-averaged |chi| and group label
group_abs_ref = np.array([np.mean(np.abs(chi_ref[g])) for g in groups])
group_abs_sos = np.array([[np.mean(np.abs(chi_sos[i, g])) for g in groups]
                           for i in range(len(N_list))])
group_q  = np.array([qG_sq[g[0]] for g in groups])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the *actual* head (G'=0) separately + the wing groups
# Wings: drop the head G' from its own group and re-aggregate.
wing_groups = []
wing_q = []
for i, g in enumerate(groups):
    g_no_head = g[~is_head[g]]
    if g_no_head.size > 0:
        wing_groups.append(g_no_head)
        wing_q.append(qG_sq[g_no_head[0]])
wing_q = np.array(wing_q)
wabs_ref = np.array([np.mean(np.abs(chi_ref[g])) for g in wing_groups])
wabs_sos = np.array([[np.mean(np.abs(chi_sos[i, g])) for g in wing_groups]
                      for i in range(len(N_list))])

# Head (single G')
head_ref = np.abs(chi_ref[i_head])
head_sos = np.abs(chi_sos[:, i_head])

# ── (A) absolute |χ| vs N: head bold, wings light ──
cmap = plt.colormaps.get_cmap('viridis').resampled(len(wing_q))
axA.plot(N_list, head_sos, 'o-', color='red', lw=2.4, ms=8,
         label=f"G'=0 head  (full={head_ref:.3e})", zorder=5)
axA.axhline(head_ref, color='red', ls='--', lw=1.5, alpha=0.7)
for i, q in enumerate(wing_q):
    axA.plot(N_list, wabs_sos[:, i], 'o-', color=cmap(i), alpha=0.7,
             label=f"|q+G'|²={q:.2f} ({len(wing_groups[i])}× G')")
    axA.axhline(wabs_ref[i], ls='--', color=cmap(i), alpha=0.4)
axA.set_xlabel("N_cond  (explicit cond-N projector size)")
axA.set_ylabel(r"$|\chi_{G'0}(q)|$ (Ry$^{-1}$)")
axA.set_title("|χ_{G'0}(q=(1/3,1/3,0))| vs N_cond")
axA.legend(fontsize=8, loc='right')
axA.grid(alpha=0.3)

# ── (B) fractional convergence ratio ──
axB.plot(N_list, head_sos / head_ref, 'o-', color='red', lw=2.4, ms=8,
         label="G'=0 head", zorder=5)
for i, q in enumerate(wing_q):
    if wabs_ref[i] < 1e-5: continue
    axB.plot(N_list, wabs_sos[:, i] / wabs_ref[i], 'o-',
             color=cmap(i), alpha=0.7, label=f"|q+G'|²={q:.2f}")
axB.axhline(1.0, color='k', lw=0.5)
axB.set_xlabel("N_cond")
axB.set_ylabel(r"$|\chi_N| \,/\, |\chi_\mathrm{full}|$")
axB.set_title("Fractional convergence  (|χ_N| / |χ_full|)")
axB.legend(fontsize=8, loc='lower right')
axB.grid(alpha=0.3)
axB.set_ylim(0, 1.4)

plt.tight_layout()
out = 'cond_sweep/cond_sweep_convergence.png'
plt.savefig(out, dpi=140)
print(f"  Saved {out}")

# ── Tabulate exact vs N=50 head/wings rel-error (head separated from wings) ──
print("\n  Convergence at N=50 (out of 56 cond bands available):")
i_N50 = list(N_list).index(50)
print(f"    HEAD  G'=0      |χ_full|={head_ref:.3e}   "
      f"|χ_N50|={head_sos[i_N50]:.3e}   "
      f"rel-err = {abs(1 - head_sos[i_N50]/head_ref):.2%}")
print(f"    {'-'*65}")
for i, q in enumerate(wing_q):
    if wabs_ref[i] < 1e-5: continue
    rel = abs(1.0 - wabs_sos[i_N50, i] / wabs_ref[i])
    print(f"    wing  |q+G'|²={q:.3f}  ({len(wing_groups[i])}× G')  "
          f"|χ_full|={wabs_ref[i]:.3e}   rel-err = {rel:.2%}")
