"""60 Ry vs 30 Ry side-by-side cond-band convergence plot.

Reads:
  cond_sweep_60Ry/sweep.npz                                  — this run
  ../../02_mos2_3x3_nosym/B_sternheimer_smoke/cond_sweep/sweep.npz  — 30 Ry

Plots fractional convergence (|χ_N| / |χ_full|) for head + wings at
each cutoff.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D60 = np.load('cond_sweep_60Ry/sweep.npz')
D30 = np.load('../../02_mos2_3x3_nosym/B_sternheimer_smoke/cond_sweep/sweep.npz')

def _summary(d, label):
    N_list  = d['N_list']
    chi_ref = d['chi_ref']
    chi_sos = d['chi_sos']
    qG_sq   = d['qG_sq']
    Gint    = d['Gint']
    is_head = np.all(Gint == 0, axis=-1)
    i_head  = int(np.where(is_head)[0][0])

    # Group wings by unique |q+G'|² (exclude the head G'=0 itself)
    groups = []
    qs     = []
    used = is_head.copy()      # mark head so it doesn't enter wing groups
    for i, q in enumerate(qG_sq):
        if used[i]: continue
        idx = np.where(np.abs(qG_sq - q) < 1e-3)[0]
        idx = idx[~is_head[idx]]
        if idx.size == 0: continue
        groups.append(idx); qs.append(q)
        used[idx] = True
    qs = np.array(qs)
    wabs_ref = np.array([np.mean(np.abs(chi_ref[g])) for g in groups])
    wabs_sos = np.array([[np.mean(np.abs(chi_sos[i, g])) for g in groups]
                          for i in range(len(N_list))])
    head_ref = np.abs(chi_ref[i_head])
    head_sos = np.abs(chi_sos[:, i_head])

    print(f"  [{label}]  ecutwfc={float(d['ecutwfc']) if 'ecutwfc' in d.files else 30.0:.0f} Ry  "
          f"N_max={N_list[-1]}  |head_full|={head_ref:.3e}")
    return dict(N=N_list, head_ref=head_ref, head_sos=head_sos,
                wabs_ref=wabs_ref, wabs_sos=wabs_sos, q_groups=qs)

s30 = _summary(D30, '30 Ry')
s60 = _summary(D60, '60 Ry')

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, s, ecut in [(axes[0], s30, 30), (axes[1], s60, 60)]:
    cmap = plt.colormaps.get_cmap('viridis').resampled(len(s['q_groups']))
    ax.plot(s['N'], s['head_sos'] / s['head_ref'], 'o-',
             color='red', lw=2.4, ms=8, label="G'=0 head", zorder=5)
    for i, q in enumerate(s['q_groups']):
        if s['wabs_ref'][i] < 1e-5: continue
        label = f"|q+G'|²={q:.2f}"
        ax.plot(s['N'], s['wabs_sos'][:, i] / s['wabs_ref'][i],
                 'o-', color=cmap(i), alpha=0.7, label=label)
    ax.axhline(1.0, color='k', lw=0.5)
    ax.set_xlabel("N_cond")
    ax.set_title(f"{ecut} Ry  (mnband-n_occ ≈ {56 if ecut==30 else 174})")
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.4)
    ax.legend(fontsize=7, loc='lower right', ncol=2)

axes[0].set_ylabel(r"$|\chi_N| \,/\, |\chi_\mathrm{full}|$")
fig.suptitle("Cond-band convergence  q crys=(1/3,1/3,0)  —  30 Ry vs 60 Ry plane-wave cutoff",
             fontsize=11)
plt.tight_layout()
out = 'cond_sweep_60Ry/cond_sweep_30vs60.png'
plt.savefig(out, dpi=140)
print(f"\n  Saved {out}")

# ── Tabulate at high-N anchors ──
def _row(s, label, N_target, group_idx=None):
    i = int(np.argmin(np.abs(s['N'] - N_target)))
    head_pct = s['head_sos'][i] / s['head_ref']
    return label, int(s['N'][i]), head_pct

print("\n══ Head fractional convergence at comparable N ══")
print(f"  30 Ry, N=50 (max=56):  head/full = {s30['head_sos'][-1]/s30['head_ref']:.4f}")
print(f"  60 Ry, N=64:           head/full = {s60['head_sos'][np.argmin(np.abs(s60['N'] - 64))]/s60['head_ref']:.4f}")
print(f"  60 Ry, N=128:          head/full = {s60['head_sos'][np.argmin(np.abs(s60['N'] - 128))]/s60['head_ref']:.4f}")
print(f"  60 Ry, N=150 (max=174):head/full = {s60['head_sos'][-1]/s60['head_ref']:.4f}")
print()
print("══ Same |q+G'|²=2.04 wing ══")
i_30 = int(np.argmin(np.abs(s30['q_groups'] - 2.04)))
i_60 = int(np.argmin(np.abs(s60['q_groups'] - 2.04)))
print(f"  30 Ry, |q+G'|²={s30['q_groups'][i_30]:.3f},  |χ_full|={s30['wabs_ref'][i_30]:.3e},  N=50: rel={s30['wabs_sos'][-1, i_30]/s30['wabs_ref'][i_30]:.4f}")
print(f"  60 Ry, |q+G'|²={s60['q_groups'][i_60]:.3f},  |χ_full|={s60['wabs_ref'][i_60]:.3e},  N=150: rel={s60['wabs_sos'][-1, i_60]/s60['wabs_ref'][i_60]:.4f}")
