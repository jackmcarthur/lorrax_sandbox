"""Orbital-magnetization-resolved bandstructure of monolayer CrI3 (FM), Gamma-M-K-Gamma.
Each (n,k) point colored by its per-state orbital moment m_z along the spin axis:
red = parallel to spin, blue = antiparallel. Diverging bwr, symmetric, robust-clipped."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
REP = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16"

d = np.load(f"{REP}/orbmag_bandpath.npz", allow_pickle=True)
E = d['E']; MZ = d['MZ_spin']; dist = d['dist']; node = d['node_idx']; lab = d['labels']
VBM = float(d['VBM']); nocc = int(d['nocc'])
E = E - VBM                                              # zero to VBM
nk, nb = E.shape
labmap = {'G': r'$\Gamma$', 'M': 'M', 'K': 'K'}
ticks = [labmap.get(str(x), str(x)) for x in lab]

# Fixed physical color scale. The per-state moment is O(0.01-0.08) muB away from
# crossings; at (avoided) band crossings 1/(En-Em)^2 spikes by orders of magnitude
# (real feature of the SOS per-state decomposition). Saturate there rather than let
# the spikes swamp the scale. vmax ~ bulk p99 of the non-divergent states.
vmax = 0.04
norm = plt.Normalize(-vmax, vmax); cmap = plt.get_cmap('bwr')

fig, ax = plt.subplots(figsize=(7.2, 5.6))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)        # for the colorbar
for n in range(nb):                                     # one colored line per band
    pts = np.array([dist, E[:, n]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c = 0.5 * (MZ[:-1, n] + MZ[1:, n])                  # segment color = mean of endpoints
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.2)
    lc.set_array(c); ax.add_collection(lc)

ax.axhline(0, color='0.4', lw=0.8, ls='--')             # VBM
for x in dist[node]: ax.axvline(x, color='0.7', lw=0.7)
ax.set_xticks(dist[node]); ax.set_xticklabels(ticks, fontsize=12)
ax.set_xlim(dist.min(), dist.max()); ax.set_ylim(-6, 3.5)
ax.set_ylabel("E − E$_{VBM}$ (eV)"); ax.set_xlabel("")
ax.set_title("Monolayer CrI$_3$ (FM): orbital-magnetization-resolved bandstructure")
cb = fig.colorbar(sm, ax=ax, pad=0.02, extend='both')
cb.set_label(r"per-state $m_z^{\rm orb}$ ($\mu_B$):  red = $\parallel$ spin,  blue = anti-$\parallel$")
ax.text(0.5, -0.09, "color saturates at $\\pm$0.04 $\\mu_B$; spikes at band crossings (SOS denominator)",
        transform=ax.transAxes, ha='center', fontsize=7.5, color='0.45')
fig.tight_layout(); fig.savefig(f"{REP}/cri3_orbmag_bandstructure.png", dpi=200, bbox_inches='tight')
print(f"saved {REP}/cri3_orbmag_bandstructure.png   vmax={vmax}  bands={nb}  k={nk}")
# which occupied bands carry the most orbital weight (clip spikes first)
w = np.abs(np.clip(MZ[:, :nocc], -vmax, vmax)).sum(axis=0)
top = np.argsort(w)[::-1][:6]
print("top occ bands by clipped |m_z| weight:", [(int(b), round(float(w[b]), 3)) for b in top])
