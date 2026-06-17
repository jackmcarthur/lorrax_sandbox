"""Generic colored bandstructure plotter for a band-path npz (E, SZ, MZ_spin, dist,
node_idx, labels, VBM, nocc). Two modes:
  spin  -> color by <sigma_z> per (n,k), red=up / blue=down  (bounded [-1,1])
  orb   -> color by per-state orbital moment m_z along spin, red=parallel / blue=anti
Usage: plot_band_colored.py <npz> <spin|orb> <out.png> "<title>"
Supersedes plot_orbmag_bandstructure.py (one plotter for spin & orbital, CrI3 & VI3)."""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

npz, mode, out, title = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = np.load(npz, allow_pickle=True)
E = d['E'] - float(d['VBM'])                            # eV, zeroed to VBM
dist, node, lab = d['dist'], d['node_idx'], d['labels']
nk, nb = E.shape

if mode == 'spin':
    C = d['SZ']; vmax = 1.0
    cblabel = r"$\langle\sigma_z\rangle$:  red = $\uparrow$,  blue = $\downarrow$"
    note = None
elif mode == 'orb':
    C = d['MZ_spin']; vmax = 0.04
    cblabel = r"per-state $m_z^{\rm orb}$ ($\mu_B$):  red = $\parallel$ spin,  blue = anti-$\parallel$"
    note = "color saturates at $\\pm$0.04 $\\mu_B$; spikes at band crossings (SOS denominator)"
else:
    raise SystemExit(f"mode must be spin|orb, got {mode}")

norm = plt.Normalize(-vmax, vmax); cmap = plt.get_cmap('bwr')
labmap = {'G': r'$\Gamma$', 'M': 'M', 'K': 'K'}
ticks = [labmap.get(str(x), str(x)) for x in lab]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
for n in range(nb):
    pts = np.array([dist, E[:, n]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c = 0.5 * (C[:-1, n] + C[1:, n])
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.2)
    lc.set_array(c); ax.add_collection(lc)
ax.axhline(0, color='0.4', lw=0.8, ls='--')
for x in dist[node]: ax.axvline(x, color='0.7', lw=0.7)
ax.set_xticks(dist[node]); ax.set_xticklabels(ticks, fontsize=12)
ax.set_xlim(dist.min(), dist.max()); ax.set_ylim(-6, 4)
ax.set_ylabel("E − E$_{VBM}$ (eV)")
ax.set_title(title)
cb = fig.colorbar(sm, ax=ax, pad=0.02, extend=('both' if mode == 'orb' else 'neither'))
cb.set_label(cblabel)
if note:
    ax.text(0.5, -0.09, note, transform=ax.transAxes, ha='center', fontsize=7.5, color='0.45')
fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"saved {out}  mode={mode} vmax={vmax} bands={nb} k={nk}")
