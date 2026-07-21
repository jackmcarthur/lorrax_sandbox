"""Publication figure: MoS2 exciton bandstructure E_S(Q) along Gamma-M-K-Gamma,
referenced to the CONVERGED G0W0 gap.

Why the GW reference needs care.  `bse.exciton_bands` builds the TDA diagonal
D_Q from `enk_sigma`, i.e. the DFT eigenvalues returned by
`htransform.initialize_wfns` -- the BSE is solved on the DFT band structure.
So the raw E_S the driver writes sits on the DFT gap.  The GW correction over
the 8v8c BSE window is, to the extent it is k- and band-flat, a rigid scissor
Delta = (GW direct gap - DFT direct gap): it shifts every conduction level by
Delta and therefore shifts every E_S(Q) by exactly Delta.  Consequences,
stated plainly because they are easy to get wrong:

  * the OPTICAL GAP moves:  E_1^GW(Gamma) = E_1^DFT(Gamma) + Delta
  * the BINDING ENERGY does not:  E_b = E_gap - E_1 is scissor-invariant,
    E_b^GW = GW_gap - E_1^GW = DFT_gap - E_1^DFT = E_b^DFT

So "binding vs the GW gap" is the same number as "binding vs the DFT gap"
BY CONSTRUCTION -- what changes, and what this figure shows, is that the
free-pair onset the binding is measured down from is now the converged
quasiparticle gap rather than the (badly underestimated) DFT one.  The script
also reports the k/band SPREAD of the QP correction inside the BSE window, so
the rigid-scissor approximation is quantified rather than assumed.

usage:
  python3 plot_exciton.py <bands.dat> <out.png> <eqp1.dat>
env: EX_NVAL (26) EX_NV (8) EX_NC (8)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

DAT = sys.argv[1]
OUT = sys.argv[2]
EQP = sys.argv[3] if len(sys.argv) > 3 else None
NVAL = int(os.environ.get("EX_NVAL", "26"))
NV = int(os.environ.get("EX_NV", "8"))
NC = int(os.environ.get("EX_NC", "8"))


# ---------------------------------------------------------------- eqp / gaps
def parse_eqp(path):
    ks, rows, cur = [], [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) == 4 and '.' in p[0]:
            ks.append([float(v) for v in p[:3]])
            cur = {}
            rows.append(cur)
        elif len(p) == 4 and cur is not None:
            cur[int(p[1])] = (float(p[2]), float(p[3]))
    return np.asarray(ks), rows


def _wrap(d):
    return (d + 0.5) % 1.0 - 0.5


dft_gap = gw_gap = delta = None
spread = {}
if EQP and os.path.exists(EQP):
    ks, rows = parse_eqp(EQP)
    nb = max(max(r) for r in rows)
    ed = np.full((len(ks), nb), np.nan)
    eq = np.full((len(ks), nb), np.nan)
    for k in range(len(ks)):
        for b, (a, c) in rows[k].items():
            ed[k, b - 1] = a
            eq[k, b - 1] = c
    # Direct gap at K -- the optical transition MoS2's A exciton lives on.
    ikK = int(np.argmin(np.linalg.norm(
        _wrap(ks[:, :2] - np.array([1 / 3, 1 / 3])), axis=1)))
    dft_gap = float(ed[ikK, NVAL] - ed[ikK, NVAL - 1])
    gw_gap = float(eq[ikK, NVAL] - eq[ikK, NVAL - 1])
    delta = gw_gap - dft_gap
    # Rigid-scissor quality inside the BSE window (bands NVAL-NV .. NVAL+NC).
    lo, hi = NVAL - NV, NVAL + NC
    shift = eq[:, lo:hi] - ed[:, lo:hi]
    dsh = shift[:, NV:] - shift[:, :NV].mean()      # conduction rel. valence
    spread = dict(mean=float(np.nanmean(shift)), std=float(np.nanstd(shift)),
                  cond_minus_val=float(np.nanmean(dsh)),
                  cond_std=float(np.nanstd(shift[:, NV:])),
                  val_std=float(np.nanstd(shift[:, :NV])))
    print(f"[gaps] at K: DFT direct {dft_gap:.4f} eV   "
          f"GW direct {gw_gap:.4f} eV   Delta = {delta:+.4f} eV")
    print(f"[scissor quality] QP shift over BSE window bands "
          f"[{lo},{hi}): mean {spread['mean']:+.3f} eV, "
          f"std {spread['std']:.3f} eV;  valence std {spread['val_std']:.3f}, "
          f"conduction std {spread['cond_std']:.3f} eV")

# ---------------------------------------------------------------- exciton dat
rows = []
with open(DAT) as fh:
    for line in fh:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        rows.append((int(p[0]), float(p[1]),
                     np.array([float(p[2]), float(p[3]), float(p[4])]),
                     np.array([float(x) for x in p[6:]])))
rows.sort(key=lambda r: r[0])
s_path = np.array([r[1] for r in rows])
Qs = np.array([r[2] for r in rows])
E = np.array([r[3] for r in rows])
nQ, neig = E.shape
print(f"[exciton] nQ={nQ} n_eig={neig}")

nodes_frac = [np.array([0, 0, 0.]), np.array([0, .5, 0.]),
              np.array([1 / 3, 1 / 3, 0.]), np.array([0, 0, 0.])]
node_idx, seen = [], -1
for tgt in nodes_frac:
    d = np.linalg.norm(_wrap(Qs - tgt), axis=1)
    cand = [i for i in np.argsort(d) if i > seen]
    j = cand[0] if cand else int(np.argmin(d))
    node_idx.append(j)
    seen = j
node_x = [s_path[i] for i in node_idx]

iG = node_idx[0]
E1_dft = float(E[iG, 0])
E_gw = E + (delta if delta is not None else 0.0)
E1_gw = float(E_gw[iG, 0])
gap_ref = gw_gap if gw_gap is not None else np.nan
binding = (gap_ref - E1_gw) if gw_gap is not None else np.nan

# C3 check: Gamma is sampled at both ends of the path; they must agree.
c3 = abs(float(E[node_idx[0], 0]) - float(E[node_idx[-1], 0]))
print(f"[exciton] E_1(Gamma) raw(DFT-based) = {E1_dft:.4f} eV")
print(f"[exciton] E_1(Gamma) GW-referenced  = {E1_gw:.4f} eV "
      f"(+{delta:.4f} rigid scissor)" if delta is not None else "")
print(f"[exciton] binding vs GW gap ({gap_ref:.4f}) = {binding*1e3:.0f} meV")
print(f"[exciton] Gamma endpoint consistency |E1(start)-E1(end)| = "
      f"{c3*1e3:.2f} meV")
imin = int(np.argmin(E_gw[:, 0]))
print(f"[exciton] E_1 min over path = {E_gw[imin,0]:.4f} eV @ s={s_path[imin]:.3f}"
      f"  (dispersion Gamma->min = {(E1_gw-E_gw[imin,0])*1e3:+.0f} meV)")
print(f"[exciton] E_1 max over path = {E_gw[:,0].max():.4f} eV"
      f"  (total bandwidth {(E_gw[:,0].max()-E_gw[:,0].min())*1e3:.0f} meV)")

np.savez(OUT.replace(".png", ".npz"), s_path=s_path, E_dft_based=E,
         E_gw=E_gw, node_idx=np.array(node_idx), E1_gw=E1_gw,
         E1_dft=E1_dft, binding=binding, gw_gap=gap_ref, dft_gap=dft_gap,
         delta=delta)

# ---------------------------------------------------------------------- plot
INK, INK2, GRID = "#1a1a19", "#55544e", "#e2e1dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)
nshow = min(6, neig)
cmap = plt.cm.viridis(np.linspace(0.10, 0.80, nshow))
for x in node_x:
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)
for b in range(nshow):
    ax.plot(s_path, E_gw[:, b], color=cmap[b], lw=1.8 if b == 0 else 1.2,
            label=(f"S$_{b+1}$" if b < 4 else None), zorder=3,
            solid_capstyle="round")
if gw_gap is not None:
    ax.axhline(gw_gap, color="#c0392b", lw=1.2, ls="--", zorder=1,
               label="G$_0$W$_0$ direct gap at K (free e-h)")
    ax.annotate("", xy=(node_x[0], E1_gw), xytext=(node_x[0], gw_gap),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(node_x[0] + 0.012 * (s_path[-1] - s_path[0]),
            0.5 * (E1_gw + gw_gap),
            f"E$_b$ = {binding*1e3:.0f} meV", color="#c0392b", fontsize=10,
            va="center", zorder=6,
            bbox=dict(fc="white", ec="none", pad=1.0))
ax.set_xticks(node_x)
ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
ax.set_xlim(s_path[0], s_path[-1])
ax.set_ylabel("Exciton energy E$_S$(Q)  (eV)")
ax.set_title("Monolayer MoS$_2$ — BSE exciton bandstructure on the\n"
             "converged G$_0$W$_0$ (12$\\times$12, 80 Ry, 2412 centroids)",
             fontsize=11, pad=10)
ax.text(0.0, 1.005,
        f"native 12$\\times$12 BSE (no coarse$\\to$fine interpolation) · "
        f"{NV}v{NC}c · TDA · E$_1(\\Gamma)$ = {E1_gw:.3f} eV",
        transform=ax.transAxes, fontsize=7.6, color=INK2,
        ha="left", va="bottom")
ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"SAVED {OUT}")
