"""Publication GW/QP bandstructure figure -- on the COMPUTED k-points, no interpolant.

Why not the htransform.  The usual figure route interpolates E_DFT and E_QP from
the coarse grid onto a dense Gamma-M-K-Gamma path with ``bandstructure.htransform``.
That interpolant is an SVD of A = (nk*nb, nspinor*n_mu); it is only well posed
when the ISDF rank EXCEEDS the state count nk*nb.  On the 6x6 predecessor it did
(36 x 60 = 2160 states vs rank 3178, ctilde orthogonality 2e-14).  On this 12x12
reference it does not: 144 x 48 = 6912 states vs rank 2472 gave orthogonality
error 4.9e-1 and an interpolated DFT band structure with a NEGATIVE indirect gap
(-1.20 eV), i.e. the interpolant had collapsed.  Restricting to a gap-centred
16-band window instead returned rank = 0.  And the fix -- more centroids -- is
barred by ``bse_setup``'s replicated fH_R (nk * (ns*n_mu)^2 * 16 B = 49.9 GiB at
n_mu = 2412).  See KNOWN_SANDBOX_ERRORS.md.

So this figure plots what was actually computed: the QP energies at the subset of
the 12x12x1 grid k-points that lie exactly on Gamma-M-K-Gamma.  No interpolation
error of any kind; the sampling is coarse and is drawn as such (markers on the
computed points, light guides between them).

usage: python3 gw_bands_ongrid.py <eqp1.dat> <out.png> [<eqp0.dat>]
env: OG_NVAL (26)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

EQP1 = sys.argv[1]
OUT = sys.argv[2]
EQP0 = sys.argv[3] if len(sys.argv) > 3 else None
NVAL = int(os.environ.get("OG_NVAL", "26"))


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
    nb = max(max(r) for r in rows)
    ed = np.full((len(ks), nb), np.nan)
    eq = np.full((len(ks), nb), np.nan)
    for k in range(len(ks)):
        for b, (a, c) in rows[k].items():
            ed[k, b - 1] = a
            eq[k, b - 1] = c
    return np.asarray(ks), ed, eq


ks, edft, eqp1 = parse_eqp(EQP1)
eqp0 = parse_eqp(EQP0)[2] if EQP0 else None
N = 12                                     # 12x12x1 grid


def key(fr):
    """Round a fractional k to the 1/N lattice, wrapped to [0,1)."""
    return tuple(int(round(v * N)) % N for v in fr[:2])


index = {key(k): i for i, k in enumerate(ks)}

# --- the path, in units of 1/12, through grid points only -------------------
# Gamma(0,0) -> M(0,1/2) -> K(1/3,1/3) -> Gamma.  Only points that ARE grid
# points are on the path; the M->K leg only hits the lattice every other step.
seg_G_M = [(0, j) for j in range(0, 7)]                       # (0,0)..(0,6)
seg_M_K = [(i, 6 - i // 2) for i in (0, 2, 4)]                # (0,6),(2,5),(4,4)
seg_K_G = [(i, i) for i in range(4, -1, -1)]                  # (4,4)..(0,0)
path = seg_G_M + seg_M_K[1:] + seg_K_G[1:]

bvec_ang = None      # path distance in fractional-metric units is enough here
pts, idx = [], []
for p in path:
    if p not in index:
        raise SystemExit(f"grid point {p} missing from eqp file")
    pts.append(p)
    idx.append(index[p])

# path abscissa: cumulative |dk| in the (a*, b*) hexagonal metric
# a* . a* = b* . b* = 1, a* . b* = -1/2  (60-deg reciprocal lattice)
def dist(p, q):
    dx, dy = (q[0] - p[0]) / N, (q[1] - p[1]) / N
    return np.sqrt(dx * dx + dy * dy - dx * dy)


x = [0.0]
for a, b in zip(pts[:-1], pts[1:]):
    x.append(x[-1] + dist(a, b))
x = np.asarray(x)
node_x = [x[0], x[len(seg_G_M) - 1], x[len(seg_G_M) + len(seg_M_K) - 2], x[-1]]

Ed = edft[idx]
E1 = eqp1[idx]
vbm_d, vbm_q = np.nanmax(edft[:, NVAL - 1]), np.nanmax(eqp1[:, NVAL - 1])
Ed = Ed - vbm_d
E1 = E1 - vbm_q

nshow_lo, nshow_hi = NVAL - 8, NVAL + 8
print(f"[ongrid] {len(idx)} grid k-points on the path; "
      f"bands {nshow_lo+1}..{nshow_hi}")
ik_K = index[(4, 4)]
ik_G = index[(0, 0)]
print(f"[ongrid] K direct: DFT {edft[ik_K,NVAL]-edft[ik_K,NVAL-1]:.4f}  "
      f"GW {eqp1[ik_K,NVAL]-eqp1[ik_K,NVAL-1]:.4f} eV")

# ---------------------------------------------------------------------- plot
C_DFT, C_GW = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#1a1a19", "#55544e", "#e2e1dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
for nx in node_x:
    ax.axvline(nx, color=GRID, lw=0.8, zorder=0)
ax.axhline(0.0, color=GRID, lw=0.8, ls="--", zorder=0)

for b in range(nshow_lo, nshow_hi):
    ax.plot(x, Ed[:, b], color=C_DFT, lw=1.0, alpha=0.55, zorder=2)
    ax.plot(x, Ed[:, b], color=C_DFT, ls="none", marker="o", ms=2.6, zorder=3)
    ax.plot(x, E1[:, b], color=C_GW, lw=1.0, alpha=0.55, zorder=4)
    ax.plot(x, E1[:, b], color=C_GW, ls="none", marker="o", ms=2.6, zorder=5)

# direct-gap annotation at K
iK = pts.index((4, 4))
for tag, E, col, dx in (("DFT", Ed, C_DFT, -0.020), ("GW", E1, C_GW, +0.020)):
    lo, hi = E[iK, NVAL - 1], E[iK, NVAL]
    xx = x[iK] + dx * (x[-1] - x[0])
    ax.annotate("", xy=(xx, hi), xytext=(xx, lo),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xx + 0.012 * (x[-1] - x[0]) * (1 if dx > 0 else -1),
            0.5 * (lo + hi), f"{hi-lo:.2f} eV", color=INK, fontsize=9,
            ha="left" if dx > 0 else "right", va="center",
            bbox=dict(fc="white", ec="none", pad=1.2), zorder=7)

ax.set_xticks(node_x)
ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-6.0, 8.0)
ax.set_ylabel("E − E$_{VBM}$  (eV)")
ax.set_title("Monolayer MoS$_2$: DFT vs G$_0$W$_0$ (GN-PPM), converged reference",
             fontsize=11.5, pad=22)
ax.plot([], [], color=C_DFT, lw=1.6, marker="o", ms=3.4,
        label="DFT (PBE, FR-SOC)")
ax.plot([], [], color=C_GW, lw=1.6, marker="o", ms=3.4,
        label="G$_0$W$_0$ quasiparticle (eqp1)")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.text(0.5, 1.012,
        "80 Ry · 12×12×1 (144 k) · 326 screening bands · 2412 band-range + "
        "D$_{3h}$ centroids · Σ for bands 1–80",
        transform=ax.transAxes, fontsize=7.8, color=INK2, ha="center",
        va="bottom")
ind_d = np.nanmin(edft[:, NVAL]) - np.nanmax(edft[:, NVAL - 1])
ind_q = np.nanmin(eqp1[:, NVAL]) - np.nanmax(eqp1[:, NVAL - 1])
ax.text(0.5, -0.115,
        f"markers = computed 12×12 k-points on the path (no interpolation)"
        f"      ·      indirect gap over the full BZ: "
        f"DFT {ind_d:.2f} eV → GW {ind_q:.2f} eV",
        transform=ax.transAxes, fontsize=8, color=INK2, ha="center", va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
np.savez(OUT.replace(".png", ".npz"), x=x, Ed=Ed, E1=E1,
         pts=np.array(pts), node_x=np.array(node_x))
print(f"SAVED {OUT}")
