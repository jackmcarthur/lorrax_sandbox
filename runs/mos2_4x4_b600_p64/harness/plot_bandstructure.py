"""Plot the b300 DFT and G0W0 quasiparticle bandstructures on one axis.

Consumes the two ``bandstructure.dat`` files written by
``bandstructure.htransform`` (columns: idx_k idx_b kx ky kz s energy;
energy in Ry, already shifted so each run's own VBM sits at 0) and emits a
single PNG plus a stdout numeric summary.

Usage:
    python plot_bandstructure_b300.py DFT.dat QP.dat OUT.png NVAL
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RYD_TO_EV = 13.605693122994

# Categorical slots 1 and 2 of the validated reference palette (light
# surface #fcfcfb): CVD separation dE 24.7 protan / 32.7 tritan, both
# >= 3:1 contrast. Identity is carried by the legend, not by color alone.
C_DFT = "#2a78d6"
C_QP = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#26241f"
MUTED = "#6b675e"
GRID = "#d8d5cd"

HI_SYM = [
    (np.array([0.0, 0.0, 0.0]), "Γ"),
    (np.array([0.0, 0.5, 0.0]), "M"),
    (np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]), "K"),
    (np.array([0.0, 0.0, 0.0]), "Γ"),
]


def load(path):
    """-> (s[nk], kfrac[nk,3], E_eV[nk, nb])."""
    raw = np.loadtxt(path)
    ik = raw[:, 0].astype(int)
    ib = raw[:, 1].astype(int)
    nk = ik.max() + 1
    nb = ib.max() + 1
    E = np.full((nk, nb), np.nan)
    E[ik, ib] = raw[:, 6] * RYD_TO_EV
    s = np.zeros(nk)
    s[ik] = raw[:, 5]
    kf = np.zeros((nk, 3))
    kf[ik] = raw[:, 2:5]
    return s, kf, E


def node_positions(s, kf):
    """Path coordinates of the high-symmetry nodes, in path order."""
    xs, labels, used = [], [], -1
    for target, label in HI_SYM:
        d = np.linalg.norm(kf - target[None, :], axis=1)
        cand = np.where(d < 1e-6)[0]
        idx = cand[cand > used][0] if np.any(cand > used) else int(np.argmin(d))
        xs.append(float(s[idx]))
        labels.append(label)
        used = int(idx)
    return xs, labels


def gaps(s, kf, E, nval, tag):
    """Print the gap/bandwidth numbers this plot is meant to be judged on."""
    vb = E[:, nval - 1]
    cb = E[:, nval]
    ivbm, icbm = int(np.argmax(vb)), int(np.argmin(cb))
    indirect = cb[icbm] - vb[ivbm]
    idirect = int(np.argmin(cb - vb))
    print(f"[{tag}] VBM {vb[ivbm]:+.4f} eV at k={kf[ivbm]} (s={s[ivbm]:.4f})")
    print(f"[{tag}] CBM {cb[icbm]:+.4f} eV at k={kf[icbm]} (s={s[icbm]:.4f})")
    print(f"[{tag}] fundamental (indirect-allowed) gap = {indirect:.4f} eV")
    print(f"[{tag}] smallest direct gap = {(cb - vb)[idirect]:.4f} eV "
          f"at k={kf[idirect]} (s={s[idirect]:.4f})")
    print(f"[{tag}] gap at Gamma = {(cb[0] - vb[0]):.4f} eV")
    print(f"[{tag}] valence manifold width (band 0 .. band {nval-1}) = "
          f"{E[:, :nval].max() - E[:, :nval].min():.4f} eV")
    print(f"[{tag}] top valence band width = {vb.max() - vb.min():.4f} eV")
    print(f"[{tag}] lowest conduction band width = {cb.max() - cb.min():.4f} eV")
    print(f"[{tag}] plotted window: {E.shape[1]} bands, "
          f"E in [{np.nanmin(E):.3f}, {np.nanmax(E):.3f}] eV")
    return indirect, (cb - vb)[idirect]


def main():
    dft_path, qp_path, out_png, nval = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    s_d, kf_d, E_d = load(dft_path)
    print(f"DFT  {dft_path}: nk={E_d.shape[0]} nb={E_d.shape[1]}")
    g_d = gaps(s_d, kf_d, E_d, nval, "DFT")

    E_q = None
    if qp_path != "NONE":
        s_q, kf_q, E_q = load(qp_path)
        print(f"QP   {qp_path}: nk={E_q.shape[0]} nb={E_q.shape[1]}")
        g_q = gaps(s_q, kf_q, E_q, nval, "QP")

    xs, labels = node_positions(s_d, kf_d)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for x in xs[1:-1]:
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.8, ls="--", dashes=(4, 3), zorder=0)

    for b in range(E_d.shape[1]):
        ax.plot(s_d, E_d[:, b], lw=1.1, color=C_DFT, zorder=2)
    if E_q is not None:
        for b in range(E_q.shape[1]):
            ax.plot(s_q, E_q[:, b], lw=1.1, color=C_QP, zorder=3)

    handles = [Line2D([], [], color=C_DFT, lw=2.0, label="DFT (Kohn-Sham eigenvalues)")]
    if E_q is not None:
        handles.append(Line2D([], [], color=C_QP, lw=2.0, label="G$_0$W$_0$ quasiparticle"))
    # Legend below the axes: with 44 bands there is no in-plot region that is
    # reliably empty, and a legend box over the marks is an anti-pattern.
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.07), ncol=len(handles),
                    frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)

    stack = [E_d] + ([E_q] if E_q is not None else [])
    lo = min(np.nanmin(E[:, max(0, nval - 6):nval]) for E in stack) - 0.6
    hi = max(np.nanmax(E[:, nval:nval + 6]) for E in stack) + 0.6
    ax.set_ylim(lo, hi)
    ax.set_xlim(s_d.min(), s_d.max())

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylabel("Energy relative to VBM  (eV)", color=INK)
    sub = f"DFT gap {g_d[0]:.2f} eV"
    if E_q is not None:
        sub += f"   |   G$_0$W$_0$ gap {g_q[0]:.2f} eV"
    nb_label = os.environ.get("LORRAX_PLOT_NBAND", "300")
    ax.set_title("MoS$_2$ 4$\\times$4 k-grid, 30 Ry, %s bands (26v + 18c $\\Sigma$ window)\n" % nb_label + sub,
                 color=INK, fontsize=11, pad=10)

    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    for t in ax.get_xticklabels():
        t.set_color(INK)
        t.set_fontsize(12)

    fig.tight_layout()
    # bbox_inches="tight": the Gamma tick labels sit exactly on the axis ends
    # and are clipped by the figure edge otherwise.
    fig.savefig(out_png, facecolor=SURFACE, bbox_inches="tight")
    print(f"WROTE {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
