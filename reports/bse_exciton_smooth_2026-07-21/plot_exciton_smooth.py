"""Publication figure: MoS2 8v8c exciton dispersion on the converged G0W0,
sampled at ARBITRARY Q (b26p V_Q interpolation) along the two segments
Gamma->M and Gamma->K.

Nothing is scissor-shifted: the BSE was solved directly on the quasiparticle
energies (``exciton_bands --eqp eqp1.dat``), so E_S(Q) IS the QP-referenced
exciton energy and the binding energy is measured against the run's own
converged G0W0 gap.

The .dat path runs M -> Gamma -> K, which is exactly the owner's two segments
radiating from Gamma with Gamma counted once; the figure keeps that layout so
both segments are visible at once, Gamma at the centre.

Colour (dataviz method): the branches are an ORDERED family (E_1 < E_2 < ...),
so they take a single-hue sequential ramp, dark = low, with the lowest branch on
the accent and direct-labelled.  The two reference levels are one distinct hue
each, dashed/dotted, both direct-labelled, so identity is never colour-alone.
Palette validated (light surface #fcfcfb, categorical, 3 slots): all six checks
PASS, worst adjacent CVD dE 24.7 protan, normal-vision dE 33.6.

usage: python3 plot_exciton_smooth.py <bands.dat> <out.png>
env: EX_GAP_DIRECT EX_GAP_INDIRECT EX_NBRANCH EX_TITLE EX_JSON
     EX_AGREE (meV, the interp-vs-ongrid max |dE1| printed in the caption)
     EX_ONGRID_DAT (the run-08 on-grid .dat, drawn as the 13-point comparison)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

DAT = sys.argv[1]
OUT = sys.argv[2]
GAP_D = float(os.environ.get("EX_GAP_DIRECT", "2.6356"))
GAP_I = float(os.environ.get("EX_GAP_INDIRECT", "2.5079"))
NBR = int(os.environ.get("EX_NBRANCH", "6"))
AGREE = os.environ.get("EX_AGREE")
ONGRID = os.environ.get("EX_ONGRID_DAT")
TITLE = os.environ.get(
    "EX_TITLE",
    "MoS$_2$ exciton dispersion $E_S(\\mathbf{Q})$ — 8v8c TDA BSE on converged "
    "G$_0$W$_0$, arbitrary $\\mathbf{Q}$")

# dataviz reference palette (light surface #fcfcfb)
ACCENT = "#2a78d6"      # slot 1 blue   — lowest branch
REF_D = "#eb6834"       # slot 2 orange — direct gap at K
REF_I = "#4a3aa7"       # slot 7 violet — indirect gap
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8b8a85"
SURFACE = "#fcfcfb"


def read_dat(path):
    rows, modes = [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        modes.append(p[5])
        rows.append([float(v) for v in p[1:5]] + [float(v) for v in p[6:]])
    arr = np.asarray([r for r, m in zip(rows, modes) if m != "refit"])
    return arr[:, 1:4], arr[:, 4:]


Q, E = read_dat(DAT)
nQ, neig = E.shape
NBR = min(NBR, neig)

# ---------------------------------------------------------------- geometry
# Signed path coordinate: |Q| in crystal units, NEGATIVE on the Gamma->M leg so
# the two segments radiate from Gamma at x = 0.  Cartesian |Q| would compress
# Gamma->K against Gamma->M by the 2/sqrt(3) hexagonal factor; using the
# fraction of each segment travelled keeps both legs on their own scale and the
# node ticks exact.
iG = int(np.argmin(np.linalg.norm(Q - np.round(Q), axis=1)))
QM, QK = Q[0], Q[-1]                      # path is M -> Gamma -> K


def frac_along(q, node):
    return float(np.dot(q, node) / np.dot(node, node))


x = np.array([-frac_along(Q[i], QM) if i <= iG else frac_along(Q[i], QK)
              for i in range(nQ)])

leg_M = np.arange(0, iG + 1)              # M .. Gamma   (x <= 0)
leg_K = np.arange(iG, nQ)                 # Gamma .. K   (x >= 0)


def qlabel(q):
    f = q - np.round(q)
    names = {(0., 0., 0.): r"$\Gamma$", (0., .5, 0.): "M", (.5, 0., 0.): "M",
             (1 / 3, 1 / 3, 0.): "K", (-1 / 3, -1 / 3, 0.): "K",
             (1 / 6, 1 / 6, 0.): r"$\Lambda$", (-1 / 6, -1 / 6, 0.): r"$\Lambda$"}
    for k, v in names.items():
        if np.allclose(f, k, atol=2e-3):
            return v
    return f"({f[0]:+.3f},{f[1]:+.3f})"


# ---------------------------------------------------------------- numbers
E1G = float(E[iG, 0])
Eb = GAP_D - E1G
# OFF-GRID COLLAPSE.  ``EX_FLAG_JSON`` (offgrid_collapse.json) lists the Q whose
# whole eigenvalue multiplet falls hundreds of meV below the local trend — the
# interpolated eps_c(k+Q) collapsing, NOT dispersion.  They are drawn as crosses
# and EXCLUDED from the E_1 line and from every reported minimum, because
# joining them would draw a curve that is smooth-looking and wrong.
_FLAGS = os.environ.get("EX_FLAG_JSON")
_bad = set(json.load(open(_FLAGS))["collapsed_indices"]) if (
    _FLAGS and os.path.exists(_FLAGS)) else set()
bad = _bad
_okM = np.array([i for i in leg_M if i not in _bad])
_okK = np.array([i for i in leg_K if i not in _bad])
iminM = int(_okM[np.argmin(E[_okM, 0])])
iminK = int(_okK[np.argmin(E[_okK, 0])])
_ok = np.array([i for i in range(nQ) if i not in _bad])
imin = int(_ok[np.argmin(E[_ok, 0])])

summary = dict(
    n_Q=nQ, n_branch=neig, n_Q_GammaM=len(leg_M), n_Q_GammaK=len(leg_K),
    E1_Gamma_eV=E1G, gap_direct_K_eV=GAP_D, gap_indirect_eV=GAP_I,
    binding_vs_direct_meV=Eb * 1e3,
    binding_vs_indirect_meV=(GAP_I - E1G) * 1e3,
    E1_min_GammaM_eV=float(E[iminM, 0]),
    E1_min_GammaM_Q=list(map(float, Q[iminM])),
    E1_min_GammaM_label=qlabel(Q[iminM]),
    E1_min_GammaK_eV=float(E[iminK, 0]),
    E1_min_GammaK_Q=list(map(float, Q[iminK])),
    E1_min_GammaK_label=qlabel(Q[iminK]),
    E1_min_eV=float(E[imin, 0]), E1_min_label=qlabel(Q[imin]),
    dispersion_Gamma_to_min_meV=(E1G - float(E[imin, 0])) * 1e3,
    bandwidth_E1_meV=float(E[:, 0].max() - E[:, 0].min()) * 1e3,
    interp_vs_ongrid_max_dE1_meV=(float(AGREE) if AGREE else None),
)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.4, 4.9))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.axhspan(GAP_D, GAP_D + 10, color=REF_D, alpha=0.06, lw=0, zorder=0)
ax.axvline(0.0, color=MUTED, lw=0.7, alpha=0.55, zorder=1)
ax.grid(axis="y", color=MUTED, lw=0.5, alpha=0.30, zorder=0)
ax.set_axisbelow(True)

# sequential ramp, dark (low branch) -> light (high branch); LINES only
cmap = plt.get_cmap("Blues_r")
keep = {leg_id: np.array([i for i in leg if i not in bad])
        for leg_id, leg in (("M", leg_M), ("K", leg_K))}
for b in range(NBR - 1, 0, -1):
    c = cmap(0.18 + 0.42 * (b - 1) / max(1, NBR - 2))
    for leg in keep.values():
        ax.plot(x[leg], E[leg, b], "-", lw=1.4, color=c, zorder=3, alpha=0.9,
                solid_capstyle="round")
for leg in keep.values():
    ax.plot(x[leg], E[leg, 0], "-", lw=2.6, color=ACCENT, zorder=5,
            solid_capstyle="round")
if bad:
    bi = np.array(sorted(bad))
    ax.plot(x[bi], E[bi, 0], "x", ms=7, mew=1.8, color=REF_D, ls="none",
            zorder=9)

# what the previous on-grid run could see: its 13 Q, same physics, plotted as
# open rings.  Only the ones on THESE two segments exist here.
if ONGRID and os.path.exists(ONGRID):
    Qo, Eo = read_dat(ONGRID)
    xs, ys = [], []
    for j in range(len(Qo)):
        d = np.linalg.norm((Q - np.round(Q))
                           - (Qo[j] - np.round(Qo[j]))[None, :], axis=1)
        i = int(np.argmin(d))
        if d[i] < 1e-6:
            xs.append(x[i])
            ys.append(Eo[j, 0])
    ax.plot(xs, ys, "o", ms=7.5, mfc="none", mec=INK2, mew=1.3, zorder=6,
            ls="none")
    summary["n_ongrid_rings"] = len(xs)

ax.axhline(GAP_D, color=REF_D, lw=1.8, ls="--", dashes=(5, 3), zorder=4)
ax.axhline(GAP_I, color=REF_I, lw=1.4, ls=":", zorder=4)

x0, x1 = float(x.min()), float(x.max())
ax.set_xlim(x0, x1)
lo = min(E[:, 0].min(), GAP_I) - 0.14
hi = max(E[:, :NBR].max(), GAP_D) + 0.10
ax.set_ylim(lo, hi)

# direct labels
ax.text(x1 - 0.012 * (x1 - x0), GAP_D - 0.022 * (hi - lo),
        f"G$_0$W$_0$ direct gap @ K  {GAP_D:.4f} eV", color=REF_D,
        ha="right", va="top", fontsize=8.6, fontweight="bold")
ax.text(x1 - 0.012 * (x1 - x0), GAP_I - 0.020 * (hi - lo),
        f"indirect gap  {GAP_I:.4f} eV", color=REF_I,
        ha="right", va="top", fontsize=8.0)
ax.text(x0 + 0.012 * (x1 - x0), float(E[0, 0]) + 0.020 * (hi - lo),
        r"$E_1$", color=ACCENT, ha="left", va="bottom",
        fontsize=10.5, fontweight="bold")

# binding energy at Gamma, drawn on the Gamma axis
ax.annotate("", xy=(0.0, GAP_D), xytext=(0.0, E1G),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3,
                            shrinkA=0, shrinkB=0), zorder=7)
ax.text(0.018 * (x1 - x0), 0.5 * (GAP_D + E1G),
        f"$E_b(\\Gamma)$ = {Eb*1e3:.0f} meV\n$E_1(\\Gamma)$ = {E1G:.4f} eV",
        color=INK, ha="left", va="center", fontsize=9.2, fontweight="bold",
        bbox=dict(fc=SURFACE, ec="none", alpha=0.92, pad=2.0), zorder=8)

# segment minima — parked at fixed anchors on their own half of the axis so the
# two labels cannot collide when both minima sit near Gamma
for i, xa, ha, dy in ((iminM, -0.78, "left", 0.030), (iminK, 0.78, "right", 0.105)):
    ax.plot([x[i]], [E[i, 0]], "o", ms=6.0, mfc=ACCENT, mec=SURFACE, mew=1.2,
            zorder=8)
    ax.annotate(f"$\\Gamma\\!\\rightarrow\\!${'M' if xa < 0 else 'K'} min: "
                f"{qlabel(Q[i])}  {E[i, 0]:.4f} eV",
                xy=(x[i], E[i, 0]), xytext=(xa, lo + dy * (hi - lo)),
                color=INK, fontsize=8.4, ha=ha, va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                shrinkB=3), zorder=8)

ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_xticklabels(["M", "", r"$\Gamma$", "", "K"], fontsize=11, color=INK)
# Lambda = 1/2 GK is the converged CBM; label it as a minor tick
ax.set_xticks([0.5], minor=True)
ax.set_xticklabels([r"$\Lambda$"], minor=True, fontsize=9.5, color=INK2)
ax.tick_params(axis="x", which="minor", length=3, pad=2)
ax.axvline(0.5, color=MUTED, lw=0.6, alpha=0.35, ls=(0, (3, 3)), zorder=1)

ax.set_xlabel(r"$\longleftarrow\ \Gamma\!\rightarrow\!$M"
              r"$\qquad\qquad\qquad\qquad$"
              r"$\Gamma\!\rightarrow\!$K$\ \longrightarrow$",
              fontsize=9.0, color=INK2, labelpad=6)
ax.set_ylabel(r"$E_S(\mathbf{Q})$  (eV)", fontsize=10.5, color=INK)
ax.set_title(TITLE, fontsize=10.6, color=INK, pad=9)
ax.tick_params(colors=INK2, labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(MUTED)

handles = [
    Line2D([], [], color=ACCENT, lw=2.6, label=r"$E_1$ (lowest exciton)"),
    Line2D([], [], color=cmap(0.42), lw=1.6,
           label=r"$E_2\ldots E_{%d}$" % NBR),
    Line2D([], [], color=REF_D, lw=1.8, ls="--",
           label="G$_0$W$_0$ direct gap (K)"),
    Line2D([], [], color=REF_I, lw=1.4, ls=":",
           label="G$_0$W$_0$ indirect gap"),
]
if ONGRID and os.path.exists(ONGRID):
    handles.append(Line2D([], [], color=INK2, lw=0, marker="o", ms=7.5,
                          mfc="none", mew=1.3,
                          label="on-grid $\\mathbf{Q}$ (run 08, exact tiles)"))
if bad:
    handles.append(Line2D([], [], color=REF_D, lw=0, marker="x", ms=7, mew=1.8,
                          label="$\\varepsilon_c(k\\!+\\!Q)$ COLLAPSED "
                                "— excluded, not physics"))
leg = ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.2,
                frameon=True, framealpha=0.92, edgecolor=MUTED, borderpad=0.5,
                bbox_to_anchor=(0.50, 0.80))
leg.get_frame().set_facecolor(SURFACE)
for t in leg.get_texts():
    t.set_color(INK2)

sub1 = (f"12$\\times$12 k · 80 Ry · $n_\\mu$=2412 · fH window 28 bands · "
        f"{len(leg_M)} $\\mathbf{{Q}}$ on $\\Gamma\\!\\rightarrow\\!$M + "
        f"{len(leg_K)} on $\\Gamma\\!\\rightarrow\\!$K, arbitrary "
        f"$\\mathbf{{Q}}$ (b26p $V_Q$ interpolation) · QP energies, no scissor")
sub2 = ("interpolated exchange vs the EXACT stored tile, over the 11 mesh "
        f"$\\mathbf{{Q}}$ of the companion on-grid run: max $|\\Delta E_1|$ = "
        f"{AGREE} meV — the EXCHANGE model is not the defect"
        if AGREE else
        "interpolated arbitrary-$\\mathbf{Q}$ exchange (b26p)")
fig.text(0.010, 0.036, sub1, fontsize=6.6, color=MUTED, ha="left")
fig.text(0.010, 0.010, sub2, fontsize=6.6, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0.062, 1, 1))
fig.savefig(OUT, dpi=220, facecolor=SURFACE)
print(f"Wrote {OUT}")

for k, v in summary.items():
    print(f"  {k:<32s} {v}")
jp = os.environ.get("EX_JSON")
if jp:
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"Wrote {jp}")
