"""Publication figure: MoS2 8v8c exciton bandstructure E_S(Q) on the CONVERGED G0W0.

Unlike the parent campaign's plot_exciton.py, nothing here is scissor-shifted.
The BSE was solved directly on the quasiparticle energies (``exciton_bands
--eqp eqp1.dat``), so E_S(Q) IS the QP-referenced exciton energy and the
binding energy is measured against the run's own converged gap rather than
inferred from a rigid shift.

Colour: the branches are an ORDERED family (E_1 < E_2 < ... ), so they get a
single-hue sequential ramp, dark = low; the lowest branch carries the accent and
a direct label.  The two reference levels (direct gap at K, indirect gap) are a
distinct hue each, dashed, direct-labelled.  Palette validated with the dataviz
validator (light surface, categorical check: all PASS; the aqua slot's contrast
WARN is relieved by direct labels).

usage: python3 plot_exciton_bands.py <bands.dat> <out.png>
env: EX_GAP_DIRECT (eV) EX_GAP_INDIRECT (eV) EX_TITLE EX_NBRANCH EX_JSON
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
TITLE = os.environ.get("EX_TITLE",
                       "MoS$_2$ exciton dispersion — 8v8c TDA BSE on converged G$_0$W$_0$")

# dataviz reference palette (light surface #fcfcfb)
ACCENT = "#2a78d6"      # slot 1 blue   — lowest branch
REF_D = "#eb6834"       # slot 2 orange — direct gap at K
REF_I = "#4a3aa7"       # slot 7 violet — indirect gap
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8b8a85"
SURFACE = "#fcfcfb"

# ---------------------------------------------------------------- read .dat
nodes = []
rows, modes = [], []
for line in open(DAT):
    if line.startswith("#"):
        if line.startswith("# nodes:"):
            # "0:Γ 6:M 8:K 12:Gamma" — but the driver echoes the K_POINTS
            # comment verbatim as the label, so a commented file yields stray
            # tokens between the index:label pairs.  Take only the pairs.
            for tok in line.split(":", 1)[1].split():
                i, sep, lbl = tok.partition(":")
                if sep and i.strip().isdigit():
                    nodes.append((int(i), lbl))
        continue
    p = line.split()
    if not p:
        continue
    modes.append(p[5])
    rows.append([float(v) for v in p[1:5]] + [float(v) for v in p[6:]])
arr = np.asarray([r for r, m in zip(rows, modes) if m != "refit"])
s = arr[:, 0]
Q = arr[:, 1:4]
E = arr[:, 4:]                                                   # (nQ, n_eig) eV
nQ, neig = E.shape
NBR = min(NBR, neig)

# the K_POINTS comment can say Gamma / G / Γ / the escaped Γ
GAMMA = {"gamma", "g", "γ", "\\u0393", "Γ"}
tick_x, tick_l = [], []
for i, lbl in nodes:
    tick_x.append(s[i])
    tick_l.append(r"$\Gamma$" if lbl.lower() in GAMMA else lbl)
if not tick_x:
    tick_x, tick_l = [s[0], s[-1]], ["", ""]

# ---------------------------------------------------------------- numbers
iG = [i for i in range(nQ) if np.linalg.norm(Q[i] - np.round(Q[i])) < 1e-9]
E1G = float(E[iG[0], 0]) if iG else float(E[0, 0])
closure = (float(E[iG[-1], 0] - E[iG[0], 0]) * 1e3) if len(iG) > 1 else float("nan")
Eb = GAP_D - E1G
imin = int(np.argmin(E[:, 0]))
E1min = float(E[imin, 0])


def qlabel(q):
    f = q - np.round(q)
    names = {(0., 0., 0.): r"$\Gamma$", (0., .5, 0.): "M", (.5, 0., 0.): "M",
             (1/3, 1/3, 0.): "K", (-1/3, -1/3, 0.): "K",
             (1/6, 1/6, 0.): r"$\Lambda$", (-1/6, -1/6, 0.): r"$\Lambda$"}
    for k, v in names.items():
        if np.allclose(f, k, atol=2e-3):
            return v
    return f"({f[0]:+.3f},{f[1]:+.3f})"


summary = dict(
    n_Q=nQ, n_branch=neig, E1_Gamma_eV=E1G,
    gap_direct_K_eV=GAP_D, gap_indirect_eV=GAP_I,
    binding_vs_direct_meV=Eb * 1e3,
    binding_vs_indirect_meV=(GAP_I - E1G) * 1e3,
    gamma_closure_meV=closure,
    E1_min_eV=E1min, E1_min_iQ=imin, E1_min_Q=list(map(float, Q[imin])),
    E1_min_label=qlabel(Q[imin]),
    dispersion_Gamma_to_min_meV=(E1G - E1min) * 1e3,
    E1_max_eV=float(E[:, 0].max()),
    bandwidth_E1_meV=float(E[:, 0].max() - E[:, 0].min()) * 1e3,
)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.0, 4.7))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# free-pair continuum onset: shade above the direct gap
ax.axhspan(GAP_D, GAP_D + 10, color=REF_D, alpha=0.06, lw=0, zorder=0)

for xv in tick_x[1:-1]:
    ax.axvline(xv, color=MUTED, lw=0.7, alpha=0.55, zorder=1)
ax.grid(axis="y", color=MUTED, lw=0.5, alpha=0.30, zorder=0)
ax.set_axisbelow(True)

# sequential ramp, dark (low branch) -> light (high branch)
cmap = plt.get_cmap("Blues_r")
for b in range(NBR - 1, 0, -1):
    c = cmap(0.18 + 0.42 * (b - 1) / max(1, NBR - 2))
    ax.plot(s, E[:, b], "-", lw=1.3, color=c, marker="o", ms=3.0,
            mfc=c, mec=SURFACE, mew=0.5, zorder=3, alpha=0.85)
ax.plot(s, E[:, 0], "-", lw=2.4, color=ACCENT, marker="o", ms=5.5,
        mfc=ACCENT, mec=SURFACE, mew=1.2, zorder=5)

# Valley-COMMENSURATE Q — the ones a 12x12 mesh can actually resolve.  MoS2's
# band edges sit at K (valence) and Λ = ½ΓK (conduction); a finite-Q exciton is
# only as low as the best (k_v, k_v+Q) pair the mesh offers, so Q that map a
# valley onto a valley (Γ, Λ, K, M here) are converged sampling points and the Q
# between them are not.  Ring them so the reader can tell which is which.
special = [(i, qlabel(Q[i])) for i in range(nQ)
           if qlabel(Q[i]) in (r"$\Gamma$", "M", "K", r"$\Lambda$")]
for i, nm in special:
    ax.plot([s[i]], [E[i, 0]], "o", ms=11, mfc="none", mec=ACCENT, mew=1.6,
            zorder=6)

ax.axhline(GAP_D, color=REF_D, lw=1.8, ls="--", dashes=(5, 3), zorder=4)
ax.axhline(GAP_I, color=REF_I, lw=1.4, ls=":", zorder=4)

x0, x1 = float(s[0]), float(s[-1])
ax.set_xlim(x0, x1)
lo = min(E[:, 0].min(), GAP_I) - 0.12
hi = max(E[:, :NBR].max(), GAP_D) + 0.10
ax.set_ylim(lo, hi)

# direct labels (identity is never colour-alone)
ax.text(x1 - 0.012 * (x1 - x0), GAP_D - 0.022 * (hi - lo),
        f"G$_0$W$_0$ direct gap @ K  {GAP_D:.4f} eV", color=REF_D,
        ha="right", va="top", fontsize=8.6, fontweight="bold")
ax.text(x1 - 0.012 * (x1 - x0), GAP_I - 0.020 * (hi - lo),
        f"indirect gap  {GAP_I:.4f} eV", color=REF_I,
        ha="right", va="top", fontsize=8.0)
ax.text(x0 + 0.010 * (x1 - x0), E1G - 0.028 * (hi - lo),
        r"$E_1$", color=ACCENT, ha="left", va="top",
        fontsize=10.5, fontweight="bold")

# binding-energy annotation, pinned on the Γ axis itself
ax.annotate("", xy=(x0, GAP_D), xytext=(x0, E1G),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3,
                            shrinkA=0, shrinkB=0), zorder=7)
ax.text(x0 + 0.010 * (x1 - x0), 0.5 * (GAP_D + E1G),
        f"$E_b(\\Gamma)$ = {Eb*1e3:.0f} meV", color=INK, ha="left", va="center",
        fontsize=9.6, fontweight="bold",
        bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=2.0), zorder=8)

# mark the dispersion minimum
ax.annotate(f"min at {qlabel(Q[imin])}: {E1min:.4f} eV\n"
            f"({(E1G-E1min)*1e3:.0f} meV below $E_1(\\Gamma)$)",
            xy=(s[imin], E1min), xytext=(s[imin], lo + 0.030 * (hi - lo)),
            color=INK, fontsize=8.4, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9), zorder=8)

# high-symmetry ticks, plus Λ as a labelled minor tick
ax.set_xticks(tick_x)
ax.set_xticklabels(tick_l, fontsize=11, color=INK)
lam = [s[i] for i, nm in special if nm == r"$\Lambda$"]
if lam:
    ax.set_xticks(lam, minor=True)
    ax.set_xticklabels([r"$\Lambda$"] * len(lam), minor=True, fontsize=9.5,
                       color=INK2)
    ax.tick_params(axis="x", which="minor", length=3, pad=2)
    for xv in lam:
        ax.axvline(xv, color=MUTED, lw=0.6, alpha=0.35, ls=(0, (3, 3)), zorder=1)
ax.set_ylabel(r"$E_S(\mathbf{Q})$  (eV)", fontsize=10.5, color=INK)
ax.set_title(TITLE, fontsize=10.8, color=INK, pad=9)
ax.tick_params(colors=INK2, labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(MUTED)

handles = [
    Line2D([], [], color=ACCENT, lw=2.4, marker="o", ms=6, mec=SURFACE,
           label=r"$E_1$ (lowest exciton)"),
    Line2D([], [], color=cmap(0.42), lw=1.6, marker="o", ms=3.4,
           label=r"$E_2\ldots E_{%d}$" % NBR),
    Line2D([], [], color=REF_D, lw=1.8, ls="--", label="G$_0$W$_0$ direct gap (K)"),
    Line2D([], [], color=REF_I, lw=1.4, ls=":", label="G$_0$W$_0$ indirect gap"),
]
leg = ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=8.2,
                frameon=True, framealpha=0.92, edgecolor=MUTED, borderpad=0.5,
                bbox_to_anchor=(0.50, 0.83))
leg.get_frame().set_facecolor(SURFACE)
for t in leg.get_texts():
    t.set_color(INK2)

sub1 = (f"12$\\times$12 k · 80 Ry · $n_\\mu$=2412 · fH window 28 bands · "
        f"{nQ} $\\mathbf{{Q}}$ on the BSE grid, exact exchange tiles · "
        "QP energies, no scissor")
sub2 = ("ringed = valley-commensurate $\\mathbf{Q}$ (resolved by a "
        "12$\\times$12 mesh); intermediate $\\mathbf{Q}$ are mesh-limited")
fig.text(0.010, 0.036, sub1, fontsize=6.6, color=MUTED, ha="left")
fig.text(0.010, 0.010, sub2, fontsize=6.6, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0.060, 1, 1))
fig.savefig(OUT, dpi=220, facecolor=SURFACE)
print(f"Wrote {OUT}")

for k, v in summary.items():
    print(f"  {k:<28s} {v}")
jp = os.environ.get("EX_JSON")
if jp:
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"Wrote {jp}")
