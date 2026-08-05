#!/usr/bin/env python3
"""GN-PPM scissor / per-k Sigma instability diagnostic for the MoS2 BSE-figures run.

READ-ONLY over the run outputs.  Uses the compare-skill v2 header-driven parser
for sigma_freq_debug.dat and a small eqp.dat parser.  Produces:
  - scissor_table.txt        (per-bin scissor stats + anomaly flags)
  - blown_im_map.txt         (per-(k,band) |Im sig_c| outliers, in/out of window)
  - symmetry_break.txt       (per-equivalence-class V_H/kin_ion/x_bare/sig_c spread)
  - scissor_vs_edft.png      (scissor vs E_dft, binned/colored)
  - im_sigma_vs_edft.png     (|Im sig_c(Edft)| vs E_dft-Ef, window overlaid)
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/00_lorrax_gw_gnppm"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_scissor_eval_2026-07-20"
RY = 13.6056980659
NELEC = 26                # VBM = band 26 (1-indexed)
WIN_LO, WIN_HI = -10.0, 10.0   # sigma_omega grid edges (eV, rel E_F)

# ---------------------------------------------------------------- parsers
def parse_sigma_freq_debug_v2(path):
    """compare-skill v2 header-driven parser. data[(k,n_phys)] -> {col:val}."""
    cols = None; data = {}
    for line in open(path):
        s = line.strip()
        if s.startswith("#"):
            p = s.lstrip("#").split()
            if len(p) >= 3 and p[0] == "k" and p[1] == "n":
                cols = p[2:]
            continue
        if not s or cols is None:
            continue
        p = s.split()
        if len(p) != len(cols) + 2:
            continue
        try:
            k, n = int(p[0]), int(p[1])
        except ValueError:
            continue
        data[(k, n + 1)] = {c: (np.nan if v == "nan" else float(v))
                            for c, v in zip(cols, p[2:])}
    return cols, data

def parse_eqp(path):
    """eqp.dat -> {(k,band1): (Edft, Eqp)}, k 0-indexed, band 1-indexed."""
    data = {}; ik = -1
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) == 4 and "." in p[0] and "." in p[1]:   # k header: kx ky kz nb
            ik += 1
            continue
        if len(p) == 4:                                    # band row
            data[(ik, int(p[1]))] = (float(p[2]), float(p[3]))
    return data, ik + 1

def parse_kcoords(path):
    ks = []
    for line in open(path):
        p = line.split()
        if len(p) == 4 and "." in p[0] and "." in p[1]:
            ks.append((float(p[0]), float(p[1]), float(p[2])))
    return ks

# ---------------------------------------------------------------- load
cols, fd = parse_sigma_freq_debug_v2(os.path.join(RUN, "sigma_freq_debug.dat"))
eqp1, nk = parse_eqp(os.path.join(RUN, "eqp1.dat"))
eqp0, _ = parse_eqp(os.path.join(RUN, "eqp0.dat"))
kcoords = parse_kcoords(os.path.join(RUN, "eqp1.dat"))
print(f"freq_debug cols = {cols}")
print(f"nk={nk}, n bands in eqp1 = {max(b for (_,b) in eqp1)}")

nband = max(b for (_, b) in eqp1)
kset = sorted({k for (k, _) in eqp1})

# per-(k,band) arrays
rows = []
for k in kset:
    for b in range(1, nband + 1):
        ed, eq = eqp1[(k, b)]
        d = fd.get((k, b), {})
        kin = d.get("kin_ion", np.nan); vh = d.get("V_H", np.nan)
        xb = d.get("x_bare", np.nan)
        scre = d.get("sig_c(Edft).Re", np.nan); scim = d.get("sig_c(Edft).Im", np.nan)
        edrel = d.get("Edft-Ef", np.nan)
        vxc = ed - kin - vh          # Vxc = E_dft - <T+Vion> - <V_H>
        rows.append(dict(k=k, b=b, Edft=ed, Eqp=eq, scissor=eq - ed,
                         Edft_rel=edrel, kin_ion=kin, VH=vh, xbare=xb,
                         sigc_re=scre, sigc_im=scim, Vxc=vxc,
                         sig_m_vxc=xb + scre - vxc))
import numpy as np
K = np.array([r["k"] for r in rows]); B = np.array([r["b"] for r in rows])
Edft = np.array([r["Edft"] for r in rows]); Eqp = np.array([r["Eqp"] for r in rows])
Scis = np.array([r["scissor"] for r in rows])
Edrel = np.array([r["Edft_rel"] for r in rows])
VH = np.array([r["VH"] for r in rows]); KIN = np.array([r["kin_ion"] for r in rows])
XB = np.array([r["xbare"] for r in rows]); SCRE = np.array([r["sigc_re"] for r in rows])
SCIM = np.array([r["sigc_im"] for r in rows]); VXC = np.array([r["Vxc"] for r in rows])

# VBM/CBM from DFT grid
vb_mask = B <= NELEC; cb_mask = B > NELEC
E_VBM = Edft[vb_mask].max(); E_CBM = Edft[cb_mask].min()
EF = 0.5 * (E_VBM + E_CBM)
print(f"E_VBM(dft)={E_VBM:.4f}  E_CBM(dft)={E_CBM:.4f}  midgap EF={EF:.4f} eV  DFT indirect gap={E_CBM-E_VBM:.4f}")

# ============================================================ (1) SCISSOR BINS
# bins relative to VBM/CBM
near = np.abs(Edft - E_VBM) < 10.0
far_below = Edft < (E_VBM - 10.0)
far_above = Edft > (E_CBM + 10.0)
mid = ~(near | far_below | far_above)   # states between +10 above VBM and CBM+10 not caught

def stats(mask, name):
    s = Scis[mask]
    if s.size == 0:
        return f"{name:12s}  n=0"
    return (f"{name:12s}  n={s.size:5d}  scissor mean={s.mean():+8.3f}  "
            f"median={np.median(s):+8.3f}  std={s.std():7.3f}  "
            f"min={s.min():+9.2f}  max={s.max():+9.2f} eV")

lines = []
lines.append("=== SCISSOR (E_qp - E_dft) BY E_dft BIN (relative to DFT VBM/CBM) ===")
lines.append(f"E_VBM={E_VBM:.4f} eV, E_CBM={E_CBM:.4f} eV, midgap E_F={EF:.4f} eV, sigma window rel-EF [{WIN_LO},{WIN_HI}] eV")
lines.append(f"  window in absolute E_dft: [{EF+WIN_LO:.3f}, {EF+WIN_HI:.3f}] eV")
lines.append("")
lines.append(stats(near, "NEAR-GAP"))
lines.append(f"             (|E_dft - VBM| < 10 eV)")
lines.append(stats(far_below, "FAR-BELOW"))
lines.append(f"             (semicore, E_dft < VBM-10)")
lines.append(stats(far_above, "FAR-ABOVE"))
lines.append(f"             (high cond, E_dft > CBM+10)")
if mid.any():
    lines.append(stats(mid, "MID"))
lines.append("")
# fraction of states with E_dft outside the sigma window (=> clamped)
outside = (Edrel < WIN_LO) | (Edrel > WIN_HI)
lines.append(f"States with E_dft-Ef OUTSIDE sigma window [{WIN_LO},{WIN_HI}] eV (=> interp CLAMPED to edge): "
             f"{outside.sum()}/{outside.size} = {100*outside.mean():.1f}%   (gw.out QSGW-clip: 68.2%)")
lines.append("")

# anomalous states: scissor sign wrong or |scissor| >> near-gap scale
near_scale = np.abs(Scis[near]).mean()
lines.append(f"Near-gap mean |scissor| scale = {near_scale:.3f} eV")
anom = np.abs(Scis) > 5.0 * max(near_scale, 0.5)
lines.append(f"States with |scissor| > 5x near-gap scale ({5*max(near_scale,0.5):.2f} eV): {anom.sum()}")
# biggest offenders
order = np.argsort(-np.abs(Scis))
lines.append("")
lines.append("Top-20 |scissor| offenders (k, band, E_dft, E_qp, scissor, Edft-Ef, in-window?):")
for i in order[:20]:
    inw = "IN " if (WIN_LO <= Edrel[i] <= WIN_HI) else "OUT"
    lines.append(f"  k={K[i]:2d} b={B[i]:3d}  Edft={Edft[i]:+9.3f}  Eqp={Eqp[i]:+10.3f}  "
                 f"scis={Scis[i]:+10.3f}  Edft-Ef={Edrel[i]:+8.2f}  [{inw}]")
open(os.path.join(OUT, "scissor_table.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ============================================================ (2) BLOWN-UP IM Sigma MAP
im_abs = np.abs(SCIM)
blines = []
blines.append("=== BLOWN-UP IMAGINARY Sigma_c(E_dft) MAP ===")
blines.append("Physical |Im Sigma_c| near the gap should be < ~1 eV (QP lifetime).")
blines.append(f"|Im sig_c| stats over all {im_abs.size} (k,band): "
              f"median={np.median(im_abs):.3f}  mean={im_abs.mean():.1f}  max={im_abs.max():.1f} eV")
for thr in (1, 5, 50, 500):
    m = im_abs > thr
    inw = m & (Edrel >= WIN_LO) & (Edrel <= WIN_HI)
    blines.append(f"  |Im sig_c| > {thr:5d} eV : {m.sum():5d} states   "
                  f"({inw.sum()} of them are IN-window |Edft-Ef|<=10)")
blines.append("")
# IN-WINDOW blown-up states (the ones the coordinator flagged as the real defect)
inwin = (Edrel >= WIN_LO) & (Edrel <= WIN_HI)
big_inwin = inwin & (im_abs > 5.0)
blines.append(f"IN-WINDOW states with |Im sig_c| > 5 eV (near-gap PPM instability): {big_inwin.sum()}")
blines.append("  (k, band, Edft-Ef, Re sig_c, Im sig_c, scissor):")
oi = np.argsort(-im_abs)
shown = 0
for i in oi:
    if not inwin[i]:
        continue
    if im_abs[i] <= 5.0:
        break
    blines.append(f"  k={K[i]:2d} b={B[i]:3d}  Edft-Ef={Edrel[i]:+7.2f}  "
                  f"Re={SCRE[i]:+9.3f}  Im={SCIM[i]:+12.2f}  scis={Scis[i]:+8.3f}")
    shown += 1
    if shown >= 40:
        blines.append("  ... (truncated)")
        break
blines.append("")
# Specifically the Gamma VBM-region state the coordinator cited
blines.append("Gamma (k=0) near-gap valence rows (bands 24-27):")
for b in (24, 25, 26, 27):
    i = np.where((K == 0) & (B == b))[0]
    if i.size:
        i = i[0]
        blines.append(f"  b={b:3d}  Edft={Edft[i]:+8.3f} (rel {Edrel[i]:+6.2f})  "
                      f"Re sig_c={SCRE[i]:+8.3f}  Im sig_c={SCIM[i]:+11.2f}  "
                      f"Eqp={Eqp[i]:+8.3f}  scis={Scis[i]:+7.3f}")
open(os.path.join(OUT, "blown_im_map.txt"), "w").write("\n".join(blines) + "\n")
print("\n".join(blines))

# ============================================================ (3) SYMMETRY BREAKING
# Group k-points into equivalence classes by identical sorted DFT spectrum.
# Symmetry-equivalent k share E_dft(all bands); V_H/kin_ion/x_bare must match
# (depend only on |psi_nk> and symmetric potentials); if sig_c also matches,
# the whole Sigma is symmetric.
spectra = {}
for k in kset:
    vec = tuple(round(Edft[(K == k) & (B == b)][0], 5) for b in range(1, nband + 1))
    spectra.setdefault(vec, []).append(k)
classes = [v for v in spectra.values() if len(v) > 1]
slines = []
slines.append("=== SYMMETRY-BREAKING ACROSS EQUIVALENT k (identical DFT spectrum) ===")
slines.append(f"{len(spectra)} distinct DFT spectra among {len(kset)} k-points; "
              f"{len(classes)} multi-member (symmetry-equivalent) classes.")
slines.append("For each class we report the MAX spread (max-min) over equivalent k, per band,")
slines.append("of each Sigma ingredient.  kin_ion/V_H/x_bare depend only on psi & symmetric")
slines.append("potentials => nonzero spread there = psi/loader symmetry broken.  If only")
slines.append("sig_c spreads => ISDF/centroid/screening path breaks symmetry.")
slines.append("")

def class_spread(karr, arr):
    """max over bands of (max-min across the equivalent k) for quantity arr."""
    worst = 0.0; worst_b = -1
    for b in range(1, nband + 1):
        vals = np.array([arr[(K == k) & (B == b)][0] for k in karr])
        sp = np.nanmax(vals) - np.nanmin(vals)
        if np.isfinite(sp) and sp > worst:
            worst = sp; worst_b = b
    return worst, worst_b

overall = {q: 0.0 for q in ("kin_ion", "VH", "xbare", "sigc_re", "scissor")}
arrs = dict(kin_ion=KIN, VH=VH, xbare=XB, sigc_re=SCRE, scissor=Scis)
for ci, karr in enumerate(sorted(classes, key=lambda c: -len(c))):
    kc = kcoords[karr[0]]
    slines.append(f"Class {ci}: k-indices {karr}  (mult={len(karr)})  rep-coord={kc}")
    for q, arr in arrs.items():
        sp, wb = class_spread(karr, arr)
        overall[q] = max(overall[q], sp)
        tag = "  <== BREAKS" if sp > 0.05 else ""
        slines.append(f"    max spread {q:9s} = {sp:9.4f} eV  (worst band {wb}){tag}")
    slines.append("")
slines.append("=== WORST-CASE SPREAD OVER ALL EQUIVALENT-k CLASSES ===")
for q in ("kin_ion", "VH", "xbare", "sigc_re", "scissor"):
    slines.append(f"  {q:9s}: {overall[q]:.4f} eV")
slines.append("")
psi_side = max(overall["kin_ion"], overall["VH"], overall["xbare"])
slines.append(f"psi-side max spread (kin_ion/V_H/x_bare) = {psi_side:.4f} eV")
slines.append(f"sig_c-only spread                        = {overall['sigc_re']:.4f} eV")
if psi_side > 0.05:
    slines.append(">>> psi/loader symmetry is BROKEN (V_H/kin_ion/x_bare differ between equivalent k).")
else:
    slines.append(">>> psi-side (V_H/kin_ion/x_bare) is symmetric; breaking is in the sig_c/ISDF path only.")
open(os.path.join(OUT, "symmetry_break.txt"), "w").write("\n".join(slines) + "\n")
print("\n".join(slines))

# ============================================================ (4) RAW GAPS
glines = []
glines.append("=== RAW QP GAPS FROM eqp1.dat (no interpolation) ===")
# global indirect: min_k Eqp(band27) - max_k Eqp(band26)
eqp_b26 = np.array([eqp1[(k, 26)][1] for k in kset])
eqp_b27 = np.array([eqp1[(k, 27)][1] for k in kset])
dft_b26 = np.array([eqp1[(k, 26)][0] for k in kset])
dft_b27 = np.array([eqp1[(k, 27)][0] for k in kset])
glines.append(f"GLOBAL indirect QP gap = min_k Eqp(27) - max_k Eqp(26) = "
              f"{eqp_b27.min():.4f} - {eqp_b26.max():.4f} = {eqp_b27.min()-eqp_b26.max():.4f} eV")
glines.append(f"GLOBAL indirect DFT gap = {dft_b27.min()-dft_b26.max():.4f} eV")
# direct gap per k, min over k
direct_qp = eqp_b27 - eqp_b26
direct_dft = dft_b27 - dft_b26
kmin = kset[int(np.argmin(direct_qp))]
glines.append(f"MIN direct QP gap (over k) = {direct_qp.min():.4f} eV at k-index {kmin} coord {kcoords[kmin]}")
# K point specifically
Kidx = [i for i, c in enumerate(kcoords)
        if abs(c[0]-1/3) < 1e-3 and abs(c[1]-1/3) < 1e-3]
if Kidx:
    ki = Kidx[0]
    glines.append(f"K-point (1/3,1/3,0) idx {ki}: DFT direct gap b27-b26 = {eqp1[(ki,27)][0]-eqp1[(ki,26)][0]:.4f}, "
                  f"QP direct gap = {eqp1[(ki,27)][1]-eqp1[(ki,26)][1]:.4f} eV")
glines.append("")
glines.append("Per-k VBM(b26)/CBM(b27) QP energies:")
for k in kset:
    glines.append(f"  k={k:2d} {kcoords[k]}  Eqp26={eqp1[(k,26)][1]:+8.3f}  Eqp27={eqp1[(k,27)][1]:+8.3f}  "
                  f"direct={eqp1[(k,27)][1]-eqp1[(k,26)][1]:+7.3f}")
open(os.path.join(OUT, "raw_gaps.txt"), "w").write("\n".join(glines) + "\n")
print("\n".join(glines[:8]))

# ============================================================ PLOTS
# scissor vs E_dft
fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
ax.scatter(Edft[near], Scis[near], s=14, c="#2c7fb8", label="near-gap (|E-VBM|<10)")
ax.scatter(Edft[far_below], Scis[far_below], s=22, c="#d95f0e", marker="v", label="far-below (semicore)")
ax.scatter(Edft[far_above], Scis[far_above], s=22, c="#c0392b", marker="^", label="far-above (high cond)")
if mid.any():
    ax.scatter(Edft[mid], Scis[mid], s=12, c="0.6", label="mid")
ax.axvline(E_VBM, color="k", ls=":", lw=0.8); ax.axvline(E_CBM, color="k", ls=":", lw=0.8)
ax.axvspan(EF+WIN_LO, EF+WIN_HI, color="green", alpha=0.08, label="sigma window (unclamped)")
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xlabel("E_dft (eV)"); ax.set_ylabel("scissor E_qp - E_dft (eV)")
ax.set_title("GN-PPM scissor vs E_dft (MoS2 6x6, all k, bands 1-100)")
ax.legend(fontsize=8, loc="upper center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "scissor_vs_edft.png"))

# |Im sig_c| vs Edft-Ef
fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
sc = ax.scatter(Edrel, np.abs(SCIM) + 1e-3, s=14, c=np.abs(Scis),
                cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=200))
ax.set_yscale("log")
ax.axvspan(WIN_LO, WIN_HI, color="green", alpha=0.1, label="unclamped window")
ax.set_xlabel("E_dft - E_F (eV)"); ax.set_ylabel("|Im sig_c(E_dft)| (eV)")
ax.set_title("Blown-up imaginary Sigma_c vs energy (color=|scissor|)")
plt.colorbar(sc, label="|scissor| (eV)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "im_sigma_vs_edft.png"))
print("\nWROTE plots + tables to", OUT)
