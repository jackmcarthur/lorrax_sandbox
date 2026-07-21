"""Tasks 1-2: scissor(k,n) = E_qp - E_dft smoothness-by-bin + clamp physics.

Pure NumPy. Parses eqp0.dat / eqp1.dat / sigma_freq_debug.dat from the
recovered-D3h GW run and characterizes the QP scissor for ALL bands, binned by
where E_dft-Ef sits relative to the +/-10 eV Sigma_c(w) window.

Window: fermi_reference=midgap, Sigma_c grid = [-10,10] eV RELATIVE to E_F.
Clamp engages (Sigma evaluated at edge) when |E_dft - E_F| > 10 eV.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.environ.get("SCISSOR_RUN",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu")
OUT = os.environ.get("SCISSOR_OUT",
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
WIN = 10.0  # +/- eV window around E_F


# ---------- parsers ----------
def parse_eqp(path):
    """{ik: {band(1-idx): (Edft, Eqp)}}, and k-frac list in file order."""
    data = {}
    kfrac = []
    ik = -1
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) == 4 and ("." in p[0]):          # k-block header
                ik += 1
                data[ik] = {}
                kfrac.append((float(p[0]), float(p[1]), float(p[2])))
            elif len(p) == 4:                           # band row
                b = int(p[1])
                data[ik][b] = (float(p[2]), float(p[3]))
    return data, np.array(kfrac)


def parse_sigma_freq_debug_v2(path):
    cols = None
    data = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#'):
            pp = s.lstrip('#').split()
            if len(pp) >= 3 and pp[0] == 'k' and pp[1] == 'n':
                cols = pp[2:]
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
        data[(k, n + 1)] = {c: (np.nan if v == 'nan' else float(v))
                            for c, v in zip(cols, p[2:])}
    return cols, data


# ---------- load ----------
eqp0, kfrac = parse_eqp(os.path.join(RUN, "eqp0.dat"))
eqp1, _ = parse_eqp(os.path.join(RUN, "eqp1.dat"))
cols, fdbg = parse_sigma_freq_debug_v2(os.path.join(RUN, "sigma_freq_debug.dat"))

nk = len(eqp0)
nb = len(eqp0[0])
print(f"nk={nk} nb={nb}  freq_debug cols={cols}")

# (nk, nb) arrays (band index 0..nb-1 <-> file band 1..nb)
Edft = np.array([[eqp0[k][b + 1][0] for b in range(nb)] for k in range(nk)])
Eqp0 = np.array([[eqp0[k][b + 1][1] for b in range(nb)] for k in range(nk)])
Eqp1 = np.array([[eqp1[k][b + 1][1] for b in range(nb)] for k in range(nk)])

# Fermi energy from freq_debug (Edft - (Edft-Ef))
ef_samples = []
for (k, n), d in fdbg.items():
    if 'E_dft' in d and 'Edft-Ef' in d:
        ef_samples.append(d['E_dft'] - d['Edft-Ef'])
EF = float(np.median(ef_samples))
print(f"E_F = {EF:.4f} eV   window (abs) = [{EF-WIN:.3f}, {EF+WIN:.3f}] eV")

Erel = Edft - EF                       # (nk, nb) E_dft - E_F
scissor0 = Eqp0 - Edft                 # on-shell scissor
scissor1 = Eqp1 - Edft                 # Z-linearized scissor (the applied one)

# freq_debug diagnostic columns as (nk,nb)
def fdcol(name):
    A = np.full((nk, nb), np.nan)
    for k in range(nk):
        for b in range(nb):
            d = fdbg.get((k, b + 1))
            if d is not None and name in d:
                A[k, b] = d[name]
    return A

xbare = fdcol('x_bare')
sigc_re = fdcol('sig_c(Edft).Re')
sigc_im = fdcol('sig_c(Edft).Im')

# ---------- k adjacency on the 6x6 mesh ----------
# index each k by its (i,j) on a 6x6 grid
kg = 6
idx_ij = np.rint(kfrac[:, :2] * kg).astype(int) % kg          # (nk,2)
pos2k = {tuple(ij): k for k, ij in enumerate(idx_ij)}
edges = []
for k, (i, j) in enumerate(idx_ij):
    for di, dj in ((1, 0), (0, 1)):
        nb_ij = ((i + di) % kg, (j + dj) % kg)
        k2 = pos2k[nb_ij]
        edges.append((k, k2))
edges = np.array(edges)                                       # (nedge,2)


def nn_jump(field_kn):
    """max |Delta field| over adjacent-k edges, per band -> (nb,)."""
    d = np.abs(field_kn[edges[:, 0], :] - field_kn[edges[:, 1], :])  # (nedge, nb)
    return d.max(axis=0)


# ---------- per-band classification ----------
rel_min = Erel.min(axis=0)
rel_max = Erel.max(axis=0)
# out-of-window if outside [-WIN, WIN]
low_out = rel_max < -WIN            # entire band below lower edge (deep, always clamped)
high_out = rel_min > WIN           # entire band above upper edge (high cond, always clamped)
straddle_lo = (rel_min < -WIN) & (rel_max >= -WIN)   # crosses lower edge across k
straddle_hi = (rel_min <= WIN) & (rel_max > WIN)     # crosses upper edge across k
in_win = (rel_min >= -WIN) & (rel_max <= WIN)

bin_name = np.array(["?"] * nb, dtype=object)
bin_name[low_out] = "deep_clamped"
bin_name[high_out] = "high_clamped"
bin_name[straddle_lo] = "straddle_lo"
bin_name[straddle_hi] = "straddle_hi"
bin_name[in_win] = "in_window"

kink0 = nn_jump(scissor0)
kink1 = nn_jump(scissor1)

# fraction of (k,n) clamped
clamped_kn = (np.abs(Erel) > WIN)
print(f"clamped fraction = {clamped_kn.mean()*100:.1f}%  ({clamped_kn.sum()}/{nk*nb})")

# ---------- summary table by bin ----------
print("\n=== per-band bin summary ===")
order = ["deep_clamped", "straddle_lo", "in_window", "straddle_hi", "high_clamped"]
rows = []
for bn in order:
    mask = bin_name == bn
    if mask.sum() == 0:
        continue
    bands = np.where(mask)[0]
    print(f"\n[{bn}]  {mask.sum()} bands  (idx {bands.min()}..{bands.max()})")
    print(f"    Erel range over these bands: [{rel_min[mask].min():+.2f}, {rel_max[mask].max():+.2f}] eV")
    print(f"    scissor1 range: [{scissor1[:,mask].min():+.2f}, {scissor1[:,mask].max():+.2f}] eV")
    print(f"    kink1 (max NN |Dscissor1|): mean={kink1[mask].mean()*1e3:.1f} meV  "
          f"max={kink1[mask].max()*1e3:.1f} meV")
    print(f"    kink0 (on-shell)          : mean={kink0[mask].mean()*1e3:.1f} meV  "
          f"max={kink0[mask].max()*1e3:.1f} meV")
    rows.append((bn, int(mask.sum()), int(bands.min()), int(bands.max()),
                 kink1[mask].mean()*1e3, kink1[mask].max()*1e3,
                 kink0[mask].mean()*1e3, kink0[mask].max()*1e3))

# per-band detail near the lower boundary (the interesting transition)
print("\n=== per-band detail (bands 0..30) ===")
print(f"{'b':>3} {'Erel_min':>9} {'Erel_max':>9} {'bin':>13} "
      f"{'sc1_mean':>9} {'sc1_range':>10} {'kink1_meV':>10} {'kink0_meV':>10}")
for b in range(min(nb, 31)):
    print(f"{b:>3} {rel_min[b]:>9.2f} {rel_max[b]:>9.2f} {bin_name[b]:>13} "
          f"{scissor1[:,b].mean():>9.2f} "
          f"{scissor1[:,b].max()-scissor1[:,b].min():>10.3f} "
          f"{kink1[b]*1e3:>10.1f} {kink0[b]*1e3:>10.1f}")

np.savez(os.path.join(OUT, "scissor_data.npz"),
         Edft=Edft, Eqp0=Eqp0, Eqp1=Eqp1, Erel=Erel, EF=EF,
         scissor0=scissor0, scissor1=scissor1,
         kink0=kink0, kink1=kink1, bin_name=bin_name.astype(str),
         xbare=xbare, sigc_re=sigc_re, sigc_im=sigc_im,
         rel_min=rel_min, rel_max=rel_max, kfrac=kfrac, idx_ij=idx_ij)

# ---------- PLOT 1: scissor(k) for representative bands of each bin ----------
# pick a representative band per bin: deepest, the straddler with biggest kink, an in-window near-gap
b_deep = 0
b_strad = None
strad_bands = np.where(straddle_lo)[0]
if len(strad_bands):
    b_strad = strad_bands[np.argmax(kink1[strad_bands])]
b_inwin = 25  # VBM-ish (nval=26 -> bands 0..25 valence)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=140)
# order k for a readable 1D trace: lexicographic by (i,j)
kord = np.lexsort((idx_ij[:, 1], idx_ij[:, 0]))
for ax, b, ttl in [
    (axes[0], b_deep, f"deep semicore b={b_deep}\nErel~{Erel[:,b_deep].mean():.1f} eV (always clamped)"),
    (axes[1], b_strad if b_strad is not None else 12,
     f"straddle-lo b={b_strad}\nErel in [{rel_min[b_strad]:.1f},{rel_max[b_strad]:.1f}] (clamp toggles)"
     if b_strad is not None else "no straddle band"),
    (axes[2], b_inwin, f"in-window b={b_inwin}\nErel~{Erel[:,b_inwin].mean():.1f} eV (unclamped)"),
]:
    ax.plot(range(nk), scissor1[kord, b], "o-", ms=4, label="scissor1 (E_qp1-E_dft)")
    ax.plot(range(nk), scissor0[kord, b], "s--", ms=3, alpha=0.6, label="scissor0 (on-shell)")
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("k index (lex-ordered 6x6)")
    ax.set_ylabel("scissor (eV)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("QP scissor E_qp-E_dft vs k for three bins (recovered-D3h MoS2 GW, 6x6)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "01_scissor_vs_k_bins.png"))
print(f"\nsaved 01_scissor_vs_k_bins.png  (b_deep={b_deep}, b_strad={b_strad}, b_inwin={b_inwin})")

# ---------- PLOT 2: kink metric vs band, colored by bin + Erel ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=140, sharex=True)
colmap = {"deep_clamped": "#1f4e79", "straddle_lo": "#c0392b", "in_window": "#2e7d32",
          "straddle_hi": "#e67e22", "high_clamped": "#7d3c98", "?": "#888"}
cols_b = [colmap[bn] for bn in bin_name]
a1.bar(range(nb), kink1 * 1e3, color=cols_b)
a1.set_yscale("log")
a1.set_ylabel("kink1 = max NN |Dscissor1| (meV)")
a1.set_title("Scissor jaggedness per band (log). Red=straddle_lo is where the clamp toggles.")
a1.grid(alpha=0.3, axis="y")
# Erel band envelope
a2.fill_between(range(nb), rel_min, rel_max, color="#bbb", alpha=0.6, label="E_dft-Ef range over k")
a2.plot(range(nb), Erel.mean(axis=0), "k-", lw=1, label="mean")
a2.axhline(-WIN, color="r", ls="--", label="-10 eV clamp edge")
a2.axhline(WIN, color="r", ls="--")
a2.set_ylabel("E_dft - E_F (eV)")
a2.set_xlabel("band index")
a2.legend(fontsize=8)
a2.grid(alpha=0.3)
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=l) for l, c in colmap.items() if l != "?"]
a1.legend(handles=handles, fontsize=7, ncol=5)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "02_kink_vs_band.png"))
print("saved 02_kink_vs_band.png")

# ---------- PLOT 3: physical plausibility of the deep-semicore clamp ----------
# scissor vs Erel for all (k,n); overlay bare-exchange-dominated expectation
fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
sc = ax.scatter(Erel.ravel(), scissor1.ravel(), c=np.abs(Erel.ravel()) > WIN,
                cmap="coolwarm", s=10, alpha=0.6)
ax.axvline(-WIN, color="k", ls="--", lw=0.8)
ax.axvline(WIN, color="k", ls="--", lw=0.8)
ax.set_xlabel("E_dft - E_F (eV)")
ax.set_ylabel("scissor1 = E_qp1 - E_dft (eV)")
ax.set_title("Scissor vs DFT energy (color = clamped). "
             "Deep bands: large smooth exchange-dominated shift.")
ax.grid(alpha=0.3)
fig.colorbar(sc, label="clamped (|Erel|>10)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "03_scissor_vs_Erel.png"))
print("saved 03_scissor_vs_Erel.png")

print("\nDONE scissor_analysis")
