"""Task 3 (lean, progress-logged): htransform smoothness sweep.
Reduced window set; per-window timing; direct-flush logging (no tail buffering).
"""
import os, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
RUN = os.environ.get("SCISSOR_RUN",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu")
OUT = os.environ.get("SCISSOR_OUT",
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
INP = os.path.join(RUN, "gwbands.in")
NVAL, NCOND, NBAND = 26, 64, 90
NKX = NKY = 6
def log(*a): print(*a, flush=True)


def parse_eqp(path):
    data = {}; ik = -1
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) == 4 and ("." in p[0]):
            ik += 1; data[ik] = {}
        elif len(p) == 4:
            data[ik][int(p[1])] = (float(p[2]), float(p[3]))
    return data


t0 = time.time()
mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input(INP)
params["nval"], params["ncond"], params["nband"] = NVAL, NCOND, NBAND
(wfn, sym, meta, _m, _S, ctilde, B, enk_dft) = ht.initialize_wfns(INP, params, log, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz)); nk = kg[0]*kg[1]*kg[2]
rank = int(ctilde.shape[2]); nb_ret = min(int(ctilde.shape[1]), rank)
enk_dft = np.asarray(jax.device_get(enk_dft))
if enk_dft.shape[0] == nk:
    enk_dft = enk_dft.T
EF = float(wfn.efermi) * RY
log(f"[cfg] kg={kg} nk={nk} nb_ret={nb_ret} rank={rank} EF={EF:.4f} eV  build={time.time()-t0:.1f}s", )

eqp1 = parse_eqp(os.path.join(RUN, "eqp1.dat"))
eqp0 = parse_eqp(os.path.join(RUN, "eqp0.dat"))
dS1 = np.zeros_like(enk_dft); dS0 = np.zeros_like(enk_dft)
for k in range(nk):
    for b in range(nb_ret):
        ed, e1 = eqp1[k][b + 1]; _, e0 = eqp0[k][b + 1]
        dS1[b, k] = (e1 - ed) / RY; dS0[b, k] = (e0 - ed) / RY
enk_qp1 = enk_dft + dS1
enk_qp0 = enk_dft + dS0

wfn0, _s0 = ht.setup_wfn_and_sym(os.path.join(RUN, "WFN.h5"))
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
node_idx = [int(n) for n in node_idx]; nQ = kpath.shape[0]
kgrid = np.stack(np.meshgrid(np.arange(NKX)/NKX, np.arange(NKY)/NKY, [0.0], indexing="ij"),
                 axis=-1).reshape(-1, 3)
log(f"[path] nQ={nQ} nodes={node_idx}")


def interp(enk, b_lo, b_hi, qlist):
    ct = ctilde[:, b_lo:b_hi, :]
    en = jax.numpy.asarray(enk[b_lo:b_hi, :])
    bnd = compute_wfns_fi(ctilde=ct, B_at_mu=B, enk_sigma=en, kgrid_co=kg,
                          band_window_fi=(0, b_hi - b_lo), mesh_xy=mesh_xy,
                          q_list=jax.numpy.asarray(qlist), a_band_index=None,
                          log_fn=(lambda *a, **k: None))
    return np.asarray(jax.device_get(bnd.enk_full)) * RY


def smooth_metric(E_path):
    nnj = np.abs(np.diff(E_path, axis=0)).max(axis=0) * 1e3
    d2 = np.abs(E_path[2:] - 2*E_path[1:-1] + E_path[:-2]).max(axis=0) * 1e3
    return nnj, d2


def recon_err(E_grid, enk, b_lo, b_hi):
    inp = np.sort((enk[b_lo:b_hi, :].T) * RY, axis=1)
    out = np.sort(E_grid, axis=1)
    return np.abs(out - inp).max(axis=0) * 1e3


# SCISSOR_WINDOWS="0:2,0:12,..." overrides the default window list (used by
# reports/gw_conduction_postfix_2026-07-21 to locate the exact b_max at which
# the QP interpolation breaks).
_w = os.environ.get("SCISSOR_WINDOWS", "")
windows0 = ([tuple(int(v) for v in w.split(":")) for w in _w.split(",")] if _w
            else [(0, 2), (0, 12), (0, 14), (0, 26), (0, 40), (0, 50), (0, 90)])
windows_clean = [(0, 30), (14, 30), (0, 14)]  # leakage: full vs no-semicore vs semicore-only(near-gap absent)

results = {}
log("\n=== sweep (start=0). d2=max|2nd-diff| along path (meV), recon=on-grid (meV) ===")
log(f"{'win':>9} {'a_Ry':>8} {'shift_eV':>9} {'DFTd2':>9} {'QPd2':>10} {'DFTrec':>8} {'QPrec':>10} {'t':>6}")
for (b0, b1) in windows0:
    tt = time.time()
    Ep_dft = interp(enk_dft, b0, b1, kpath); Ep_qp = interp(enk_qp1, b0, b1, kpath)
    Eg_dft = interp(enk_dft, b0, b1, kgrid); Eg_qp = interp(enk_qp1, b0, b1, kgrid)
    nnj_d, d2_d = smooth_metric(Ep_dft); nnj_q, d2_q = smooth_metric(Ep_qp)
    rc_d = recon_err(Eg_dft, enk_dft, b0, b1); rc_q = recon_err(Eg_qp, enk_qp1, b0, b1)
    _, a_f, n_f, shift = ht.f_transform_eigs(jax.numpy.asarray(enk_dft[b0:b1, :]), None)
    results[(b0, b1)] = dict(Ep_dft=Ep_dft, Ep_qp=Ep_qp, d2_d=d2_d, d2_q=d2_q, rc_d=rc_d, rc_q=rc_q,
                             a=float(a_f), shift=float(shift)*RY)
    log(f"[{b0:>2},{b1:>3}) {float(a_f):>8.3f} {float(shift)*RY:>9.2f} {d2_d.max():>9.1f} "
        f"{d2_q.max():>10.1f} {rc_d.max():>8.1f} {rc_q.max():>10.1f} {time.time()-tt:>5.1f}s")

log("\n=== leakage: near-gap abs bands 24,25(VBM) 26,27(CBM) recon(meV, DFT) & path d2(meV,DFT) ===")
leak = {}
for (b0, b1) in windows_clean:
    tt = time.time()
    Eg_dft = interp(enk_dft, b0, b1, kgrid); Ep_dft = interp(enk_dft, b0, b1, kpath)
    Ep_qp = interp(enk_qp1, b0, b1, kpath)
    rc_d = recon_err(Eg_dft, enk_dft, b0, b1)
    nnj_d, d2_d = smooth_metric(Ep_dft); _, d2_q = smooth_metric(Ep_qp)
    leak[(b0, b1)] = dict(Ep_dft=Ep_dft, Ep_qp=Ep_qp, rc_d=rc_d, d2_d=d2_d, d2_q=d2_q)
    def cell(babs, arr):
        l = babs - b0
        return f"{arr[l]:.1f}" if 0 <= l < (b1 - b0) else "--"
    log(f"[{b0:>2},{b1:>3}) recon24,25,26,27 = "
        f"{cell(24,rc_d):>7} {cell(25,rc_d):>7} {cell(26,rc_d):>7} {cell(27,rc_d):>7}   "
        f"d2(DFT)24,25,26,27 = {cell(24,d2_d):>7} {cell(25,d2_d):>7} {cell(26,d2_d):>7} {cell(27,d2_d):>7}  {time.time()-tt:.1f}s")

np.savez(os.path.join(OUT, "htransform_sweep.npz"),
         x_path=x_path, node_idx=node_idx, EF=EF, nb_ret=nb_ret, rank=rank,
         windows0=np.array(windows0),
         **{f"w{b0}_{b1}_Ep_dft": r["Ep_dft"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_Ep_qp": r["Ep_qp"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_rc_d": r["rc_d"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_rc_q": r["rc_q"] for (b0, b1), r in results.items()})

# ---- plots ---- (only for the default window set)
if _w:
    log(f"\nDONE htransform_sweep (custom windows) total={time.time()-t0:.1f}s"); raise SystemExit(0)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
labs = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
def draw_nodes(ax):
    for n in node_idx:
        ax.axvline(x_path[n], color="0.85", lw=0.7, zorder=0)
    ax.set_xticks([x_path[n] for n in node_idx]); ax.set_xticklabels(labs); ax.set_xlim(x_path[0], x_path[-1])

# A: semicore-only [0,2] and deep [0,12] DFT vs QP
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
for col, (b0, b1) in enumerate([(0, 2), (0, 12)]):
    r = results[(b0, b1)]
    for row, (key, ttl, c, d2k) in enumerate([("Ep_dft", "DFT", "#2c6e8f", "d2_d"),
                                              ("Ep_qp", "QP(eqp1)", "#b5432c", "d2_q")]):
        ax = axes[row, col]; E = r[key]
        for b in range(E.shape[1]):
            ax.plot(x_path, E[:, b], lw=1.1, color=c)
        draw_nodes(ax); ax.set_ylabel("E (eV)")
        ax.set_title(f"window [{b0},{b1})  {ttl}   d2max={r[d2k].max():.0f} meV", fontsize=10)
fig.suptitle("Semicore htransform interpolation (deepest manifolds)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "04_semicore_interp.png")); log("saved 04")

# B: sweep max-d2 vs window top
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
xw = [b1 for (b0, b1) in windows0]
ax.plot(xw, [results[w]["d2_d"].max() for w in windows0], "o-", label="DFT interp (machinery)")
ax.plot(xw, [results[w]["d2_q"].max() for w in windows0], "s-", label="QP interp (DFT+scissor)")
ax.set_yscale("log"); ax.set_xlabel("window top b_max ([0,b_max))"); ax.set_ylabel("max |2nd-diff| path (meV)")
ax.axvline(12, color="0.6", ls=":"); ax.axvline(26, color="g", ls=":"); ax.axvline(46, color="purple", ls=":")
ax.text(12, ax.get_ylim()[1]*0.5, "semicore top", rotation=90, fontsize=7)
ax.text(26, ax.get_ylim()[1]*0.5, "VBM", rotation=90, fontsize=7, color="g")
ax.text(46, ax.get_ylim()[1]*0.5, "far-cond start", rotation=90, fontsize=7, color="purple")
ax.legend(); ax.grid(alpha=0.3); ax.set_title("htransform smoothness vs window (where does it break?)")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "05_sweep_smoothness.png")); log("saved 05")

# D: full window [0,90] DFT vs QP + capped [0,50]
r90 = results[(0, 90)]; r50 = results[(0, 50)]
fig, axs = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
for ax, r, tt in [(axs[0], r90, "[0,90) full"), (axs[1], r50, "[0,50) capped")]:
    vbm_d = r["Ep_dft"][:, :26].max(); vbm_q = r["Ep_qp"][:, :26].max()
    for b in range(r["Ep_dft"].shape[1]):
        ax.plot(x_path, r["Ep_dft"][:, b] - vbm_d, lw=0.9, color="#2c6e8f")
        ax.plot(x_path, r["Ep_qp"][:, b] - vbm_q, lw=0.9, color="#b5432c")
    draw_nodes(ax); ax.set_ylim(-8, 10); ax.set_title(f"{tt}: DFT(blue) vs QP(red)"); ax.set_ylabel("E-VBM (eV)")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "07_fullwindow_dft_vs_qp.png")); log("saved 07")

log(f"\nDONE htransform_sweep  total={time.time()-t0:.1f}s")
