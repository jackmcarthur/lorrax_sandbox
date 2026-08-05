"""Task 3: htransform smoothness sweep.

Build the htransform fH from band SUBWINDOWS (starting at the deepest semicore
and extending upward), interpolate along Gamma-M-K-Gamma, for BOTH DFT energies
(smooth baseline / machinery test) and QP energies (E_dft + eqp1 scissor).

Deliverables:
  * semicore-only interpolation (smooth + near-flat?)
  * window sweep -> where does smoothness break (band/energy)?
  * leakage: does including deep/jagged bands degrade near-gap interp?
    (compare windows starting at 0 vs starting above the semicore; on-grid recon)
Outputs npz + plots.  Runs on 1 GPU.
"""
import os, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20"
PLOTS = os.path.join(OUT, "plots")
INP = os.path.join(RUN, "gwbands.in")
NVAL, NCOND, NBAND = 26, 64, 90        # ctilde over abs bands [0,90)
NKX = NKY = 6


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


# ---- basis + DFT energies (single expensive Galerkin build) ----
mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input(INP)
params["nval"], params["ncond"], params["nband"] = NVAL, NCOND, NBAND
(wfn, sym, meta, _m, _S, ctilde, B, enk_dft) = ht.initialize_wfns(INP, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz)); nk = kg[0]*kg[1]*kg[2]
rank = int(ctilde.shape[2]); nb_ret = min(int(ctilde.shape[1]), rank)
enk_dft = np.asarray(jax.device_get(enk_dft))
if enk_dft.shape[0] == nk:
    enk_dft = enk_dft.T                       # -> (nb, nk) Ry
EF = float(wfn.efermi) * RY
print(f"[cfg] kg={kg} nk={nk} nb_ret={nb_ret} rank={rank} EF={EF:.4f} eV enk_dft{enk_dft.shape}", flush=True)

# ---- QP energies: enk_qp = enk_dft + (Eqp - Edft)/RY, per DFT-band-label ----
eqp1 = parse_eqp(os.path.join(RUN, "eqp1.dat"))
eqp0 = parse_eqp(os.path.join(RUN, "eqp0.dat"))
dS1 = np.zeros_like(enk_dft); dS0 = np.zeros_like(enk_dft)
for k in range(nk):
    for b in range(nb_ret):
        ed, e1 = eqp1[k][b + 1]; _, e0 = eqp0[k][b + 1]
        dS1[b, k] = (e1 - ed) / RY
        dS0[b, k] = (e0 - ed) / RY
enk_qp1 = enk_dft + dS1
enk_qp0 = enk_dft + dS0

# ---- k-path + coarse grid ----
wfn0, _s0 = ht.setup_wfn_and_sym(os.path.join(RUN, "WFN.h5"))
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
node_idx = [int(n) for n in node_idx]; nQ = kpath.shape[0]
kgrid = np.stack(np.meshgrid(np.arange(NKX)/NKX, np.arange(NKY)/NKY, [0.0], indexing="ij"),
                 axis=-1).reshape(-1, 3)
print(f"[path] nQ={nQ} nodes={node_idx} labels={node_labels}", flush=True)


def interp(enk, b_lo, b_hi, qlist, a_band=None):
    """fH from bands [b_lo,b_hi) of enk/ctilde; return energies (nq, b_hi-b_lo) eV."""
    ct = ctilde[:, b_lo:b_hi, :]
    en = jax.numpy.asarray(enk[b_lo:b_hi, :])
    bnd = compute_wfns_fi(ctilde=ct, B_at_mu=B, enk_sigma=en, kgrid_co=kg,
                          band_window_fi=(0, b_hi - b_lo), mesh_xy=mesh_xy,
                          q_list=jax.numpy.asarray(qlist), a_band_index=a_band,
                          log_fn=(lambda *a, **k: None))
    return np.asarray(jax.device_get(bnd.enk_full)) * RY


def smooth_metric(E_path):
    """(nQ, nb) -> per-band (max NN |dE| meV, max |2nd-diff| meV) along path."""
    nnj = np.abs(np.diff(E_path, axis=0)).max(axis=0) * 1e3
    d2 = np.abs(E_path[2:] - 2*E_path[1:-1] + E_path[:-2]).max(axis=0) * 1e3
    return nnj, d2


def recon_err(E_grid, enk, b_lo, b_hi):
    """on-grid recon: |sort(interp) - sort(input)| per rank (meV)."""
    inp = np.sort((enk[b_lo:b_hi, :].T) * RY, axis=1)     # (nk, nbw)
    out = np.sort(E_grid, axis=1)
    return np.abs(out - inp).max(axis=0) * 1e3            # (nbw,) meV


# ---- window sweep (start at 0 = includes semicore, owner constraint) ----
windows0 = [(0, 2), (0, 4), (0, 8), (0, 12), (0, 14), (0, 20), (0, 26),
            (0, 30), (0, 40), (0, 50), (0, 90)]
# ---- leakage windows (exclude semicore, still include near-gap) ----
windows_clean = [(12, 30), (14, 30), (20, 30), (12, 40), (0, 30)]

results = {}
print("\n=== window sweep (start=0, DFT & QP) ===")
print(f"{'window':>10} {'a_Ry':>8} {'shift_eV':>9} "
      f"{'DFTd2max':>9} {'QPd2max':>9} {'DFTrecon':>9} {'QPrecon':>9}  (meV unless noted)")
for (b0, b1) in windows0:
    Ep_dft = interp(enk_dft, b0, b1, kpath)
    Ep_qp = interp(enk_qp1, b0, b1, kpath)
    Eg_dft = interp(enk_dft, b0, b1, kgrid)
    Eg_qp = interp(enk_qp1, b0, b1, kgrid)
    nnj_d, d2_d = smooth_metric(Ep_dft)
    nnj_q, d2_q = smooth_metric(Ep_qp)
    rc_d = recon_err(Eg_dft, enk_dft, b0, b1)
    rc_q = recon_err(Eg_qp, enk_qp1, b0, b1)
    # transform params (recompute for reporting)
    _, a_f, n_f, shift = ht.f_transform_eigs(jax.numpy.asarray(enk_dft[b0:b1, :]), None)
    results[(b0, b1)] = dict(Ep_dft=Ep_dft, Ep_qp=Ep_qp, nnj_d=nnj_d, d2_d=d2_d,
                             nnj_q=nnj_q, d2_q=d2_q, rc_d=rc_d, rc_q=rc_q,
                             a=float(a_f), shift=float(shift)*RY)
    print(f"[{b0:>2},{b1:>3})  {float(a_f):>8.4f} {float(shift)*RY:>9.2f} "
          f"{d2_d.max():>9.1f} {d2_q.max():>9.1f} {rc_d.max():>9.1f} {rc_q.max():>9.1f}", flush=True)

print("\n=== leakage windows (near-gap bands 24-29 focus) ===")
# near-gap abs bands of interest: 24,25 (VBM), 26,27 (CBM), 28,29
leak = {}
for (b0, b1) in windows_clean:
    Eg_dft = interp(enk_dft, b0, b1, kgrid)
    Eg_qp = interp(enk_qp1, b0, b1, kgrid)
    Ep_dft = interp(enk_dft, b0, b1, kpath)
    Ep_qp = interp(enk_qp1, b0, b1, kpath)
    rc_d = recon_err(Eg_dft, enk_dft, b0, b1)
    rc_q = recon_err(Eg_qp, enk_qp1, b0, b1)
    nnj_d, d2_d = smooth_metric(Ep_dft)
    nnj_q, d2_q = smooth_metric(Ep_qp)
    # map near-gap abs band -> local rank in window
    def loc(babs):
        return babs - b0 if b0 <= babs < b1 else None
    ng = {babs: loc(babs) for babs in (24, 25, 26, 27, 28, 29)}
    ng_recon_d = {babs: (float(rc_d[l]) if l is not None else None) for babs, l in ng.items()}
    ng_recon_q = {babs: (float(rc_q[l]) if l is not None else None) for babs, l in ng.items()}
    ng_d2_d = {babs: (float(d2_d[l]) if l is not None else None) for babs, l in ng.items()}
    leak[(b0, b1)] = dict(Ep_dft=Ep_dft, Ep_qp=Ep_qp, rc_d=rc_d, rc_q=rc_q,
                          ng_recon_d=ng_recon_d, ng_recon_q=ng_recon_q, ng_d2_d=ng_d2_d)
    print(f"[{b0:>2},{b1:>3})  near-gap DFT recon(meV): "
          + " ".join(f"b{b}={('%.1f'%v) if v is not None else '--':>6}" for b, v in ng_recon_d.items()))
    print(f"          near-gap DFT d2(meV):    "
          + " ".join(f"b{b}={('%.0f'%v) if v is not None else '--':>6}" for b, v in ng_d2_d.items()), flush=True)

np.savez(os.path.join(OUT, "htransform_sweep.npz"),
         x_path=x_path, node_idx=node_idx, EF=EF, nb_ret=nb_ret, rank=rank,
         windows0=np.array(windows0), enk_dft=enk_dft, enk_qp1=enk_qp1,
         **{f"w{b0}_{b1}_Ep_dft": r["Ep_dft"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_Ep_qp": r["Ep_qp"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_rc_d": r["rc_d"] for (b0, b1), r in results.items()},
         **{f"w{b0}_{b1}_rc_q": r["rc_q"] for (b0, b1), r in results.items()})

# ================= PLOTS =================
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
labs = ["$\\Gamma$", "M", "K", "$\\Gamma$"]

def draw_nodes(ax):
    for n in node_idx:
        ax.axvline(x_path[n], color="0.85", lw=0.7, zorder=0)
    ax.set_xticks([x_path[n] for n in node_idx]); ax.set_xticklabels(labs)
    ax.set_xlim(x_path[0], x_path[-1])

# PLOT A: semicore-only window [0,2) and [0,12) DFT vs QP
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
for col, (b0, b1) in enumerate([(0, 2), (0, 12)]):
    r = results[(b0, b1)]
    for row, (key, ttl, c) in enumerate([("Ep_dft", "DFT", "#2c6e8f"), ("Ep_qp", "QP(eqp1)", "#b5432c")]):
        ax = axes[row, col]
        E = r[key]
        for b in range(E.shape[1]):
            ax.plot(x_path, E[:, b], lw=1.1, color=c)
        draw_nodes(ax)
        ax.set_title(f"window [{b0},{b1})  {ttl}   d2max={r['d2_d' if row==0 else 'd2_q'].max():.0f} meV", fontsize=10)
        ax.set_ylabel("E (eV)")
fig.suptitle("Semicore htransform interpolation (deepest manifold = the -65 & -42 eV bands)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "04_semicore_interp.png"))
print("saved 04_semicore_interp.png")

# PLOT B: sweep — max d2 (curvature spike) vs window top, DFT & QP
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
xw = [b1 for (b0, b1) in windows0]
ax.plot(xw, [results[w]["d2_d"].max() for w in windows0], "o-", label="DFT interp (machinery)")
ax.plot(xw, [results[w]["d2_q"].max() for w in windows0], "s-", label="QP interp (DFT+scissor)")
ax.set_yscale("log"); ax.set_xlabel("window top band b_max (window = [0,b_max))")
ax.set_ylabel("max |2nd-diff| along path (meV)")
ax.axvline(12, color="0.6", ls=":", label="top of semicore (band 12)")
ax.axvline(26, color="g", ls=":", label="VBM (band 26)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title("Where does htransform smoothness break as the window grows?")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "05_sweep_smoothness.png"))
print("saved 05_sweep_smoothness.png")

# PLOT C: leakage — near-gap DFT recon vs window choice
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
wlabels = [f"[{b0},{b1})" for (b0, b1) in windows_clean]
for babs in (24, 25, 26, 27, 28, 29):
    vals = [leak[w]["ng_recon_d"][babs] for w in windows_clean]
    ax.plot(range(len(windows_clean)), [v if v is not None else np.nan for v in vals],
            "o-", label=f"band {babs}")
ax.set_xticks(range(len(windows_clean))); ax.set_xticklabels(wlabels)
ax.set_yscale("log"); ax.set_ylabel("on-grid DFT recon err (meV)")
ax.set_xlabel("fH window (does including semicore/deep bands hurt near-gap?)")
ax.set_title("Leakage test: near-gap band recon vs whether deep bands are in the fH subspace")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "06_leakage_recon.png"))
print("saved 06_leakage_recon.png")

# PLOT D: full-window DFT vs QP bands (near-gap zoom), the actual figure input
r90 = results[(0, 90)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
vbm_d = r90["Ep_dft"][:, :26].max(); vbm_q = r90["Ep_qp"][:, :26].max()
for b in range(90):
    a1.plot(x_path, r90["Ep_dft"][:, b] - vbm_d, lw=0.9, color="#2c6e8f")
    a1.plot(x_path, r90["Ep_qp"][:, b] - vbm_q, lw=0.9, color="#b5432c")
draw_nodes(a1); a1.set_ylim(-8, 10); a1.set_ylabel("E - VBM (eV)")
a1.set_title("Full window [0,90): DFT (blue) vs QP (red)")
for b in range(90):
    a2.plot(x_path, r90["Ep_dft"][:, b] - vbm_d, lw=1.1, color="#2c6e8f")
    a2.plot(x_path, r90["Ep_qp"][:, b] - vbm_q, lw=1.1, color="#b5432c")
draw_nodes(a2); a2.set_ylim(-3, 6); a2.set_title("near-gap zoom")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "07_fullwindow_dft_vs_qp.png"))
print("saved 07_fullwindow_dft_vs_qp.png")

print("\nDONE htransform_sweep")
