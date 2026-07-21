"""Publication GW/QP bandstructure figure: DFT vs G0W0 along Gamma-M-K-Gamma.

Interpolates BOTH the DFT eigenvalues and the QP energies (eqp1.dat, one-shot
G0W0 with GN-PPM) from the 6x6 coarse grid onto the k-path with the SAME
htransform fH, so the overlay is an apples-to-apples comparison of the same
interpolation machinery -- any wiggle common to both is the interpolant, any
separation is physics.

Run from reports/gw_bandrange_centroids_2026-07-21 with PYTHONPATH pointing at
the lorrax_A_figures worktree (its compute_wfns_fi carries the arbitrary-q
`q_list` kwarg the k-path needs).

env: BF_RUN (run dir with gwbands.in / WFN.h5 / eqp0.dat / eqp1.dat)
     BF_OUT (output dir)  BF_WIN (interp window "0:48")  BF_TAG
"""
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input                     # noqa: E402
from bandstructure import htransform as ht                     # noqa: E402
from bandstructure.bse_setup import compute_wfns_fi            # noqa: E402
from bse.bse_w_exact import _create_mesh_xy                    # noqa: E402

RY = 13.6056980659
RUN = os.environ["BF_RUN"]
OUT = os.environ.get("BF_OUT", ".")
TAG = os.environ.get("BF_TAG", "gw_bands")
B0, B1 = (int(v) for v in os.environ.get("BF_WIN", "0:48").split(":"))
NVAL = int(os.environ.get("BF_NVAL", "26"))
NCOND = int(os.environ.get("BF_NCOND", "34"))
NBAND = NVAL + NCOND
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def parse_eqp(path):
    data, ik = {}, -1
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) == 4 and "." in p[0]:
            ik += 1
            data[ik] = {}
        elif len(p) == 4:
            data[ik][int(p[1])] = (float(p[2]), float(p[3]))
    return data


t0 = time.time()
mesh_xy = _create_mesh_xy(1, 1)
INP = os.path.join(RUN, "gwbands.in")
params = read_lorrax_input(INP)
params["nval"], params["ncond"], params["nband"] = NVAL, NCOND, NBAND
(wfn, sym, meta, _m, _S, ctilde, B, enk_dft) = ht.initialize_wfns(
    INP, params, log, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz))
nk = kg[0] * kg[1] * kg[2]
rank = int(ctilde.shape[2])
nb_ret = min(int(ctilde.shape[1]), rank)
enk_dft = np.asarray(jax.device_get(enk_dft))
if enk_dft.shape[0] == nk:
    enk_dft = enk_dft.T
log(f"[cfg] kg={kg} nk={nk} nb_ret={nb_ret} rank={rank} "
    f"window=[{B0},{B1})  build={time.time()-t0:.1f}s")

eqp1 = parse_eqp(os.path.join(RUN, "eqp1.dat"))
dS1 = np.zeros_like(enk_dft)
for k in range(nk):
    for b in range(min(nb_ret, NBAND)):
        ed, e1 = eqp1[k][b + 1]
        dS1[b, k] = (e1 - ed) / RY
enk_qp = enk_dft + dS1

wfn0, _s0 = ht.setup_wfn_and_sym(os.path.join(RUN, "WFN.h5"))
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac)
x_path = np.asarray(x_path)
node_idx = [int(n) for n in node_idx]
log(f"[path] nQ={kpath.shape[0]} nodes={node_idx} labels={node_labels}")


def interp(enk, qlist):
    ct = ctilde[:, B0:B1, :]
    en = jax.numpy.asarray(enk[B0:B1, :])
    bnd = compute_wfns_fi(ctilde=ct, B_at_mu=B, enk_sigma=en, kgrid_co=kg,
                          band_window_fi=(0, B1 - B0), mesh_xy=mesh_xy,
                          q_list=jax.numpy.asarray(qlist), a_band_index=None,
                          log_fn=(lambda *a, **k: None))
    return np.asarray(jax.device_get(bnd.enk_full)) * RY


E_dft = interp(enk_dft, kpath)         # (nQ, nb_win)
E_qp = interp(enk_qp, kpath)
log(f"[interp] DFT {E_dft.shape}  QP {E_qp.shape}  t={time.time()-t0:.1f}s")

# Band identity on the path: htransform returns eigenvalues sorted per q, so
# "VBM" = the NVAL-th lowest at each q (the window starts at band 0).
iv = NVAL - 1 - B0
ic = NVAL - B0
vb_d, cb_d = E_dft[:, iv], E_dft[:, ic]
vb_q, cb_q = E_qp[:, iv], E_qp[:, ic]

res = {}
for tag, vb, cb in (("DFT", vb_d, cb_d), ("GW", vb_q, cb_q)):
    idir = int(np.argmin(cb - vb))
    ivbm, icbm = int(np.argmax(vb)), int(np.argmin(cb))
    res[tag] = dict(direct=float(cb[idir] - vb[idir]), i_direct=idir,
                    indirect=float(cb[icbm] - vb[ivbm]),
                    i_vbm=ivbm, i_cbm=icbm, vbm=float(vb[ivbm]))
    log(f"[{tag}] direct {res[tag]['direct']:.3f} eV @ x={x_path[idir]:.3f}   "
        f"indirect {res[tag]['indirect']:.3f} eV  "
        f"(VBM x={x_path[ivbm]:.3f}, CBM x={x_path[icbm]:.3f})")

np.savez(os.path.join(OUT, f"{TAG}.npz"), x_path=x_path,
         node_idx=np.array(node_idx), E_dft=E_dft, E_qp=E_qp,
         window=np.array([B0, B1]), nval=NVAL)

# ── figure ───────────────────────────────────────────────────────────────
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

C_DFT, C_GW = "#2a78d6", "#eb6834"      # validated categorical slots 1 & 2
INK, INK2, GRID = "#1a1a19", "#55544e", "#e2e1dc"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.linewidth": 0.8, "figure.facecolor": "white",
})

# Align both to their own VBM so the gap opening is the visible quantity.
Ed = E_dft - vb_d.max()
Eq = E_qp - vb_q.max()

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
for n in node_idx:
    ax.axvline(x_path[n], color=GRID, lw=0.8, zorder=0)
ax.axhline(0.0, color=GRID, lw=0.8, ls="--", zorder=0)

for b in range(Ed.shape[1]):
    ax.plot(x_path, Ed[:, b], lw=1.4, color=C_DFT, zorder=2,
            solid_capstyle="round")
    ax.plot(x_path, Eq[:, b], lw=1.4, color=C_GW, zorder=3,
            solid_capstyle="round")

# Gap annotations at the direct-gap k-point of each theory.
for tag, E, col, dx in (("DFT", Ed, C_DFT, -0.018), ("GW", Eq, C_GW, +0.018)):
    i = res[tag]["i_direct"]
    x = x_path[i] + dx * (x_path[-1] - x_path[0])
    lo = E[i, iv]
    hi = E[i, ic]
    ax.annotate("", xy=(x, hi), xytext=(x, lo),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.text(x + 0.012 * (x_path[-1] - x_path[0]), 0.5 * (lo + hi),
            f"{res[tag]['direct']:.2f} eV", color=INK, fontsize=9,
            ha="left" if dx > 0 else "right", va="center",
            bbox=dict(fc="white", ec="none", pad=1.2), zorder=6)

ax.set_xticks([x_path[n] for n in node_idx])
ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
ax.set_xlim(x_path[0], x_path[-1])
ax.set_ylim(-6.0, 8.0)
ax.set_ylabel("E − E$_{VBM}$  (eV)")
ax.set_title("Monolayer MoS$_2$: DFT vs G$_0$W$_0$ (GN-PPM)", fontsize=11,
             pad=10)
ax.plot([], [], color=C_DFT, lw=1.8, label="DFT (PBE, FR-SOC)")
ax.plot([], [], color=C_GW, lw=1.8, label="G$_0$W$_0$ quasiparticle")
ax.legend(loc="upper right", frameon=False, fontsize=9)
sub = (f"6×6 k-grid · {NBAND} bands Σ-corrected · 200 bands screening · "
       f"{B1-B0}-band htransform interpolation")
ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=7.6, color=INK2,
        ha="left", va="bottom")
ax.text(1.0, -0.115,
        f"indirect: DFT {res['DFT']['indirect']:.2f} eV → "
        f"GW {res['GW']['indirect']:.2f} eV",
        transform=ax.transAxes, fontsize=8, color=INK2, ha="right", va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
p = os.path.join(OUT, f"{TAG}.png")
fig.savefig(p, bbox_inches="tight")
log(f"saved {p}")
log(f"DONE total={time.time()-t0:.1f}s")
