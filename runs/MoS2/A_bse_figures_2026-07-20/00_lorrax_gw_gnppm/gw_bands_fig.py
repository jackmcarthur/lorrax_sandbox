"""GW/QP bandstructure figure for MoS2 (GN-PPM G0W0, 6x6).

DFT bands come from the htransform sp-bands machinery (Galerkin interpolation of
the coarse-grid DFT energies onto the Gamma-M-K-Gamma path).  The QP bands are
the SAME interpolation applied to (E_dft + [E_qp - E_dft]) where the smooth
QP correction is read from eqp1.dat (Z-linearized) at the 36 grid points.
Exact grid-point gaps are read straight from eqp0/eqp1 (no interpolation).

Outputs: ../figures/mos2_gw_bandstructure.png  and  gw_bands.npz
Run from 00_lorrax_gw_gnppm/ (WFN.h5, centroids, eqp0/eqp1, gwbands.in present).
"""
import os, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
INP = "gwbands.in"
# htransform interp window [0,50): a MODEST window (run-10 lesson) — too many
# interp bands corrupt the on-grid energies (nband=90 gave 465 meV recon).
# nband=50 = 26 valence + 24 conduction guards; plotted bands 16..35 stay interior.
NVAL, NCOND, NBAND = 26, 24, 50
A_BAND = int(os.environ.get("A_BAND", "30"))
B_PLOT_LO, B_PLOT_HI = 16, 36           # band indices plotted (interior to [0,50))
NKX = NKY = 6
OUTDIR = "../figures"

# ---------- parse a BGW-columnar eqp file -> {k: {band: (Edft, Eqp)}} ----------
def parse_eqp(path):
    data = {}; ik = -1; nb = 0
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) == 4 and ("." in p[0]):          # k-block header: kx ky kz nbands
                ik += 1; nb = int(p[3]); data[ik] = {}
            elif len(p) == 4:                            # band row: spin band Edft Eqp
                b = int(p[1]); data[ik][b] = (float(p[2]), float(p[3]))
    return data

def grid_gaps(eqp, nval=26):
    """Direct/indirect gaps (eV) from grid Eqp values. valence bands 1..nval."""
    ks = sorted(eqp.keys())
    bands = sorted(next(iter(eqp.values())).keys())
    vb = [b for b in bands if b <= nval]; cb = [b for b in bands if b > nval]
    Edft = np.array([[eqp[k][b][0] for b in bands] for k in ks])   # (nk, nb)
    Eqp  = np.array([[eqp[k][b][1] for b in bands] for k in ks])
    def gaps(E):
        vmax = E[:, [bands.index(b) for b in vb]].max()
        cmin = E[:, [bands.index(b) for b in cb]].min()
        vmax_k = E[:, [bands.index(b) for b in vb]].max(axis=1)
        cmin_k = E[:, [bands.index(b) for b in cb]].min(axis=1)
        return dict(indirect=cmin - vmax, direct=float((cmin_k - vmax_k).min()),
                    vmax=vmax, cmin=cmin)
    return gaps(Edft), gaps(Eqp)

eqp0 = parse_eqp("eqp0.dat"); eqp1 = parse_eqp("eqp1.dat")
dft_g, qp0_g = grid_gaps(eqp0); _, qp1_g = grid_gaps(eqp1)
print(f"[gap] DFT   direct={dft_g['direct']:.3f}  indirect={dft_g['indirect']:.3f} eV")
print(f"[gap] eqp0  direct={qp0_g['direct']:.3f}  indirect={qp0_g['indirect']:.3f} eV")
print(f"[gap] eqp1  direct={qp1_g['direct']:.3f}  indirect={qp1_g['indirect']:.3f} eV", flush=True)

# ---------- htransform interp basis + DFT energies ----------
mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input(INP)
params["nval"], params["ncond"], params["nband"] = NVAL, NCOND, NBAND
(wfn, sym, meta, _m, _S, ctilde, B, enk_dft) = ht.initialize_wfns(INP, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz)); nk = kg[0]*kg[1]*kg[2]
rank = int(ctilde.shape[2]); nb_ret = min(int(ctilde.shape[1]), rank)
enk_dft = np.asarray(jax.device_get(enk_dft))
if enk_dft.shape[0] == nk:            # normalize to (nb, nk)
    enk_dft = enk_dft.T
print(f"[cfg] kg={kg} nb_ret={nb_ret} rank={rank} enk_dft{enk_dft.shape} A_BAND={A_BAND}", flush=True)

# QP scissor (Ry) from eqp1: dSigma[b,k] = Eqp1 - Edft, band b (0-idx) = file band b+1
dSig = np.zeros_like(enk_dft)
for k in range(nk):
    for b in range(nb_ret):
        ed, eq = eqp1[k][b + 1]
        dSig[b, k] = (eq - ed) / RY
enk_qp = enk_dft + dSig

# ---------- k-path ----------
wfn0, _s0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
node_idx = [int(n) for n in node_idx]; nQ = kpath.shape[0]
print(f"[path] nQ={nQ} nodes={node_idx} labels={node_labels}", flush=True)

def bands_on(qlist, enk):
    bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=jax.numpy.asarray(enk),
                          kgrid_co=kg, band_window_fi=(0, nb_ret), mesh_xy=mesh_xy,
                          q_list=jax.numpy.asarray(qlist), a_band_index=A_BAND, log_fn=print)
    return np.asarray(jax.device_get(bnd.enk_full)) * RY

E_dft = bands_on(kpath, enk_dft)          # (nQ, nb_ret) eV
E_qp  = bands_on(kpath, enk_qp)

# ---------- on-grid reconstruction check ----------
kgrid = np.stack(np.meshgrid(np.arange(NKX)/NKX, np.arange(NKY)/NKY, [0.0], indexing="ij"),
                 axis=-1).reshape(-1, 3)
E_grid = bands_on(kgrid, enk_dft)                 # (nk, nb_ret) eV
recon = np.abs(E_grid.T - enk_dft * RY)           # (nb, nk)
recon_plot = recon[B_PLOT_LO:B_PLOT_HI].max() * 1e3
print(f"[recon] on-grid DFT interp max-err bands[{B_PLOT_LO},{B_PLOT_HI})={recon_plot:.2f} meV, "
      f"all={recon.max()*1e3:.2f} meV", flush=True)

# ---------- align each to its own VBM ----------
vbm_dft = E_dft[:, :NVAL].max(); vbm_qp = E_qp[:, :NVAL].max()
np.savez(f"{OUTDIR}/gw_bands.npz", x_path=x_path, node_idx=node_idx,
         E_dft=E_dft, E_qp=E_qp, vbm_dft=vbm_dft, vbm_qp=vbm_qp,
         dft_gap=dft_g, qp0_gap=qp0_g, qp1_gap=qp1_g,
         recon_plot_meV=recon_plot, recon_all_meV=recon.max()*1e3)

# ---------- plot ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.2, 6.4), dpi=200)
C_DFT, C_GW = "#7a7a7a", "#c0392b"
lab_dft = lab_gw = True
for b in range(nb_ret):
    if not (B_PLOT_LO <= b < B_PLOT_HI):
        continue
    ax.plot(x_path, E_dft[:, b] - vbm_dft, color=C_DFT, lw=1.1, ls="--",
            label=("DFT (PBE)" if lab_dft else None), zorder=2); lab_dft = False
    ax.plot(x_path, E_qp[:, b] - vbm_qp, color=C_GW, lw=1.6,
            label=("GW (GN-PPM G$_0$W$_0$)" if lab_gw else None), zorder=3); lab_gw = False

ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
for n in node_idx:
    ax.axvline(x_path[n], color="0.85", lw=0.7, zorder=0)
ax.set_xticks([x_path[n] for n in node_idx])
ax.set_xticklabels(["$\\Gamma$", "M", "K", "$\\Gamma$"])
ax.set_xlim(x_path[0], x_path[-1]); ax.set_ylim(-6.5, 8.0)
ax.set_ylabel("E $-$ E$_{\\mathrm{VBM}}$ (eV)")
ax.set_title("MoS$_2$ monolayer — GW/QP bandstructure\n"
             "GN-PPM G$_0$W$_0$, 6$\\times$6, 1558 centroids, 200 bands", fontsize=12)
# gap annotation
g_dft = dft_g["direct"]; g_qp = qp1_g["direct"]; g_qp_i = qp1_g["indirect"]
txt = (f"DFT direct gap:  {g_dft:.2f} eV\n"
       f"GW direct gap:   {g_qp:.2f} eV\n"
       f"GW indirect gap: {g_qp_i:.2f} eV")
ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
ax.legend(loc="lower center", fontsize=10.5, framealpha=0.9, ncol=2)
fig.tight_layout()
png = f"{OUTDIR}/mos2_gw_bandstructure.png"
fig.savefig(png)
print(f"SAVED {png}  vbm_dft={vbm_dft:.3f} vbm_qp={vbm_qp:.3f}", flush=True)
