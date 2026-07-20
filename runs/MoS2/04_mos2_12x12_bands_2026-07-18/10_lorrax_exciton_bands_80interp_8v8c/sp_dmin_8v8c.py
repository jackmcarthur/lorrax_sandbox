"""12x12 nband=80 SP htransform bands + free-pair floor D_min(Q) in the 8v8c
BSE window (valence idx 18-25, conduction idx 26-33), overlaid with the run-10
80-interp/8v8c exciton E_1.  Adapts 09_spbands_12x12_fullband/sp_dmin_12.py to
the full 80-band basis and the 8v8c window.  DFT energies, single fH build over
ALL 80 bands.  a_band_index from the sweep (env A_BAND).
Saves E_raw_80.npz then sp_dmin_80.npz before plotting."""
import os, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
# WORKING interp basis (nband=40): the 80-band basis is broken (955 meV on-grid
# conduction error — see sp_reconcile / capacity_probe), so the D_min diagnostic
# uses the same nband=40 basis as the exciton deliverable.
NBAND = int(os.environ.get("NBAND", "40"))
A_BAND = int(os.environ.get("A_BAND", "33"))
INP = "hts80.in"                                 # path/centroids source; nband overridden
EXC_DAT = f"exciton_bands_{NBAND}interp_8v8c.dat"  # run-10 final (optional overlay)
V0, V1 = 18, 26      # 8v8c valence window (top-8 valence)
C0, C1 = 26, 34      # 8v8c conduction window (bottom-8 conduction)

mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input(INP)
params["nband"] = NBAND; params["ncond"] = NBAND - 26
(wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(INP, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz)); nkx, nky, nkz = kg
rank = int(ctilde.shape[2]); nb_ret = min(int(ctilde.shape[1]), rank)
print(f"[cfg] kgrid={kg} nb_ret={nb_ret} rank={rank} A_BAND={A_BAND}", flush=True)

wfn0, _s0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
node_idx = [int(n) for n in node_idx]; nQ = kpath.shape[0]
print(f"[cfg] path nQ={nQ} nodes={node_idx}", flush=True)

k_frac = np.stack(np.meshgrid(np.arange(nkx)/nkx, np.arange(nky)/nky, [0.0], indexing="ij"),
                  axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]
q_big = (kpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk, kgrid_co=kg,
                      band_window_fi=(0, nb_ret), mesh_xy=mesh_xy, q_list=q_big,
                      a_band_index=A_BAND, log_fn=print)
E = np.asarray(jax.device_get(bnd.enk_full)) * RY
E = E.reshape(nQ, nk, nb_ret)
print(f"[t] compute {time.time()-t0:.1f}s  E {E.shape}", flush=True)

j0 = int(np.argmin(np.linalg.norm(k_frac, axis=1)))    # k=(0,0,0)
E_path = E[:, j0, :]                                    # (nQ, nb_ret)
vbm = E_path[:, :26].max()
np.savez(f"E_raw_{NBAND}.npz", E=E, x_path=x_path, node_idx=node_idx, k_frac=k_frac, j0=j0, vbm=vbm, A_BAND=A_BAND)

# free-pair floor in the 8v8c window
vtop_k = E[0, :, V0:V1].max(axis=1)                    # (nk,) top-8 valence at base k
cbot_kQ = E[:, :, C0:C1].min(axis=2)                   # (nQ, nk) bottom-8 conduction at k+Q
dmin = (cbot_kQ - vtop_k[None, :]).min(axis=1)         # (nQ,) 8v8c free-pair floor
dmin_full = (E[:, :, 26:].min(axis=2) - E[0, :, :26].max(axis=1)[None, :]).min(axis=1)

ex_x = ex_E1 = None
if os.path.isfile(EXC_DAT):
    ex = np.loadtxt(EXC_DAT, usecols=(1, 6)); ex_x = ex[:, 0]; ex_E1 = ex[:, 1]
    print(f"[overlay] loaded {EXC_DAT}: {ex.shape[0]} rows", flush=True)
else:
    print(f"[overlay] {EXC_DAT} not present yet; D_min-only panel", flush=True)

np.savez(f"sp_dmin_{NBAND}_8v8c.npz", x_path=x_path, E_path=E_path, vbm=vbm, dmin=dmin,
         dmin_full=dmin_full, node_idx=node_idx,
         ex_x=(ex_x if ex_x is not None else []), ex_E1=(ex_E1 if ex_E1 is not None else []))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 9.0), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
for b in range(nb_ret):
    if b < 26:        c, lw = "#2c6e8f", 0.7           # valence
    elif b < C1:      c, lw = "#b5432c", 1.3           # 8v8c conduction (26-33)
    else:             c, lw = "#c9a24a", 0.5           # guards 34-79
    a1.plot(x_path, E_path[:, b] - vbm, lw=lw, color=c)
a1.set_ylim(-8, 14); a1.set_ylabel("E − E$_{VBM}$ (eV)")
a1.set_title(f"MoS₂ 12×12 SP-80 htransform bands (a_band={A_BAND})\n"
             f"blue=val, red=8v8c cond(26-33), gold=guard(34-79)")

off = (ex_E1[0] - dmin[0]) if ex_E1 is not None else 0.0
if ex_E1 is not None:
    a2.plot(ex_x, ex_E1, lw=1.8, color="#b5432c", label="exciton E$_1$(Q) 8v8c/80 [interp]")
a2.plot(x_path, dmin + off, lw=1.4, color="#333", ls="--",
        label="8v8c free-pair floor D$_{min}$(Q) − binding")
a2.plot(x_path, dmin_full + off, lw=1.0, color="#888", ls=":",
        label="full-window D$_{min}$(Q) − binding")
a2.set_ylabel("energy (eV)"); a2.legend(loc="upper right", fontsize=8)
a2.set_title("exciton E$_1$ vs free-pair floor (shifted by Γ binding)")

labs = ["Γ", "M", "K", "Γ"]
for ax in (a1, a2):
    for n in node_idx:
        ax.axvline(x_path[n], color="0.8", lw=.6, zorder=0)
    ax.set_xlim(x_path[0], x_path[-1])
a2.set_xticks([x_path[n] for n in node_idx]); a2.set_xticklabels(labs)
fig.tight_layout(); fig.savefig(f"sp_dmin_{NBAND}_8v8c.png")
print(f"SAVED sp_dmin_{NBAND}_8v8c.png  vbm=%.4f  dmin_range=[%.3f,%.3f]" %
      (vbm, dmin.min(), dmin.max()), flush=True)
