"""12x12 full-band SP htransform bands + free-pair floor D_min(Q) along the
exciton 40-pt Gamma-M-K-Gamma path, overlaid with the 08 exciton E_1.
Full basis (nval=26, ncond=14, nband=40), DFT energies. Single fH build."""
import time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
inp = "hts12.in"
params = read_lorrax_input(inp)

# fH build over the FULL band window
(wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(inp, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz))
nkx, nky, nkz = kg
print(f"[cfg] kgrid={kg} nband={enk.shape[-1]}", flush=True)

# path
wfn0, _sym0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
nQ = kpath.shape[0]
print(f"[cfg] path nQ={nQ} nodes={node_idx} labels={node_labels}", flush=True)

# coarse k grid (12x12)
k_frac = np.stack(np.meshgrid(np.arange(nkx)/nkx, np.arange(nky)/nky, [0.0], indexing="ij"),
                  axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]
q_big = (kpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)   # (nQ*nk, 3)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk, kgrid_co=kg,
                      band_window_fi=(0, 40), mesh_xy=mesh_xy, q_list=q_big,
                      a_band_index=28, log_fn=print)
E = np.asarray(jax.device_get(bnd.enk_full)) * RY          # (nQ*nk, 40) eV
E = E.reshape(nQ, nk, 40)
print(f"[t] compute {time.time()-t0:.1f}s  E shape {E.shape}", flush=True)

# SP bands along the path = the k_coarse=Gamma column (j=0)
j0 = int(np.argmin(np.linalg.norm(k_frac, axis=1)))         # index of k=(0,0,0)
E_path = E[:, j0, :]                                        # (nQ, 40)
vbm = E_path[:, :26].max()

# SAVE raw compute immediately (never lose it again)
np.savez("E_raw.npz", E=E, x_path=x_path, node_idx=node_idx, k_frac=k_frac, j0=j0, vbm=vbm)

# free-pair floor in the BSE 4v4c window: valence idx 22-25, conduction idx 26-29
vtop_k = E[0, :, 22:26].max(axis=1)                        # (nk,) top-4 valence at base k
cbot_kQ = E[:, :, 26:30].min(axis=2)                       # (nQ, nk) bottom-4 conduction at k+Q
gap = cbot_kQ - vtop_k[None, :]
dmin = gap.min(axis=1)                                     # (nQ,)  4v4c free-pair floor
# also full-window floor for context
dmin_full = (E[:, :, 26:].min(axis=2) - E[0, :, :26].max(axis=1)[None, :]).min(axis=1)

# 08 exciton E_1 (col 1 = xdist, col 6 = E1; col 5 is the 'interp' string)
ex = np.loadtxt("../08_lorrax_exciton_bands_fullbasis/exciton_bands_fullbasis.dat",
                usecols=(1, 6))
ex_x = ex[:, 0]; ex_E1 = ex[:, 1]

np.savez("sp_dmin_12.npz", x_path=x_path, E_path=E_path, vbm=vbm, dmin=dmin,
         dmin_full=dmin_full, node_idx=node_idx, ex_x=ex_x, ex_E1=ex_E1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 8.4), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
for b in range(40):
    c = "#2c6e8f" if b < 26 else "#b5432c"
    a1.plot(x_path, E_path[:, b] - vbm, lw=1.0, color=c)
a1.set_ylim(-8, 6); a1.set_ylabel("E − E$_{VBM}$ (eV)")
a1.set_title("MoS₂ 12×12 htransform SP bands — full basis (26v+14c), DFT")

# align D_min shape to exciton at Gamma (compare dispersion, not absolute binding)
off = ex_E1[0] - dmin[0]
a2.plot(ex_x, ex_E1, lw=1.8, color="#b5432c", label="exciton E$_1$(Q) [interp]")
a2.plot(x_path, dmin + off, lw=1.4, color="#333", ls="--",
        label="4v4c free-pair floor D$_{min}$(Q) − binding")
a2.plot(x_path, dmin_full + off, lw=1.0, color="#888", ls=":",
        label="full-window D$_{min}$(Q) − binding")
a2.set_ylabel("energy (eV)"); a2.legend(loc="upper right", fontsize=8)
a2.set_title("exciton E$_1$ vs free-pair floor (shifted by Γ binding) — does the dip track the pair floor?")

nodes = [node_idx[0], node_idx[1], node_idx[2], node_idx[3]] if len(node_idx) >= 4 else node_idx
labs = ["Γ", "M", "K", "Γ"]
for ax in (a1, a2):
    for n in nodes:
        ax.axvline(x_path[n], color="0.8", lw=.6, zorder=0)
    ax.set_xlim(x_path[0], x_path[-1])
a2.set_xticks([x_path[n] for n in nodes]); a2.set_xticklabels(labs)
fig.tight_layout(); fig.savefig("sp_dmin_12x12_fullband.png")
print("SAVED sp_dmin_12x12_fullband.png  vbm=%.4f  dmin_range=[%.3f,%.3f]" %
      (vbm, dmin.min(), dmin.max()), flush=True)
