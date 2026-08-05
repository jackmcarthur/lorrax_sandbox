"""Probe the iQ=9 (Q=(0,0.3)) single-point dip: scan the htransform
conduction energies eps_c(k+Q) across the dip and report the free
pair-continuum floor D_min(Q) = min_{k,c,v} [eps_c(k+Q) - eps_v(k)].
The exchange/stencil V_Q is smooth by construction; a spike in D_min (or
in min_k eps_c) pins the dip on the htransform 8-band-window caches."""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_w_exact import _create_mesh_xy

RY2EV = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input("exciton_bands_nw.in")
data = load_bse_data_from_restart_sharded(
    _find_restart_file("exciton_bands_nw.in"), n_val=4, n_cond=4,
    mesh_xy=mesh_xy, input_file="exciton_bands_nw.in", inject_head=True)
eps_v = np.asarray(jax.device_get(data["eps_v"]))[:, :4]      # (nk, 4)
(wfn, sym, meta, _m, _S, ctilde, B_at_mu,
 enk_sigma) = ht.initialize_wfns("exciton_bands_nw.in", params, print,
                                 mesh_xy=mesh_xy)
nkx = nky = 12
k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                              [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
Qs = [(0, 0.2667, 0), (0, 0.28, 0), (0, 0.29, 0), (0, 0.30, 0),
      (0, 0.31, 0), (0, 0.32, 0), (0, 0.3333, 0)]
q_list = np.concatenate([np.asarray(Q)[None, :] + k_frac for Q in Qs], 0)
bundle = compute_wfns_fi(
    ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
    kgrid_co=(nkx, nky, 1), band_window_fi=(int(params["nval"]),
                                            int(params["nval"]) + 4),
    mesh_xy=mesh_xy, q_list=q_list, log_fn=print)
eps = np.asarray(jax.device_get(bundle.enk_full)).reshape(len(Qs), 144, -1)
print("\nQ_y      min_k eps_c1..c4 (eV)                D_min (eV)")
for i, Q in enumerate(Qs):
    e = eps[i][:, :4] * RY2EV                    # (nk, 4) conduction eV
    dmin = np.min(e[:, :, None] - eps_v[:, None, :] * RY2EV)
    mins = " ".join(f"{v:8.4f}" for v in e.min(axis=0))
    kam = int(np.argmin(e[:, 0]))
    print(f"{Q[1]:.4f}  {mins}   {dmin:8.4f}  (argmin k={kam} "
          f"{np.array2string(k_frac[kam], precision=3)})")
