"""Per-row artifact mask for the 40-point path: htransform conduction
energies under two guard windows (24-32 production vs 23-31 shifted-down),
plus the exact stored ε at on-grid k+Q.  Rows where the two windows
disagree in min_k eps_c beyond tol are window-truncation-artifact suspects
(the iQ 9 mechanism); rows where they agree to sub-meV are trustworthy.
Solver-free (epsilon-side only — the demonstrated artifact mechanism)."""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import h5py

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_io import _find_restart_file
from bse.bse_w_exact import _create_mesh_xy

RY2EV = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)

# the production path (from the driver input)
params = read_lorrax_input("exciton_bands_nw.in")
wfn0, _ = ht.setup_wfn_and_sym("WFN.h5")
Qpath, node_idx, labels = ht.generate_kpath_from_qe_segments(params, wfn0)
Qpath = np.asarray(Qpath)
nQ = Qpath.shape[0]
k_frac = np.stack(np.meshgrid(np.arange(12) / 12, np.arange(12) / 12, [0.0],
                              indexing="ij"), axis=-1).reshape(-1, 3)
q_list = (Qpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)

with h5py.File(_find_restart_file("exciton_bands_nw.in"), "r") as f:
    enk = f["enk_full"][()]              # (nk, 80) Ry, stored truth on-grid
k_int = np.rint(k_frac * np.array([12, 12, 1])).astype(int) % \
    np.array([12, 12, 1])
k_lookup = {tuple(v): i for i, v in enumerate(k_int)}

eps_by_win = {}
for tag, inp in [("w2432", "exciton_bands_nw.in"),
                 ("w2331", "exciton_dip_w2331.in")]:
    p = read_lorrax_input(inp)
    (wfn, sym, meta, _m, _S, ctilde, B_at_mu,
     enk_sigma) = ht.initialize_wfns(inp, p, print, mesh_xy=mesh_xy)
    b0 = int(p["nval"])
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=(12, 12, 1), band_window_fi=(b0, b0 + 4),
        mesh_xy=mesh_xy, q_list=q_list, log_fn=print)
    eps_by_win[tag] = np.asarray(jax.device_get(bundle.enk_full)) \
        .reshape(nQ, 144, 4)

print("\niQ  Qy_frac  min_eps_c: w2432    w2331    |dmin| meV   on-grid-err meV  flag")
for iQ in range(nQ):
    e1 = eps_by_win["w2432"][iQ][:, 0] * RY2EV
    e2 = eps_by_win["w2331"][iQ][:, 0] * RY2EV
    d = abs(e1.min() - e2.min()) * 1e3
    # stored truth where k+Q is on-grid
    og = ""
    kq = Qpath[iQ] + k_frac
    ki = np.rint(kq * np.array([12, 12, 1]))
    if np.max(np.abs(kq * np.array([12, 12, 1]) - ki)) < 1e-6:
        idx = [k_lookup[tuple(v.astype(int) % np.array([12, 12, 1]))]
               for v in ki]
        e_st = enk[idx][:, 26:30].min() * RY2EV
        og = f"{abs(e1.min()-e_st)*1e3:10.3f}"
    flag = "ARTIFACT?" if d > 5.0 else ""
    print(f"{iQ:3d}  {Qpath[iQ][1]:.4f}  {e1.min():9.4f} {e2.min():9.4f} "
          f"{d:10.3f}   {og:>14s}  {flag}")
