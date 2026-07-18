"""diag_dense_gamma — dense ground truth for the Γ TDA problem (n=144)
+ htransform Γ-subspace decomposition.

1. Dense H at Γ via the production stack matvec applied to identity
   columns, for BOTH ψ sources.  eigh → exact spectra.  Compares:
   dense(stored) vs solve_bse_sharded(stored) [Lanczos health],
   dense(ht) vs dense(stored) [pure ψ-source physics delta],
   and the driver row [scan + n_reorth health].
2. Where do the htransform Γ conduction states live in the stored 80-band
   basis?  (weights over bands 20..40)
"""
import sys

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import h5py

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_lanczos import solve_bse_sharded
from bse.bse_serial import compute_pair_amplitude
from bse.bse_stack_matvec import build_bse_stack_matvec
from bse.bse_w_exact import _create_mesh_xy
from bse.bse_ring_comm import make_bse_shardings
from common.fft_helpers import make_sharded_ifftn_3d

RY2EV = 13.6056980659
RD = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17"
INP = f"{RD}/exciton_smoke.in"
NV = NC = 4

mesh = _create_mesh_xy(1, 1)
sh = make_bse_shardings(mesh)
restart = _find_restart_file(INP)
data = load_bse_data_from_restart_sharded(
    restart, n_val=NV, n_cond=NC, mesh_xy=mesh, input_file=INP,
    inject_head=True)
nk = 9
n_flat = NC * NV * nk
matvec = build_bse_stack_matvec(mesh, 3, 3, 1, kernel="bse")
_ifftn = make_sharded_ifftn_3d(mesh, sh.W.spec, sh.W.spec, axes=(2, 3, 4),
                               norm="ortho")
W_R = _ifftn(data["W_q"])


def dense_H(psi_c, eps_c):
    psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
    psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
    M_X = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, data["psi_v_X"]), sh.psi_x)
    M_Y = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_Y, data["psi_v_Y"]), sh.psi_y)
    cols = []
    eye = np.eye(n_flat)
    bs = 16
    for i in range(0, n_flat, bs):
        X = jnp.asarray(eye[i:i + bs].reshape(bs, NC, NV, nk))
        HX = matvec(X, psi_c_X, psi_c_Y, data["psi_v_X"], data["psi_v_Y"],
                    eps_c, data["eps_v"], W_R, data["V_q0"], M_X, M_Y)
        cols.append(np.asarray(jax.device_get(HX)).reshape(bs, n_flat))
    H = np.concatenate(cols, 0).T
    herm = np.linalg.norm(H - H.conj().T) / np.linalg.norm(H)
    return H, herm


H_st, herm_st = dense_H(data["psi_c_X"], data["eps_c"])
ev_st = np.linalg.eigvalsh(0.5 * (H_st + H_st.conj().T))[:8]
print(f"dense stored-psi (herm resid {herm_st:.2e}):",
      " ".join(f"{e*RY2EV:.6f}" for e in ev_st))

with mesh:
    evsA, _v, _n = solve_bse_sharded(data, mesh, n_eig=8, max_iter=40,
                                     block_size=8, include_W=True)
print("solve_bse_sharded stored-psi          :",
      " ".join(f"{e*RY2EV:.6f}" for e in np.sort(np.asarray(evsA))[:8]))

# htransform psi at Q=0
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi

params = read_lorrax_input(INP)
(wfn, sym, meta, _m, _S, ctilde, B_at_mu, enk_sigma) = ht.initialize_wfns(
    INP, params, print, mesh_xy=mesh)
k_frac = np.stack(np.meshgrid(np.arange(3) / 3, np.arange(3) / 3,
                              np.arange(1), indexing="ij"),
                  axis=-1).reshape(-1, 3)
nval_in = int(params["nval"])
bundle = compute_wfns_fi(
    ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma, kgrid_co=(3, 3, 1),
    band_window_fi=(nval_in, nval_in + NC), mesh_xy=mesh, q_list=k_frac)
psi_ht = jnp.asarray(bundle.psi_rmu_Y)
eps_ht = jnp.asarray(bundle.enk_full)
H_ht, herm_ht = dense_H(psi_ht, eps_ht)
ev_ht = np.linalg.eigvalsh(0.5 * (H_ht + H_ht.conj().T))[:8]
print(f"dense htrans-psi (herm resid {herm_ht:.2e}):",
      " ".join(f"{e*RY2EV:.6f}" for e in ev_ht))
print("driver Γ row: 0.083293 0.089466 0.129965 0.171928 0.179359 0.179359")

# ---- Γ subspace decomposition of the ht states over stored bands 20..40 ----
with h5py.File(restart, "r") as f:
    psi_full = np.asarray(f["psi_full_y"][:])
ph = np.asarray(jax.device_get(psi_ht))[0].reshape(NC, -1)    # k=0 rows
ph = ph / np.linalg.norm(ph, axis=1, keepdims=True)
print("\nht Γ conduction states over stored bands 20..40 (|overlap|>0.15):")
B20 = psi_full[0, 20:40].reshape(20, -1)
B20 = B20 / np.linalg.norm(B20, axis=1, keepdims=True)
ov = np.abs(ph.conj() @ B20.T)
for i in range(NC):
    hits = [(20 + j, float(ov[i, j])) for j in range(20) if ov[i, j] > 0.15]
    print(f"  ht c{i}: " + "  ".join(f"b{b}:{o:.3f}" for b, o in hits))
print("DONE")
