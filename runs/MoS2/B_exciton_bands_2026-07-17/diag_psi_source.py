"""diag_psi_source — isolate the Γ-row mismatch.

A/B: production solve with STORED conduction ψ vs the SAME solver with
htransform conduction ψ (Q=0 grid).  If B reproduces the driver's Γ row,
the driver plumbing is exact and the delta is purely the ψ source.
Then: per-k conduction-window diagnostics — band energies near the window
edge (degeneracy check) + per-band ht-vs-stored overlaps.
"""
import sys

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_lanczos import solve_bse_sharded
from bse.bse_serial import compute_pair_amplitude
from bse.bse_w_exact import _create_mesh_xy
from bse.bse_ring_comm import make_bse_shardings

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

with mesh:
    evsA, _v, _n = solve_bse_sharded(data, mesh, n_eig=6, max_iter=40,
                                     block_size=8, include_W=True)
evsA = np.sort(np.asarray(jax.device_get(evsA)))[:6]
print("A stored-psi   :", " ".join(f"{e*RY2EV:.6f}" for e in evsA))

# ---- htransform conduction psi on the Q=0 grid ----
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi

params = read_lorrax_input(INP)
(wfn, sym, meta, _m, _S, ctilde, B_at_mu, enk_sigma) = ht.initialize_wfns(
    INP, params, print, mesh_xy=mesh)
nkx, nky, nkz = 3, 3, 1
k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                              np.arange(nkz) / nkz, indexing="ij"),
                  axis=-1).reshape(-1, 3)
nval_in = int(params["nval"])
bundle = compute_wfns_fi(
    ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
    kgrid_co=(nkx, nky, nkz), band_window_fi=(nval_in, nval_in + NC),
    mesh_xy=mesh, q_list=k_frac)
psi_ht = np.asarray(jax.device_get(bundle.psi_rmu_Y))   # (9, NC, 2, 640)
eps_ht = np.asarray(jax.device_get(bundle.enk_full))

d2 = dict(data)
p = jnp.asarray(psi_ht)
d2["psi_c_X"] = jax.device_put(p, sh.psi_x)
d2["psi_c_Y"] = jax.device_put(p, sh.psi_y)
d2["eps_c"] = jnp.asarray(eps_ht)
d2["M_X"] = jax.lax.with_sharding_constraint(
    compute_pair_amplitude(d2["psi_c_X"], d2["psi_v_X"]), sh.psi_x)
d2["M_Y"] = jax.lax.with_sharding_constraint(
    compute_pair_amplitude(d2["psi_c_Y"], d2["psi_v_Y"]), sh.psi_y)
with mesh:
    evsB, _v, _n = solve_bse_sharded(d2, mesh, n_eig=6, max_iter=40,
                                     block_size=8, include_W=True)
evsB = np.sort(np.asarray(jax.device_get(evsB)))[:6]
print("B htrans-psi   :", " ".join(f"{e*RY2EV:.6f}" for e in evsB))
print("driver Γ row   : 0.083293 0.089466 0.129965 0.171928 0.179359 0.179359")

# ---- window-edge diagnostics ----
enk = np.asarray(jax.device_get(data["eps_c"]))          # stored (nk, NC)
with jax.default_device(jax.devices()[0]):
    pass
import h5py
with h5py.File(restart, "r") as f:
    enk_full = np.asarray(f["enk_full"][:])              # (nk, 80)
    psi_full = np.asarray(f["psi_full_y"][:])            # (nk, 80, 2, 640)
n_occ = 26
print("\nconduction bands about the window edge (eV, per k; window = c0..c3):")
for k in range(9):
    e = (enk_full[k, n_occ:n_occ + 6]) * RY2EV
    print(f"  k={k}: " + " ".join(f"{x:8.4f}" for x in e)
          + f"   gap c3-c4 = {(e[4]-e[3])*1e3:7.1f} meV")

print("\nper-k per-band |<psi_ht|psi_stored>| (row-normalized):")
for k in range(9):
    A = psi_ht[k].reshape(NC, -1)
    B = psi_full[k, n_occ:n_occ + NC].reshape(NC, -1)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    ov = np.abs(A.conj() @ B.T)
    print(f"  k={k}: diag " + " ".join(f"{ov[i,i]:.4f}" for i in range(NC))
          + f"   min-sval {np.linalg.svd(ov, compute_uv=False).min():.4f}")
print("\nDONE")
