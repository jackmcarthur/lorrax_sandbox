"""Confirm: jit-wrapping the SEED (gen) and PROJECT (snapshot) boundary
shard_maps removes the per-call trace/lower overhead AND is bit-identical.

usage: experiment_jit_boundary.py <input.in> <px> <py>
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_feast import gmres_solve_sharded_jit, _apply_shifted_matvec
from bse.bse_w_exact import (
    _build_rpa_resolvent, _select_compare_cols, apply_screening_resolvent_block)

inp = sys.argv[1]; px = int(sys.argv[2]); py = int(sys.argv[3])
restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:px*py]).reshape(px, py), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh, input_file=inp, inject_head=False)
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"]); nk = nkx*nky*nkz
n_rmu = int(data["V_q0"].shape[0]); nlog = int(data["n_rmu"])
matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)

W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
V0 = np.asarray(jax.device_get(data["V_q0"])); T = W0 - V0
cols, _ = _select_compare_cols(T, nlog, 8, seed=7)
n_pad = int(np.ceil(len(cols)/py)*py)
G = np.zeros((n_pad, n_rmu), np.float64)
for i, c in enumerate(cols):
    G[i, int(c)] = 1.0

def sync(x): jax.block_until_ready(x); return x
def timeit(fn, n=8, warmup=3):
    for _ in range(warmup): sync(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); sync(fn()); ts.append(time.perf_counter()-t0)
    return min(ts)

# Build the seed input r and the readout s once (shared across bare/jit).
Gj = jnp.asarray(G, dtype=jnp.float64)
r = jax.device_put(jnp.broadcast_to(Gj[:, :, None], (n_pad, n_rmu, nk)), sh.S)

# ---- SEED boundary: bare gen vs jax.jit(gen) ----
gen_j = jax.jit(gen)
f_bare = sync(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
f_jit = sync(gen_j(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
d_seed = float(jnp.max(jnp.abs(f_bare - f_jit)))
t_seed_bare = timeit(lambda: gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
t_seed_jit = timeit(lambda: gen_j(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))

# Build a readout s (pair-basis) to feed the PROJECT boundary.
f = jax.lax.with_sharding_constraint(f_bare, sh.X)
s_all = jax.lax.with_sharding_constraint(
    jnp.broadcast_to(f[:, None], (n_pad, 1, f.shape[1], f.shape[2], f.shape[3]))[:, 0], sh.X)

# ---- PROJECT boundary: bare snapshot vs jax.jit(snapshot) ----
snap_j = jax.jit(snapshot)
w_bare = sync(snapshot(s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
w_jit = sync(snap_j(s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
d_proj = float(jnp.max(jnp.abs(w_bare - w_jit)))
t_proj_bare = timeit(lambda: snapshot(s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
t_proj_jit = timeit(lambda: snap_j(s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))

print(f"\n[px={px} py={py}]  boundary bare-vs-jit  (warm min, 8 runs)")
print(f"  SEED (gen)      bare={1e3*t_seed_bare:8.2f} ms   jit={1e3*t_seed_jit:8.3f} ms   "
      f"speedup={t_seed_bare/t_seed_jit:6.1f}x   max|bare-jit|={d_seed:.2e}")
print(f"  PROJECT (snap)  bare={1e3*t_proj_bare:8.2f} ms   jit={1e3*t_proj_jit:8.3f} ms   "
      f"speedup={t_proj_bare/t_proj_jit:6.1f}x   max|bare-jit|={d_proj:.2e}")
print(f"  bit-identical: SEED {'YES' if d_seed==0.0 else 'NO'}  PROJECT {'YES' if d_proj==0.0 else 'NO'}")
print(f"  out sharding: SEED {f_jit.sharding.spec}  PROJECT {w_jit.sharding.spec}")

# ---- whole-function jit (stages fused) vs eager apply_screening_resolvent_block ----
max_iter, tol = 200, 1e-10
psi_c_X, psi_v_X = data["psi_c_X"], data["psi_v_X"]
psi_c_Y, psi_v_Y, V_q0 = data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]

@jax.jit
def _resolvent_jit(Gin, zin):
    Gr = jax.lax.with_sharding_constraint(
        jnp.broadcast_to(jnp.asarray(Gin, jnp.float64)[:, :, None], (n_pad, n_rmu, nk)), sh.S)
    ff = jax.lax.with_sharding_constraint(gen(Gr, psi_c_X, psi_v_X, V_q0), sh.X)
    rhs = jax.lax.with_sharding_constraint(jnp.stack([ff, -ff], axis=0).astype(jnp.complex128), sh.X_full)
    rhs_scan = jnp.moveaxis(rhs, 1, 0)
    def _solve_col(carry, rhs_col):
        rhs_i = rhs_col[:, None]
        x, _ = gmres_solve_sharded_jit(matvec, diag_h, zin, rhs_i, data, max_iter=max_iter, tol=tol)
        r_true = rhs_i - _apply_shifted_matvec(matvec, x, zin, data)
        nrhs = jnp.linalg.norm(rhs_i)
        resid = jnp.where(nrhs == 0.0, jnp.asarray(0.0, nrhs.dtype), jnp.linalg.norm(r_true) / nrhs)
        s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
        return carry, (s[0], resid)
    _, (s_all, resids) = jax.lax.scan(_solve_col, None, rhs_scan)
    s_all = jax.lax.with_sharding_constraint(s_all, sh.X)
    W = snapshot(s_all, psi_c_Y, psi_v_Y, V_q0)
    return W, resids

W_ref, res_ref = apply_screening_resolvent_block(
    G, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh, max_iter=max_iter, tol=tol)
W_ref = np.asarray(jax.device_get(W_ref)); res_ref = np.asarray(jax.device_get(res_ref))
W_wf, res_wf = _resolvent_jit(jnp.asarray(G, jnp.float64), jnp.asarray(0.0 + 0.0j, jnp.complex128))
W_wf = np.asarray(jax.device_get(W_wf)); res_wf = np.asarray(jax.device_get(res_wf))
d_W = float(np.max(np.abs(W_wf - W_ref))); d_res = float(np.max(np.abs(res_wf - res_ref)))
t_wf = timeit(lambda: _resolvent_jit(jnp.asarray(G, jnp.float64), jnp.asarray(0.0 + 0.0j, jnp.complex128)), n=5)
t_eager = timeit(lambda: apply_screening_resolvent_block(
    G, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh, max_iter=max_iter, tol=tol), n=5)
print(f"\n[whole-fn jit vs eager]  eager={1e3*t_eager:8.1f} ms  wf-jit={1e3*t_wf:8.2f} ms  "
      f"speedup={t_eager/t_wf:6.1f}x")
print(f"  max|W_wf - W_eager| = {d_W:.2e}   max|res_wf - res_eager| = {d_res:.2e}   "
      f"W bit-identical: {'YES' if d_W==0.0 else 'NO'}")
