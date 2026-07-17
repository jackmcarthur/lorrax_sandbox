"""Profile apply_screening_resolvent_block: per-stage timing + matvec throughput.

usage: profile_resolvent.py <input.in> <px> <py> [n_cols]

Reports (all warm, min-of-N after warmup unless labeled COLD):
  * end-to-end resolve (cold first call incl. compile; warm)
  * per-stage: SEED / SOLVE(scan-GMRES) / PROJECT
  * SOLVE with vs without the per-column true-residual recompute (redundant matvec)
  * matvec batch-scaling: time(_apply_shifted_matvec) at b in {1,2,4,8,16,...}
    -> the utilization question (flat => underutilized => batching wins)
  * GMRES iteration count per column
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_feast import gmres_solve_sharded_jit, _apply_shifted_matvec
from bse.bse_w_exact import (
    _build_rpa_resolvent, apply_screening_resolvent_block, _select_compare_cols)

inp = sys.argv[1]
px = int(sys.argv[2]); py = int(sys.argv[3])
n_cols = int(sys.argv[4]) if len(sys.argv) > 4 else 8

restart = _find_restart_file(inp)
ndev = px * py
mesh = Mesh(np.array(jax.devices()[:ndev]).reshape(px, py), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh, input_file=inp, inject_head=False)

nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
nk = nkx * nky * nkz
n_rmu = int(data["V_q0"].shape[0]); nlog = int(data["n_rmu"])
n_cond_pad = int(data["n_cond_pad"]); n_val_pad = int(data["n_val_pad"])
matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)
z = 0.0 + 0.0j
max_iter, tol = 200, 1e-10

W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
V0 = np.asarray(jax.device_get(data["V_q0"])); T = W0 - V0
cols, _ = _select_compare_cols(T, nlog, n_cols, seed=7)
n_pad = int(np.ceil(len(cols) / py) * py)
G = np.zeros((n_pad, n_rmu), np.float64)
for i, c in enumerate(cols):
    G[i, int(c)] = 1.0

print(f"[cfg] px={px} py={py} ndev={ndev}  N_mu={nlog} (pad {n_rmu})  "
      f"nc={n_cond_pad} nv={n_val_pad} nk={nk}  n_cols={len(cols)} n_pad={n_pad}")

def sync(x):
    jax.block_until_ready(x); return x

def timeit(fn, n=8, warmup=2):
    for _ in range(warmup): sync(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); sync(fn()); ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))

# ---- end-to-end (cold + warm) ----
t0 = time.perf_counter()
Wt, res = apply_screening_resolvent_block(G, z, data, matvec, diag_h, gen, snapshot, sh,
                                          max_iter=max_iter, tol=tol)
sync((Wt, res))
cold = time.perf_counter() - t0
e2e_min, e2e_med = timeit(
    lambda: apply_screening_resolvent_block(G, z, data, matvec, diag_h, gen, snapshot, sh,
                                            max_iter=max_iter, tol=tol), n=5)
print(f"\n[end-to-end] COLD(1st,incl compile)={cold:8.3f}s   WARM min={e2e_min:8.3f}s med={e2e_med:8.3f}s")

# ---- stage 1: SEED ----
def stage1():
    Gj = jnp.asarray(G, dtype=jnp.float64)
    r = jax.device_put(jnp.broadcast_to(Gj[:, :, None], (n_pad, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)
    return rhs
rhs = sync(stage1())
s1_min, s1_med = timeit(stage1)

rhs_scan = jnp.moveaxis(rhs, 1, 0)

# ---- stage 2: SOLVE full (with residual recompute) ----
def _solve_full(carry, rhs_col):
    rhs_i = rhs_col[:, None]
    x, kit = gmres_solve_sharded_jit(matvec, diag_h, z, rhs_i, data, max_iter=max_iter, tol=tol)
    r_true = rhs_i - _apply_shifted_matvec(matvec, x, z, data)
    nrhs = jnp.linalg.norm(rhs_i)
    resid = jnp.where(nrhs == 0.0, jnp.asarray(0.0, nrhs.dtype), jnp.linalg.norm(r_true) / nrhs)
    s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
    return carry, (s[0], resid, kit)
def stage2_full():
    _, (s_all, resids, kits) = jax.lax.scan(_solve_full, None, rhs_scan)
    return s_all, resids, kits
sA, rA, kA = sync(stage2_full())
s2f_min, s2f_med = timeit(stage2_full, n=5)

# ---- stage 2: SOLVE no-recompute (drop the redundant matvec) ----
def _solve_nr(carry, rhs_col):
    rhs_i = rhs_col[:, None]
    x, kit = gmres_solve_sharded_jit(matvec, diag_h, z, rhs_i, data, max_iter=max_iter, tol=tol)
    s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
    return carry, (s[0], kit)
def stage2_nr():
    _, (s_all, kits) = jax.lax.scan(_solve_nr, None, rhs_scan)
    return s_all, kits
sync(stage2_nr())
s2n_min, s2n_med = timeit(stage2_nr, n=5)

# ---- stage 3: PROJECT ----
def stage3():
    return snapshot(sA, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
sync(stage3())
s3_min, s3_med = timeit(stage3)

kits = np.asarray(jax.device_get(kA))
print(f"\n[stages, warm min]  SEED={s1_min:7.4f}s  SOLVE_full={s2f_min:7.4f}s  "
      f"SOLVE_norecompute={s2n_min:7.4f}s  PROJECT={s3_min:7.4f}s")
print(f"[residual-recompute cost] SOLVE_full-SOLVE_nr = {s2f_min - s2n_min:7.4f}s "
      f"({100*(s2f_min-s2n_min)/s2f_min:.1f}% of SOLVE)")
print(f"[gmres iters/col] {kits.tolist()}  (mean {kits[:len(cols)].mean():.1f})")
tot = s1_min + s2f_min + s3_min
print(f"[sum of stages] {tot:.4f}s   (end-to-end warm {e2e_min:.4f}s)")
print(f"\n[HOTSPOT %]  SEED {100*s1_min/tot:5.1f}%   SOLVE {100*s2f_min/tot:5.1f}%   "
      f"PROJECT {100*s3_min/tot:5.1f}%")

# ---- matvec batch-scaling (the utilization question) ----
print("\n[matvec batch scaling]  time of ONE _apply_shifted_matvec on b columns")
print(f"{'b':>4} {'min[ms]':>9} {'ms/col':>9} {'speedup/b':>10}")
key = jax.random.PRNGKey(0)
base = None
for b in [1, 2, 4, 8, 16, 32]:
    if b > 4 * n_rmu:
        break
    k1, k2 = jax.random.split(key)
    x = (jax.random.normal(k1, (2, b, n_cond_pad, n_val_pad, nk))
         + 1j * jax.random.normal(k2, (2, b, n_cond_pad, n_val_pad, nk)))
    x = jax.lax.with_sharding_constraint(x.astype(jnp.complex128), sh.X_full)
    mmin, _ = timeit(lambda: _apply_shifted_matvec(matvec, x, z, data), n=10, warmup=3)
    if base is None:
        base = mmin
    print(f"{b:>4} {1e3*mmin:9.3f} {1e3*mmin/b:9.4f} {b*base/mmin:10.2f}")
