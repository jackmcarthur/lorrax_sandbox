"""Confirm the SEED/PROJECT hotspot is bare-shard_map eager re-lowering.

Times gen/snapshot called raw (bare shard_map) vs wrapped in jax.jit, and
checks the outputs are bit-identical.  usage: probe <input.in> <px> <py>
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_w_exact import _build_rpa_resolvent, _select_compare_cols

inp = sys.argv[1]; px = int(sys.argv[2]); py = int(sys.argv[3])
restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:px*py]).reshape(px, py), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh, input_file=inp, inject_head=False)
nk = int(data["nkx"]*data["nky"]*data["nkz"])
n_rmu = int(data["V_q0"].shape[0]); nlog = int(data["n_rmu"])
matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)

W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
V0 = np.asarray(jax.device_get(data["V_q0"])); T = W0 - V0
cols, _ = _select_compare_cols(T, nlog, 8, seed=7)
n_pad = int(np.ceil(len(cols)/py)*py)
G = np.zeros((n_pad, n_rmu), np.float64)
for i, c in enumerate(cols): G[i, int(c)] = 1.0

def sync(x): jax.block_until_ready(x); return x
def timeit(fn, n=10, warmup=3):
    for _ in range(warmup): sync(fn())
    ts = [time.perf_counter() for _ in range(0)]
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); sync(r); out.append(time.perf_counter()-t0)
    return min(out), float(np.median(out))

# seed inputs
Gj = jnp.asarray(G, dtype=jnp.float64)
r = jax.device_put(jnp.broadcast_to(Gj[:, :, None], (n_pad, n_rmu, nk)), sh.S)

gen_jit = jax.jit(gen, in_shardings=(sh.S, sh.psi_x, sh.psi_x, sh.V), out_shardings=sh.X)

f_raw = sync(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
f_jit = sync(gen_jit(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
gen_match = float(jnp.max(jnp.abs(f_raw - f_jit)))

graw_min, graw_med = timeit(lambda: gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
gjit_min, gjit_med = timeit(lambda: gen_jit(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))

# project inputs (use f as a stand-in s)
s = f_raw
snap_jit = jax.jit(snapshot, in_shardings=(sh.X, sh.psi_y, sh.psi_y, sh.V), out_shardings=sh.V)
w_raw = sync(snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
w_jit = sync(snap_jit(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
snap_match = float(jnp.max(jnp.abs(w_raw - w_jit)))

sraw_min, sraw_med = timeit(lambda: snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
sjit_min, sjit_med = timeit(lambda: snap_jit(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))

print(f"[cfg] px={px} py={py}  n_pad={n_pad} N_mu={nlog}")
print(f"[gen ]  raw min={1e3*graw_min:8.2f}ms  jit min={1e3*gjit_min:8.3f}ms  "
      f"speedup={graw_min/gjit_min:7.1f}x   max|raw-jit|={gen_match:.2e}")
print(f"[snap]  raw min={1e3*sraw_min:8.2f}ms  jit min={1e3*sjit_min:8.3f}ms  "
      f"speedup={sraw_min/sjit_min:7.1f}x   max|raw-jit|={snap_match:.2e}")
