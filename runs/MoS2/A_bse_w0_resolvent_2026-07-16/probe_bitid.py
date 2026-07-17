"""Bit-identity investigation (run with builders in their BARE, pre-fix state).

Per mesh, reports:
  * gen/snapshot: plain jax.jit(shard_map) vs bare, and sharded-jit vs bare
    (same mesh) -> does jit change fp at all, and does plain-jit preserve it?
  * emits the BARE _resolve_wc_columns W tile to npz for a cross-mesh
    (1x1 vs 2x2) device-invariance baseline compare.

usage: probe_bitid.py <input.in> <px> <py> <out.npz>
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_w_exact import _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols

inp = sys.argv[1]; px = int(sys.argv[2]); py = int(sys.argv[3]); out = sys.argv[4]
restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:px*py]).reshape(px, py), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh, input_file=inp, inject_head=False)
nk = int(data["nkx"]*data["nky"]*data["nkz"]); n_rmu = int(data["V_q0"].shape[0]); nlog = int(data["n_rmu"])
matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)
T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0])) - np.asarray(jax.device_get(data["V_q0"])))
cols, _ = _select_compare_cols(T, nlog, 8, seed=7)
n_pad = int(np.ceil(len(cols)/py)*py)

def sync(x): jax.block_until_ready(x); return x
def mx(a, b): return float(jnp.max(jnp.abs(a - b)))

G = np.zeros((n_pad, n_rmu), np.float64)
for i, c in enumerate(cols): G[i, int(c)] = 1.0
r = jax.device_put(jnp.broadcast_to(jnp.asarray(G)[:, :, None], (n_pad, n_rmu, nk)), sh.S)

# gen: bare vs plain-jit vs sharded-jit (same mesh)
gp = jax.jit(gen)
gs = jax.jit(gen, in_shardings=(sh.S, sh.psi_x, sh.psi_x, sh.V), out_shardings=sh.X)
fb = sync(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
fp = sync(gp(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
fs = sync(gs(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]))
print(f"[gen  px{px}py{py}] plainjit-vs-bare={mx(fp,fb):.3e}  shardjit-vs-bare={mx(fs,fb):.3e}")

# snapshot: bare vs plain-jit vs sharded-jit (use fb as stand-in s, sh.X)
sp = jax.jit(snapshot)
ss = jax.jit(snapshot, in_shardings=(sh.X, sh.psi_y, sh.psi_y, sh.V), out_shardings=sh.V)
wb = sync(snapshot(fb, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
wp = sync(sp(fb, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
ws = sync(ss(fb, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"]))
print(f"[snap px{px}py{py}] plainjit-vs-bare={mx(wp,wb):.3e}  shardjit-vs-bare={mx(ws,wb):.3e}")

# BARE end-to-end tile for cross-mesh device-invariance baseline
Wt, _ = _resolve_wc_columns(cols, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh,
                            max_iter=200, tol=1e-10)
wc = np.asarray(jax.device_get(Wt))[:nlog, :len(cols)]
np.savez(out, cols=np.asarray(cols), wc=wc)
print(f"wrote {out} (BARE baseline)")
