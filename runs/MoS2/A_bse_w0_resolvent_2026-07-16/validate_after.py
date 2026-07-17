"""Post-fix validation of apply_screening_resolvent_block / _resolve_wc_columns.

Exercises the jitted seed/project boundaries.  Prints per-column closure rel_err
+ gmres_resid (gate-equivalent) and dumps W columns + resids to an npz so 1x1 and
2x2 runs can be compared bit-for-bit.

usage: validate_after.py <input.in> <px> <py> <out.npz>
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
nlog = int(data["n_rmu"])
T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
     - np.asarray(jax.device_get(data["V_q0"])))

matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)
cols, _ = _select_compare_cols(T, nlog, 8, seed=7)
W_tile, resids = _resolve_wc_columns(
    cols, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh, max_iter=200, tol=1e-10)

assert W_tile.sharding.spec == P("x", "y"), W_tile.sharding.spec
wc = np.asarray(jax.device_get(W_tile))      # (n_rmu, n_pad)
resids = np.asarray(jax.device_get(resids))

print(f"[cfg] px={px} py={py}  N_mu={nlog}  spec={W_tile.sharding.spec}")
print(f"{'nu':>5} {'||T_col||':>13} {'rel_err':>12} {'gmres_resid':>12}")
rels = []
for i, nu0 in enumerate(cols):
    tcol = T[:nlog, int(nu0)]
    rel = float(np.linalg.norm(wc[:nlog, i] - tcol) / np.linalg.norm(tcol))
    rels.append(rel)
    print(f"{int(nu0):5d} {np.linalg.norm(tcol):13.4e} {rel:12.4e} {resids[i]:12.4e}")
rels = np.asarray(rels)
print(f"max rel_err={rels.max():.4e}  median={np.median(rels):.4e}  "
      f"max gmres_resid={resids[:len(cols)].max():.4e}")
np.savez(out, cols=np.asarray(cols), wc=wc[:nlog, :len(cols)], resids=resids[:len(cols)], rels=rels)
print(f"wrote {out}")
