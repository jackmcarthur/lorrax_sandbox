"""Instrument the sharded finite-q resolvent: V_q magnitude, diag_h range,
GMRES convergence, and sharded-vs-dense chi0 agreement (roll +q, no phase).
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse import bse_io
from bse.bse_feast import build_preconditioner_diagonal_sharded
from bse.bse_w_exact import _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols

RESTART, INPUT = sys.argv[1], sys.argv[2]


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
    a = jnp.roll(a, shift=(int(shift[0]), int(shift[1]), int(shift[2])), axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def finite_q_data(data, q, mesh):
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    dq = dict(data)
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    dq["V_q0"] = data["V_q_full"][:, :, qx, qy, qz]
    pcx = roll_k(data["psi_c_X"], q, nkx, nky, nkz)
    dq["psi_c_X"] = jax.lax.with_sharding_constraint(pcx, NamedSharding(mesh, P(None, None, None, "x")))
    dq["psi_c_Y"] = jax.lax.with_sharding_constraint(pcx, NamedSharding(mesh, P(None, None, None, "y")))
    dq["eps_c"] = jax.lax.with_sharding_constraint(
        roll_k(data["eps_c"], q, nkx, nky, nkz), NamedSharding(mesh, P(None, None)))
    return dq


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=INPUT, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])

    for q in [(0, 0, 0), (0, 1, 0)]:
        dq = finite_q_data(data, q, mesh)
        Vq = np.asarray(jax.device_get(dq["V_q0"]))[:nlog, :nlog]
        diag = build_preconditioner_diagonal_sharded(dq, mesh, include_W=False, use_tda=False)
        dloc = np.asarray(jax.device_get(diag))
        print(f"\nq={q}: ||V_q||_F={np.linalg.norm(Vq):.3e}  "
              f"max|V_q|={np.max(np.abs(Vq)):.3e}  V_q[0,0]={Vq[0,0]:.3e}")
        print(f"   diag_h: min|.|={np.min(np.abs(dloc)):.3e}  Re range=[{dloc.real.min():.3e},{dloc.real.max():.3e}]")

        for mi in (200, 400, 800):
            matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, dq)
            W0 = np.asarray(jax.device_get(data["W_q"][:, :, q[0], q[1], q[2]]))
            Vf = np.asarray(jax.device_get(data["V_q_full"][:, :, q[0], q[1], q[2]]))
            T = W0 - Vf
            cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
            W_tile, resids = _resolve_wc_columns(cols, 0.0 + 0.0j, dq, matvec, diag_h, gen, snapshot, sh,
                                                 max_iter=mi, tol=1e-10)
            wc = np.asarray(jax.device_get(W_tile))
            rr = np.asarray(jax.device_get(resids))
            rels = [float(np.linalg.norm(wc[:nlog, i] - T[:nlog, int(n0)]) / np.linalg.norm(T[:nlog, int(n0)]))
                    for i, n0 in enumerate(cols)]
            print(f"   max_iter={mi}: max gmres_resid={rr.max():.3e}  max tile rel_err={max(rels):.3e}")


if __name__ == "__main__":
    main()
