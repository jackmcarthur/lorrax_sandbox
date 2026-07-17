"""Determine the finite-q W_q resolvent convention empirically.

Sweeps (roll_shift = +q vs -q) x (umklapp phase off vs on) for the smallest
nonzero symmetry-reduced q on the MoS2 gnppm fixture, comparing the RPA
screening resolvent against the on-disk (W0_qmunu - V_qmunu)[q_flat] tile.
Also checks q=0 through the same roll machinery (identity) as a control.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from bse import bse_io
from bse.bse_w_exact import _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols
from common.symmetry_maps import kgrid_shift_map

RESTART = sys.argv[1]
INPUT = sys.argv[2]   # cohsex .in (resolves wfn_file -> WFN.h5)
CENTROIDS = sys.argv[3]


def roll_k(arr, shift, nkx, nky, nkz):
    tail = arr.shape[1:]
    a = arr.reshape((nkx, nky, nkz) + tail)
    a = jnp.roll(a, shift=(int(shift[0]), int(shift[1]), int(shift[2])),
                 axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def load_centroids_frac(path, n_pad):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    s = np.asarray(rows, dtype=np.float64)
    out = np.zeros((n_pad, 3), dtype=np.float64)
    out[:s.shape[0]] = s
    return out


def build_finite_q_data(data, q, mesh_xy, *, roll_shift, umklapp_phase, cfrac):
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    dq = dict(data)
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    dq["V_q0"] = data["V_q_full"][:, :, qx, qy, qz]
    psi_c_X = roll_k(data["psi_c_X"], roll_shift, nkx, nky, nkz)
    eps_c = roll_k(data["eps_c"], roll_shift, nkx, nky, nkz)
    if umklapp_phase:
        q_off = (-int(roll_shift[0]), -int(roll_shift[1]), -int(roll_shift[2]))
        _, G_umk = kgrid_shift_map(nkx, nky, nkz, q_off)   # (nk,3)
        n_pad = int(data["n_rmu_pad"])
        phase = np.exp(-2j * np.pi * (G_umk.astype(np.float64) @ cfrac[:n_pad].T))  # (nk, n_pad)
        psi_c_X = psi_c_X * jnp.asarray(phase)[:, None, None, :]
    psi_c_X = jax.lax.with_sharding_constraint(
        psi_c_X, jax.sharding.NamedSharding(mesh_xy, P(None, None, None, "x")))
    dq["psi_c_X"] = psi_c_X
    dq["psi_c_Y"] = jax.lax.with_sharding_constraint(
        psi_c_X, jax.sharding.NamedSharding(mesh_xy, P(None, None, None, "y")))
    dq["eps_c"] = jax.lax.with_sharding_constraint(
        eps_c, jax.sharding.NamedSharding(mesh_xy, P(None, None)))
    return dq


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=INPUT, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])
    n_pad = int(data["n_rmu_pad"])
    cfrac = load_centroids_frac(CENTROIDS, n_pad)

    # self-check: V_q_full[...,0] == V_q0 (both head-less)
    dv = float(np.max(np.abs(np.asarray(jax.device_get(data["V_q_full"][:, :, 0, 0, 0]))
                             - np.asarray(jax.device_get(data["V_q0"])))))
    print(f"self-check V_q_full[...,0]==V_q0 : max|Δ|={dv:.2e}")

    # roll==gather self-check
    kpq, _ = kgrid_shift_map(3, 3, 1, (0, -1, 0))
    a = np.arange(9)
    rolled = np.asarray(roll_k(jnp.asarray(a.reshape(9, 1).astype(np.float64)),
                               (0, 1, 0), 3, 3, 1)).reshape(-1)
    print(f"roll==gather(kpq) : {np.array_equal(rolled.astype(int), a[kpq])}")

    def resolve_and_compare(q, roll_shift, phase):
        qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
        W0 = np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
        V0 = np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz]))
        T = W0 - V0
        dq = build_finite_q_data(data, q, mesh, roll_shift=roll_shift,
                                 umklapp_phase=phase, cfrac=cfrac)
        matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, dq)
        cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
        W_tile, resids = _resolve_wc_columns(
            cols, 0.0 + 0.0j, dq, matvec, diag_h, gen, snapshot, sh,
            max_iter=200, tol=1e-10)
        wc = np.asarray(jax.device_get(W_tile))
        rels = []
        for i, nu0 in enumerate(cols):
            tcol = T[:nlog, int(nu0)]
            rels.append(float(np.linalg.norm(wc[:nlog, i] - tcol) / np.linalg.norm(tcol)))
        return float(np.max(rels)), float(np.max(np.asarray(jax.device_get(resids))))

    print("\n--- q=0 control through roll machinery (roll_shift=0) ---")
    r, gr = resolve_and_compare((0, 0, 0), (0, 0, 0), False)
    print(f"q=(0,0,0) roll=0 phase=off : max rel_err={r:.3e} gmres={gr:.2e}")

    q = (0, 1, 0)
    print(f"\n--- smallest nonzero q={q} (flat={q[1]}) : convention sweep ---")
    for phase in (False, True):
        for sign in (+1, -1):
            rs = (sign * q[0], sign * q[1], sign * q[2])
            try:
                r, gr = resolve_and_compare(q, rs, phase)
                print(f"roll_shift={rs} phase={'on ' if phase else 'off'} : "
                      f"max rel_err={r:.3e}  gmres_resid={gr:.2e}")
            except Exception as e:
                print(f"roll_shift={rs} phase={'on ' if phase else 'off'} : ERROR {e}")


if __name__ == "__main__":
    main()
