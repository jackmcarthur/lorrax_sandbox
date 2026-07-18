"""Solve-stage attribution probe (12x12 exciton driver shapes).

1. Bare stack-matvec warm time (8-trial block) -> how much of the measured
   ~115 ms/iteration is the matvec vs solver overhead (QR/reorth/alpha).
2. Ritz-value drift vs max_iter (20/40/80) at Gamma -> is Krylov 320
   converged, and how much iteration headroom exists (values question,
   reported not changed).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--py", type=int, default=1)
    args = ap.parse_args()

    from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
    from bse.bse_w_exact import _create_mesh_xy
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from bse.bse_serial import compute_pair_amplitude
    from common.fft_helpers import make_sharded_ifftn_3d
    from solvers.lanczos import block_lanczos_eig_jit
    from functools import partial

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=4, n_cond=4, mesh_xy=mesh_xy,
        input_file=args.input, inject_head=True)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad, nv_pad = int(data["n_cond_pad"]), int(data["n_val_pad"])
    n_flat = nc_pad * nv_pad * nk
    sh = make_bse_shardings(mesh_xy)

    _ifftn = make_sharded_ifftn_3d(mesh_xy, sh.W.spec, sh.W.spec,
                                   axes=(2, 3, 4), norm="ortho")
    W_R = _ifftn(data["W_q"])
    matvec = build_bse_stack_matvec(mesh_xy, nkx, nky, nkz, kernel="bse")
    M_X = data["M_X"]; M_Y = data["M_Y"]

    key = jax.random.PRNGKey(0)
    X = (jax.random.normal(key, (8, nc_pad, nv_pad, nk), dtype=jnp.float64)
         + 0j)
    X = jax.device_put(X, sh.X)

    args10 = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"],
              data["psi_v_Y"], data["eps_c"], data["eps_v"], W_R,
              data["V_q0"], M_X, M_Y)
    HX = matvec(X, *args10)
    jax.block_until_ready(HX)
    t0 = time.time()
    NREP = 20
    for _ in range(NREP):
        HX = matvec(HX, *args10)
    jax.block_until_ready(HX)
    dt = (time.time() - t0) / NREP
    print(f"[probe] bare stack matvec (bs=8) warm: {dt*1e3:.1f} ms/call",
          flush=True)

    # full solve at several max_iter -> per-iter overhead + Ritz drift
    ref = None
    for mi in (20, 40, 80):
        @jax.jit
        def _solve(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                   W_R, V_q0, M_X, M_Y, _mi=mi):
            def mv(Vb):
                Xb = Vb.reshape(8, nc_pad, nv_pad, nk)
                Xb = jax.lax.with_sharding_constraint(Xb, sh.X)
                return matvec(Xb, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                              eps_c, eps_v, W_R, V_q0, M_X, M_Y
                              ).reshape(8, -1)
            evs, _ = block_lanczos_eig_jit(
                mv, n_flat, n_eig=8, block_size=8, max_iter=_mi,
                n_reorth=_mi)
            return evs.real
        evs = _solve(*args10)
        jax.block_until_ready(evs)
        t0 = time.time()
        evs = _solve(*args10)
        jax.block_until_ready(evs)
        dt = time.time() - t0
        evs = np.asarray(evs)[:8]
        line = " ".join(f"{e:.9f}" for e in evs)
        print(f"[probe] solve max_iter={mi:3d}: warm {dt*1e3:8.1f} ms  evs(Ry): {line}",
              flush=True)
        if ref is None and mi == 40:
            pass
        if mi == 40:
            ref = evs
        if mi == 80 and ref is not None:
            d = np.max(np.abs(evs - ref)) * 13.6056980659e3
            print(f"[probe] max|evs(80) - evs(40)| = {d:.6f} meV", flush=True)


if __name__ == "__main__":
    main()
