"""Sub-stage wall-clock profiler for the exciton-bandstructure pipeline.

Times the driver's stages at a finer grain than exciton_bands.main's tick():
  vq    — load_zeta / build_cq / run_gates / prepare_coarse (eigh vs host split)
          / lr_design_blocks / fit_lr_model / run_nulls / eval_vq cold+warm
  ht    — initialize_wfns / compute_wfns_fi q-list (per-batch warm marginal)
  solve — build_path_solver warm time vs max_iter (fixed overhead + slope),
          optional jax.profiler.trace of one warm scan

Run through lxrun_perf.sh (module-free srun+shifter, PYTHONPATH set there).
    python profile_stages.py -i <input> --px 2 --py 2 --stage vq,ht,solve
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

T: dict[str, float] = {}


def tic():
    return time.time()


def toc(name, t0, sync=None):
    if sync is not None:
        jax.block_until_ready(sync)
    T[name] = time.time() - t0
    print(f"[T] {name:<42s} {T[name]:9.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-cond", type=int, default=4)
    ap.add_argument("--stage", default="vq,ht,solve")
    ap.add_argument("--nq-path", type=int, default=3, help="fake path points for ht/solve")
    ap.add_argument("--max-iters", default="1,5,10,20,40",
                    help="solve stage: warm-time ladder over max_iter")
    ap.add_argument("--trace-dir", default=None, help="jax.profiler trace of one warm solve")
    ap.add_argument("--eigh-backend", default="off")
    args = ap.parse_args()
    stages = set(args.stage.split(","))

    from gw.gw_config import read_lorrax_input
    from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
    from bse.bse_w_exact import _create_mesh_xy
    from bse import vq_interp

    mesh_xy = _create_mesh_xy(args.px, args.py)
    params = read_lorrax_input(args.input)
    restart_file = _find_restart_file(args.input)

    t0 = tic()
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond,
        mesh_xy=mesh_xy, input_file=args.input, inject_head=True)
    toc("load_bse", t0, sync=data["W_q"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])

    # ══════════════ vq trainer ══════════════
    if "vq" in stages:
        zeta_file = os.path.join(os.path.dirname(restart_file), "zeta_q.h5")
        t0 = tic()
        zx = vq_interp.load_zeta_coarse(restart_file, zeta_file)
        toc("vq.load_zeta_coarse", t0)
        t0 = tic()
        C_q = vq_interp.build_cq(zx)
        toc("vq.build_cq", t0)
        gacc = {}
        for fname in ("recon", "to_sphere", "_batched_vq_relF"):
            if not hasattr(vq_interp, fname):
                continue

            def _mk(f, key):
                def wrap(*a, **kw):
                    tw = time.time()
                    out = f(*a, **kw)
                    gacc[key] = gacc.get(key, 0.0) + time.time() - tw
                    return out
                return wrap
            gacc[fname + "_orig"] = getattr(vq_interp, fname)
            setattr(vq_interp, fname, _mk(gacc[fname + "_orig"], fname))
        t0 = tic()
        vq_interp.run_gates(zx, C_q)
        toc("vq.run_gates", t0)
        for fname in ("recon", "to_sphere", "_batched_vq_relF"):
            if fname + "_orig" in gacc:
                setattr(vq_interp, fname, gacc[fname + "_orig"])
                print(f"[T]   run_gates.{fname:<24s} "
                      f"{gacc.get(fname, 0.0):9.2f} s", flush=True)

        # instrument prepare_coarse internals: eigh + device_get accounting
        acc = {"eigh": 0.0, "n_eigh": 0}
        orig_eigh = vq_interp._eigh_backend

        def timed_eigh(C_dev, mesh, backend):
            te = time.time()
            lam, R = orig_eigh(C_dev, mesh, backend)
            jax.block_until_ready(R)
            acc["eigh"] += time.time() - te
            acc["n_eigh"] += 1
            return lam, R
        vq_interp._eigh_backend = timed_eigh
        t0 = tic()
        prep = vq_interp.prepare_coarse(zx, C_q, mesh_xy,
                                        eigh_backend=args.eigh_backend)
        toc("vq.prepare_coarse", t0)
        vq_interp._eigh_backend = orig_eigh
        print(f"[T]   prepare_coarse.eigh (blocked)          {acc['eigh']:9.2f} s"
              f"  ({acc['n_eigh']} calls)", flush=True)
        print(f"[T]   prepare_coarse.non-eigh (host+xfer)    "
              f"{T['vq.prepare_coarse']-acc['eigh']:9.2f} s", flush=True)

        t0 = tic()
        des = vq_interp.lr_design_blocks(zx, prep)
        toc("vq.lr_design_blocks", t0)
        t0 = tic()
        coeffs = vq_interp.fit_lr_model(des)
        toc("vq.fit_lr_model", t0)
        t0 = tic()
        vq_interp.run_nulls(zx, prep, des, coeffs)
        toc("vq.run_nulls", t0)

        t0 = tic()
        eval_vq = vq_interp.make_eval_vq(zx, prep, des, mesh_xy, n_rmu_pad)
        pinvF = jnp.asarray(vq_interp.stencil_pinv(
            zx["qfr"], vq_interp.stencil_r7(zx)))
        coeffs_packed = vq_interp.pack_coeffs(des, coeffs)
        q_probe = jnp.asarray(np.array([0.11, 0.23, 0.0]))
        V1 = eval_vq(q_probe, prep["V_SRc"], pinvF, coeffs_packed)
        toc("vq.eval_vq_cold(1)", t0, sync=V1)
        t0 = tic()
        for _ in range(10):
            V1 = eval_vq(q_probe, prep["V_SRc"], pinvF, coeffs_packed)
        toc("vq.eval_vq_warm(10)", t0, sync=V1)

    # ══════════════ htransform q-list ══════════════
    if "ht" in stages or "solve" in stages:
        from bandstructure import htransform as ht
        from bandstructure.bse_setup import compute_wfns_fi
        t0 = tic()
        (wfn, sym, meta, _mesh, _S, ctilde, B_at_mu,
         enk_sigma) = ht.initialize_wfns(args.input, params, print,
                                         mesh_xy=mesh_xy)
        toc("ht.initialize_wfns", t0)

        nval_in = int(params["nval"])
        b_min, b_max = nval_in, nval_in + args.n_cond
        k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                                      np.arange(nkz) / nkz, indexing="ij"),
                          axis=-1).reshape(-1, 3)
        rng = np.random.default_rng(7)
        Qfake = rng.uniform(-0.4, 0.4, size=(args.nq_path, 3))
        Qfake[:, 2] = 0.0
        q_list = (Qfake[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)

        t0 = tic()
        bundle = compute_wfns_fi(
            ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
            kgrid_co=(nkx, nky, nkz), band_window_fi=(b_min, b_max),
            mesh_xy=mesh_xy, q_list=q_list, log_fn=print)
        toc(f"ht.compute_wfns_fi({args.nq_path}Q x {nk}k = {len(q_list)}q)",
            t0, sync=bundle.psi_rmu_Y)
        # warm marginal: one more identical call (compile cached)
        t0 = tic()
        bundle = compute_wfns_fi(
            ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
            kgrid_co=(nkx, nky, nkz), band_window_fi=(b_min, b_max),
            mesh_xy=mesh_xy, q_list=q_list, log_fn=print)
        toc("ht.compute_wfns_fi warm(same)", t0, sync=bundle.psi_rmu_Y)
        n_batches = (len(q_list) + 31) // 32
        print(f"[T]   warm per-32-batch: "
              f"{T['ht.compute_wfns_fi warm(same)']/n_batches*1e3:9.1f} ms "
              f"({n_batches} batches)", flush=True)

    # ══════════════ solve scan ══════════════
    if "solve" in stages:
        from bse.exciton_bands import (build_path_solver, build_conduction_stacks,
                                       PAD_EPS_GUARD_RY)
        from bse.bse_ring_comm import make_bse_shardings
        from common.fft_helpers import make_sharded_ifftn_3d
        from jax.sharding import NamedSharding, PartitionSpec as P

        n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
        nv_pad, nc_pad = int(data["n_val_pad"]), int(data["n_cond_pad"])
        if nv_pad > n_val:
            eps_v = np.asarray(jax.device_get(data["eps_v"]))
            eps_v[:, n_val:] = -PAD_EPS_GUARD_RY
            data["eps_v"] = jnp.asarray(eps_v)
        nQ = args.nq_path
        psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
            bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy)
        grid_xy = NamedSharding(mesh_xy, P("x", "y"))
        v_gamma = jax.device_put(data["V_q0"], grid_xy)
        V_stack = jax.device_put(
            jnp.stack([0.5 * (v_gamma + jnp.conj(v_gamma).T)] * nQ),
            NamedSharding(mesh_xy, P(None, "x", "y")))
        sh = make_bse_shardings(mesh_xy)
        _ifftn = make_sharded_ifftn_3d(mesh_xy, sh.W.spec, sh.W.spec,
                                       axes=(2, 3, 4), norm="ortho")
        W_R = _ifftn(data["W_q"])
        jax.block_until_ready(W_R)

        for mi in [int(s) for s in args.max_iters.split(",")]:
            solver = build_path_solver(
                mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=8,
                block_size=8, max_iter=mi)
            t0 = tic()
            evs = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                         data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R)
            toc(f"solve.cold nQ={nQ} max_iter={mi}", t0, sync=evs)
            t0 = tic()
            evs = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                         data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R)
            toc(f"solve.warm nQ={nQ} max_iter={mi}", t0, sync=evs)
            print(f"[T]   warm per-Q at max_iter={mi}: "
                  f"{T[f'solve.warm nQ={nQ} max_iter={mi}']/nQ*1e3:9.1f} ms",
                  flush=True)
            if args.trace_dir and mi == 40:
                with jax.profiler.trace(args.trace_dir):
                    evs = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                                 data["psi_v_X"], data["psi_v_Y"],
                                 data["eps_v"], W_R)
                    jax.block_until_ready(evs)
                print(f"[T] trace written to {args.trace_dir}", flush=True)

    print("\n[T] ===== summary =====")
    for k, v in T.items():
        print(f"[T] {k:<42s} {v:9.2f} s")


if __name__ == "__main__":
    main()
