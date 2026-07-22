"""Separate the TWO off-grid suspects of the arbitrary-Q exciton path — the
interpolated htransform leg eps_c(k+Q) and the interpolated exchange tile
V_Q — with NO BSE solve.

Both legs are dumped over the SAME 39-Q path the failed production run used
(M -> Gamma -> K, 19+19 intervals) and gated ABSOLUTELY (a few meV), never
against a local trend: the Gamma-K leg of that run is bad end to end, which
an outlier detector structurally cannot see.

  A. eps leg   compute_wfns_fi over {k_sub + Q} for every Q, for several
               --a-band settings.  Reported per (k, band): second difference
               of eps_c along each leg (absolute meV), the recovered f'
               (the newton_inv amplification 1/f'), and the minimum
               transition min_k[eps_c(k+Q) - eps_v(k)] vs Q.
  B. V_q leg   eval_vq at every Q: Hermiticity residual, ||V||_F, the
               pair-projected block ||M0^H V M0||_F with a FIXED on-grid
               gap-window pair amplitude M0 (a physically weighted, gauge-
               correct scalar), and the eigenvalue spectrum.
  C. symmetry  the decisive reference-free test.  Point-group operations are
               DISCOVERED numerically (integer 2x2 blocks that map the k-grid
               to itself and leave the stored enk invariant), then applied to
               OFF-GRID Q.  eps_n(S p) must equal eps_n(p); the eigenvalue
               spectrum of V(S q) must equal that of V(q) (a permutation
               similarity is spectrum-preserving, so this is gauge-free).

Everything is JSON + npz; no plotting, no solve.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

RY2EV = 13.6056980659


def gather(x):
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(jax.device_get(x))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def second_diffs(y):
    """|y[i-1] - 2 y[i] + y[i+1]| along axis 0 (leg-local, absolute)."""
    y = np.asarray(y)
    if y.shape[0] < 3:
        return np.zeros((0,) + y.shape[1:])
    return np.abs(y[:-2] - 2.0 * y[1:-1] + y[2:])


def discover_symops(k_int, kgrid, enk, tol=1e-5):
    """Integer 2x2 in-plane blocks S with |det|=1 that map the stored k-grid
    onto itself AND leave the stored band energies invariant.  Self-validating:
    nothing is assumed about the lattice setting or the k vs r convention.

    ``tol`` is 1e-5 Ry (0.14 meV): the stored MoS2 QP bands are symmetric only
    to 7.9e-6 Ry, so a ULP tolerance keeps ONLY {I, -I} and throws away the
    whole C3/mirror family — which is exactly the part of the group that gives
    non-trivial off-grid images on the Gamma-M and Gamma-K legs."""
    lookup = {tuple(v): i for i, v in enumerate(k_int)}
    ops = []
    vals = (-1, 0, 1)
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    if abs(a * d - b * c) != 1:
                        continue
                    S = np.array([[a, b, 0], [c, d, 0], [0, 0, 1]], dtype=np.int64)
                    perm = np.empty(len(k_int), dtype=np.int64)
                    ok = True
                    for i, kv in enumerate(k_int):
                        img = tuple((S @ kv) % kgrid)
                        j = lookup.get(img)
                        if j is None:
                            ok = False
                            break
                        perm[i] = j
                    if not ok:
                        continue
                    err = float(np.max(np.abs(enk[perm] - enk)))
                    if err < tol:
                        ops.append((S, err))
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--eqp", default="eqp1.dat")
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-cond", type=int, default=8)
    ap.add_argument("--px", type=int, default=4)
    ap.add_argument("--py", type=int, default=4)
    ap.add_argument("--k-stride", type=int, default=4,
                    help="k-subset stride for the eps sweep (36 of 144 at 4)")
    ap.add_argument("--k-stride-sweep", type=int, default=12,
                    help="coarser k-subset for the extra --a-band settings")
    ap.add_argument("--a-bands", default="none,21,17",
                    help="comma list of a_band_index (window-relative); "
                         "'none' = the default (top of the fH window)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--skip-vq", action="store_true")
    ap.add_argument("--skip-eps", action="store_true")
    ap.add_argument("--out", default="offgrid_diag")
    args = ap.parse_args()

    t_all = time.time()
    rank0 = jax.process_index() == 0

    def log(*a):
        if rank0:
            print(*a, flush=True)

    from bse.bse_w_exact import _create_mesh_xy
    from bse.bse_io import (_find_restart_file,
                            load_bse_data_from_restart_sharded,
                            apply_eqp_corrections, resolve_n_occ)
    from bse import vq_interp
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi
    from gw.gw_config import read_lorrax_input
    import h5py

    mesh_xy = _create_mesh_xy(args.px, args.py)
    log(f"[dist] devices={jax.device_count()} mesh={dict(mesh_xy.shape)}")

    params = read_lorrax_input(args.input)
    restart_file = _find_restart_file(args.input)
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond, mesh_xy=mesh_xy,
        input_file=args.input, inject_head=True, load_v_full=False)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_cond = int(data["n_cond"])
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    log(f"  grid {nkx}x{nky}x{nkz}, n_rmu={n_rmu} (pad {n_rmu_pad})")

    with h5py.File(restart_file, "r") as f:
        enk_dft_full = np.asarray(f["enk_full"][:])
    n_occ_in = resolve_n_occ(enk_dft_full, input_file=args.input)
    enk_qp_full = apply_eqp_corrections(enk_dft_full, args.eqp)
    log(f"  [eqp] n_occ={n_occ_in}, shifts "
        f"{(enk_qp_full - enk_dft_full).min()*RY2EV:+.3f} / "
        f"{(enk_qp_full - enk_dft_full).max()*RY2EV:+.3f} eV")

    (wfn, sym, meta, _m, _S, ctilde, B_at_mu,
     enk_sigma) = ht.initialize_wfns(args.input, params, log, mesh_xy=mesh_xy)
    b0 = int(wfn.nelec) - int(params["nval"])
    b1 = int(wfn.nelec) + int(params["ncond"])
    enk_sigma = jnp.asarray(enk_qp_full[:, b0:b1].T)
    nb_window = int(ctilde.shape[1])
    nval_in = int(params["nval"])
    b_min, b_max = nval_in, nval_in + n_cond
    log(f"  fH window {nb_window} bands; BSE conduction [{b_min},{b_max})")

    kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn, params)
    Qpath = np.asarray(kpath_frac, dtype=np.float64)
    nQ = Qpath.shape[0]
    node_idx = [int(i) for i in node_idx]
    log(f"  Q path: {nQ} points, nodes {node_idx} {node_labels}")

    kgrid = np.array([nkx, nky, nkz], dtype=np.int64)
    k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                                  np.arange(nkz) / nkz,
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    k_int = np.rint(k_frac * kgrid[None, :]).astype(np.int64) % kgrid[None, :]

    # ── on-grid / off-grid classification of the path (absolute) ──────────
    frac = Qpath * kgrid[None, :]
    off_units = np.max(np.abs(frac - np.round(frac)), axis=1)
    ongrid_mask = off_units < 1e-6
    log(f"  on-grid Q: {int(ongrid_mask.sum())} of {nQ}; "
        f"max off-grid displacement {off_units.max():.4f} mesh units")

    out = {"nQ": int(nQ), "Qpath": Qpath.tolist(), "x_path": np.asarray(x_path).tolist(),
           "node_idx": node_idx, "node_labels": list(node_labels),
           "ongrid_mask": ongrid_mask.astype(int).tolist(),
           "offgrid_units": off_units.tolist()}
    npz = {}

    # ── discovered symmetry operations ────────────────────────────────────
    ops = discover_symops(k_int, kgrid, enk_qp_full)
    log(f"  symmetry: {len(ops)} integer point-group ops leave the stored "
        f"QP bands invariant on the {nkx}x{nky} grid")
    out["n_symops"] = len(ops)
    out["symops"] = [op[0][:2, :2].tolist() for op in ops]

    # legs of the path (split at the Gamma node)
    iG = node_idx[1]
    legs = {"M-Gamma": list(range(0, iG + 1)), "Gamma-K": list(range(iG, nQ))}

    # =====================================================================
    # A.  eps leg
    # =====================================================================
    if not args.skip_eps:
        # QP bands straight from the eqp table — unambiguous ordering, and it
        # keeps the on-grid reference on the SAME (quasiparticle) footing as
        # the interpolated leg.
        eps_vbm = enk_qp_full[:, n_occ_in - 1]                     # (nk,) Ry
        eps_st_qp = enk_qp_full[:, n_occ_in:n_occ_in + n_cond]     # (nk, nc) Ry
        a_bands = []
        for s in args.a_bands.split(","):
            s = s.strip()
            if s:
                a_bands.append(None if s.lower() in ("none", "default") else int(s))
        eps_res = {}
        for ia, ab in enumerate(a_bands):
            stride = args.k_stride if ia == 0 else args.k_stride_sweep
            ksub = np.arange(0, nk, stride)
            q_list = (Qpath[:, None, :] + k_frac[None, ksub, :]).reshape(-1, 3)
            t0 = time.time()
            bundle = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                band_window_fi=(b_min, b_max), mesh_xy=mesh_xy,
                q_list=q_list, a_band_index=ab, log_fn=log)
            eps = gather(bundle.enk_full).reshape(nQ, len(ksub), n_cond)
            lam = gather(bundle.lam_fi).reshape(nQ, len(ksub), n_cond)
            dt = time.time() - t0
            del bundle
            tag = "default" if ab is None else str(ab)
            # f' at the recovered energies (the newton_inv amplification)
            a_f, n_f, shift = ht._f_params_from_energies(
                enk_sigma, top_band_index=nb_window - 1, a_band_index=ab)
            dfp = np.asarray(ht.dfun(a_f, n_f, shift, jnp.asarray(eps)))
            # absolute smoothness, per leg, on the LOWEST conduction band
            leg_stats = {}
            for name, idx in legs.items():
                d2 = second_diffs(eps[idx]) * RY2EV * 1e3        # meV
                leg_stats[name] = {
                    "max_meV": float(d2.max()), "mean_meV": float(d2.mean()),
                    "max_meV_band0": float(d2[..., 0].max()),
                    "mean_meV_band0": float(d2[..., 0].mean()),
                    "per_Q_max_meV": d2.max(axis=(1, 2)).tolist()}
            # min transition vs Q (over the k-subset)
            dmin = np.min(eps[:, :, 0] - eps_vbm[None, ksub], axis=1)
            d2min = second_diffs(dmin[list(legs["M-Gamma"])]) * RY2EV * 1e3
            d2min_k = second_diffs(dmin[list(legs["Gamma-K"])]) * RY2EV * 1e3
            # on-grid reference: recovered eps at Q on the mesh vs the stored
            og = np.where(ongrid_mask)[0]
            eps_st = eps_st_qp
            lut = {tuple(v): j for j, v in enumerate(k_int)}
            og_err = []
            for i in og:
                sh = np.rint(Qpath[i] * kgrid).astype(np.int64)
                tgt = (k_int[ksub] + sh[None, :]) % kgrid[None, :]
                jj = np.array([lut[tuple(t)] for t in tgt])
                og_err.append(float(np.max(np.abs(eps[i] - eps_st[jj]))))
            eps_res[tag] = {
                "a_band": ab, "a_f_Ry": float(a_f), "shift_Ry": float(shift),
                "n_k_sub": int(len(ksub)), "seconds": dt,
                "legs": leg_stats,
                "min_transition_eV": (dmin * RY2EV).tolist(),
                "d2_min_transition_MG_meV": d2min.tolist(),
                "d2_min_transition_GK_meV": d2min_k.tolist(),
                "d2_min_transition_MG_max_meV": float(d2min.max()) if d2min.size else None,
                "d2_min_transition_GK_max_meV": float(d2min_k.max()) if d2min_k.size else None,
                "dfprime_min": float(dfp.min()), "dfprime_median": float(np.median(dfp)),
                "ongrid_max_abs_deps_meV": (max(og_err) * RY2EV * 1e3) if og_err else None,
                "lam_min": float(lam.min()), "lam_max": float(lam.max())}
            npz[f"eps_{tag}"] = eps
            npz[f"lam_{tag}"] = lam
            npz[f"ksub_{tag}"] = ksub
            log(f"  [eps a_band={tag}] {dt:.1f}s  a_f={a_f:.4f} Ry  "
                f"min f'={dfp.min():.3e}  ongrid|deps|={eps_res[tag]['ongrid_max_abs_deps_meV']} meV")
            for name in legs:
                s = leg_stats[name]
                log(f"      2nd-diff {name:9s} max {s['max_meV']:9.1f} meV  "
                    f"mean {s['mean_meV']:8.1f} meV")
            log(f"      2nd-diff min-transition  M-G max "
                f"{eps_res[tag]['d2_min_transition_MG_max_meV']} meV  "
                f"G-K max {eps_res[tag]['d2_min_transition_GK_max_meV']} meV")

        # ── eps symmetry covariance at OFF-GRID points ────────────────────
        # Test points: k_sub + Q at a handful of off-grid Q, and their images.
        if ops:
            offQ = [i for i in range(nQ) if not ongrid_mask[i]]
            pick = offQ[:: max(1, len(offQ) // 8)][:8]
            base = np.concatenate(
                [Qpath[i][None, :] + k_frac[np.arange(0, nk, 24)] for i in pick])
            pts = [base]
            for S, _ in ops:
                pts.append(base @ np.asarray(S, dtype=np.float64).T)
            pts = np.concatenate(pts)
            bundle = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                band_window_fi=(b_min, b_max), mesh_xy=mesh_xy,
                q_list=pts, a_band_index=a_bands[0], log_fn=log)
            e = gather(bundle.enk_full).reshape(len(ops) + 1, len(base), n_cond)
            del bundle
            dev = np.abs(e[1:] - e[0][None]) * RY2EV * 1e3           # meV
            out["eps_symmetry"] = {
                "n_test_points": int(len(base)), "n_ops": len(ops),
                "max_dev_meV": float(dev.max()),
                "mean_dev_meV": float(dev.mean()),
                "per_op_max_meV": dev.max(axis=(1, 2)).tolist(),
                "band0_max_meV": float(dev[..., 0].max())}
            npz["eps_sym"] = e
            log(f"  [eps symmetry] {len(base)} off-grid points x {len(ops)} ops: "
                f"max |Δeps| = {dev.max():.3f} meV (band0 {dev[...,0].max():.3f})")
        out["eps"] = eps_res
        del ctilde, B_at_mu

    # =====================================================================
    # B.  V_q leg
    # =====================================================================
    if not args.skip_vq:
        t0 = time.time()
        kw = {} if args.alpha is None else {"alpha": args.alpha}
        vqm = vq_interp.build_vq_evaluator(restart_file, mesh_xy, n_rmu_pad,
                                           log_fn=log, **kw)
        zx, prep = vqm.zx, vqm.prep
        log(f"  vq model built in {time.time()-t0:.1f}s")

        # FIXED on-grid gap-window pair amplitude M0 -> a physical, gauge-
        # correct scalar functional of V(Q).
        M0 = vq_interp.gap_window_pairs(zx, 0, nvw=2, ncw=2)       # (npair, n_mu)
        M0d = jnp.asarray(M0)

        @jax.jit
        def _metrics(V):
            Vl = V[:n_rmu, :n_rmu]
            herm = (jnp.linalg.norm(Vl - jnp.conj(Vl).T)
                    / jnp.linalg.norm(Vl))
            B = jnp.conj(M0d) @ Vl @ M0d.T
            ev = jnp.linalg.eigvalsh(0.5 * (Vl + jnp.conj(Vl).T))
            return (jnp.linalg.norm(Vl), herm, jnp.linalg.norm(B),
                    jnp.trace(B).real, ev[-12:], ev[:12])

        def eval_at(qfrac):
            q = np.asarray(qfrac, dtype=np.float64)
            q = q - np.round(q)
            return vqm.eval_vq(jnp.asarray(q), prep["V_SRc"], vqm.pinvF,
                               vqm.coeffs_packed)

        rows = []
        Vprev = None
        cont = []
        for i in range(nQ):
            qt = -Qpath[i]
            V = eval_at(qt)
            fro, herm, bfro, btr, evhi, evlo = [np.asarray(gather(z))
                                                for z in _metrics(V)]
            rows.append({"iQ": i, "ongrid": bool(ongrid_mask[i]),
                         "fro": float(fro), "herm_rel": float(herm),
                         "B_fro": float(bfro), "B_trace": float(btr),
                         "ev_max": float(evhi[-1]), "ev_min": float(evlo[0])})
            if Vprev is not None:
                cont.append(float(np.asarray(gather(
                    jnp.linalg.norm(V - Vprev) / jnp.linalg.norm(V)))))
            Vprev = V
        out["vq_rows"] = rows
        out["vq_consecutive_relF"] = cont
        bfro = np.array([r["B_fro"] for r in rows])
        btr = np.array([r["B_trace"] for r in rows])
        fro = np.array([r["fro"] for r in rows])
        out["vq_smoothness"] = {}
        for name, idx in legs.items():
            out["vq_smoothness"][name] = {
                "d2_B_fro_rel_max": float((second_diffs(bfro[idx])
                                           / max(np.abs(bfro[idx]).max(), 1e-300)).max()),
                "d2_B_trace_rel_max": float((second_diffs(btr[idx])
                                             / max(np.abs(btr[idx]).max(), 1e-300)).max()),
                "d2_fro_rel_max": float((second_diffs(fro[idx])
                                         / max(fro[idx].max(), 1e-300)).max()),
                "d2_B_fro": second_diffs(bfro[idx]).tolist()}
        out["vq_max_herm_rel"] = float(max(r["herm_rel"] for r in rows))
        npz["vq_B_fro"] = bfro
        npz["vq_fro"] = fro
        log(f"  [V_q] max Hermiticity residual {out['vq_max_herm_rel']:.3e}; "
            f"||V||_F range {fro.min():.4g}..{fro.max():.4g}")
        for name in legs:
            s = out["vq_smoothness"][name]
            log(f"      rel 2nd-diff {name:9s} B_fro {s['d2_B_fro_rel_max']:.3e}  "
                f"||V||_F {s['d2_fro_rel_max']:.3e}")

        # ── V_q symmetry: eigenvalue spectra at Q vs S Q (gauge-free) ─────
        if ops:
            offQ = [i for i in range(nQ) if not ongrid_mask[i]]
            pick = offQ[:: max(1, len(offQ) // 6)][:6]

            @jax.jit
            def _spec(V):
                Vl = V[:n_rmu, :n_rmu]
                return jnp.linalg.eigvalsh(0.5 * (Vl + jnp.conj(Vl).T))

            sym_rows = []
            for i in pick:
                qt = -Qpath[i]
                s0 = np.asarray(gather(_spec(eval_at(qt))))
                devs = []
                for S, _ in ops:
                    qi = np.asarray(S, dtype=np.float64) @ qt
                    if np.max(np.abs((qi - qt) - np.round(qi - qt))) < 1e-12:
                        continue          # trivial image
                    s1 = np.asarray(gather(_spec(eval_at(qi))))
                    devs.append(float(np.max(np.abs(s1 - s0))
                                      / max(np.max(np.abs(s0)), 1e-300)))
                sym_rows.append({"iQ": i, "Q": Qpath[i].tolist(),
                                 "n_images": len(devs),
                                 "max_spec_relerr": max(devs) if devs else None,
                                 "spec_relerrs": devs})
                log(f"  [V_q symmetry] iQ={i} Q={np.round(Qpath[i],5).tolist()}: "
                    f"max spectrum rel-dev over {len(devs)} images = "
                    f"{max(devs) if devs else float('nan'):.3e}")
            out["vq_symmetry"] = sym_rows

    if rank0:
        with open(args.out + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
        np.savez_compressed(args.out + ".npz", **npz)
        print(f"Wrote {args.out}.json / .npz   TOTAL {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
