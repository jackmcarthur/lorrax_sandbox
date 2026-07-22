"""fH-window sweep for the off-grid eps_c(k+Q) leg — the FIX search.

``offgrid_diag.py`` localised the arbitrary-Q exciton failure to the
interpolated htransform leg (2nd differences of eps_c in Q: max 2.9 eV,
mean 0.27 eV, on BOTH legs; f' >= 0.71 everywhere in the BSE conduction
window, so the f-transform / ``--a-band`` is NOT the amplifier).  What is
left is the interpolation BASIS: the Galerkin alpha-basis is built from the
pair matrix (nk*nb, nspinor*n_mu), and at the failing setting that matrix is
numerically singular (sigma_min/sigma_max ~ 1e-16 at nb=28, nk=144,
n_mu=2412 — only a 1.20x margin on nspinor*n_mu > nk*nb).

This sweeps the fH window (nval, ncond) at a FIXED 8-band BSE conduction
selection and gates the recovered eps_c(k+Q) ABSOLUTELY:

    on-grid   max |eps_ht - eps_stored| over the on-grid path Q   (meV)
    off-grid  max/mean |2nd difference of eps_c along Q|, per leg (meV)
    off-grid  max |2nd difference of min_k[eps_c(k+Q) - eps_VBM]| (meV)

A real exciton band needs the last of these at the few-meV level.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp

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


def d2(y):
    y = np.asarray(y)
    return np.abs(y[:-2] - 2.0 * y[1:-1] + y[2:]) if y.shape[0] >= 3 \
        else np.zeros((0,) + y.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--eqp", default="eqp1.dat",
                    help="BGW-format eqp1.dat, or 'none' to gate on the DFT "
                         "energies.  'none' is what the 2026-07-20 known-good "
                         "run needs as a CONTROL: it was solved on DFT "
                         "energies (no --eqp), so gating it on QP energies "
                         "would not be the same calculation.")
    ap.add_argument("--n-cond", type=int, default=8, help="BSE conduction bands")
    ap.add_argument("--windows", default="14,14;12,12;10,10;8,10",
                    help="';'-separated 'nval,ncond' fH windows")
    ap.add_argument("--a-band", default="",
                    help="f-transform 'a' band (window-RELATIVE), the same "
                         "knob as exciton_bands --a-band.  '' = the driver "
                         "default (a from the TOP of the fH window, which the "
                         "flag's own help warns can collapse off-grid eps_c by "
                         "eV); an INT pins it; 'auto' = nval + n_cond - 1, the "
                         "top of the BSE conduction selection — this is what "
                         "the known-good 2026-07-20 run used (nval 26, "
                         "n_cond 8 -> --a-band 33) and it tracks the window so "
                         "every sweep point is gated at ITS production value.")
    ap.add_argument("--px", type=int, default=4)
    ap.add_argument("--py", type=int, default=4)
    ap.add_argument("--k-stride", type=int, default=24)
    ap.add_argument("--debug-bands", action="store_true",
                    help="at Q=0 print the recovered vs stored conduction\n"
                         "energies band by band — the fastest way to tell a\n"
                         "band OFFSET or a k PERMUTATION from a genuinely bad\n"
                         "interpolation basis")
    ap.add_argument("--out", default="eps_window_sweep")
    args = ap.parse_args()

    rank0 = jax.process_index() == 0

    def log(*a):
        if rank0:
            print(*a, flush=True)

    from bse.bse_w_exact import _create_mesh_xy
    from bse.bse_io import (_find_restart_file, apply_eqp_corrections,
                            resolve_n_occ)
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi
    from gw.gw_config import read_lorrax_input
    import h5py

    mesh_xy = _create_mesh_xy(args.px, args.py)
    params0 = read_lorrax_input(args.input)
    restart_file = _find_restart_file(args.input)
    with h5py.File(restart_file, "r") as f:
        enk_dft_full = np.asarray(f["enk_full"][:])
    n_occ = resolve_n_occ(enk_dft_full, input_file=args.input)
    enk_qp = (enk_dft_full if args.eqp.lower() == "none"
              else apply_eqp_corrections(enk_dft_full, args.eqp))

    windows = [tuple(int(v) for v in s.split(","))
               for s in args.windows.split(";") if s.strip()]
    out = {"n_occ": int(n_occ), "n_cond_bse": args.n_cond, "windows": {}}

    for (nv_in, nc_in) in windows:
        t0 = time.time()
        params = dict(params0)
        params["nval"], params["ncond"] = nv_in, nc_in
        params["nband"] = n_occ + nc_in
        (wfn, sym, meta, _m, _S, ctilde, B_at_mu,
         enk_sigma) = ht.initialize_wfns(args.input, params, log,
                                         mesh_xy=mesh_xy)
        nb_window = int(ctilde.shape[1])
        rank = int(ctilde.shape[2])
        # ``enk_win`` (nk, nb_window) is BOTH the energy fH is built from and
        # the reference the recovered eps_c is gated against — one array, so
        # the gate can never be measuring a k-ORDERING mismatch between two
        # different sources instead of the htransform.  It follows the driver:
        #   --eqp <file>  the restart's enk_full, QP-corrected  (bse.exciton_bands
        #                 does exactly this for its interpolated leg)
        #   --eqp none    the energies ``initialize_wfns`` read from the WFN,
        #                 i.e. what the driver uses when no --eqp is given.
        #                 Overriding these with the restart's enk_full is what
        #                 made the 2026-07-20 known-good CONTROL report 1742 meV
        #                 at Γ where the driver's own gate reports 0.855 meV.
        if args.eqp.lower() == "none":
            enk_win = np.asarray(jax.device_get(enk_sigma)).T   # (nk, nb_window)
        else:
            enk_win = enk_qp[:, n_occ - nv_in:n_occ + nc_in]
            enk_sigma = jnp.asarray(enk_win.T)
        b_min, b_max = nv_in, nv_in + args.n_cond
        if b_max > nb_window:
            log(f"  SKIP window ({nv_in},{nc_in}): BSE window exceeds fH")
            continue
        kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(
            wfn, params)
        Qpath = np.asarray(kpath_frac, dtype=np.float64)
        nQ = Qpath.shape[0]
        nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
        nk = nkx * nky * nkz
        kgrid = np.array([nkx, nky, nkz], dtype=np.int64)
        k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx,
                                      np.arange(nky) / nky,
                                      np.arange(nkz) / nkz,
                                      indexing="ij"), axis=-1).reshape(-1, 3)
        k_int = np.rint(k_frac * kgrid).astype(np.int64) % kgrid
        lut = {tuple(v): j for j, v in enumerate(k_int)}
        ksub = np.arange(0, nk, args.k_stride)
        q_list = (Qpath[:, None, :] + k_frac[None, ksub, :]).reshape(-1, 3)
        if args.a_band == "":
            a_band = None
        elif args.a_band == "auto":
            a_band = b_max - 1
        else:
            a_band = int(args.a_band)
        bundle = compute_wfns_fi(
            ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
            kgrid_co=(nkx, nky, nkz), band_window_fi=(b_min, b_max),
            mesh_xy=mesh_xy, q_list=q_list, a_band_index=a_band, log_fn=log)
        eps = gather(bundle.enk_full).reshape(nQ, len(ksub), args.n_cond)
        del bundle, ctilde, B_at_mu

        frac = Qpath * kgrid[None, :]
        ongrid = np.max(np.abs(frac - np.round(frac)), axis=1) < 1e-6
        eps_st = enk_win[:, b_min:b_max]        # window-relative == absolute
        eps_vbm = enk_win[:, b_min - 1]
        og_err = 0.0
        og_per_Q = {}
        for i in np.where(ongrid)[0]:
            sh = np.rint(Qpath[i] * kgrid).astype(np.int64)
            jj = np.array([lut[tuple(t)] for t in
                           (k_int[ksub] + sh[None, :]) % kgrid[None, :]])
            e_i = float(np.max(np.abs(eps[i] - eps_st[jj])))
            # Q = 0 is the ONE on-grid point whose k-index map is the identity,
            # so it isolates the htransform from any k-ordering assumption in
            # ``lut``: a large error at Γ is the basis, a large error only at
            # M/K would be the map.  (It is also the only point the driver's
            # own gate checks.)
            og_per_Q[f"{Qpath[i][0]:+.4f},{Qpath[i][1]:+.4f},{Qpath[i][2]:+.4f}"] \
                = e_i * RY2EV * 1e3
            og_err = max(og_err, e_i)
            if args.debug_bands and np.allclose(Qpath[i], 0.0, atol=1e-9):
                log("    [dbg] Q=0, k=k_frac[ksub[0]] — recovered vs stored "
                    "conduction (eV):")
                log("      ht  " + " ".join(f"{v*RY2EV:9.4f}" for v in eps[i, 0]))
                log("      st  " + " ".join(f"{v*RY2EV:9.4f}"
                                            for v in eps_st[jj[0]]))
                log("      full window (stored, eV): "
                    + " ".join(f"{v*RY2EV:.4f}" for v in enk_win[jj[0]]))
        iG = int(node_idx[1])
        legs = {"M-Gamma": list(range(0, iG + 1)),
                "Gamma-K": list(range(iG, nQ))}
        dmin = np.min(eps[:, :, 0] - eps_vbm[None, ksub], axis=1)
        rec = {"nb_window": nb_window, "rank": rank,
               "a_band": (None if a_band is None else int(a_band)),
               "window_anchored_at_Emin": bool(nv_in == n_occ),
               "capacity_margin": float(B_at_mu_shape_margin(rank, nk, nb_window)),
               "seconds": time.time() - t0,
               "ongrid_max_deps_meV": og_err * RY2EV * 1e3,
               "ongrid_deps_per_Q_meV": og_per_Q,
               "min_transition_eV": (dmin * RY2EV).tolist()}
        for name, idx in legs.items():
            g = d2(eps[idx]) * RY2EV * 1e3
            gm = d2(dmin[idx]) * RY2EV * 1e3
            rec[name] = {"eps_d2_max_meV": float(g.max()),
                         "eps_d2_mean_meV": float(g.mean()),
                         "mintrans_d2_max_meV": float(gm.max()),
                         "mintrans_d2_mean_meV": float(gm.mean())}
        out["windows"][f"{nv_in},{nc_in}"] = rec
        rec["rank_over_nk_nb"] = rank / float(nk * nb_window)
        log(f"[window {nv_in}v+{nc_in}c  nb={nb_window}  rank={rank}"
            f" ({rank/float(nk*nb_window):.3f}·nk·nb)  "
            f"a_band={a_band}  anchored@Emin={nv_in == n_occ}]  "
            f"{rec['seconds']:.0f}s  on-grid |deps| = "
            f"{rec['ongrid_max_deps_meV']:.2f} meV  per-Q "
            + " ".join(f"{k}:{v:.1f}" for k, v in og_per_Q.items()))
        for name in legs:
            r = rec[name]
            log(f"    {name:9s} eps 2nd-diff max {r['eps_d2_max_meV']:9.2f} "
                f"mean {r['eps_d2_mean_meV']:8.2f} meV | min-transition "
                f"2nd-diff max {r['mintrans_d2_max_meV']:9.2f} mean "
                f"{r['mintrans_d2_mean_meV']:8.2f} meV")

    if rank0:
        with open(args.out + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"Wrote {args.out}.json")
    return 0


def B_at_mu_shape_margin(rank, nk, nb):
    """rank of the alpha basis over the pair count it has to carry."""
    return rank / max(nk * nb, 1)


if __name__ == "__main__":
    raise SystemExit(main())
