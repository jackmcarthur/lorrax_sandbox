"""fH_q eigh BACKEND sweep — memory high-water mark + the correctness gate.

Same experiment as ``reports/bse_exciton_smooth_2026-07-21/eps_window_sweep.py``
(whose gate math this file IMPORTS, so there is one definition of the metric),
with two additions the distributed-eigh initiative needs:

  * ``--eigh-backend off|cusolvermp|slate`` — which eigensolver decomposes
    fH_q inside ``bandstructure.bse_setup.compute_wfns_fi``.
  * per-window instrumentation: XLA high-water mark AND the true device arena
    (nvidia-smi), plus the wall clock split into Galerkin / htransform, so the
    "widest window that fits" question has a number attached.

Gated quantities, ABSOLUTE, per fH window:

    on-grid   max |eps_ht - eps_stored| over the on-grid Q                (meV)
    off-grid  max/mean |2nd difference of eps_c along Q|, per leg         (meV)
    off-grid  max |2nd difference of min_k[eps_c(k+Q) - eps_VBM]|         (meV)

On-grid agreement is BLIND to off-grid breakage in this pipeline (the
2026-07-21 failure passed at 0.855 meV on-grid while ringing at 2905 meV
off-grid), so a backend swap is only "correct" when BOTH move together.

usage (16 ranks, 4x4 mesh):
    python3 -u eigh_backend_gate.py -i exciton.in --eqp eqp1.dat \\
        --windows 26,8 --a-band auto --eigh-backend cusolvermp --out gate_mp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

# The sibling sweep IS the gate definition (d2 / gather / capacity margin) and
# it performs the env + distributed init at import time.  Import it FIRST, and
# reuse its helpers rather than restating them.
_SMOOTH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "bse_exciton_smooth_2026-07-21")
sys.path.insert(0, os.path.normpath(_SMOOTH))
import eps_window_sweep as ews                       # noqa: E402

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

RY2EV = ews.RY2EV
d2 = ews.d2
gather = ews.gather


# ---------------------------------------------------------------------------
# Memory instrumentation
# ---------------------------------------------------------------------------

def _gpu_used_gb():
    """Device memory already in use on THIS process's GPU, GB — read ONCE.

    The allocation is a shared pool: a co-tenant step with --overlap lands on
    the same GPUs, and its VRAM makes an OOM look like a capacity ceiling when
    it is really contention.  Every measurement records what was already
    resident before JAX allocated anything, so a contended point can be thrown
    out instead of quoted.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        h = pynvml.nvmlDeviceGetHandleByIndex(int(vis))
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
    except Exception:
        return None


def _host_hwm_gb():
    """Peak host RSS of THIS process (VmHWM), GB.  Four ranks share a node's
    RAM; a host-side OOM shows up as a bare SIGKILL with no Python traceback,
    so the number has to be logged, not inferred."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return None


def _xla_peak_gb():
    """XLA-visible high-water mark on this process's device, GB.

    ``peak_bytes_in_use`` is what the BFC allocator reports; under
    ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` (required by the FFI backends —
    NCCL/CAL starve behind a 0.95 BFC pool) the key can be missing, which is
    exactly why the nvidia-smi arena sampler below exists as well.
    """
    try:
        st = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return None
    v = st.get("peak_bytes_in_use")
    return None if v is None else v / 1e9


class ArenaSampler(threading.Thread):
    """Poll the true per-device memory arena (GB) in a background thread.

    The XLA HWM under-reports the arena by ~1.44x at this scale
    (project_planner_conservative_8x), and it is not even defined under the
    platform allocator, so the arena is the number that decides whether a
    window "fits".  ``pynvml`` is used when importable — an in-process NVML
    query, no fork; forking nvidia-smi out of a CUDA + NCCL + Cray-MPICH
    process is a hazard, so the subprocess reader is opt-in
    (``--arena-source smi``) and OFF by default.
    """

    def __init__(self, period=0.5, source="nvml"):
        super().__init__(daemon=True)
        self.period, self.peak, self._ev = period, 0.0, threading.Event()
        self.source, self._h = source, None
        if source == "nvml":
            try:
                import pynvml
                pynvml.nvmlInit()
                vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                self._h = pynvml.nvmlDeviceGetHandleByIndex(int(vis))
                self._nvml = pynvml
            except Exception:
                self._h = None

    def _sample(self):
        if self.source == "nvml":
            if self._h is None:
                return 0.0
            try:
                return self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 1e9
            except Exception:
                return 0.0
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.split()
            return max(float(v) for v in out) / 1024.0
        except Exception:
            return 0.0

    def run(self):
        if self.source == "off":
            return
        while not self._ev.wait(self.period):
            self.peak = max(self.peak, self._sample())

    def stop(self):
        self._ev.set()
        self.join(timeout=2)
        return self.peak


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--eqp", default="eqp1.dat",
                    help="BGW-format eqp1.dat, or 'none' for DFT energies")
    ap.add_argument("--n-cond", type=int, default=8,
                    help="BSE conduction bands returned from the fH window")
    ap.add_argument("--windows", default="26,8",
                    help="';'-separated 'nval,ncond' fH windows.  ANCHOR AT "
                         "E_min: nval = n_occ (26 for MoS2) makes the window "
                         "absolute bands [0, nval+ncond) — semicore included, "
                         "the owner's hard constraint.  Widening then means "
                         "raising ncond, i.e. raising `rank`.")
    ap.add_argument("--a-band", default="auto",
                    help="'' = driver default (top of fH window), an INT pins "
                         "it, 'auto' = nval + n_cond - 1 (top of the BSE "
                         "conduction selection) — the known-good setting")
    ap.add_argument("--eigh-backend", default="off",
                    choices=("auto", "off", "cusolvermp", "slate"))
    ap.add_argument("--px", type=int, default=4)
    ap.add_argument("--py", type=int, default=4)
    ap.add_argument("--k-stride", type=int, default=24,
                    help="sub-sample the k-sum inside each Q (the gate needs "
                         "the Q dependence, not every k)")
    ap.add_argument("--arena-source", default="nvml",
                    choices=("nvml", "smi", "off"),
                    help="how to sample the true device arena: in-process NVML "
                         "(default), a forked nvidia-smi, or off")
    ap.add_argument("--out", default="eigh_backend_gate")
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
    log(f"[dist] devices={jax.device_count()} processes={jax.process_count()} "
        f"mesh={dict(mesh_xy.shape)}  eigh_backend={args.eigh_backend}")
    params0 = read_lorrax_input(args.input)
    restart_file = _find_restart_file(args.input)
    with h5py.File(restart_file, "r") as f:
        enk_dft_full = np.asarray(f["enk_full"][:])
    n_occ = resolve_n_occ(enk_dft_full, input_file=args.input)
    enk_qp = (enk_dft_full if args.eqp.lower() == "none"
              else apply_eqp_corrections(enk_dft_full, args.eqp))

    windows = [tuple(int(v) for v in s.split(","))
               for s in args.windows.split(";") if s.strip()]
    out = {"n_occ": int(n_occ), "n_cond_bse": args.n_cond,
           "eigh_backend": args.eigh_backend,
           "mesh": [args.px, args.py], "windows": {}, "failures": {}}

    # Contention detector.  pynvml is not in the container, but the BFC pool's
    # ``bytes_limit`` is MEM_FRACTION x the FREE VRAM at backend init — so a
    # limit well below 0.95 x 80 GB means a co-tenant step (the allocation is a
    # shared pool) already holds memory on this GPU, and any OOM below is
    # CONTENTION, not a capacity ceiling.  Quote no ceiling from a contended run.
    _st = {}
    try:
        _st = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        pass
    lim = _st.get("bytes_limit")
    out["gpu_used_before_jax_gb"] = _gpu_used_gb()
    out["bfc_bytes_limit_gb"] = None if lim is None else lim / 1e9
    log(f"[gpu] BFC bytes_limit = {None if lim is None else round(lim/1e9,1)} GB"
        f" (clean 80 GB card at MEM_FRACTION 0.95 ~= 77-78 GB; much less means a"
        f" co-tenant holds VRAM and an OOM here is contention, not a ceiling)")

    for (nv_in, nc_in) in windows:
        tag = f"{nv_in},{nc_in}"
        arena = ArenaSampler(source=args.arena_source)
        arena.start()
        t0 = time.time()
        try:
            params = dict(params0)
            params["nval"], params["ncond"] = nv_in, nc_in
            params["nband"] = n_occ + nc_in
            (wfn, sym, meta, _m, _S, ctilde, B_at_mu,
             enk_sigma) = ht.initialize_wfns(args.input, params, log,
                                             mesh_xy=mesh_xy)
            t_gal = time.time() - t0
            nb_window = int(ctilde.shape[1])
            rank = int(ctilde.shape[2])
            if args.eqp.lower() == "none":
                enk_win = np.asarray(jax.device_get(enk_sigma)).T
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

            t1 = time.time()
            bundle = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(nkx, nky, nkz), band_window_fi=(b_min, b_max),
                mesh_xy=mesh_xy, q_list=q_list, a_band_index=a_band,
                eigh_backend=args.eigh_backend, log_fn=log)
            jax.block_until_ready(bundle.psi_rmu_Y)
            t_ht = time.time() - t1
            log(f"    [mem] rank0 host VmHWM {_host_hwm_gb():.1f} GB, "
                f"xla peak {_xla_peak_gb()} GB")
            eps = gather(bundle.enk_full).reshape(nQ, len(ksub), args.n_cond)
            del bundle, ctilde, B_at_mu
        except Exception as exc:                       # OOM or FFI refusal
            arena_gb = arena.stop()
            out["failures"][tag] = {
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "arena_peak_gb": arena_gb,
                "gpu_used_at_failure_gb": _gpu_used_gb(),
                "seconds": time.time() - t0}
            log(f"[window {tag}] FAILED after {time.time()-t0:.0f}s "
                f"(arena peak {arena_gb:.1f} GB): {type(exc).__name__}: "
                f"{str(exc)[:400]}")
            if rank0:
                with open(args.out + ".json", "w") as fh:
                    json.dump(out, fh, indent=1)
            continue

        arena_gb = arena.stop()
        xla_gb = _xla_peak_gb()

        frac = Qpath * kgrid[None, :]
        ongrid = np.max(np.abs(frac - np.round(frac)), axis=1) < 1e-6
        eps_st = enk_win[:, b_min:b_max]
        eps_vbm = enk_win[:, b_min - 1]
        og_err, og_per_Q = 0.0, {}
        for i in np.where(ongrid)[0]:
            sh = np.rint(Qpath[i] * kgrid).astype(np.int64)
            jj = np.array([lut[tuple(t)] for t in
                           (k_int[ksub] + sh[None, :]) % kgrid[None, :]])
            e_i = float(np.max(np.abs(eps[i] - eps_st[jj])))
            og_per_Q[f"{Qpath[i][0]:+.4f},{Qpath[i][1]:+.4f},"
                     f"{Qpath[i][2]:+.4f}"] = e_i * RY2EV * 1e3
            og_err = max(og_err, e_i)
        iG = int(node_idx[1])
        legs = {"M-Gamma": list(range(0, iG + 1)),
                "Gamma-K": list(range(iG, nQ))}
        dmin = np.min(eps[:, :, 0] - eps_vbm[None, ksub], axis=1)
        rec = {"nb_window": nb_window, "rank": rank,
               "a_band": (None if a_band is None else int(a_band)),
               "window_anchored_at_Emin": bool(nv_in == n_occ),
               "capacity_margin": float(rank / max(nk * nb_window, 1)),
               "nq": int(q_list.shape[0]),
               "seconds": time.time() - t0,
               "seconds_galerkin": t_gal,
               "seconds_htransform": t_ht,
               "ms_per_q": 1e3 * t_ht / max(q_list.shape[0], 1),
               "arena_peak_gb": arena_gb,
               "xla_peak_gb": xla_gb,
               "host_hwm_gb": _host_hwm_gb(),
               "ongrid_max_deps_meV": og_err * RY2EV * 1e3,
               "ongrid_deps_per_Q_meV": og_per_Q,
               "eps_c_offgrid": eps.tolist(),
               "min_transition_eV": (dmin * RY2EV).tolist()}
        for name, idx in legs.items():
            g = d2(eps[idx]) * RY2EV * 1e3
            gm = d2(dmin[idx]) * RY2EV * 1e3
            rec[name] = {"eps_d2_max_meV": float(g.max()),
                         "eps_d2_mean_meV": float(g.mean()),
                         "mintrans_d2_max_meV": float(gm.max()),
                         "mintrans_d2_mean_meV": float(gm.mean())}
        out["windows"][tag] = rec
        log(f"[window {nv_in}v+{nc_in}c  nb={nb_window}  rank={rank}  "
            f"a_band={a_band}  backend={args.eigh_backend}]  "
            f"{rec['seconds']:.0f}s (galerkin {t_gal:.0f}s, htransform "
            f"{t_ht:.0f}s = {rec['ms_per_q']:.1f} ms/q over {rec['nq']} q)  "
            f"arena {arena_gb:.1f} GB  xla {xla_gb if xla_gb is None else round(xla_gb,1)} GB  "
            f"on-grid |deps| = {rec['ongrid_max_deps_meV']:.3f} meV")
        for name in legs:
            r = rec[name]
            log(f"    {name:9s} eps 2nd-diff max {r['eps_d2_max_meV']:9.2f} "
                f"mean {r['eps_d2_mean_meV']:8.2f} meV | min-transition "
                f"2nd-diff max {r['mintrans_d2_max_meV']:9.2f} mean "
                f"{r['mintrans_d2_mean_meV']:8.2f} meV")
        if rank0:
            with open(args.out + ".json", "w") as fh:
                json.dump(out, fh, indent=1)

    if rank0:
        with open(args.out + ".json", "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"Wrote {args.out}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
