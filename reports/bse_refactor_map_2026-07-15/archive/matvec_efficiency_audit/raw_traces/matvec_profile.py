"""BSE matvec efficiency profiler (TRACER dossier harness).

Builds SYNTHETIC in-device data (no GW run) mirroring
``bse_ring_comm.ring_matvec_smoke_test`` and drives one of the three matvecs
under study:

  kind = ring  -> build_bse_ring_matvec      (legacy TDA, ppermute-ring encode)
  kind = stack -> build_bse_stack_matvec     (production TDA, scan-in-shard_map)
  kind = full  -> build_bse_ring_matvec_full (non-TDA S=[[A,B],[-B,-A]])

Two regimes:
  fixture  : MoS2 gnppm gate size  (nc=2, nv=2, ns=2, nk=9,  mu=400)   latency
  inflated : compute/bandwidth     (nc=48,nv=48,ns=2, nk=16, mu=800)   compute

usage:
  matvec_profile.py <kind> <regime> <px> <py> [--ntrials N] [--trace DIR]
                    [--hlo DIR] [--mem] [--perterm] [--reps R]
"""
import argparse, os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

SRC = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bse.bse_ring_comm import (
    make_bse_shardings, build_bse_ring_matvec, build_bse_ring_matvec_full)
from bse.bse_stack_matvec import build_bse_stack_matvec

REGIMES = {
    "fixture":  dict(nc=2,  nv=2,  ns=2, nkx=3, nky=3, nkz=1, mu=400),
    "inflated": dict(nc=48, nv=48, ns=2, nkx=4, nky=4, nkz=1, mu=800),
}


def build_synth(regime, px, py, n_trials, seed=0):
    p = REGIMES[regime]
    nc, nv, ns = p["nc"], p["nv"], p["ns"]
    nkx, nky, nkz = p["nkx"], p["nky"], p["nkz"]
    mu = p["mu"]
    nk = nkx * nky * nkz
    assert nc % px == 0 and nv % py == 0, (nc, nv, px, py)
    assert mu % px == 0 and mu % py == 0, (mu, px, py)

    rng = np.random.default_rng(seed)
    def cplx(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    psi_c = cplx((nk, nc, ns, mu))
    psi_v = cplx((nk, nv, ns, mu))
    eps_c = rng.uniform(0.1, 0.5, (nk, nc)).astype(np.float64)
    eps_v = rng.uniform(-0.5, -0.1, (nk, nv)).astype(np.float64)
    W_R = (cplx((mu, mu, nkx, nky, nkz)) * 0.01)
    V_q0 = cplx((mu, mu)) * 0.05
    X = cplx((n_trials, nc, nv, nk))
    return dict(psi_c=psi_c, psi_v=psi_v, eps_c=eps_c, eps_v=eps_v,
                W_R=W_R, V_q0=V_q0, X=X, nkx=nkx, nky=nky, nkz=nkz, nk=nk,
                nc=nc, nv=nv, ns=ns, mu=mu, n_trials=n_trials)


def place(d, sh, mesh, full=False):
    """Return the 9-tuple of matvec args on-device with the right shardings."""
    def put(a, s):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, s))
    X = d["X"]
    if full:
        # X_full = [X, Y], shape (2, n_trials, nc, nv, nk), sharded sh.X_full
        Xf = np.stack([d["X"], d["X"][::-1] if d["X"].shape[0] > 1 else d["X"]], axis=0)
        X_arg = put(Xf, P(None, None, "x", "y", None))
    else:
        X_arg = put(X, P(None, "x", "y", None))
    psi_c_X = put(d["psi_c"], P(None, None, None, "x"))
    psi_c_Y = put(d["psi_c"], P(None, None, None, "y"))
    psi_v_X = put(d["psi_v"], P(None, None, None, "x"))
    psi_v_Y = put(d["psi_v"], P(None, None, None, "y"))
    eps_c = put(d["eps_c"], P(None, None))
    eps_v = put(d["eps_v"], P(None, None))
    W_R = put(d["W_R"], P("x", "y", None, None, None))
    V_q0 = put(d["V_q0"], P("x", "y"))
    return (X_arg, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)


def build_matvec(kind, mesh, d):
    if kind == "ring":
        return build_bse_ring_matvec(mesh, d["nkx"], d["nky"], d["nkz"])
    if kind == "stack":
        return build_bse_stack_matvec(mesh, d["nkx"], d["nky"], d["nkz"], kernel="bse")
    if kind == "full":
        return build_bse_ring_matvec_full(mesh, d["nkx"], d["nky"], d["nkz"])
    raise ValueError(kind)


def timeit(fn, reps=20, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(min=float(ts.min()), med=float(np.median(ts)), mean=float(ts.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ring", "stack", "full"])
    ap.add_argument("regime", choices=["fixture", "inflated"])
    ap.add_argument("px", type=int)
    ap.add_argument("py", type=int)
    ap.add_argument("--ntrials", type=int, default=1)
    ap.add_argument("--trace", default=None, help="xprof logdir")
    ap.add_argument("--hlo", default=None, help="write compiled HLO text here")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--perterm", action="store_true")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    ndev = args.px * args.py
    devs = jax.devices()
    assert len(devs) >= ndev, f"need {ndev} devices, have {len(devs)}"
    mesh = Mesh(np.array(devs[:ndev]).reshape(args.px, args.py), axis_names=("x", "y"))

    d = build_synth(args.regime, args.px, args.py, args.ntrials)
    full = args.kind == "full"
    argt = place(d, None, mesh, full=full)
    mv = build_matvec(args.kind, mesh, d)

    tag = f"{args.kind}/{args.regime}/{args.px}x{args.py}/nt{args.ntrials}"
    p = REGIMES[args.regime]
    print(f"[cfg] {tag}  nc={p['nc']} nv={p['nv']} ns={p['ns']} nk={d['nk']} "
          f"mu={p['mu']}  ndev={ndev}", flush=True)

    # ---- AOT compile: HLO text + memory analysis ----
    compiled = mv.lower(*argt).compile()
    if args.hlo:
        os.makedirs(os.path.dirname(args.hlo), exist_ok=True)
        with open(args.hlo, "w") as f:
            f.write(compiled.as_text())
        print(f"[hlo] wrote {args.hlo}  ({os.path.getsize(args.hlo)} bytes)", flush=True)
    if args.mem:
        m = compiled.memory_analysis()
        info = dict(temp=int(m.temp_size_in_bytes),
                    argument=int(m.argument_size_in_bytes),
                    output=int(m.output_size_in_bytes),
                    alias=int(m.alias_size_in_bytes))
        info["peak_per_rank_MB"] = (info["temp"] + info["argument"]
                                    + info["output"] - info["alias"]) / 1e6
        info["temp_MB"] = info["temp"] / 1e6
        print(f"[mem] {json.dumps(info)}", flush=True)
        try:
            c = compiled.cost_analysis()
            flops = c.get("flops") if isinstance(c, dict) else None
            byts = c.get("bytes accessed") if isinstance(c, dict) else None
            print(f"[cost] flops={flops} bytes_accessed={byts}", flush=True)
        except Exception as e:
            print(f"[cost] unavailable: {e}", flush=True)

    # ---- warm timing ----
    def call():
        return mv(*argt)
    t = timeit(call, reps=args.reps)
    print(f"[warm] min={1e3*t['min']:.4f} ms  med={1e3*t['med']:.4f} ms  "
          f"mean={1e3*t['mean']:.4f} ms  (reps={args.reps})", flush=True)

    # ---- per-term timing (ring/full only: use timed=True builder) ----
    if args.perterm and args.kind in ("ring", "full"):
        import common.timing as timing
        if args.kind == "ring":
            mvt = build_bse_ring_matvec(mesh, d["nkx"], d["nky"], d["nkz"], timed=True)
        else:
            mvt = build_bse_ring_matvec_full(mesh, d["nkx"], d["nky"], d["nkz"], timed=True)
        timing.reset() if hasattr(timing, "reset") else None
        for _ in range(args.reps):
            jax.block_until_ready(mvt(*argt))
        try:
            timing.report()
        except Exception as e:
            print(f"[perterm] timing.report failed: {e}", flush=True)

    # ---- xprof trace ----
    if args.trace:
        os.makedirs(args.trace, exist_ok=True)
        jax.block_until_ready(call())  # ensure compiled
        with jax.profiler.trace(args.trace):
            for _ in range(10):
                jax.block_until_ready(call())
        print(f"[trace] wrote {args.trace}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
