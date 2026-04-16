#!/usr/bin/env python3
"""run_profiled.py — one-shot wrapper that runs a Python module under the
full LORRAX profiling stack.

Delegates to ``python -m <module> [args...]`` after wiring up:

    XLA_FLAGS=... --xla_dump_to=<artifacts>/xla_dump --xla_dump_hlo_as_text
    JAX_LOG_COMPILES=1
    JAX_EXPLAIN_CACHE_MISSES=1
    JAX_DUMP_IR_TO=<artifacts>/jax_ir

And wraps the run in a jax.profiler trace + periodic device memory snapshots
(opt-in via --mem-snapshots N).

Example (inside the Shifter container)::

    lxrun python3 -u scripts/profiling/run_profiled.py \\
        --out runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/profile \\
        -m psp.run_nscf -i nscf.in

After the run finishes, run the analyzers::

    python3 scripts/profiling/analyze_hlo_dump.py <out>
    python3 scripts/profiling/analyze_compile_log.py <out>

Notes
-----
This wrapper is kept deliberately thin — it only calls ``runpy.run_module``
and handles env/trace setup. If the target module's main() already uses
``argparse`` it will see our stripped argv (everything after the first ``--``
or after ``-m <module>``).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _split_target_args(raw_argv: list[str]) -> tuple[str, list[str], list[str]]:
    """Split our own argv vs the target's. Canonical form:

        run_profiled.py [--opts] -m <module> [<target args…>]

    Everything after ``-m <module>`` is passed through to the module.
    Returns (module_name, target_args, our_args).
    """
    if "-m" not in raw_argv:
        raise SystemExit("run_profiled.py: missing '-m <module>'")
    idx = raw_argv.index("-m")
    if idx + 1 >= len(raw_argv):
        raise SystemExit("run_profiled.py: '-m' requires a module name")
    return raw_argv[idx + 1], raw_argv[idx + 2:], raw_argv[:idx]


def main() -> int:
    # We have to parse our own args BEFORE importing jax so env vars land.
    raw = sys.argv[1:]
    module, target_args, our_args = _split_target_args(raw)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="Artifacts directory (created if needed)")
    ap.add_argument("--no-hlo", action="store_true",
                    help="Skip XLA HLO dump (saves disk on huge runs)")
    ap.add_argument("--no-log-compiles", action="store_true",
                    help="Disable JAX_LOG_COMPILES stderr capture")
    ap.add_argument("--no-trace", action="store_true",
                    help="Skip jax.profiler trace capture (xprof)")
    ap.add_argument("--mem-snapshot-at-start", action="store_true",
                    help="Save a pprof memory snapshot after module import")
    ap.add_argument("--mem-sample-interval", type=float, default=0.25,
                    help="Live-array sampler interval in seconds "
                         "(set 0 to disable). Output: memory_timeline.{txt,json}")
    ap.add_argument("--persistent-cache", action="store_true",
                    help="Use <out>/compilation_cache as persistent cache dir")
    ap.add_argument("--extra-xla-flags", default="",
                    help="Extra tokens appended to XLA_FLAGS")
    args = ap.parse_args(our_args)

    # Make scripts/profiling importable as a plain module name
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    import pf  # noqa: E402  (intentional — must come BEFORE jax import)

    artifacts = pf.setup_env(
        args.out,
        hlo=not args.no_hlo,
        log_compiles=not args.no_log_compiles,
        persistent_cache=args.persistent_cache,
        extra_xla_flags=[args.extra_xla_flags] if args.extra_xla_flags else None,
    )

    # Attach file handler for the compile log (stderr already carries it
    # via JAX_LOG_COMPILES, but tail -f of the run is easier with a file).
    if not args.no_log_compiles:
        pf.attach_compile_log(artifacts / "compile.log")

    # Fabricate the argv the target module expects.
    sys.argv = [module, *target_args]

    # Wrap the whole module run in a single trace + outer region so the agent
    # sees one top-level block in the xprof timeline.
    import contextlib as _c
    import time as _t

    if args.no_trace:
        trace_ctx = _c.nullcontext()
    else:
        trace_ctx = pf.trace_profile(artifacts)

    if args.mem_snapshot_at_start:
        try:
            pf.snapshot_memory(artifacts / "memprof" / "start.prof", label="start")
        except Exception as e:
            print(f"[run_profiled] start snapshot failed: {e}", file=sys.stderr)

    if args.mem_sample_interval > 0:
        pf.start_memory_sampler(interval_s=args.mem_sample_interval)

    t0 = _t.perf_counter()
    with trace_ctx:
        with pf.region(f"run_module:{module}"):
            try:
                runpy.run_module(module, run_name="__main__", alter_sys=True)
            except SystemExit as se:
                rc = int(se.code) if isinstance(se.code, int) else 0
                if rc != 0:
                    raise

    # Stop the sampler BEFORE the pprof snapshot so we get a clean peak
    if args.mem_sample_interval > 0:
        pf.stop_memory_sampler()
        pf.write_memory_timeline(artifacts / "memory_timeline.txt")

    # Always try a final memory snapshot
    try:
        pf.snapshot_memory(artifacts / "memprof" / "end.prof", label="end")
    except Exception as e:
        print(f"[run_profiled] end snapshot failed: {e}", file=sys.stderr)

    dt = _t.perf_counter() - t0
    print(f"[run_profiled] {module} finished in {dt:.2f}s. Artifacts: {artifacts}",
          file=sys.stderr)
    print("[run_profiled] Next: run", file=sys.stderr)
    print(f"  python3 scripts/profiling/analyze_hlo_dump.py {artifacts}",
          file=sys.stderr)
    print(f"  python3 scripts/profiling/analyze_compile_log.py {artifacts}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
