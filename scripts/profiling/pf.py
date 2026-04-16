"""pf.py — LORRAX JAX profiling helper.

A small, drop-in module used alongside (not inside) LORRAX source. The intent
is that an agent can instrument a module entry point with 5-10 lines at the
top of a wrapper script and get a complete picture of:

  1. memory bottlenecks       (per-module HLO "memory-usage-report" + device snapshot)
  2. compute time             (jax.profiler trace + named regions)
  3. sharding / communication (optimized HLO collectives + rematerialization warnings)
  4. compilation time         (JAX_LOG_COMPILES stderr + explain_cache_misses)

Two usage patterns:

  A. Run-level instrumentation (heavy functions, full pipelines):
       from scripts.profiling import pf
       pf.setup_env("artifacts")           # before any jax import
       import jax; ...
       with pf.trace_profile("artifacts"): # wraps jax.profiler.start_trace
           with pf.region("davidson"):     # TraceAnnotation
               run_nscf(...)
       pf.snapshot_memory("artifacts/final.memprof")

  B. Leaf-function inspection (AOT, cheap, surgical):
       @jax.jit
       def kernel(x, y): ...
       pf.aot_report(kernel, x, y, out="artifacts/aot/kernel")

The analysis scripts (analyze_hlo_dump.py, analyze_compile_log.py) then
produce agent-readable summaries of everything written under "artifacts/".

This module contains no LORRAX imports and no mandatory JAX import — it is
safe to import before jax.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────────────────────────────────
#  Environment setup — must run BEFORE jax is imported
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_XLA_FLAGS = [
    # HLO dumping: the "memory-usage-report" + "buffer-assignment" + optimized
    # HLO text are our richest static-analysis sources. Text dumps only — proto
    # and HTML blow up artifact size for no agent-visible gain.
    "--xla_dump_hlo_as_text",
    # Include source-line metadata in dumps so the agent can map an op back
    # to a Python line without opening a trace viewer.
    "--xla_dump_include_timestamp=false",
    # SPMD partitioner: emit readable names and before/after-partitioner
    # dumps so collectives can be traced back to the originating op.
    "--xla_dump_hlo_pass_re=spmd-partitioner|sharding-propagation",
]


_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"


def _maybe_init_jax_distributed() -> None:
    """Same logic as gw.gw_jax — run once, before any jax import triggers the
    XLA backend. Safe on single-process; a no-op when called a second time.
    """
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return
    proc_count = int(
        os.environ.get("JAX_PROCESS_COUNT",
        os.environ.get("JAX_NUM_PROCESSES",
        os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count <= 1:
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return
    import jax  # OK: we are about to initialize distributed right now
    try:
        jax.distributed.initialize()
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return
    except Exception:
        pass
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coord is None:
        import subprocess
        nodelist = os.environ.get("SLURM_NODELIST")
        if nodelist:
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True, text=True, check=True)
                coord = f"{result.stdout.strip().splitlines()[0]}:12355"
            except Exception:
                pass
        if coord is None:
            host = (os.environ.get("SLURMD_NODENAME")
                    or os.environ.get("HOSTNAME") or "localhost")
            coord = f"{host}:12355"
    proc_id = int(os.environ.get("JAX_PROCESS_INDEX",
                                 os.environ.get("SLURM_PROCID", "0")))
    jax.distributed.initialize(coordinator_address=coord,
                               num_processes=proc_count,
                               process_id=proc_id)
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


def setup_env(
    artifacts_dir: str | os.PathLike,
    *,
    hlo: bool = True,
    log_compiles: bool = True,
    explain_cache_misses: bool = True,
    ir_dump: bool = True,
    persistent_cache: bool = False,
    enable_x64: bool = True,
    extra_xla_flags: Optional[list[str]] = None,
) -> Path:
    """Configure environment for a profiled run.

    Call this once at the top of a wrapper script, before importing jax.
    Creates <artifacts_dir>/ with the following layout populated later:

        xla_dump/       HLO text, buffer-assignment, memory-usage-report per module
        jax_ir/         jax-level MLIR IR dump (jax_dump_ir_to)
        compile.log     stderr capture of JAX_LOG_COMPILES
        xprof/          jax.profiler trace (xplane.pb + plugins/)
        aot/            per-function AOT reports (jaxpr, HLO, memory, cost)
        memprof/        device memory pprof snapshots

    Returns the absolute artifacts Path.
    """
    artifacts = Path(artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    for sub in ("xla_dump", "jax_ir", "xprof", "aot", "memprof"):
        (artifacts / sub).mkdir(exist_ok=True)

    # XLA_FLAGS — HLO dump enables memory-usage-report which is the
    # most compact, agent-readable per-module memory summary.
    if hlo:
        flags = list(_DEFAULT_XLA_FLAGS)
        flags.append(f"--xla_dump_to={artifacts/'xla_dump'}")
        if extra_xla_flags:
            flags.extend(extra_xla_flags)
        existing = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (existing + " " + " ".join(flags)).strip()

    # JAX_LOG_COMPILES — one WARNING line per XLA compilation with timings.
    # We also unconditionally enable cache-miss explanations; cheap and
    # essential for catching silent recompiles.
    if log_compiles:
        os.environ["JAX_LOG_COMPILES"] = "1"
    if explain_cache_misses:
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"

    # JAX-level IR dump (MLIR, pre-XLA). Complements XLA HLO by showing
    # jax primitive structure before lowering.
    if ir_dump:
        os.environ["JAX_DUMP_IR_TO"] = str(artifacts / "jax_ir")

    if persistent_cache:
        os.environ.setdefault(
            "JAX_COMPILATION_CACHE_DIR", str(artifacts / "compilation_cache"))

    # LORRAX always runs in float64; any early jax import (e.g. via start_trace)
    # would otherwise lock the config to float32 before the target module's
    # own setdefault runs. Opt out with enable_x64=False for rare cases.
    if enable_x64:
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    # Multi-process bootstrap: if we're running under SLURM with >1 task,
    # initialize jax.distributed here — BEFORE trace_profile or any other
    # code that triggers XLA backend creation. This mirrors the pattern in
    # gw.gw_jax._maybe_init_jax_distributed and ensures every process sees
    # one GPU and participates in collectives.
    _maybe_init_jax_distributed()

    # Save the config for downstream analysis scripts.
    (artifacts / "pf_setup.json").write_text(json.dumps({
        "artifacts_dir": str(artifacts),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "JAX_LOG_COMPILES": os.environ.get("JAX_LOG_COMPILES", ""),
        "JAX_EXPLAIN_CACHE_MISSES": os.environ.get("JAX_EXPLAIN_CACHE_MISSES", ""),
        "JAX_DUMP_IR_TO": os.environ.get("JAX_DUMP_IR_TO", ""),
    }, indent=2))

    return artifacts


def attach_compile_log(path: str | os.PathLike) -> None:
    """Capture jax compile-log warnings to a file.

    JAX emits compile-log lines as Python logging WARNINGs on the jax._src
    loggers. Route them through a file handler so the wrapper script's stderr
    stays clean and the log is persistent.
    """
    import logging as _logging
    fmt = _logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh = _logging.FileHandler(path, mode="w")
    fh.setLevel(_logging.WARNING)
    fh.setFormatter(fmt)
    for name in ("jax._src.dispatch", "jax._src.interpreters.pxla",
                 "jax._src.compiler", "jax._src.cache_key", "jax._src.pjit"):
        lg = _logging.getLogger(name)
        lg.addHandler(fh)
        lg.setLevel(_logging.WARNING)
        # Avoid double-emission on stderr via Python's "lastResort" handler.
        # The FileHandler above keeps the full log; JAX_LOG_COMPILES already
        # routes the important lines to stderr via jax's own print path.
        lg.propagate = False


# ─────────────────────────────────────────────────────────────────────────
#  Profiler trace wrapper
# ─────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def trace_profile(artifacts_dir: str | os.PathLike, *, create_perfetto_link: bool = False):
    """Capture a jax.profiler trace into <artifacts>/xprof/.

    The resulting xplane.pb file is viewable with:
        pip install xprof
        xprof <artifacts>/xprof --port=8791

    On multi-process runs, each process writes its own .xplane.pb with a
    per-host filename. Only rank 0 creates the aggregated perfetto trace
    (otherwise every rank races on the same file).
    """
    import jax, jax.profiler as jp
    outdir = str(Path(artifacts_dir) / "xprof")
    # Perfetto aggregation races across processes (rank 0 reads the xplane
    # while non-zero ranks are still writing). Disable in multi-process mode;
    # each process still emits its own xplane.pb, openable by xprof.
    multi_proc = jax.process_count() > 1
    perfetto = (not multi_proc)
    jp.start_trace(outdir,
                   create_perfetto_link=create_perfetto_link and perfetto,
                   create_perfetto_trace=perfetto)
    try:
        yield outdir
    finally:
        jp.stop_trace()


@contextlib.contextmanager
def region(name: str):
    """Named region in the xprof trace + wall-clock stderr timer.

    Use as a drop-in around any pipeline stage:
        with pf.region("davidson_warmup"):
            warmup_davidson_jit(...)

    On the xprof trace this shows up as a named event under XLAOps > Python.
    The wall-clock timer hits stderr so it is visible in the run log even
    if the trace never gets opened.
    """
    import jax.profiler as jp
    t0 = time.perf_counter()
    print(f"[pf] ▶ {name}", file=sys.stderr, flush=True)
    with jp.TraceAnnotation(name):
        try:
            yield
        finally:
            # block_until_ready on any single device ensures async work
            # completes before we stamp a wall-clock time.
            try:
                import jax
                jax.effects_barrier()
            except Exception:
                pass
            dt = time.perf_counter() - t0
            print(f"[pf] ■ {name}  {dt:.3f}s", file=sys.stderr, flush=True)


def annotate(name: Optional[str] = None):
    """Decorator form of region()."""
    def deco(fn):
        nm = name or fn.__qualname__
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with region(nm):
                return fn(*args, **kwargs)
        return wrapper
    return deco


# ─────────────────────────────────────────────────────────────────────────
#  Device memory snapshot
# ─────────────────────────────────────────────────────────────────────────

def snapshot_memory(path: str | os.PathLike, label: str = "") -> None:
    """Save a pprof-format device memory snapshot.

    Python-side allocations only — jit-compiled internal buffers are opaque
    (use the XLA memory-usage-report for those). Use this to catch leaks
    across pipeline stages and to verify buffer donation.

    View with:
        pprof -text <path>            # terminal summary
        pprof -http=: <path>          # interactive browser view
    """
    import jax.profiler as jp
    jp.save_device_memory_profile(str(path))
    if label:
        print(f"[pf] mem snapshot '{label}' → {path}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────
#  Ahead-of-time report for a single jit-compiled function
# ─────────────────────────────────────────────────────────────────────────

def aot_report(
    fn: Callable,
    *args: Any,
    out: str | os.PathLike,
    save_hlo: bool = True,
    save_stablehlo: bool = True,
    save_jaxpr: bool = True,
    timing_runs: int = 0,
    **kwargs: Any,
) -> dict:
    """Lower + compile a function ahead of time and dump everything
    interesting to disk as compact text files.

    Produces under <out>/:
        jaxpr.txt                 jax.make_jaxpr output (primitive-level IR)
        stablehlo.mlir            pre-XLA MLIR
        optimized_hlo.txt         compiled.as_text() (XLA HLO w/ collectives)
        memory_analysis.txt       input/output/temp bytes per device
        cost_analysis.txt         flops + bytes-accessed estimates
        input_shardings.txt       NamedSharding / PartitionSpec of each arg
        summary.md                one-page agent-readable digest

    If timing_runs > 0, also runs the compiled function that many times and
    writes a timing histogram (first call = includes compile, subsequent = cached).

    Returns a dict with the most-important numbers for programmatic use.
    """
    import jax
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Tracing / lowering / compiling happens without execution.
    if save_jaxpr:
        try:
            (out / "jaxpr.txt").write_text(str(jax.make_jaxpr(fn)(*args, **kwargs)))
        except Exception as e:  # noqa: BLE001
            (out / "jaxpr.txt").write_text(f"# make_jaxpr failed: {e!r}\n")

    jit_fn = fn if getattr(fn, "lower", None) is not None else jax.jit(fn)
    lowered = jit_fn.lower(*args, **kwargs)
    if save_stablehlo:
        try:
            (out / "stablehlo.mlir").write_text(lowered.as_text())
        except Exception as e:  # noqa: BLE001
            (out / "stablehlo.mlir").write_text(f"# lowered.as_text failed: {e!r}\n")

    compiled = lowered.compile()
    mem = compiled.memory_analysis()
    cost = compiled.cost_analysis()

    if save_hlo:
        try:
            (out / "optimized_hlo.txt").write_text(compiled.as_text())
        except Exception as e:  # noqa: BLE001
            (out / "optimized_hlo.txt").write_text(f"# compiled.as_text failed: {e!r}\n")

    # memory_analysis is a dataclass-like struct — keep only numeric fields.
    mem_dict: dict[str, int | float] = {}
    for k in dir(mem):
        if k.startswith("_"):
            continue
        v = getattr(mem, k, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mem_dict[k] = int(v)
    (out / "memory_analysis.txt").write_text(
        "\n".join(f"{k:40s} {v:>20d}" for k, v in mem_dict.items()) + "\n"
    )
    if cost is not None:
        (out / "cost_analysis.txt").write_text(
            "\n".join(f"{k:40s}{v!r}" for k, v in cost.items()) + "\n"
        )

    shardings = []
    try:
        shardings = [str(s) for s in compiled.input_shardings[0]]
    except Exception:  # noqa: BLE001
        pass
    (out / "input_shardings.txt").write_text("\n".join(shardings) + "\n")

    # Timing (optional — needs actual execution).
    timings: list[float] = []
    if timing_runs > 0:
        for _ in range(timing_runs):
            t0 = time.perf_counter()
            r = jit_fn(*args, **kwargs)
            jax.tree_util.tree_map(lambda x: getattr(x, "block_until_ready",
                                                    lambda: None)(), r)
            timings.append(time.perf_counter() - t0)
        (out / "timings.txt").write_text(
            "\n".join(f"{i:3d}  {t*1e3:10.3f} ms" for i, t in enumerate(timings)) + "\n"
        )

    # Summary.md — the single page we want an agent to read.
    def _hb(n: int) -> str:
        if n is None or not isinstance(n, (int, float)):
            return str(n)
        x = float(n)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(x) < 1024.0:
                return f"{x:.2f} {unit}"
            x /= 1024.0
        return f"{x:.2f} PiB"
    mb = _hb
    summary = [
        f"# AOT report: {getattr(fn, '__qualname__', fn)}",
        "",
        "## Memory",
        f"- arguments : {mb(mem.argument_size_in_bytes)}",
        f"- outputs   : {mb(mem.output_size_in_bytes)}",
        f"- temp      : {mb(mem.temp_size_in_bytes)}",
        f"- alias     : {mb(mem.alias_size_in_bytes)}",
        f"- code      : {mb(mem.generated_code_size_in_bytes)}",
        "",
        "## Cost",
    ]
    if cost is not None:
        for k in ("flops", "bytes accessed", "optimal_seconds"):
            if k in cost:
                summary.append(f"- {k}: {cost[k]}")
    if timings:
        import statistics
        summary += [
            "",
            "## Timings (after compile)",
            f"- first call (incl compile): {timings[0]*1e3:.2f} ms",
            f"- median of remaining: {statistics.median(timings[1:])*1e3:.2f} ms"
            if len(timings) > 1 else "",
            f"- min/max: {min(timings)*1e3:.2f} / {max(timings)*1e3:.2f} ms",
        ]
    shard_lines = [f"  {s}" for s in shardings] or ["  (none reported)"]
    summary += ["", "## Input shardings", *shard_lines]
    (out / "summary.md").write_text("\n".join(summary) + "\n")

    return {
        "name": getattr(fn, "__qualname__", str(fn)),
        "out": str(out),
        "memory": mem_dict,
        "cost": dict(cost) if cost is not None else None,
        "timings": timings,
    }
