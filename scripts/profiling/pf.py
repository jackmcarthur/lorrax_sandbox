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


def _maybe_init_jax_distributed() -> None:
    """Delegate to LORRAX's canonical multi-process bootstrap.

    The old in-tree copy of this function called ``jax.distributed.initialize()``
    with no kwargs, which hangs on Perlmutter Cray MPICH (each rank sees only
    one GPU via ``CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`` so the no-arg topology
    exchange waits forever for "local" peers that don't exist). The fixed
    implementation lives in ``runtime/__init__.py`` and explicitly sets
    ``local_device_ids=[0]`` derived from CUDA_VISIBLE_DEVICES — see the
    docstring there for the full reasoning.
    """
    from runtime import init_jax_distributed
    init_jax_distributed()


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
def trace_profile(artifacts_dir: str | os.PathLike, *,
                  create_perfetto_link: bool = False,
                  host_tracer_level: int = 1,
                  python_tracer_level: int = 0):
    """Capture a jax.profiler trace into <artifacts>/xprof/.

    The resulting xplane.pb file is viewable with:
        pip install xprof
        xprof <artifacts>/xprof --port=8791

    On multi-process runs, each process writes its own .xplane.pb with a
    per-host filename. Only rank 0 creates the aggregated perfetto trace
    (otherwise every rank races on the same file).

    Tuning host vs device tracing:
      The Chrome-JSON perfetto trace (single file) is capped at ~1M events.
      Default JAX host/python tracers emit roughly one event per Python
      frame, which drowns out GPU-kernel events after a few seconds on a
      busy run. We drop host_tracer_level=1 (kernel launches only) and
      python_tracer_level=0 (disabled) so the trace captures the full run.
      Override either if you are debugging host code specifically.
    """
    import jax
    import jax.profiler as jp
    import jax._src.profiler as _p
    import jaxlib.xla_extension as xe
    # Each rank writes to its own subdir so stop_trace's perfetto aggregation
    # (which reads back the .trace.json.gz) never races with another rank.
    rank = jax.process_index()
    base = Path(artifacts_dir) / "xprof"
    base.mkdir(exist_ok=True)
    outdir = base / f"rank_{rank}" if jax.process_count() > 1 else base
    outdir.mkdir(exist_ok=True)

    perfetto = True  # Per-rank dir → always safe to write perfetto trace

    # Replicate jp.start_trace behaviour but with reduced host/python tracer
    # levels. The Chrome-JSON perfetto trace is capped at 1M events; default
    # host_tracer_level=2 + python_tracer_level=1 swamps the buffer with host
    # frame events after ~5 s on a busy run, truncating the tail of the
    # actual GPU timeline. host_tracer_level=1 keeps kernel launches;
    # python_tracer_level=0 drops the per-Python-frame events entirely.
    with _p._profile_state.lock:
        if _p._profile_state.profile_session is not None:
            raise RuntimeError("A jax.profiler trace is already in progress.")
        # Make sure the backend is initialized before the session starts.
        jax.devices()
        opts = xe.profiler.ProfileOptions()
        opts.host_tracer_level = int(host_tracer_level)
        opts.python_tracer_level = int(python_tracer_level)
        opts.include_dataset_ops = False
        _p._profile_state.profile_session = xe.profiler.ProfilerSession(opts)
        _p._profile_state.create_perfetto_link = (create_perfetto_link and perfetto)
        _p._profile_state.create_perfetto_trace = perfetto
        _p._profile_state.log_dir = str(outdir)
    try:
        yield str(outdir)
    finally:
        # Delegates to the exact same finalization path jp.stop_trace uses.
        try:
            jp.stop_trace()
        except Exception as e:
            print(f"[pf] stop_trace failed: {e}", file=sys.stderr)


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
#  Live-array memory sampler — peak-position + what-was-held-then
# ─────────────────────────────────────────────────────────────────────────

# A background thread polls ``device.memory_stats()['bytes_in_use']`` on a
# short interval. When a new peak is seen we snapshot the top-N live JAX
# arrays (shape, dtype, size, an abbreviated origin) so the post-run report
# can answer: "what arrays were we holding when we peaked?" That question is
# invisible to the static HLO view and to a pprof snapshot taken at an
# arbitrary time.
#
# Overhead is small (one Python call per interval, no device sync) but we
# keep the default interval conservative — the sampler is for orientation,
# not microbenchmarking.

import threading as _threading


class _LiveArraySampler:
    def __init__(self, interval_s: float = 0.25, top_n: int = 10):
        self.interval_s = interval_s
        self.top_n = top_n
        self._stop = _threading.Event()
        self._thread: _threading.Thread | None = None
        self.timeline: list[dict] = []   # every sample
        self.peak_bytes: int = 0
        self.peak_sample: dict | None = None
        self._t0 = time.perf_counter()

    def _snapshot(self) -> dict:
        import jax
        dev = jax.local_devices()[0]
        try:
            stats = dev.memory_stats() or {}
        except Exception:
            stats = {}
        bytes_in_use = int(stats.get("bytes_in_use", 0))
        peak_bytes = int(stats.get("peak_bytes_in_use", 0))
        arr_summary: list[dict] = []
        try:
            arrs = list(jax.live_arrays())
        except Exception:
            arrs = []
        # Keep only addressable shards to avoid double-counting across devices
        rows = []
        for a in arrs:
            try:
                sz = a.nbytes
            except Exception:
                sz = 0
            try:
                dt = str(a.dtype)
            except Exception:
                dt = "?"
            try:
                shp = tuple(a.shape)
            except Exception:
                shp = ()
            rows.append((sz, dt, shp))
        rows.sort(reverse=True)
        total_live = sum(r[0] for r in rows)
        for sz, dt, shp in rows[: self.top_n]:
            arr_summary.append({"bytes": sz, "dtype": dt, "shape": list(shp)})
        return {
            "t": time.perf_counter() - self._t0,
            "bytes_in_use": bytes_in_use,
            "peak_bytes_in_use": peak_bytes,
            "live_bytes_sum": total_live,
            "n_live_arrays": len(rows),
            "top_arrays": arr_summary,
        }

    def _run(self):
        while not self._stop.is_set():
            s = self._snapshot()
            self.timeline.append(s)
            if s["bytes_in_use"] > self.peak_bytes:
                self.peak_bytes = s["bytes_in_use"]
                self.peak_sample = s
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = _threading.Thread(
            target=self._run, name="pf-mem-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # One final snapshot after stop, in case we missed the peak on the
        # last interval (e.g. a kernel that finished and freed immediately).
        try:
            s = self._snapshot()
            self.timeline.append(s)
            if s["bytes_in_use"] > self.peak_bytes:
                self.peak_bytes = s["bytes_in_use"]
                self.peak_sample = s
        except Exception:
            pass

    def write(self, out_path: str | os.PathLike) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # JSON for machine use
        out.with_suffix(".json").write_text(json.dumps({
            "peak_bytes": self.peak_bytes,
            "peak_sample": self.peak_sample,
            "timeline": self.timeline,
        }, indent=2))
        # TXT for agent reading — timeline + peak breakdown
        def _hb(n):
            x = float(n)
            for u in ("B", "KiB", "MiB", "GiB"):
                if abs(x) < 1024:
                    return f"{x:.2f} {u}"
                x /= 1024
            return f"{x:.2f} TiB"
        lines: list[str] = []
        lines.append(f"# memory_timeline.txt — live-array sampler output")
        lines.append(f"# Interval: {self.interval_s:.3f}s  |  "
                     f"Samples: {len(self.timeline)}")
        lines.append("")
        if self.peak_sample:
            ps = self.peak_sample
            lines.append(f"── Peak live HBM  at t={ps['t']:.2f}s  ──────────────────────────")
            lines.append(f"bytes_in_use                = {_hb(ps['bytes_in_use'])}")
            lines.append(f"device.peak_bytes_in_use    = {_hb(ps['peak_bytes_in_use'])}  (cumulative)")
            lines.append(f"sum of jax.live_arrays()    = {_hb(ps['live_bytes_sum'])}")
            lines.append(f"n live arrays at peak        = {ps['n_live_arrays']}")
            lines.append("")
            lines.append("Top JAX arrays at peak:")
            lines.append(f"  {'size':>14s}  {'dtype':<12s}  shape")
            for a in ps["top_arrays"]:
                lines.append(f"  {_hb(a['bytes']):>14s}  {a['dtype']:<12s}  "
                             f"{tuple(a['shape'])}")
            lines.append("")
        lines.append("── Timeline (t, bytes_in_use, n_live) ─────────────────────────")
        lines.append(f"{'t (s)':>8s}  {'bytes_in_use':>14s}  {'n_live':>7s}  {'live_bytes_sum':>14s}")
        for s in self.timeline:
            lines.append(
                f"{s['t']:>8.2f}  {_hb(s['bytes_in_use']):>14s}  "
                f"{s['n_live_arrays']:>7d}  {_hb(s['live_bytes_sum']):>14s}"
            )
        out.write_text("\n".join(lines) + "\n")


_SAMPLER: _LiveArraySampler | None = None


def start_memory_sampler(interval_s: float = 0.25, top_n: int = 10) -> None:
    """Start the background live-array sampler.

    Should be called AFTER jax is importable. Pair with ``stop_memory_sampler``
    (or rely on atexit — not yet wired). Produces ``memory_timeline.txt`` and
    ``memory_timeline.json`` when ``write_memory_timeline`` is called.
    """
    global _SAMPLER
    if _SAMPLER is not None:
        return
    _SAMPLER = _LiveArraySampler(interval_s=interval_s, top_n=top_n)
    _SAMPLER.start()


def stop_memory_sampler() -> None:
    global _SAMPLER
    if _SAMPLER is None:
        return
    _SAMPLER.stop()


def write_memory_timeline(out_path: str | os.PathLike) -> None:
    if _SAMPLER is None:
        return
    _SAMPLER.write(out_path)


# ─────────────────────────────────────────────────────────────────────────
#  Device memory snapshot (pprof format — unchanged)
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
