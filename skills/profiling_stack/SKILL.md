# Profiling Stack

Goes from "unfamiliar LORRAX module" → "ranked list of bottlenecks across
memory, compute, sharding, compilation" with **zero edits to the target
module**. One launcher runs the module, three analyzers produce markdown
summaries, each row points at a source_file:line or jit name.

## The run in 5 lines

```bash
cd <my_run_dir>                             # anywhere with the module's input files
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out profile -m gw.gw_jax -i cohsex.in
cd /pscratch/sd/j/jackm/lorrax_sandbox
python3 scripts/profiling/analyze_hlo_dump.py     <my_run_dir>/profile
python3 scripts/profiling/analyze_compile_log.py  <my_run_dir>/profile
python3 scripts/profiling/analyze_trace.py        <my_run_dir>/profile
```

The target module is never modified. The launcher delegates to
`runpy.run_module` after setting env vars and starting a background memory
sampler in a sidecar thread.

## Artifacts — what lands under `<my_run_dir>/profile/`

| File | Ranked view of | Read first? |
|---|---|---|
| `hlo_summary.md` | biggest HBM modules, collectives by bytes, rematerialization, retrace counts | ✓ |
| `compile_summary.md` | slowest XLA compiles, tracing cache misses with source locations | ✓ |
| `trace_summary.md` | top GPU kernels, H2D/D2H totals, **async-overlap fraction**, peak-window bandwidth, low-occupancy kernels | ✓ |
| `memory_timeline.txt` | peak live HBM timestamp + the top JAX arrays held at that moment | ✓ |
| `memory_details.txt` | top-N modules' full `memory-usage-report.txt` concatenated | drill-in |
| `collectives_details.txt` | each top collective with ±3 lines HLO context + source:line | drill-in |
| `remat_details.txt` | every remat warning with 5 lines HLO context + source:line | drill-in |
| `retrace_details.txt` | per retraced jit name, one ENTRY-signature per module instance | drill-in |
| `trace_details.txt` | densest per-event dump of top copies + top kernels | drill-in |
| `memprof/end.prof` | pprof-format live buffer snapshot with Python tracebacks | `pprof -tree` |
| `xprof/plugins/profile/…/xplane.pb` | raw trace for xprof / Perfetto UI | UI only |
| `xla_dump/module_XXXX.*` | per-jit HLO, buffer assignment, thunk sequence | when a detail txt points here |
| `compile.log` | full captured `JAX_LOG_COMPILES` output | grep |

## Cold-start read order — ~2 minutes, cold, no prior familiarity

1. `hlo_summary.md` top-to-bottom — Memory, Sharding, Remat, Retrace.
2. `compile_summary.md` — Wall-clock totals + top cache misses.
3. `trace_summary.md` — top kernels + overlap + bandwidth.
4. `memory_timeline.txt` — peak timestamp + top arrays at peak.
5. Pick the single highest-ranking issue → open `drilldowns.md` at the
   matching § (Memory / Compute time / Sharding / Compilation).
6. *Only now* open LORRAX source at the file:line the summary pointed to.

If you find yourself opening LORRAX source before step 5, stop — you are
prematurely narrowing.

## Zero source edits — what's on by default vs opt-in

**On by default, no edits needed anywhere:**
  * HLO dump (`XLA_FLAGS=--xla_dump_to=…`)
  * `JAX_LOG_COMPILES` + cache-miss explanations
  * `JAX_ENABLE_X64=1` (set before jax import so `os.environ.setdefault` in
    the target module stays consistent)
  * jax.distributed bootstrap on multi-process runs
  * jax.profiler trace (per-rank xprof subdirectories, reduced host/python
    tracer levels so the 1M-event perfetto cap doesn't truncate long runs)
  * Background live-array sampler (default interval 0.25 s, disable with
    `--mem-sample-interval 0`)
  * End-of-run pprof snapshot

**Opt-in (edit the target module only if you want these):**
  * `pf.region("name"):` blocks — named bars in xprof + `[pf] ■ name Xs`
    stderr timings. One-liner per region.
  * `@pf.annotate("name")` decorator form — same, for whole functions.
  * `pf.aot_report(fn, *args, out=…)` in a **separate probe script** —
    per-function AOT report with jaxpr / StableHLO / optimized HLO /
    memory / cost. Target source stays untouched. See `aot_reports.md`.

## Launcher variants

```bash
# Single-GPU (k-parallel off)
LORRAX_NGPU=1 lxrun python3 -u .../run_profiled.py --out p -m psp.run_nscf -i nscf.in

# Compile-time sweep (fastest iteration, skip HLO + trace)
LORRAX_NGPU=1 lxrun python3 -u .../run_profiled.py --out p --no-trace --no-hlo \
    -m psp.run_nscf -i nscf.in

# Persistent compile cache (second run re-uses first)
LORRAX_NGPU=4 lxrun python3 -u .../run_profiled.py --out p --persistent-cache \
    -m gw.gw_jax -i cohsex.in

# Disable memory sampler (truly zero overhead)
LORRAX_NGPU=4 lxrun python3 -u .../run_profiled.py --out p --mem-sample-interval 0 \
    -m gw.gw_jax -i cohsex.in
```

Full option list: `scripts/profiling/run_profiled.py --help`.

## A/B comparison — when refactoring

```bash
git -C sources/lorrax checkout main          && lxrun ... --out profile_main ...
git -C sources/lorrax checkout agent/my-fix  && lxrun ... --out profile_fix  ...

diff <(head -40 profile_main/hlo_summary.md)    <(head -40 profile_fix/hlo_summary.md)
diff <(head -40 profile_main/compile_summary.md) <(head -40 profile_fix/compile_summary.md)
diff <(head -40 profile_main/trace_summary.md)   <(head -40 profile_fix/trace_summary.md)
```

Small diffs pinpoint exactly what changed (new collective, lost fusion,
new retrace group, peak-bandwidth regression).

## Memory-leak hunt across pipeline stages

When peak HBM grows across stages rather than being dominated by one
module:

```python
# leak_probe.py — separate file, target module unchanged
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf; pf.setup_env("leak_probe", hlo=False, log_compiles=False)

from gw.gw_jax import main_phase          # or any staged API the module exposes

pf.snapshot_memory("leak_probe/memprof/00_start.prof")
main_phase("isdf");  pf.snapshot_memory("leak_probe/memprof/10_isdf.prof")
main_phase("chi0");  pf.snapshot_memory("leak_probe/memprof/20_chi0.prof")
main_phase("sigma"); pf.snapshot_memory("leak_probe/memprof/30_sigma.prof")
```

```bash
LORRAX_NGPU=4 lxrun python3 -u leak_probe.py
pprof -diff_base leak_probe/memprof/10_isdf.prof leak_probe/memprof/20_chi0.prof
```

The diff shows what was held after stage 2 that wasn't after stage 1 —
usually a buffer not donated or not deleted.

## Drop-in instrumentation without using `run_profiled.py`

When you already have a hand-rolled driver:

```python
# at the top, BEFORE any jax import
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("my_artifacts")
pf.attach_compile_log("my_artifacts/compile.log")

# ... rest of script ...
import jax
from gw.gw_jax import main

pf.start_memory_sampler(interval_s=0.25)
with pf.trace_profile("my_artifacts"):
    with pf.region("chi0"):
        chi = compute_chi0(...)
    with pf.region("W"):
        W = solve_w(...)
pf.stop_memory_sampler()
pf.write_memory_timeline("my_artifacts/memory_timeline.txt")
pf.snapshot_memory("my_artifacts/memprof/end.prof")
```

Run the three analyzers on `my_artifacts/` exactly as with the launcher.

## Layout

```
scripts/profiling/
    pf.py                  # helpers (imported by launcher; usable standalone too)
    run_profiled.py        # the launcher you invoke
    analyze_hlo_dump.py    # → hlo_summary.md + 4 detail txts
    analyze_compile_log.py # → compile_summary.md
    analyze_trace.py       # → trace_summary.md + trace_details.txt

skills/profiling_stack/
    SKILL.md               # this file, the cold entry point
    drilldowns.md          # Memory / Compute / Sharding / Compilation interpretations + fixes
    aot_reports.md         # secondary tool: one-function AOT probe
```

## Rules

1. **Always `lxrun`** inside a Shifter container — never bare `python3`.
2. **Read the four summaries before opening LORRAX source.** That's the
   whole point.
3. **End every profiling session with a report** under
   `reports/profiling_<module>_<date>/report.md` linking the summaries,
   top-3 bottlenecks with `source_file:line` + proposed next steps, and
   any LORRAX source changes with commit hash + branch.
4. **Don't turn on `--xla_dump_hlo_pass_re=.*`** (40× artifact blowup) unless
   you're specifically hunting a compiler pass.
