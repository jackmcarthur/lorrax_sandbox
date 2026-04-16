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
5. Pick the single highest-ranking issue → open the category doc
   (`memory.md` / `compute_time.md` / `sharding.md` / `compilation.md`).
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

## Common launcher variants

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
    memory.md              # drill: Memory table → fix
    compute_time.md        # drill: kernel ranking + overlap → fix
    sharding.md            # drill: collectives + rematerialization → fix
    compilation.md         # drill: retrace groups + cache misses → fix
    aot_reports.md         # secondary tool: one-function AOT probe
```

## A/B comparison — when refactoring

```bash
git -C sources/lorrax checkout main          && lxrun ... --out profile_main ...
git -C sources/lorrax checkout agent/my-fix  && lxrun ... --out profile_fix  ...

diff <(head -40 profile_main/hlo_summary.md)    <(head -40 profile_fix/hlo_summary.md)
diff <(head -40 profile_main/compile_summary.md) <(head -40 profile_fix/compile_summary.md)
diff <(head -40 profile_main/trace_summary.md)   <(head -40 profile_fix/trace_summary.md)
```

Small diffs; they pinpoint exactly what changed (new collective, lost
fusion, new retrace group, peak-bandwidth regression).

## Rules

1. **Always `lxrun`** inside a Shifter container — never bare `python3`.
2. **Read the four summaries before opening LORRAX source.** That's the
   whole point.
3. **End every profiling session with a report** under
   `reports/profiling_<module>_<date>/report.md` linking the summaries +
   top-3 bottlenecks + proposed next step.
4. **Don't turn on `--xla_dump_hlo_pass_re=.*`** (40× artifact blowup) unless
   you're specifically hunting a compiler pass.

## Escape hatches

| Question summaries can't answer | Tool |
|---|---|
| Per-op GPU kernel timings in an interactive UI | `xprof profile/xprof --port=8791`, or upload `perfetto_trace.json.gz` to ui.perfetto.dev |
| Memory timeline across a fori_loop | xprof Memory Profile tab |
| Which Python line allocated a live buffer | `pprof -tree profile/memprof/end.prof` |
| Does my sharding constraint survive a reshape | open the specific module's `sm_*_gpu_after_optimizations.txt` |
| SPMD-partitioner decisions | add `--xla_dump_hlo_pass_re=spmd-partitioner\|sharding-propagation` for that run only |
