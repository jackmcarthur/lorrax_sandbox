# Cookbook

Concrete recipes, each self-contained. Pattern: "wide-scope run first, then
narrow". Assumes `module load lorrax` and `SLURM_JOBID` exported (see the
main sandbox `AGENTS.md` for `lxalloc` setup).

## Recipe 1 — cold start on an unfamiliar module

```bash
# From the target run directory
cd runs/<system>/<run>/<lorrax_variant>
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out profile -m gw.gw_jax -i cohsex.in

cd /pscratch/sd/j/jackm/lorrax_sandbox
python3 scripts/profiling/analyze_hlo_dump.py     <run>/profile
python3 scripts/profiling/analyze_compile_log.py  <run>/profile
```

Read order (≈1 min total):

1. `<run>/profile/hlo_summary.md`  — top of each section (Memory, Compute,
   Sharding, Retrace groups).
2. `<run>/profile/compile_summary.md` — Wall-clock totals + Cache misses.
3. Pick the single highest-ranking issue across the four categories.
4. *Only now* open the per-category doc (memory.md / compute_time.md /
   sharding.md / compilation.md) for the drill-in.

## Recipe 2 — single-GPU compile-time sweep

Fast iteration while changing one line of a function to see if a retrace
count drops. Skips xprof and HLO to make each iteration <30 s.

```bash
LORRAX_NGPU=1 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out quick --no-trace --no-hlo -m psp.run_nscf -i nscf.in
python3 scripts/profiling/analyze_compile_log.py quick
```

## Recipe 3 — multi-GPU sharding audit

```bash
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out profile4 -m gw.gw_jax -i cohsex.in
python3 scripts/profiling/analyze_hlo_dump.py profile4 --top 30
```

The Sharding + Rematerialization sections of `hlo_summary.md` are the
first things to read. Any Involuntary full rematerialization warning
should lead every follow-up investigation.

## Recipe 4 — A/B comparison of two code paths

Same run directory, different source branches:

```bash
# main
git -C sources/lorrax checkout main
LORRAX_NGPU=4 lxrun python3 ... --out profile_main ...

# candidate
git -C sources/lorrax checkout agent/my-refactor
LORRAX_NGPU=4 lxrun python3 ... --out profile_candidate ...

diff <(head -40 profile_main/hlo_summary.md) \
     <(head -40 profile_candidate/hlo_summary.md)
diff <(head -40 profile_main/compile_summary.md) \
     <(head -40 profile_candidate/compile_summary.md)
```

The diffs are usually small and pinpoint exactly what changed (a new
collective, a lost fusion, a retrace group).

## Recipe 5 — memory-leak hunt

Driver that takes periodic snapshots:

```python
# leak_probe.py
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf; pf.setup_env("leak_probe", hlo=False, log_compiles=False)

from gw.gw_jax import main_phase  # hypothetical API

pf.snapshot_memory("leak_probe/memprof/00_start.prof")
main_phase("isdf");  pf.snapshot_memory("leak_probe/memprof/10_isdf.prof")
main_phase("chi0");  pf.snapshot_memory("leak_probe/memprof/20_chi0.prof")
main_phase("sigma"); pf.snapshot_memory("leak_probe/memprof/30_sigma.prof")
```

Diff consecutive snapshots to see which stage accumulates memory:

```bash
pprof -diff_base leak_probe/memprof/10_isdf.prof leak_probe/memprof/20_chi0.prof
```

## Recipe 6 — drilling into one named function

AFTER the ranked summaries have identified a bottleneck:

```python
# probe_<fn>.py
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf; pf.setup_env("probe", hlo=False, log_compiles=False)

from gw.w_isdf import compute_chi0_q  # the function the summaries pointed at
from gw.gw_init import load_inputs
args = load_inputs("cohsex.in")
pf.aot_report(compute_chi0_q, *args,
              out="probe/aot/compute_chi0_q", timing_runs=3)
```

```bash
LORRAX_NGPU=4 lxrun python3 -u probe_compute_chi0_q.py
cat probe/aot/compute_chi0_q/summary.md
```

See `aot_reports.md` for the API.

## Recipe 7 — drop-in instrumentation without using run_profiled.py

When you already have a driver script and want to add profiling without
switching launchers:

```python
# At the very top, BEFORE any jax import
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("my_artifacts")
pf.attach_compile_log("my_artifacts/compile.log")

# ... rest of script ...
import jax
from gw.gw_jax import main

with pf.trace_profile("my_artifacts"):
    with pf.region("chi0"):
        compute_chi0(...)
    with pf.region("W"):
        solve_w(...)

pf.snapshot_memory("my_artifacts/memprof/end.prof")
```

Then run the analyzers on `my_artifacts/` as usual. No LORRAX source edits
required.

## Recipe 8 — persistent cache for fast iteration

When changing one file and comparing two runs:

```bash
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out run1 --persistent-cache -m psp.run_nscf -i nscf.in
# (edit a source file)
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out run2 --persistent-cache -m psp.run_nscf -i nscf.in
```

`run1/compilation_cache/` is populated after the first run; `run2` hits it
for most modules. `run2/compile_summary.md` should show near-zero XLA
compile time — anything that didn't cache is a lead for further work (see
`compilation.md`).

## Recipe 9 — ending a profiling session

Every session produces a report under `reports/profiling_<module>_<date>/`
with:

1. A two-sentence summary of the module profiled and the size of the run.
2. The top-3 bottlenecks across the four categories, each with:
   - a one-line symptom from the summaries
   - the file:line pointed at
   - a proposed next step
3. Links (relative paths) to `hlo_summary.md`, `compile_summary.md`,
   and the xprof trace.
4. Any code changes made under `sources/lorrax/` with commit hash.

See `reports/profiling_run_nscf_2026-04-16/` for an example.
