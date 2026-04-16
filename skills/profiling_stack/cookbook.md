# Cookbook — recipes not in SKILL.md

`SKILL.md` covers the default flow + common launcher variants (compile-time
sweep, persistent cache, A/B comparison, mem-sampler off). This file holds
a few special-case recipes.

## Recipe A — memory-leak hunt across pipeline stages

When peak HBM grows across stages rather than being dominated by one
module. Run a probe script that inserts pprof snapshots at stage
boundaries:

```python
# leak_probe.py
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

## Recipe B — drop-in instrumentation without using `run_profiled.py`

When you already have a hand-rolled driver and don't want to switch to the
launcher.

```python
# At the top, BEFORE any jax import
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("my_artifacts")
pf.attach_compile_log("my_artifacts/compile.log")

# ... rest of your script ...
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

## Recipe C — ending a profiling session

Every profiling session closes with a short report under
`reports/profiling_<module>_<date>/report.md` containing:

1. Two-sentence summary of the module + run size.
2. Top-3 bottlenecks across the four categories, each with:
   - symptom (one line from a summary artifact)
   - `source_file:line` pointed to
   - proposed next step
3. Relative-path links to `hlo_summary.md`, `compile_summary.md`,
   `trace_summary.md`, `memory_timeline.txt`, xprof dir.
4. Any LORRAX source changes with commit hash + branch.

See `reports/profiling_stack_2026-04-16/report.md` for an example.
