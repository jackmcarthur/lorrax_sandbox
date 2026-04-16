# Profiling Stack

**Purpose:** start from a large, unfamiliar LORRAX module and end with a
ranked list of concrete bottlenecks. The stack is designed so an agent who
has never opened the source can run one command, read two markdown
summaries, and know exactly which files/functions are responsible for the
top ~10 memory/compute/sharding/compile-time costs.

Drilling into a specific function is a second step you only take after the
summaries point there. Start wide; narrow down only when data justifies it.

## The 30-second workflow

```bash
# 1. Run the module end-to-end under the profiler.
cd <my_run_dir>
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out profile \
    -m gw.gw_jax -i cohsex.in

# 2. Produce two ranked-list markdown files.
cd /pscratch/sd/j/jackm/lorrax_sandbox
python3 scripts/profiling/analyze_hlo_dump.py <my_run_dir>/profile
python3 scripts/profiling/analyze_compile_log.py <my_run_dir>/profile
```

After this you have, in `<my_run_dir>/profile/`:

| File | What it tells you (all ranked top-first) |
|------|-------------------------------------------|
| `hlo_summary.md` | **Biggest HBM modules** (→ memory bottleneck), **compute op inventory**, **collectives ranked by bytes**, **rematerialization warnings**, **retrace groups** (→ compile bloat) |
| `compile_summary.md` | **Slowest XLA compiles**, per-function total compile time, **tracing cache misses with source locations** |
| `xprof/…/xplane.pb` | **Per-op GPU timeline**, openable with xprof / Perfetto |
| `memprof/end.prof` | **Live HBM allocations** at end of run (pprof format) |

**Every ranked table ends at the source file/line or module name that caused
the entry**, so the agent's next action is to Read that file, not scan the
whole codebase. That is the point of the stack.

## The four categories — covered in the same two reports

| Category | Where in the reports | What ranking means |
|---|---|---|
| **Memory** | `hlo_summary.md` → Memory table | Module with the biggest peak HBM + name of the largest allocation inside it |
| **Compute time** | `hlo_summary.md` → Compute + Custom calls; `xprof` for timeline | Aggregate op counts; cuBLAS/cuDNN/cuFFT usage; per-op timings in xprof |
| **Sharding** | `hlo_summary.md` → Sharding + Rematerialization | Collectives ranked by output bytes; remat warnings name the source file/line |
| **Compilation** | `compile_summary.md` + `hlo_summary.md` → Retrace groups | Which jit names got recompiled the most; which source lines emitted cache misses and why |

Read order when you have no prior knowledge of the module:

1. `hlo_summary.md` Memory table → the worst module by peak HBM
2. `hlo_summary.md` Retrace groups → any jit name compiled >5 times is a
   compile-cost bomb
3. `compile_summary.md` Cache misses → the source lines responsible
4. `hlo_summary.md` Sharding → only meaningful on multi-GPU; shows data
   movement cost
5. Drill down into one specific module (see the category docs)

## What the ranking looks like in practice

Example `hlo_summary.md` Memory table from Si 2×2×2 NSCF:

```
| Module                               | Peak HBM | Top allocation |
| module_0225.jit__apply_H_sparse      | 214 MiB  | 205 MiB — preallocated-temp: |
| module_0223.jit__apply_H_sparse      | 161 MiB  | 154 MiB — preallocated-temp: |
| module_0221.jit__apply_H_sparse      | 110 MiB  | 104 MiB — preallocated-temp: |
| module_0087.jit__ritz_and_residuals  |  22 MiB  |  13 MiB — preallocated-temp: |
| module_0065.jit_compute_V_H_and_V_xc |  14 MiB  |  11 MiB — preallocated-temp: |
```

Without ever opening the source, the agent now knows:
  * `_apply_H_sparse` is the dominant memory consumer, and it exists as
    seven separate compiled copies (shape polymorphism)
  * `_ritz_and_residuals` and `compute_V_H_and_V_xc` come next
  * 205/214 MiB is a preallocated-temp — not any named buffer — which
    suggests a fusion-group working set, not a user-held array

The follow-up actions are obvious:

  1. Open `sources/lorrax/src/psp/h_dft.py` around `_apply_H_sparse`.
  2. Look at `profile/xla_dump/module_0225.jit__apply_H_sparse.sm_*_gpu_after_optimizations-memory-usage-report.txt`
     for the specific allocations inside the largest version.
  3. Check `profile/xla_dump/module_0225.jit__apply_H_sparse.sm_*_gpu_after_optimizations-buffer-assignment.txt`
     for the live-range that owns that preallocated-temp.

None of that required any prior familiarity with the module structure.

## Layout

```
scripts/profiling/
    pf.py                    # drop-in helper: setup_env, region, trace_profile, aot_report
    run_profiled.py          # one-shot launcher around `python -m <module>`
    analyze_hlo_dump.py      # post-hoc XLA dump → hlo_summary.md
    analyze_compile_log.py   # post-hoc JAX compile log → compile_summary.md

skills/profiling_stack/
    SKILL.md                 # this file
    memory.md                # drilling into the Memory table
    compute_time.md          # drilling into timing / op inventory
    sharding.md              # drilling into collectives + remat
    compilation.md           # drilling into retrace groups + cache misses
    aot_reports.md           # per-function AOT inspection (secondary tool)
    cookbook.md              # concrete recipes
```

## Decision tree for a cold start

```
You just received a LORRAX module to profile and know nothing about it.

  1. Run the module under run_profiled.py (30s setup, then run).
  2. Read hlo_summary.md top-to-bottom once.        ← 30 seconds
     - Memory table →  which modules are memory-heavy?
     - Retrace groups → which functions recompile?
     - Sharding →     which collectives dominate?
  3. Read compile_summary.md top-to-bottom once.    ← 20 seconds
     - Which jit names burn compile time?
     - Which source lines trigger cache misses?
  4. Pick ONE bottleneck (highest-ranking entry in the worst category).
  5. Only NOW open the category-specific doc to drill in.
     - memory.md / compute_time.md / sharding.md / compilation.md
  6. Only NOW open LORRAX source for the offending function.
```

If you find yourself opening LORRAX source before step 4, stop — you are
prematurely narrowing. The stack exists precisely so that step 4 is
evidence-driven.

## Non-negotiables

1. **Always run inside the Shifter container via `lxrun`**, never bare
   `python3`.
2. **Always read the two summaries first, before drilling into source.**
   That's what they exist for.
3. **Commit a short report** under `reports/profiling_<module>_<date>/report.md`
   when you finish a session. Link the summaries and list the top-3
   bottlenecks with next steps.
4. **Don't turn on `--xla_dump_hlo_pass_re=.*`** unless you're hunting a
   specific compiler pass — that multiplies artifact size by ~40×.

## Per-function AOT inspection — the secondary tool

When you already know which function to look at (usually after step 4
above), `pf.aot_report(fn, *args)` gives you the full per-function picture:
jaxpr, StableHLO, optimized HLO, peak memory, FLOPs, optional timing runs.
See `aot_reports.md`. Use it for A/B comparison of alternative
implementations of the same kernel, or for cost-sizing a refactor before
running a full pipeline.

## When the summaries aren't enough

| Question the summaries can't answer | Escape hatch |
|---|---|
| Per-op GPU kernel timings | open the xprof trace in Perfetto / xprof CLI |
| Memory timeline across a fori_loop | xprof Memory Profile tab |
| A custom-call's internal algorithm | grep the optimized HLO for the module |
| Live buffer tracebacks | `pprof -tree memprof/end.prof` |

Each category doc enumerates these escape hatches in its own "When the
summary isn't enough" section. But nearly every real LORRAX performance
task is solvable from the two markdown summaries alone — that is the
primary claim of this stack.
