# JAX profiling stack — initial build and validation

**Branch (LORRAX):** `agent/run-nscf-kpar`
**Branch (sandbox):** `main` (uncommitted at time of writing)
**Validation system:** Si 2×2×2, 60 Ry, 12 bands
**Date:** 2026-04-16

## Summary

First-pass build of `skills/profiling_stack/` and the backing
`scripts/profiling/` helpers. The stack covers all four requested categories
from a single run:

| Category | Ranked source |
|---|---|
| Memory | `hlo_summary.md` → Memory table + `memprof/*.prof` |
| Compute time | `hlo_summary.md` → Compute/Custom calls + xprof trace |
| Sharding / communication | `hlo_summary.md` → Sharding + Rematerialization |
| Compilation | `compile_summary.md` → Retrace groups + Cache misses |

Design principle (set after user feedback): **wide-scope-first**. The top
of each ranked table names the Python source file or jit function name
responsible, so an agent with no prior familiarity with the module can go
from `run_profiled.py --out profile -m <module>` to "this file:line is the
bottleneck" in under two minutes of reading.

## Deliverables

### New scripts

| Path | Purpose |
|---|---|
| `scripts/profiling/pf.py` | Helper library: setup_env, trace_profile, region, annotate, snapshot_memory, aot_report, attach_compile_log |
| `scripts/profiling/run_profiled.py` | One-shot launcher wrapping `python -m <module>` with full env setup |
| `scripts/profiling/analyze_hlo_dump.py` | XLA dump → ranked `hlo_summary.{md,json}` (Memory, Compute, Sharding, Rematerialization, Retrace groups) |
| `scripts/profiling/analyze_compile_log.py` | JAX compile log → ranked `compile_summary.{md,json}` (wall-clock totals, cache misses by source location, persistent cache misses) |

### New skill docs

| Path | Topic |
|---|---|
| `skills/profiling_stack/SKILL.md` | Entry point — the 30-second workflow + decision tree |
| `skills/profiling_stack/memory.md` | Drilling into the Memory table |
| `skills/profiling_stack/compute_time.md` | Drilling into op inventory + xprof |
| `skills/profiling_stack/sharding.md` | Drilling into collectives + rematerialization |
| `skills/profiling_stack/compilation.md` | Drilling into retrace groups + cache misses |
| `skills/profiling_stack/aot_reports.md` | Per-function AOT (secondary tool — only after identifying a target) |
| `skills/profiling_stack/cookbook.md` | Recipe book — 9 concrete flows |

### LORRAX source changes (on `agent/run-nscf-kpar`)

`sources/lorrax/src/psp/run_nscf.py`:
  * Added `_maybe_init_jax_distributed()` (same pattern as `gw.gw_jax`).
  * Strided the Davidson k-loop over ranks (`ik % n_proc == rank`).
  * Added `process_allgather` of evals and packed coefficients at loop end.
  * Only rank 0 opens the `WFNWriter` and writes `WFN.h5`.
  * Guarded verbose prints so non-zero ranks stay silent.

This makes `psp.run_nscf` multi-process-correct without changing its
single-process behaviour.

## Validation

### Single-GPU smoke (Si 2×2×2, 12 bands)

```
[rank 0] k=  0/8: 1.060s  evals[0]=-0.418717 Ry
Davidson: 7.91s (0.988s/k, 1 rank)
Total NSCF: 28.6s
Finished in 50.6s.
```

Analyzer output:

```
[analyze_hlo_dump]    433 modules, peak-sum=1.20 GiB, 0 collectives, 0 remat warnings
[analyze_compile_log] traces=510 compiles=215 cache-misses=163
```

What the stack immediately revealed without any prior knowledge:

| Finding | Evidence |
|---|---|
| Memory peak: `jit__apply_H_sparse` at 214 MiB with 7 compiled copies (shape polymorphism) | `hlo_summary.md` Memory table |
| 33 % of wallclock spent in XLA compile (14.6 s / 44 s) | `compile_summary.md` Wall-clock totals |
| `jit_multiply` compiled 58 times; `jit_broadcast_in_dim` 45 times | `hlo_summary.md` Retrace groups |
| `_lu_solve` retraced 16 times at `solvers/davidson.py:41:12` | `compile_summary.md` Cache misses |

### 4-GPU (k-parallel) run

```
[rank 0] k=  0/8: 1.052s  evals[0]=-0.418717 Ry
Davidson: 6.99s (0.873s/k, 4 ranks)
Total NSCF: 24.9s
```

WFN.h5 diff vs 1-GPU:

```
eigenvalue maxabs diff (mRy): 0.0
coefs maxabs diff:            0.0
```

Bit-identical output across 1-GPU and 4-GPU runs.

Analyzer now populates the Sharding section:

```
| Module                         | Op                | Output bytes | Output type |
| module_0335.jit__identity_fn   | all-gather-start  | 31.05 MiB    | c128[1,8,12,2,2120] |
| module_0421.jit__identity_fn   | all-gather-start  | 31.05 MiB    | c128[1,8,12,2,2120] |
| module_0333.jit__identity_fn   | all-gather-start  |  3.75 KiB    | f64[1,8,12] |
| module_0419.jit__identity_fn   | all-gather-start  |  3.75 KiB    | f64[1,8,12] |
```

These four collectives are exactly the expected process_allgather calls
for evals (f64) and packed coefficients (c128) — the stack correctly
identifies them as cross-process data movement, with byte sizes that
match the expected payload.

### Pytest

`uv run python -m pytest -q` → **14 passed, 1 warning in 22.95s**.

## Bugs caught during implementation

1. **JAX_ENABLE_X64 was latched too late** when `run_profiled.py` imported
   jax before runpy ran the target module. Fix: `pf.setup_env` now sets
   `JAX_ENABLE_X64=1` before any jax import. Otherwise NSCF eigenvalues
   came out NaN.
2. **jax.distributed had to be initialized before the XLA backend.** The
   first 4-GPU run failed with `RuntimeError: jax.distributed.initialize()
   must be called before any JAX calls that might initialise the XLA
   backend.` Fix: `pf.setup_env` calls `_maybe_init_jax_distributed()`
   before the trace starts.
3. **Perfetto aggregation raced across ranks.** stop_trace on multi-process
   runs tried to read other ranks' xplane.pb before they were flushed,
   producing `gzip.BadGzipFile`. Fix: disable `create_perfetto_trace` on
   multi-process runs — per-process `xplane.pb` files still work with xprof.
4. **Initial `aot_report` summary crashed on bytes-typed struct fields.**
   Fix: filter `memory_analysis` dict to numeric fields only.

## Top-3 bottlenecks found (from the very first profile)

Order intentionally matches the user's priority: memory ≥ compute ≥ sharding ≥ compile.

1. **Memory — `_apply_H_sparse` retrace set holds 200+ MiB temps.** Seven
   compiled copies, each with a large preallocated-temp. Fix likely lies
   in collapsing the retrace (see #3) — shape polymorphism is the root.
2. **Compute — `[pf]` region wall-clock was not reported in this run.**
   Next step: add `pf.region("davidson_warmup")`, `pf.region("davidson_kloop")`
   inside `run_nscf.py` to surface per-stage timings. (Not done in this
   session to keep the code change focused on k-parallelism.)
3. **Compilation — 163 cache misses, 14.6 s XLA compile.** Top offenders
   are elementwise ops recompiled per k-point (`jit_multiply` x58,
   `jit_broadcast_in_dim` x45). Root cause: Davidson per-k is not
   encapsulated in a single outer jit. Fix: wrap the k-loop body in one
   jit with padded shapes, or move to `lax.scan`.

## Known follow-ups

* **Communication-heavy smoke test**: `run_nscf` is embarrassingly
  k-parallel (only `process_allgather` at the end), so the sharding
  section is sparse. A real test of the Sharding + Rematerialization
  view wants `gw.gw_jax` on a multi-GPU run. Waiting on user direction
  on which target module to profile next.
* **Automatic xprof installation** to the Shifter container would let
  agents run `xprof` directly without uploading .pb files — not
  essential for the skill, but would close a loop.
* **Persistent cache validation**: the `--persistent-cache` flag is
  plumbed but not validated on this run. A two-run A/B would confirm.

## Files

- `scripts/profiling/pf.py`
- `scripts/profiling/run_profiled.py`
- `scripts/profiling/analyze_hlo_dump.py`
- `scripts/profiling/analyze_compile_log.py`
- `skills/profiling_stack/*.md`
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/{00_davidson_only,02_davidson_4gpu}/profile/`
- Source branch: `sources/lorrax/agent/run-nscf-kpar`
