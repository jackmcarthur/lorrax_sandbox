# JAX profiling stack — initial build + Phase 2 (trace parser, detail txts, live-array sampler)

**Branch (LORRAX):** `agent/run-nscf-kpar`
**Branch (sandbox):** `main`
**Validation system:** Si 2×2×2, 60 Ry, 12 bands
**Date:** 2026-04-16

## Summary

Two-phase build of `skills/profiling_stack/` + `scripts/profiling/`. The
stack now answers, from one command, "where is memory going, where is time
going, is communication overlapping, what retraced":

| Category | Ranked view | Dense detail txt |
|---|---|---|
| Memory | `hlo_summary.md` Memory table | `memory_details.txt` |
| Sharding / collectives | `hlo_summary.md` Sharding table + source attribution | `collectives_details.txt` (HLO context per collective) |
| Rematerialization | `hlo_summary.md` Remat table | `remat_details.txt` (5-line HLO context each) |
| Retrace groups | `hlo_summary.md` Retrace table | `retrace_details.txt` (input signatures per module) |
| Compilation | `compile_summary.md` | (compile log in `compile.log`) |
| GPU kernel timing | `trace_summary.md` Top kernels | `trace_details.txt` |
| **Async H2D/D2H overlap** | `trace_summary.md` Overlap table | — |
| **Bandwidth saturation** | `trace_summary.md` Peak window table | — |
| **Peak-memory position + top live arrays** | `memory_timeline.txt` | — |

All detail txts are agent-readable plaintext with ~30-line sections per
issue — signal-dense, no graphs, no gzip, no raw HLO dumps.

## Deliverables

### Scripts (in `scripts/profiling/`)

| File | Purpose |
|---|---|
| `pf.py` | Library: `setup_env`, `trace_profile`, `region`, `annotate`, `aot_report`, `attach_compile_log`, **`start/stop_memory_sampler`**, **`write_memory_timeline`**, `snapshot_memory`. Handles jax.distributed bootstrap + multi-process tracing via per-rank xprof subdirs. |
| `run_profiled.py` | One-shot launcher with `--mem-sample-interval`, `--no-trace`, `--no-hlo`, `--persistent-cache`, etc. |
| `analyze_hlo_dump.py` | XLA dump → `hlo_summary.{md,json}` **plus** the four detail txts. Sharding / Remat / Memory rows all carry the source_file:line from HLO metadata when available. |
| `analyze_compile_log.py` | JAX compile log → `compile_summary.{md,json}`. |
| `analyze_trace.py` | **NEW.** Perfetto trace → `trace_summary.{md,json}` + `trace_details.txt`. Extracts top GPU kernels, H2D/D2H transfer totals, per-direction async-overlap fraction, sliding-window peak bandwidth, low-occupancy kernel list. |

### Phase 1 deliverables retained from the previous report

- `skills/profiling_stack/` — 7 documents (SKILL.md + memory / compute_time /
  sharding / compilation / aot_reports / cookbook).
- `sources/lorrax/src/psp/run_nscf.py` refactored for k-point parallelism
  via `jax.distributed` + `multihost_utils.process_allgather` on branch
  `agent/run-nscf-kpar` (`4617f6e`).

### Phase 2 — what was added this session

1. **Per-rank xprof subdirectories.** Multi-process traces now land in
   `<profile>/xprof/rank_<n>/` instead of racing on a single dir.
2. **`python_tracer_level=0` + `host_tracer_level=1`** in `pf.trace_profile`.
   The Chrome-JSON perfetto trace is capped at ~1M events; previously the
   default tracers flooded it with Python frames and truncated the GPU
   timeline after ~5 s. With reduced host tracing, a 25 s Si NSCF run now
   captures 23 k GPU events instead of 84.
3. **Live-array sampler** in `pf.py`. Background thread polls
   `device.memory_stats()['bytes_in_use']` + `jax.live_arrays()`; writes
   `memory_timeline.txt` with peak timestamp + top-N JAX arrays at the
   peak — answers "what was I holding when I peaked?".
4. **Source-line attribution** on every collective + remat row in
   `hlo_summary.md`. Parsed from HLO `metadata={source_file=… source_line=…}`.
5. **Four detail txts**: `memory_details.txt`, `collectives_details.txt`,
   `remat_details.txt`, `retrace_details.txt`. Each focuses on one kind of
   issue with exactly the context needed to act — no filler.
6. **Trimmed** `hlo_summary.md`: removed the Aggregate-op-counts table
   (fusion: 625, reshape: 28 — not actionable) and the Per-module file
   index (docs already say where to look). Added companion-file link block
   near the top.

## Validation — Si 2×2×2 / 60 Ry / 12 bands

### 1-GPU profile

```
[analyze_hlo_dump] 215 modules, peak-sum=869.94 MiB, 0 collectives, 0 remat
[analyze_compile_log] traces=510 compiles=215 cache-misses=163
[analyze_trace] duration 22.60s, 3673 GPU events, 999 H2D, 2650 D2H
```

**Top finding, memory side:**
`module_0225.jit__apply_H_sparse` holds 214 MiB with a 205 MiB
preallocated-temp; only 3 MiB is user-held input/output. Retrace group
shows `_apply_H_sparse` has 4 different module ids, one per Davidson basis
size (12, 24, 36, 48 bands × nspinor × ngkmax=2120).

**Top finding, compile side:** 163 cache misses, 14.6 s XLA compile time
out of 44 s wall. `jit_multiply` alone retraced **31 times** — its
`retrace_details.txt` block shows 31 distinct (dtype, shape) signatures,
including `c128[68,2120]` vs `c128[68,2109]` vs `c128[68,2100]` (ngk
varies per k-point). Padding `ngk` to `max(ngk)` would collapse all 31
into 1.

**Top finding, live-array sampler:**
```
Peak live HBM  at t=14.58s  (during Davidson k-loop)
bytes_in_use             = 33.94 MiB      (JAX-visible)
device.peak_bytes_in_use = 220.35 MiB     (XLA including temps)

Top arrays at peak:
  3.11 MiB  complex128  (48, 2, 2120)
  3.11 MiB  complex128  (48, 2, 2120)
  2.20 MiB  complex128  (68, 2120)
```

The 220 MiB XLA peak ≫ 34 MiB JAX-visible peak confirms the dominant
memory consumer is the temp buffer inside a fused kernel, not a user
array. Agent immediately knows the fix lives inside the Davidson kernel
fusion, not in anything we allocate.

### 4-GPU (k-parallel) profile

```
[analyze_hlo_dump] 216 modules, peak-sum=232.67 MiB, 4 collectives, 0 remat
[analyze_compile_log] traces=522 compiles=219 cache-misses=159
[analyze_trace] duration 21.29s, 23202 GPU events, 606 H2D, 683 D2H
```

**Collectives extracted with HLO context (from `collectives_details.txt`):**
```
[module_0335.jit__identity_fn]  op=all-gather-start  bytes=31.05 MiB
  ENTRY %main.4_spmd (param.1: c128[1,8,12,2,2120]) -> c128[4,8,12,2,2120] {
    %param.1 = c128[1,8,12,2,2120] parameter(0), sharding={devices=[4,1,1,1,1]<=[4]}
>>  %all-gather-start = ... all-gather-start(%param.1),
      channel_id=1, replica_groups=[1,4]<=[4], dimensions={0},
      backend_config={"collective_backend_config":{"is_sync":true, ...}}
```

Reads as: "4-way synchronous all-gather on dim 0 of a
`(1, 8, 12, 2, 2120)` tensor, 31 MiB output." This is the
`process_allgather` at the end of the Davidson k-loop, exactly as
expected.

**Async overlap (trace_summary.md):**
```
| Direction | Count | Total time | Exposed | Overlap frac |
| H2D       | 606   | 17.89 ms   | 17.89 ms|     0.000   |
| D2H       | 683   |  2.32 ms   |  2.24 ms|     0.032   |
```

H2D copies total 17.9 ms — all at startup, no compute to overlap with, so
the 0 overlap is harmless. D2H is 2.3 ms, tiny. Neither direction is a
bottleneck for run_nscf; this view will be much more interesting on
gw.gw_jax (where async D2H from collectives legitimately happens during
Davidson-like inner loops).

**Top GPU kernels:**
```
| Op                | Count | Total ms | Occ % | Source |
| custom-call.4.0   | 9680  | 67.30   | 100   | jit(_ritz_and_residuals)/jit(main)/jit(eigh)/eigh |
| all-gather-start  |    2  |  9.16   |   0   | _identity_fn |
| custom-call.5.0   |  816  |  8.52   | 100   | jit(_ritz_and_residuals)/jit(inv)/lu |
| fft.2.0           |  545  |  8.33   |  94   | jit(_apply_H_sparse)/jit(apply_H_k)/jit(fft)/fft |
```

The eigh solver (cuSOLVER `syevd`) dominates — 9680 calls at 67 ms total.
This is the Davidson inner-eigenvalue solve at every iteration.
Optimisation lead: batch the 8 k-points into one eigh call (saving most
of the per-call kernel-launch overhead, since each call is ~7 µs).

### Bit-identical output

Both runs produce `WFN.h5` with eigenvalue maxabs diff 0.0 and coefs
maxabs diff 0.0 vs each other (re-verified this session).

## Phase 3 candidates — what the user flagged next

1. **Host ↔ device transfer detection in `gw.gw_jax`.** Now tractable —
   `analyze_trace.py` already reports overlap_frac and exposed time. Next
   step: run the stack on gw.gw_jax where async D2H is expected to be
   heavy, and see whether the pattern is "saturation-bound" (PCIe
   limited) or "schedule-bound" (copy dispatched too late).
2. **Baseline comparison mode.** `analyze_hlo_dump.py --diff-against old/profile`
   would diff two summaries, highlight regressions. Deferred.
3. **Per-region memory deltas.** `pf.region(...)` could capture
   bytes_in_use at enter/exit — instantly pinpoints which stage grows the
   buffer pool. Deferred.
4. **Merging trace views across ranks** would give a cluster-wide
   view instead of just rank 0. Low priority since rank 0's trace already
   shows all collectives (the per-rank events are symmetric on a
   well-balanced workload).

## Files touched this session

- `scripts/profiling/pf.py` — per-rank xprof subdirs, live-array sampler,
  custom `ProfileOptions` wiring for reduced host tracing.
- `scripts/profiling/run_profiled.py` — `--mem-sample-interval`.
- `scripts/profiling/analyze_hlo_dump.py` — trimmed noise, source-line
  attribution, emits 4 detail txts.
- `scripts/profiling/analyze_trace.py` — NEW.
- `sources/lorrax/src/psp/run_nscf.py` — k-parallel (Phase 1).
- `reports/profiling_stack_2026-04-16/report.md` — this file.
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/{00_davidson_only,02_davidson_4gpu}/profile/` — regenerated artifacts.
