# Memory bottlenecks

Follow this doc AFTER reading the top of `hlo_summary.md` — you should
already know which modules rank highest by peak HBM. Here we turn that
ranking into actionable diagnosis.

## What the ranked summary told you already

`hlo_summary.md` → `## Memory — largest modules by peak HBM` lists every
compiled module sorted by the XLA compiler's own "Total bytes used" from
the after-optimizations memory-usage-report. Each row also shows the
largest single allocation inside that module with the first line of XLA's
description. The agent sees one of three patterns:

| Pattern in the top row | Interpretation | What to do next |
|---|---|---|
| One module dominates, its top alloc is `preallocated-temp` | A fused kernel has a big scratch working set | drill into its `buffer-assignment.txt` and `optimized HLO` — look for big reshape/copy/fft intermediates |
| One module dominates, its top alloc is a named `parameter` or `output` | A user array is too big for the shard/layout | look at the function's source — probably a missing with_sharding_constraint |
| Many modules with nearly identical names all near the top | Shape-polymorphism retrace — each signature gets its own XLA plan, each with its own peak | fix at retrace level (compile.md) first; memory reductions usually follow |

Everything below tells you how to go from those identifications to a fix.

## Drill-in #1 — open the top module's three sibling files

For the worst module row, pick up its module id (e.g. `module_0225.jit__apply_H_sparse`)
and open three files in order:

```
profile/xla_dump/module_0225.jit__apply_H_sparse.sm_*_gpu_after_optimizations-memory-usage-report.txt
profile/xla_dump/module_0225.jit__apply_H_sparse.sm_*_gpu_after_optimizations-buffer-assignment.txt
profile/xla_dump/module_0225.jit__apply_H_sparse.sm_*_gpu_after_optimizations.txt
```

**memory-usage-report.txt** — the most compact view. Every allocation with
its cumulative share:

```
Total bytes used: 224494040 (214.09MiB)
Allocations sorted by size:
cumulative_size; total_size - cumulative_size; allocation
   205.03MiB( 95%);    9.07MiB; allocation 0: size 205.03MiB, preallocated-temp:
   208.90MiB( 97%);    5.19MiB; allocation 1: size 3.88MiB, parameter 0, shape |c64[12,2,36,36,36]|
```

**buffer-assignment.txt** — exhaustive live ranges; search for the
allocation's byte offset when you need to know which op's output lives
there.

**optimized HLO .txt** — grep for `copy(` or `reshape(` that produce the
same shape as the big preallocated-temp. Those are almost always the
fusion-breaker. Also grep for `all-gather` / `reduce-scatter` which hint
at resharding memory cost (cross-reference with `sharding.md`).

## Drill-in #2 — live allocation pprof snapshot

`profile/memprof/end.prof` is a pprof-format snapshot at run end of every
live JAX-visible buffer, tagged with the Python stack that allocated it.
Complements the HLO view for long-lived (not kernel-temp) buffers:

```bash
pprof -text   profile/memprof/end.prof    # ranked list in terminal
pprof -tree   profile/memprof/end.prof    # with call stacks
pprof -http=: profile/memprof/end.prof    # interactive browser
```

For leak hunting insert extra snapshots via `pf.snapshot_memory(...)` at
specific points (e.g. after each main pipeline stage) and diff with
`pprof -diff_base`.

Limit: **jit-compiled internal allocations are opaque to pprof.** For those,
the HLO memory-usage-report is the only source — that's exactly why
`hlo_summary.md` leads with it.

## Drill-in #3 — AOT on the offending function

Once the ranked table has identified the bad function, you can probe it in
isolation without running the whole pipeline:

```python
from psp.h_dft import setup_H_k_from_kvec, make_apply_H
H_k = setup_H_k_from_kvec(...)               # realistic shapes matter
apply_H = make_apply_H(H_k)
psi = jnp.zeros((48, 2, 2120), dtype=jnp.complex128)
pf.aot_report(apply_H, psi, out="probe/aot/apply_H_sparse", timing_runs=0)
```

Read `probe/aot/apply_H_sparse/summary.md` → `Memory` block. Change one
thing (chunk size, remove a copy, add a `jax.lax.remat`), re-run — compare
the bytes. See `aot_reports.md`.

## Drill-in #4 — xprof memory timeline

When the peak HBM is fine but the run still OOMs, the order matters: two
heavy arrays that don't overlap in time can co-exist; two that overlap
cannot. Open the xprof trace:

```
xprof profile/xprof --port=8791
```

and navigate to the Memory Profile tab. Look for where the HBM curve
peaks and which op is running at that moment.

## Common diagnoses and their fixes

| Symptom in the Memory table | Root cause | Fix |
|---|---|---|
| Single module has `temp` >> input + output | broken fusion across a reshape or resharding | insert with_sharding_constraint, or `jax.lax.optimization_barrier` to split fusion |
| Dozens of near-duplicate rows for same jit name | shape polymorphism retrace | see `compilation.md`; collapse retraces first |
| Peak rises across pprof snapshots | buffer not donated / deleted | use `donate_argnums=` on the jit, or explicit `jax.block_until_ready(...)` + `del x` |
| HLO shows copy right before a collective | involuntary reshard | see `sharding.md` |

## When the summary isn't enough

| Question | Tool |
|---|---|
| "Which op's output lives at offset X?" | grep `buffer-assignment.txt` for the offset |
| "Does the peak happen during op foo?" | xprof Memory Profile tab |
| "Which Python line allocated this live buffer?" | `pprof -tree end.prof` |
| "Would chunking halve peak?" | `aot_reports.md` + comparison driver |
