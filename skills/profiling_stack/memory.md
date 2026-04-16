# Memory — drilling in

Read this AFTER `hlo_summary.md` Memory table + `memory_timeline.txt` have
identified the peak-HBM module and the JAX arrays held at the peak.

## What each summary artifact tells you

| File | Gives you |
|---|---|
| `hlo_summary.md` Memory table | Top-N XLA-compiled modules by peak "Total bytes used", with the top allocation inside each |
| `memory_details.txt` | Same as above but the full memory-usage-report per module concatenated — see every allocation, not just the top one |
| `memory_timeline.txt` | Wall-clock peak timestamp, JAX-visible `bytes_in_use` at peak, XLA's cumulative `peak_bytes_in_use`, and the top-10 `jax.live_arrays()` at peak with shapes + dtypes |
| `memprof/end.prof` | pprof snapshot at run end — live JAX buffers with Python stack traces |

## Reading the top row of the Memory table

Three patterns; fix depends on which one:

| Top-row pattern | Interpretation | Next step |
|---|---|---|
| `preallocated-temp` ≫ inputs + outputs | Fused kernel has a big scratch working set | Open the module's `buffer-assignment.txt` + optimized HLO, grep for `copy` / `reshape` on the same shape as the temp — usually the fusion-breaker |
| Named `parameter` or `output` dominates | A user array too big for its layout | Open the function source; probably a missing `with_sharding_constraint` or an over-replicated axis |
| Dozens of near-duplicate rows, same jit name | Shape polymorphism retrace, each compiled separately | Fix at retrace level first (see `compilation.md`) — peaks usually collapse too |

The peak-sum at the top of `hlo_summary.md` is a LOOSE upper bound (peaks
happen at different times). The actual concurrent peak is in
`memory_timeline.txt`'s `device.peak_bytes_in_use`.

## Drill-in sequence

1. **`memory_timeline.txt`** — confirm the peak moment and which JAX arrays
   were held at it. If JAX-visible peak ≪ XLA peak, the bottleneck is a
   temp inside a compiled kernel, not a user-held array.
2. **`memory_details.txt`** — find the section for the worst module.
3. For that module, open:
   ```
   profile/xla_dump/module_XXXX.<fn>.sm_*_gpu_after_optimizations-buffer-assignment.txt
   profile/xla_dump/module_XXXX.<fn>.sm_*_gpu_after_optimizations.txt
   ```
   Grep the HLO for `copy(` / `reshape(` producing the same shape as the
   preallocated-temp.
4. For leaks (peak growing across stages): add `pf.snapshot_memory(...)` at
   stage boundaries in a probe script, then `pprof -diff_base a.prof b.prof`.
5. For A/B sizing without re-running the pipeline: `pf.aot_report(fn, *args)`
   — see `aot_reports.md`.

## Common diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `temp` ≫ inputs + outputs in one module | Broken fusion across a reshape or resharding | Insert `with_sharding_constraint` or `jax.lax.optimization_barrier` to split the fusion |
| Many near-duplicate modules | Shape polymorphism | See `compilation.md`; pad shapes to a fixed size |
| Peak growing across pprof snapshots | Buffer not donated / freed | `donate_argnums=` on the jit, or explicit `del x` + `jax.block_until_ready` |
| HLO has `copy` right before a collective | Involuntary reshard | See `sharding.md` |

## Escape hatches

| Question | Tool |
|---|---|
| Which op's output lives at byte offset X? | grep `buffer-assignment.txt` for that offset |
| Does the peak happen DURING op foo? | xprof Memory Profile tab |
| Which Python line allocated this live buffer? | `pprof -tree profile/memprof/end.prof` |
| Would chunking halve the peak? | `pf.aot_report(fn, *args, out=…)` twice with different args, diff summaries |
