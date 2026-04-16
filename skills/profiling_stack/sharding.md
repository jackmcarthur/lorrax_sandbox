# Sharding and communication — drilling in

Read this AFTER `hlo_summary.md` Sharding table + `collectives_details.txt`
+ Rematerialization section have identified the biggest collective / remat.

## What each summary artifact tells you

| File | Gives you |
|---|---|
| `hlo_summary.md` → Sharding table | Top collectives (all-gather / reduce-scatter / all-reduce / collective-permute / all-to-all) ranked by output bytes, with source_file:line from HLO metadata |
| `hlo_summary.md` → Rematerialization | Every `Involuntary full rematerialization` warning with the source_file:line |
| `collectives_details.txt` | Each top collective with ±3 lines of HLO context, including `channel_id`, `replica_groups`, `is_sync`, and the paired `-done` op |
| `remat_details.txt` | Each remat warning with 5 lines of context before/after from the HLO |

## Reading rules of thumb

| You see | Interpret as |
|---|---|
| No rows under Sharding | Run was single-device — re-run with `LORRAX_NGPU ≥ 2` if that's wrong |
| Top collective > ~200 MiB | Worth investigating (A100 NVLink ~600 GB/s ⇒ 1 GiB ≈ 1.7 ms per ring pass) |
| Multiple `all-gather`s on the same tensor in one module | Two consecutive reshardings — usually one intermediate sharding fixes both |
| ANY remat warning | **Priority #1** — XLA could not bridge two shardings without materialising the full tensor. The source:line is the culprit |
| `is_sync=true` on a hot-path collective | XLA decided not to overlap it; check latency-hiding scheduler is on |

## Drill-in sequence

1. **`collectives_details.txt`** — for each top collective, read its block.
   The ENTRY sharding annotation plus `dimensions={k}` on the collective
   tells you "dim k got gathered/scattered" — usually obvious from context
   whether that's the intended sharding axis.
2. **`remat_details.txt`** — for each warning, read the 5-line context.
   Look for a `copy` or `reshape` of the large tensor just above the warning.
3. Open the HLO at that file:line:
   ```
   profile/xla_dump/module_XXXX.<fn>.sm_*_gpu_after_optimizations.txt
   ```
   Search for the collective's name; walk UPWARD. The op that produced the
   input is the reshard origin.

## Common diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Big `all-gather` then `reshape` then another `all-gather` | Two consecutive reshardings | Merge via an intermediate `with_sharding_constraint` |
| `collective-permute` between specific device IDs | Wrong axis name in a NamedSharding | Double-check `P('x','y')` vs `P('y','x')` in the call |
| Remat warning with a huge `copy` | `_finalize()`-style cross-mesh transition | Split into two `with_sharding_constraint` calls with a compatible intermediate |
| `all-reduce` inside a `fori_loop` | Loop body reduces something per iteration | Hoist the reduce out, or pre-batch the loop dimension |

## Known LORRAX sharding patterns

See `sources/lorrax/docs/PROFILING_SUGGESTIONS.md` §4 for canonical worked
examples. Summary:

| Pattern | Location | Typical fix |
|---|---|---|
| `b_XY` → `b_X` → `m_X,n_Y` | `common/load_wfns.py _finalize()` | two-step reshard via `P(None,'x',None,None)` |
| Serial fori_loop over q with `rep_shard` | `gw/w_isdf.py solve_body` | batch across q with 2D parallelism |
| FFT + reshape + accumulate inside loop | `gw/w_isdf.py _chi_kernel` | preserve sharding via `with_sharding_constraint` after the reshape |

## Escape hatches

| Question | Tool |
|---|---|
| Which op produced the bad sharding? | Grep optimized HLO for the collective, walk upward |
| Did the SPMD partitioner accept my hint? | Re-run once with `--extra-xla-flags="--xla_dump_hlo_pass_re=spmd-partitioner\|sharding-propagation"`; compare before/after files |
| Is my array really sharded at runtime? | `jax.debug.visualize_array_sharding(x)` in a one-off probe |
| Which kernel is stalled on the collective? | xprof Trace Viewer, correlate with the collective's timestamp |
