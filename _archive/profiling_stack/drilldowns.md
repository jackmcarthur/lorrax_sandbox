# Drill-downs — turning ranked summaries into fixes

This is the single reference for interpreting the four ranked summaries and
going from "this row is the worst offender" → "here's the file:line to
open and here's the likely fix". Jump to the § that matches your
bottleneck. Each § uses the same three tables:

  1. **Summary artifacts** — which file gives you what
  2. **Reading rules of thumb** — symptom → interpretation
  3. **Diagnoses → fixes** — common pattern → concrete change

The "escape hatches" (when a summary isn't enough) live at the bottom for
all four categories together.

---

## Memory

### Summary artifacts

| File | Gives you |
|---|---|
| `hlo_summary.md` Memory table | Top-N XLA-compiled modules by peak "Total bytes used", with the top allocation inside each |
| `memory_details.txt` | Same, but full memory-usage-report per module — every allocation, not just the top |
| `memory_timeline.txt` | Wall-clock peak timestamp, JAX-visible `bytes_in_use` at peak, XLA cumulative `peak_bytes_in_use`, top-10 `jax.live_arrays()` at peak with shapes + dtypes |
| `memprof/end.prof` | pprof snapshot at run end — live JAX buffers with Python stack traces |

### Reading rules of thumb

| Top-row pattern | Interpretation |
|---|---|
| `preallocated-temp` ≫ inputs + outputs | Fused kernel has a big scratch working set |
| Named `parameter` or `output` dominates | A user array too big for its layout |
| Dozens of near-duplicate rows, same jit name | Shape polymorphism retrace — each compiled separately |
| JAX-visible peak ≪ XLA cumulative peak | Bottleneck is temps inside a compiled kernel, not user arrays |

### Diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `temp` ≫ inputs + outputs in one module | Broken fusion across a reshape or resharding | `with_sharding_constraint` or `jax.lax.optimization_barrier` to split the fusion |
| Many near-duplicate modules | Shape polymorphism | See § Compilation below; pad shapes to a fixed size |
| Peak growing across pprof snapshots | Buffer not donated / freed | `donate_argnums=` on the jit, or explicit `del x` + `jax.block_until_ready` |
| HLO has `copy` right before a collective | Involuntary reshard | See § Sharding below |

### Drill sequence

1. `memory_timeline.txt` → confirm the peak moment and the JAX arrays held.
2. `memory_details.txt` → find the section for the worst module.
3. Open its `sm_*_gpu_after_optimizations-buffer-assignment.txt` and
   `_gpu_after_optimizations.txt`; grep for `copy(` / `reshape(` on the
   same shape as the preallocated-temp.
4. For leak hunts: insert `pf.snapshot_memory(...)` at stage boundaries in
   a probe script, then `pprof -diff_base a.prof b.prof`.
5. For A/B sizing without re-running the pipeline: `pf.aot_report` —
   see `aot_reports.md`.

---

## Compute time

### Summary artifacts

| File | Gives you |
|---|---|
| `trace_summary.md` Top kernels | Ranked by total GPU time, with count, max ms, occupancy %, HLO module, Python source |
| `trace_summary.md` Transfers | H2D/D2H/D2D totals (count, bytes, time, average GB/s) |
| `trace_summary.md` Async overlap | Per-direction `overlap_frac` — close to 1 means the copy was hidden behind compute |
| `trace_summary.md` Peak window bandwidth | Sliding-window peak GB/s; compare vs A100 PCIe Gen4 ~32 GB/s |
| `trace_summary.md` Low-occupancy kernels | Compute kernels with theoretical occupancy < 50 %, ranked by wasted time |
| `trace_details.txt` | Dense per-event dump of the top copies + top kernels |
| `hlo_summary.md` Custom calls | cuBLAS / cuDNN / cuFFT / cuSOLVER call counts |

### Reading rules of thumb

| You see | Interpret as |
|---|---|
| One kernel >30 % of total GPU time | The thing to optimise first; open its module's HLO |
| `overlap_frac ≈ 0` on D2H AND D2H total > ~5 % of wall | Async D2H is blocking — likely `.block_until_ready()` or host code waiting on the value |
| Peak window bandwidth > ~20 GB/s sustained | PCIe link saturated; combine with `overlap_frac` — saturated + low overlap = the bottleneck |
| Many low-occupancy kernels | Bad block/grid sizes or tiny shapes — candidates for batching / fusing |
| cuSOLVER call count very high | A per-k / per-batch solver call — consider batching |
| cuFFT call count high with many shapes | Plan cache thrash; pad FFT shapes to a fixed size |

### Diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Single kernel dominates + low occupancy | Bad block size or small shape | Batch / fuse; check `autotune_results.pbtxt` |
| `overlap_frac ≈ 0` for D2H, meaningful total | Host code waiting on device result | Delay `jax.block_until_ready` or use async wait APIs |
| cuFFT dominant + many FFT modules | Shape polymorphism | Pad FFT shapes; see § Compilation |
| Many tiny kernels | No fusion — layout-breakers between them | HLO grep for `copy` / `reshape` / `convert_element_type` between kernels |

### Drill sequence

1. Top-kernels table → is ONE kernel dominant? If yes, open its
   `module_XXXX.<fn>.thunk_sequence.txt` + `sm_*_gpu_after_optimizations.txt`.
2. Overlap table → decide if H2D/D2H volume is hiding or blocking.
   Blocking < few ms → ignore. Blocking > 10 % of wall → bottleneck.
3. Peak window bandwidth → if D2H near PCIe ceiling for a long window,
   memory movement is the real limiter, not compute.
4. Low-occupancy table → pick offenders that ALSO cost >1 ms total.

Opt-in finer granularity: add `pf.region("name"):` blocks in the driver
script to get named bars in xprof + `[pf] ■ name Xs` stderr lines. Target
module is NOT edited for the default flow — this is only when you want
sub-pipeline timing.

---

## Sharding and communication

### Summary artifacts

| File | Gives you |
|---|---|
| `hlo_summary.md` Sharding table | Top collectives (all-gather / reduce-scatter / all-reduce / collective-permute / all-to-all) ranked by output bytes, with source_file:line from HLO metadata |
| `hlo_summary.md` Rematerialization | Every `Involuntary full rematerialization` warning with source_file:line |
| `collectives_details.txt` | Each top collective with ±3 lines of HLO context, including `channel_id`, `replica_groups`, `is_sync`, paired `-done` op |
| `remat_details.txt` | Each remat warning with 5 lines of context before/after |

### Reading rules of thumb

| You see | Interpret as |
|---|---|
| No rows under Sharding | Run was single-device — re-run with `LORRAX_NGPU ≥ 2` if that's wrong |
| Top collective > ~200 MiB | Worth investigating (A100 NVLink ~600 GB/s ⇒ 1 GiB ≈ 1.7 ms per ring pass) |
| Multiple `all-gather`s on same tensor in one module | Two consecutive reshardings — one intermediate sharding fixes both |
| Any remat warning | **Priority #1** — XLA couldn't bridge two shardings without materialising the full tensor |
| `is_sync=true` on a hot-path collective | XLA decided not to overlap it; check the latency-hiding scheduler is on |

### Diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Big `all-gather` then `reshape` then another `all-gather` | Two consecutive reshardings | Merge via an intermediate `with_sharding_constraint` |
| `collective-permute` between specific device IDs | Wrong axis name in a NamedSharding | Check `P('x','y')` vs `P('y','x')` in the call |
| Remat warning with a huge `copy` | `_finalize()`-style cross-mesh transition | Split into two `with_sharding_constraint` calls with a compatible intermediate |
| `all-reduce` inside a `fori_loop` | Loop body reduces something per iteration | Hoist the reduce out, or pre-batch the loop dimension |

### Drill sequence

1. `collectives_details.txt` → each top collective's block. `dimensions={k}`
   on the collective tells you which axis got gathered/scattered.
2. `remat_details.txt` → for each warning, look for a `copy` or `reshape`
   of the large tensor just above it.
3. Open HLO at that file:line, search for the collective's name, walk
   UPWARD to find the op that produced the input.

### Known LORRAX sharding patterns

From `sources/lorrax/docs/PROFILING_SUGGESTIONS.md` §4:

| Pattern | Location | Typical fix |
|---|---|---|
| `b_XY` → `b_X` → `m_X,n_Y` | `common/load_wfns.py _finalize()` | two-step reshard via `P(None,'x',None,None)` |
| Serial fori_loop over q with `rep_shard` | `gw/w_isdf.py solve_body` | batch across q with 2D parallelism |
| FFT + reshape + accumulate inside loop | `gw/w_isdf.py _chi_kernel` | preserve sharding via `with_sharding_constraint` after the reshape |

---

## Compilation

### Summary artifacts

| File | Gives you |
|---|---|
| `compile_summary.md` Wall-clock totals | trace+transform / jaxpr→MLIR / XLA compile totals — if XLA > 15 % of wall, compilation is the bottleneck |
| `compile_summary.md` Top XLA compilations | Per-jit-name compile time |
| `compile_summary.md` Cache misses | Source `file:line` + sample reason for every retrace |
| `compile_summary.md` Persistent cache misses | Which modules couldn't re-use the persistent compile cache |
| `hlo_summary.md` Retrace groups | Number of XLA-compiled modules per jit name — authoritative "how many compiles did we burn" |
| `retrace_details.txt` | Per retraced jit name: ENTRY signature of every module instance. **Diffing signatures within a block reveals the shape/dtype that changed between calls** |
| `compile.log` | Full `JAX_LOG_COMPILES` stderr — grep for the file:line to get the multi-line "because:" block |

### Reading rules of thumb

| You see | Interpret as |
|---|---|
| `XLA compile` > 15 % of wall | Compilation is your bottleneck — start here |
| `jit_X` has > 5 modules in Retrace groups | Almost always shape polymorphism — check `retrace_details.txt` |
| `never seen function:` in cache-miss reasons | A Python closure is being recreated per iteration — move `def` outside the loop |
| `never seen input type signature:` | Genuine shape change — pad to `max` or hoist into `lax.scan` |
| `explanation unavailable` from fft/eigh/cholesky | Internal JAX shape cache missed — warm the function once with the max shape |

### Diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `jit_multiply` x50+ in Retrace groups | Elementwise ops inside a Python loop, retraced per iteration | Wrap the loop body in one `@jax.jit` so inner ops compile once |
| A kernel retraced per k-point | ngk varies across k-points | Pad to `max(ngk)`, use a mask to ignore pad |
| Closures as cache-miss culprit | `def foo():` inside the loop | Define once outside, close over a container |
| Huge single jit dominates compile time | Function too large / too much unrolling | Consider splitting; or `--persistent-cache` + warmup step |

### Drill sequence

1. Wall-clock totals → confirm compile is a real fraction of wall.
2. `retrace_details.txt` → find the worst-offending jit name. Each block
   lists N module ids with ENTRY signatures. The shape/dtype differing
   across entries is the root cause.
3. `compile_summary.md` Cache misses `file:line` → open that line in
   LORRAX source.
4. `compile.log` → grep for the same file:line to get the full multi-line
   reason with "closest seen signature" diff.

### Fast-iteration modes

```bash
# Compile-time only, no trace or HLO (iteration <30 s)
LORRAX_NGPU=1 lxrun python3 -u .../run_profiled.py --out p --no-trace --no-hlo \
    -m psp.run_nscf -i nscf.in
python3 scripts/profiling/analyze_compile_log.py p

# Persistent compile cache across runs
LORRAX_NGPU=4 lxrun python3 -u .../run_profiled.py --out p --persistent-cache ...
```

---

## Escape hatches (all categories)

When the ranked summaries aren't enough:

| Question | Tool |
|---|---|
| Per-op GPU kernel timings in an interactive UI | `xprof profile/xprof --port=8791`, or upload `perfetto_trace.json.gz` to ui.perfetto.dev |
| Memory timeline across a fori_loop | xprof Memory Profile tab |
| Which Python line allocated a live buffer | `pprof -tree profile/memprof/end.prof` |
| Which op's output lives at byte offset X | grep `buffer-assignment.txt` for that offset |
| Does the peak happen DURING op foo | xprof Memory Profile tab |
| Which op produced a bad sharding | grep optimized HLO for the collective, walk upward |
| Did the SPMD partitioner accept my hint | Re-run once with `--extra-xla-flags="--xla_dump_hlo_pass_re=spmd-partitioner\|sharding-propagation"` and diff before/after files |
| Is my array really sharded at runtime | `jax.debug.visualize_array_sharding(x)` in a one-off probe |
| What exact shape triggered this retrace | grep `compile.log` for the file:line — "closest seen signature" diff lists the dim that changed |
| Is the persistent cache writing | `ls <out>/compilation_cache/` after a run |
| MLIR cache working | In `compile_summary.md`, if `jaxpr→MLIR` count ≪ `XLA compile` count, MLIR cache is helping |
| A/B timing / memory of a candidate refactor | `pf.aot_report` — see `aot_reports.md` |
