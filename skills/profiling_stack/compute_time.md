# Compute time — drilling in

Read this AFTER `trace_summary.md` has ranked GPU kernels and
host↔device transfers.

## What each summary artifact tells you

| File | Gives you |
|---|---|
| `trace_summary.md` → Top GPU kernels | Ranked by total GPU time across the run, with count, max ms, theoretical occupancy %, HLO module, Python source |
| `trace_summary.md` → Transfers table | H2D/D2H/D2D totals (count, bytes, time, average GB/s) |
| `trace_summary.md` → Async overlap | Per-direction `overlap_frac` — close to 1 means the copy was hidden behind compute |
| `trace_summary.md` → Peak window bandwidth | Sliding-window (default 100 ms) peak GB/s for H2D and D2H — compare vs A100 PCIe Gen4 ≈32 GB/s ceiling |
| `trace_summary.md` → Low-occupancy kernels | Compute kernels with theoretical occupancy < 50 %, ranked by wasted time |
| `trace_details.txt` | Dense per-event dump of the top copies + top kernels |
| `hlo_summary.md` → Custom calls | cuBLAS / cuDNN / cuFFT / cuSOLVER call counts |

## Reading rules of thumb

| You see | Interpret as |
|---|---|
| One kernel >30 % of total GPU time | The thing to optimise first; open its module's HLO |
| `overlap_frac ≈ 0` on D2H AND D2H total > ~5 % of wall | Async D2H is blocking — likely `.block_until_ready()` or host code depending on the value |
| Peak window bandwidth > ~20 GB/s sustained | PCIe link saturated; combine with `overlap_frac` — saturated + low overlap = the bottleneck |
| Many low-occupancy kernels | Bad block/grid sizes or tiny shapes — candidates for batching / fusing |
| cuSOLVER call count very high | A per-k / per-batch solver call — consider batching |
| cuFFT call count high with many shapes | Plan cache thrash; pad FFT shapes to a fixed size |

## Drill-in sequence

1. **`trace_summary.md` Top kernels** — is ONE kernel dominant? If yes, go
   to its `module_XXXX.<fn>.thunk_sequence.txt` for the exact kernel order,
   and `sm_*_gpu_after_optimizations.txt` for the HLO.
2. **Overlap table** — decide if the H2D/D2H volume is hiding or blocking.
   If blocking is small (< few ms), ignore. If blocking > 10 % of wall,
   it's the bottleneck.
3. **Peak window bandwidth** — if D2H bandwidth peaks near PCIe ceiling
   during a long window, memory movement is the real limiter, not compute.
4. **Low-occupancy table** — pick offenders that ALSO cost >1 ms total.

## Adding finer-grained regions (opt-in, zero-source-touch stays default)

If the top kernel is inside a long pipeline stage and you want to attribute
the time back to a specific call site, add `pf.region("name"):` blocks to
the driver. Each region produces:

  * a stderr `[pf] ■ name Xs` line
  * a named bar in the xprof Trace Viewer

```python
with pf.region("davidson_kloop"):
    for ik in range(nk): ...
```

One-liner, no API changes. Re-run the profile; new regions show up
immediately. This is the ONLY opt-in source edit the stack wants from you,
and it's unnecessary in most cases because `trace_summary.md` already
attributes every kernel to its `py_name`.

## Common diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Single kernel dominates + low occupancy | Bad block size or small shape | Batch / fuse; check `autotune_results.pbtxt` |
| `overlap_frac ≈ 0` for D2H, total time meaningful | Host code waiting on device result | `jax.block_until_ready` moved too early; or use `jax.experimental.array_api.async_wait` |
| cuFFT dominant, retrace groups show many FFT modules | Shape polymorphism | Pad FFT shapes; see `compilation.md` |
| Many tiny kernels | No fusion — layout-breakers between them | HLO grep for `copy` / `reshape` / `convert_element_type` between kernels |

## Escape hatches

| Question | Tool |
|---|---|
| Exact wall-clock cost of op X | xprof Trace Viewer |
| Kernels run inside module Y | `profile/xla_dump/module_Y*.thunk_sequence.txt` |
| Is the GPU idle during stage Z | xprof Overview idle-fraction |
| A/B timing of an alternative kernel | `pf.aot_report(..., timing_runs=N)` |
