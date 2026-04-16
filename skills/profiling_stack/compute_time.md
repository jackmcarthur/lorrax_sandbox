# Compute time

Follow this doc AFTER reading `hlo_summary.md` — you already have two
ranked views of the module's compute behaviour. Here we turn that into a
diagnosis.

## What the ranked summary told you already

`hlo_summary.md` → `## Compute — aggregate op counts` and `### Custom calls`
give the module's global op inventory: how much fusion, how many dots,
how many FFTs, how many cuBLAS / cuDNN / cuFFT / cuSOLVER calls. Example:

```
| Op          | Count |
| fusion      | 625   |
| transpose   | 130   |
| copy        |  97   |
| fft         |  44   |

| __cublas$gemm             | 78 |
| __cublas$triangularSolve  | 32 |
| cusolver_getrf_ffi        | 16 |
| cusolver_syevd_ffi        | 10 |
```

Interpretation rules of thumb:

| Pattern | What it tells you |
|---|---|
| `fusion` dominates | XLA is doing its job; good baseline |
| `copy` count > `fusion` / 5 | layout churn; likely missed fusion across a sharding boundary |
| `transpose` count very high | repeated axis-reordering — frequent cause of serialisation |
| cuSOLVER calls unexpectedly large | an inner solve isn't batched (e.g. per-k `eigh`) |
| Many cuFFT with shape polymorphism | plan cache thrash; see `compilation.md` |

The `[pf] ■ <region> Xs` lines in the run's stderr also rank pipeline stages
by wall clock. Paste them at the top of your session report — they are the
single clearest orientation signal.

## Drill-in #1 — open the xprof trace

The primary GPU-timeline view. Open `<run>/profile/xprof/plugins/profile/<ts>/`
either in Perfetto (no install — upload the `perfetto_trace.json.gz` on
single-process runs; multi-process runs ship the per-rank `xplane.pb` only)
or with xprof CLI:

```bash
xprof profile/xprof --port=8791
```

Useful tabs:

| Tab | What to look for |
|---|---|
| Overview | Step-time breakdown, GPU idle fraction — jump-off point |
| Trace Viewer | Per-op timing; see your `pf.region(...)` annotations as coloured blocks |
| Graph Viewer | HLO graph — chase a kernel back to its producers |
| Memory Profile | HBM timeline — intersects with `memory.md` |

The `pf.region("name")` annotations you added in your driver show up in
Trace Viewer as named horizontal bars — use them to match wall-clock costs
to pipeline stages.

## Drill-in #2 — thunk sequence for a specific module

For a single compiled module, the cheapest "what does the GPU actually do"
view is its `thunk_sequence.txt`:

```
profile/xla_dump/module_0225.jit__apply_H_sparse.thunk_sequence.txt
```

It lists the ordered kernel launches. Useful when the trace viewer shows a
kernel taking an unexpected fraction of its module's time — the thunk
sequence shows the exact preceding / following kernels.

## Drill-in #3 — named regions in your driver

Nothing pinpoints compute cost faster than a `pf.region("name")` block:

```python
with pf.region("setup_potentials"):
    V_scf, V_loc, vnl_setup = _build_potentials(crystal, pseudos)
with pf.region("davidson"):
    for ik in range(nk):
        ...
```

This costs nothing to add and gives you:
  * a stderr `[pf] ■ name  Xs` timestamp per region
  * a named block in the xprof trace

Use this as soon as your first profile run tells you "stage X is slow" and
you want finer resolution. If you have a 10-line `main()`, you can wrap
each line this way at zero reading cost.

## Drill-in #4 — AOT microbenchmark of one function

```python
pf.aot_report(compute_chi0_q, psi_l, psi_r, quad, meta,
              out="probe/aot/compute_chi0_q", timing_runs=5)
```

Reads `probe/aot/compute_chi0_q/summary.md` → `Timings` block:
```
- first call (incl compile): 4.12 s
- median of remaining:       18.2 ms
```

A/B compare two implementations by swapping the first argument and diffing
the two summaries.

## Common diagnoses and their fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Dominant `[pf]` region, GPU idle in xprof | Python-side / HDF5 I/O / kpoint loop | wrap the loop with `lax.scan`, move work into jit, stream HDF5 in a thread |
| Many tiny GPU kernels | No fusion — boundaries cut it | inspect HLO for `copy` / `convert_element_type` / `reshape` between kernels; remove unnecessary ones |
| Single GEMM >1 s | Bad autotune plan | check `autotune_results.pbtxt`, try `--xla_gpu_autotune_level=4` |
| cuFFT dominates | Plan cache thrash across shapes | pad FFT shapes to a fixed size across iterations |
| Many `copy` around an axis | Layout mismatch before a dot or collective | see `sharding.md` |

## When the summary isn't enough

| Question | Tool |
|---|---|
| "What is op X's wall-clock cost?" | xprof Trace Viewer |
| "Which kernels run inside module Y?" | `module_Y.thunk_sequence.txt` |
| "Is the GPU actually idle here?" | xprof Overview + Trace |
| "Which Python line called this kernel?" | xprof Trace Viewer + op metadata |
| "What would chunking cost in ms?" | `pf.aot_report(..., timing_runs=N)` |
