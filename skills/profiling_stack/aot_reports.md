# AOT reports — probing one function in isolation

Secondary tool. Use when the ranked summaries from the full profile have
pointed at a specific function and you want to inspect or A/B it without
re-running the whole pipeline. **The target module is not edited.** You
write a small standalone probe script.

## What `pf.aot_report` produces

`pf.aot_report(fn, *args, out=…, timing_runs=N)` lowers + compiles `fn`
with the exact shapes/shardings of your args, then dumps:

```
<out>/
    summary.md           ← one-page digest — read first
    jaxpr.txt            primitive-level IR (cheap, shape-only)
    stablehlo.mlir       after jit lowering, before XLA
    optimized_hlo.txt    final GPU HLO (collectives, fusions, cuBLAS calls)
    memory_analysis.txt  arguments / outputs / temp / alias / code bytes
    cost_analysis.txt    flops + bytes-accessed (some entries may be -1)
    input_shardings.txt  NamedSharding / PartitionSpec per arg
    timings.txt          (only if timing_runs > 0)
```

`summary.md` is the file to read first:

```
# AOT report: compute_chi0_q

## Memory
- arguments : 1.25 GiB
- outputs   : 480.00 MiB
- temp      : 4.00 GiB     ← usually the number that matters
- alias     : 0.00 B
- code      : 1.15 MiB

## Cost
- flops: 1.2e12
- bytes accessed: 8.7e10
- optimal_seconds: 0.018

## Timings (after compile)
- first call (incl compile): 4.12 s
- median of remaining:       18.2 ms
```

## When to reach for this (and when not)

| Scenario | AOT? |
|---|---|
| Summaries pointed at X; I want to see if halving centroids halves peak | ✓ |
| Comparing two implementations A, B of the same kernel | ✓ |
| Cost-sizing a refactor before touching the full pipeline | ✓ |
| "Which function is slow?" | ✗ — use `trace_summary.md` instead |
| "Which jit retraced most?" | ✗ — `hlo_summary.md` Retrace groups |

## Typical probe script

```python
# probe_compute_chi0.py — put anywhere; does NOT edit LORRAX source
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("probe_artifacts", hlo=False, log_compiles=False)

from gw.gw_init import load_inputs            # reuse the module's own setup
from gw.w_isdf import compute_chi0_q

args = load_inputs("cohsex.in")               # realistic shapes + shardings
pf.aot_report(compute_chi0_q, *args,
              out="probe_artifacts/aot/baseline", timing_runs=3)
```

Run with `LORRAX_NGPU=4 lxrun python3 -u probe_compute_chi0.py`, then
`cat probe_artifacts/aot/baseline/summary.md`.

A/B: run again with a second arg set, diff the two `summary.md`s.

## Caveats

1. **Argument shapes and shardings matter.** Use realistic sizes — toy
   shapes give toy answers.
2. **`cost_analysis` may contain `-1`** for ops with no analytic model
   (eigh, cholesky, fft sometimes). Memory analysis is always reliable.
3. **`timing_runs=0` skips execution.** Lower+compile still runs (that's
   how we get the HLO + memory), but we never call the compiled function
   — useful on huge kernels you can't afford to run.
4. **The `code` bucket is PTX size**, not working set. >50 MiB usually
   means an unwanted `lax.scan` unroll.

## Grep recipes on the generated HLO

```bash
# All collectives
grep -E "all-gather|reduce-scatter|all-reduce|collective-permute|all-to-all" \
     probe_artifacts/aot/baseline/optimized_hlo.txt

# All cuBLAS / cuDNN / cuFFT calls, counted
grep 'custom_call_target=' probe_artifacts/aot/baseline/optimized_hlo.txt \
    | sed 's/.*custom_call_target="\([^"]*\)".*/\1/' | sort | uniq -c
```

For anything that requires running the whole pipeline, go back to
`run_profiled.py` + the three analyzers — `SKILL.md` top.
