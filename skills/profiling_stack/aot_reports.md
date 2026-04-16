# AOT reports — drilling into one known function

This is the **secondary** tool in the stack. Use it only after the ranked
summaries from `hlo_summary.md` / `compile_summary.md` have pointed at a
specific function. If you don't yet have an identified bottleneck, go back
to `SKILL.md` and run the full profile first.

## What it does

`pf.aot_report(fn, *args, out=..., timing_runs=N)` lowers and compiles
the function once with the exact input shapes and shardings you pass, then
dumps every static attribute:

```
<out>/
    jaxpr.txt              # primitive-level IR (cheap, shape-only)
    stablehlo.mlir         # after lowering, before XLA
    optimized_hlo.txt      # final GPU HLO (collectives, fusions, cuBLAS calls)
    memory_analysis.txt    # per-bucket byte counts
    cost_analysis.txt      # flops + bytes-accessed estimates
    input_shardings.txt    # PartitionSpec / NamedSharding per arg
    summary.md             # one-page agent-readable digest
    timings.txt            # iff timing_runs > 0
```

`summary.md` is the file to read first. Example:

```
# AOT report: compute_chi0_q

## Memory
- arguments : 1.25 GiB
- outputs   : 480.00 MiB
- temp      : 4.00 GiB       ← the number you usually care about
- alias     : 0.00 B
- code      : 1.15 MiB

## Cost
- flops: 1.2e12
- bytes accessed: 8.7e10
- optimal_seconds: 0.018

## Timings (after compile)
- first call (incl compile): 4.12 s
- median of remaining:       18.2 ms

## Input shardings
  NamedSharding(mesh=Mesh('x': 4, 'y': 1), spec=PartitionSpec(None, 'x', None, None))
  ...
```

## When to reach for this

| Scenario | Yes/No |
|---|---|
| Summaries pointed at `compute_chi0_q` as the biggest memory module; I want to know if reducing centroids halves peak | ✓ |
| Comparing two implementations A, B of the same kernel | ✓ |
| Cost-sizing a refactor before running the full pipeline | ✓ |
| "I want to know which function is slow" | ✗ — use the full profile instead |
| "I want to know which jit retraced most" | ✗ — `hlo_summary.md` Retrace groups already answers |

## Typical driver

```python
# probe_chi0.py
import sys; sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("probe_artifacts", hlo=False, log_compiles=False)   # no trace needed

import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from gw.gw_init import load_inputs                    # reuse real setup
from gw.w_isdf import compute_chi0_q                  # target

args = load_inputs("cohsex.in")                       # realistic shapes + shardings

# Baseline
pf.aot_report(compute_chi0_q, *args,
              out="probe_artifacts/aot/baseline", timing_runs=3)

# A candidate refactor (with different centroid count)
args2 = load_inputs("cohsex.in", n_centroids_override=400)
pf.aot_report(compute_chi0_q, *args2,
              out="probe_artifacts/aot/400c", timing_runs=3)
```

Then:

```bash
diff probe_artifacts/aot/baseline/summary.md \
     probe_artifacts/aot/400c/summary.md
```

## Caveats

1. **Argument shapes and shardings matter.** The report reflects the exact
   args you pass. Use realistic sizes — toy shapes give toy answers.
2. **`cost_analysis` may return -1s** for ops without an analytic model
   (eigh, cholesky, fft sometimes). Memory analysis is always reliable.
3. **`timing_runs=0` skips execution entirely.** The lower+compile cycle
   still happens (that's how we get the HLO + memory), but we never call
   the compiled function — useful on huge kernels you can't afford to run.
4. **The `code` bucket is PTX size**, not working set. If it balloons >50
   MiB, XLA unrolled something — often a `lax.scan` that degenerated.

## Grep recipes on the AOT HLO

```bash
# All collectives
grep -E "all-gather|reduce-scatter|all-reduce|collective-permute|all-to-all" \
     probe_artifacts/aot/baseline/optimized_hlo.txt

# All cuBLAS / cuDNN / cuFFT calls
grep 'custom_call_target=' probe_artifacts/aot/baseline/optimized_hlo.txt | \
     sed 's/.*custom_call_target="\([^"]*\)".*/\1/' | sort | uniq -c

# Largest allocations
head -30 probe_artifacts/aot/baseline/memory_analysis.txt
```

That's it. For anything that requires running the whole pipeline, use
`run_profiled.py` + the ranked summaries instead.
