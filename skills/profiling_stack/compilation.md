# Compilation time

LORRAX deploys on 100+ GPUs under tight wallclock budgets, so every retrace
is visible. This category has the simplest diagnosis workflow because the
two reports already give almost everything.

## What the ranked summary told you already

Two complementary rankings:

**`compile_summary.md` → Wall-clock totals**:

```
| Stage             | Count | Total s | Max single |
| trace+transform   | 510   |  1.566  | 0.325 |
| jaxpr→MLIR        | 215   |  1.157  | 0.147 |
| XLA compile       | 215   | 14.608  | 0.674 |
```

If `XLA compile` total > 15 % of wall-clock, compilation is your
bottleneck. In the Si 2×2×2 NSCF smoke run (≈44 s), XLA compile alone
consumed ≈33 %.

**`compile_summary.md` → Top XLA compilations** and **`hlo_summary.md` →
Retrace groups** — two views of the same effect:

```
# compile_summary.md (from JAX log)
| jit() name             | Count | Total s | Max s |
| _ritz_and_residuals    |   4   | 2.003   | 0.674 |
| multiply               |  31   | 1.473   | 0.052 |
| _apply_H_sparse        |   4   | 1.180   | 0.310 |

# hlo_summary.md (from XLA dump)
| jit fn                 | #modules | max peak | Σ peak |
| jit_multiply           | 58       | 6.60 MiB | 97.29 MiB |
| jit_broadcast_in_dim   | 45       | 3.11 MiB | 22.55 MiB |
```

Both tell you "this jit name got recompiled N times". Any entry with
#modules > 2 is a candidate; > 5 is almost always a shape-polymorphism bug.

**`compile_summary.md` → Tracing cache misses** — the source lines that
triggered a retrace:

```
| Location                                                           | Misses | Sample reason |
| sources/lorrax/src/solvers/davidson.py:41:12                       | 16 | never seen function: _lu_solve |
| sources/lorrax/src/psp/vnl_ops.py:324:10                           |  9 | never seen function: add |
| sources/lorrax/src/psp/dft_operators.py:617:12                     |  4 | for fft never seen input type signature: x: c128[12,2,36,36,36] |
```

Each row is a clickable hint: open that exact file:line and read the
surrounding code. The "Sample reason" tells you what XLA saw as new (new
function id = a re-created closure; new input type = shape change; etc.)

## Drill-in #1 — read the actual cache-miss reasons

Open `profile/compile.log` and search for the location from the cache-miss
table. The log entries look like:

```
WARNING jax._src.pjit: TRACING CACHE MISS at davidson.py:41:12 (davidson) because:
  for _lu_solve defined at /opt/jax/jax/_src/lax/linalg.py:1566
  never seen input type signature:
    args[0]: c128[8,8]
    args[1]: c128[8,16]
  closest seen input type signature has 2 mismatches,
    including:
      * args[0]: c128[8,8] vs c128[12,12]
      * args[1]: c128[8,16] vs c128[12,24]
```

The shape diff at the bottom tells you *exactly* which dimension changed
between calls. That's your fix target — pad to max, or hoist the loop
into lax.scan.

## Drill-in #2 — confirm with the XLA Retrace groups

`hlo_summary.md` → Retrace groups counts actual XLA-compiled module ids
per jit name, independent of the tracing cache. Sometimes tracing-cache
misses don't translate into XLA compiles (the MLIR was cached) — the
Retrace groups table is the authoritative view of "how many times did we
burn a GPU compile".

## Drill-in #3 — first-call vs steady-state

Compare the first-call and median-call timings from an `aot_report`:

```
## Timings (after compile)
- first call (incl compile): 4.12 s
- median of remaining:       18.2 ms
```

Ratio tells you how much of an agent's observed runtime is compilation. On
a 5-minute run that ratio can be invisible; on a 30-second run it can be
dominant.

## Drill-in #4 — persistent cache

Rerun with `--persistent-cache` on `run_profiled.py`. The first run
populates `<out>/compilation_cache/`; subsequent runs hit it. The compile
summary's "Persistent cache misses" section tells you which modules didn't
cache (usually because a closure had a non-deterministic id, or an op
was stripped by `jax_compilation_cache_min_compile_time_secs`).

## Common diagnoses and their fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `jit_multiply` x58 in Retrace groups | Elementwise ops inside a Python loop retrace per-iteration | Wrap the loop body in a single `@jax.jit` so inner ops compile once |
| A non-trivial kernel retraced per k-point | Shape polymorphism (ngkmax varies) | Pad to `max(ngkmax)`, use a mask to ignore pad |
| `never seen function: <closure>` in every cache miss | Python closure recreated per iter | Define the closure once, outside the loop |
| `explanation unavailable` from fft / eigh / cholesky | Internal JAX shape cache missed | Warm the function once at setup with the max shape |
| XLA compile time dominated by one big jit | the function itself is huge | consider splitting; or accept it and add `--persistent-cache` |

## Turning down the noise

Compile-time-focused sweeps don't need the HLO dump or trace:

```bash
python3 scripts/profiling/run_profiled.py --no-trace --no-hlo \
    --out quick_compile \
    -m psp.run_nscf -i nscf.in
python3 scripts/profiling/analyze_compile_log.py quick_compile
```

That keeps only `compile.log` + `compile_summary.md` — ideal for iterating
on a fix.

## When the summary isn't enough

| Question | Tool |
|---|---|
| "What shape triggered this retrace?" | grep `compile.log` for the file:line |
| "Is my persistent cache actually being written?" | `ls <out>/compilation_cache/` after a run |
| "Is the MLIR cache hit or miss?" | compare `compile_summary.md` `jaxpr→MLIR` count vs `XLA compile` count — if MLIR count << compile count, the MLIR cache is working |
| "What's the cheapest way to kill the first-call stall?" | move the compile into a warmup step; see `psp/run_nscf.py::warmup_davidson_jit` |
