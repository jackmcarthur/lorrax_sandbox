# Compilation — drilling in

Read this AFTER `compile_summary.md` wall-clock totals + `hlo_summary.md`
Retrace groups + `retrace_details.txt` have identified which jit names
recompiled and what changed between signatures.

## What each summary artifact tells you

| File | Gives you |
|---|---|
| `compile_summary.md` → Wall-clock totals | total (count, seconds, max) across `trace+transform`, `jaxpr→MLIR`, `XLA compile` — if XLA compile > 15 % of wall, compilation is the bottleneck |
| `compile_summary.md` → Top XLA compilations | Per-jit-name compile time |
| `compile_summary.md` → Cache misses | Source `file:line` + sample reason for every retrace |
| `compile_summary.md` → Persistent cache misses | Which modules couldn't re-use the persistent compile cache |
| `hlo_summary.md` → Retrace groups | Number of XLA-compiled modules per jit name — authoritative "how many compiles did we burn" view |
| `retrace_details.txt` | For each retraced jit name: ENTRY signature of every module instance. **Diffing signatures within a block reveals the shape/dtype that changed between calls** |
| `compile.log` | Full captured `JAX_LOG_COMPILES` stderr — grep for the file:line to get the multi-line "because: …" block |

## Reading rules of thumb

| You see | Interpret as |
|---|---|
| `XLA compile` > 15 % of wall | Compilation is your bottleneck — start here |
| `jit_X` has > 5 modules in Retrace groups | Almost always shape polymorphism — check `retrace_details.txt` |
| `never seen function:` in cache-miss reasons | A Python closure is being recreated per iteration — move the `def` outside the loop |
| `never seen input type signature:` | Genuine shape change — pad to `max` or hoist into `lax.scan` |
| `explanation unavailable` from fft/eigh/cholesky | Internal JAX shape cache missed — warm the function once with the max shape |

## Drill-in sequence

1. **`compile_summary.md` Wall-clock totals** — confirm compile is a real
   fraction of wall.
2. **`retrace_details.txt`** — find the worst-offending jit name. Each
   block lists its N module ids with ENTRY signatures. The shape or dtype
   that differs across entries is the root cause.
3. **`compile_summary.md` Cache misses** — the `file:line` column is a
   direct jump target. Open that line in LORRAX source.
4. **`compile.log`** — grep for the same file:line to get the full
   multi-line reason (including the "closest seen signature" diff).

## Common diagnoses → fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `jit_multiply` x50+ in Retrace groups | Elementwise ops inside a Python loop, retraced per iteration | Wrap the loop body in one `@jax.jit` so inner ops compile once |
| A kernel retraced per k-point | ngk varies across k-points | Pad to `max(ngk)`, use a mask to ignore pad |
| Closures as cache-miss culprit | `def foo():` inside the loop | Define once outside the loop, close over a container |
| Huge single jit dominates compile time | Function too large / too much unrolling | Consider splitting; or `--persistent-cache` + warmup step |

## Fast-iteration modes

```bash
# Compile-time only, no trace or HLO (iteration time ~30 s)
LORRAX_NGPU=1 lxrun python3 -u .../run_profiled.py --out p --no-trace --no-hlo \
    -m psp.run_nscf -i nscf.in
python3 scripts/profiling/analyze_compile_log.py p

# Persistent compile cache across runs
LORRAX_NGPU=4 lxrun python3 -u .../run_profiled.py --out p --persistent-cache ...
```

The `--persistent-cache` second-run shows near-zero XLA compile time;
anything that didn't cache becomes a lead (see `compile_summary.md`
Persistent cache misses).

## Escape hatches

| Question | Tool |
|---|---|
| What exact shape triggered this retrace? | grep `compile.log` for the file:line — the "closest seen signature" diff lists the dim that changed |
| Is the persistent cache actually writing? | `ls <out>/compilation_cache/` after a run |
| MLIR cache working? | In `compile_summary.md`, if `jaxpr→MLIR` count ≪ `XLA compile` count, MLIR cache is helping |
| Cheapest way to kill first-call stall | Wrap the first call in a warmup step (cf. `psp/run_nscf.py::warmup_davidson_jit`) |
