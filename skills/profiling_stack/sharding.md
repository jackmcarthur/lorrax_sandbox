# Sharding and communication

Follow this doc AFTER reading `hlo_summary.md` — you already have two
ranked views of the module's inter-device behaviour. Here we turn that
into a diagnosis.

## What the ranked summary told you already

`hlo_summary.md` contains two sharding-relevant sections:

```
## Sharding — collectives (largest by output bytes)
| Module                        | Op                 | Output bytes | Output type |
| module_0123.jit_gw_main       | all-gather         | 3.50 GiB     | c128[36,13,2,216000] |
| module_0123.jit_gw_main       | reduce-scatter     | 1.20 GiB     | c128[...] |
| module_0214.jit_chi0          | collective-permute | 256.00 MiB   | c128[...] |

## Rematerialization warnings
- module_0123.jit_gw_main: Involuntary full rematerialization from ...
  source_file="load_wfns.py" source_line=361
```

Reading rules:

| What you see | What to do |
|---|---|
| No rows under Sharding | Run was single-device or single-process — nothing to say. Re-run with `LORRAX_NGPU >= 2` if that's wrong. |
| Top collective output > ~200 MiB | Worth investigating — at A100 NVLink 600 GB/s, 1 GiB ≈ 1.7 ms |
| Multiple all-gathers in the same module | Likely an unnecessary reshard between two closely-related ops |
| Any Remat warning | **Priority #1** — XLA couldn't find a memory-efficient sharding transition. The source_file/line points directly at the culprit. |

## Drill-in #1 — open the module's optimized HLO

For the worst offender (e.g. `module_0123.jit_gw_main`):

```
profile/xla_dump/module_0123.jit_gw_main.sm_*_gpu_after_optimizations.txt
```

Grep for the collective name:

```bash
grep -n -E "all-gather|reduce-scatter|all-reduce|collective-permute|all-to-all" \
     profile/xla_dump/module_0123.jit_gw_main.sm_*_gpu_after_optimizations.txt
```

Each hit line looks like:

```
%all-gather.42 = c128[36,13,2,216000]{4,3,2,1,0}
    all-gather(%param.3), channel_id=12, dimensions={3}, ...
```

Walk upward in the file from that line — you'll find the op that produced
the input. If the input came from a `reshape` or `copy` or
`transpose` that preceded the collective, **that is the reshard that caused
it**.

## Drill-in #2 — Rematerialization warnings

The most important signal. A warning looks like:

```
[spmd] Involuntary full rematerialization from
  {devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate} to
  {devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}
for HLO: %copy.3 = c128[36,13,2,216000] copy(%reshape.110)
source_file="load_wfns.py" source_line=361
```

Read this as: "I was holding a tensor at sharding A, I needed it at
sharding B, and I had to materialise the entire tensor to bridge them."

The source_file/line tells you exactly which Python line caused it.
Typical fixes:
  * Insert an intermediate sharding. LORRAX's standard pattern:
    `b_XY` → `b_X` → `m_X,n_Y` via two `with_sharding_constraint` calls
    instead of one.
  * Move the reshape/copy before the sharding change, or vice versa, so
    XLA can keep one axis stationary.
  * See `sources/lorrax/docs/PROFILING_SUGGESTIONS.md` §4 for the
    canonical worked examples.

## Drill-in #3 — sharding annotations in the HLO

Each op in the optimized HLO carries its output sharding:

```
%mul.42 = c128[36,13,2,216000]{4,3,2,1,0} multiply(%a, %b),
    sharding={devices=[1,4,4,1,1]<=[16]}
```

The brackets are per-axis slicing factors on a 16-GPU mesh. Reading them
by eye is tedious but worthwhile when you suspect one op in particular —
greping for a bad split pattern like `devices=[1,1,1,1,16]` reveals
over-replication.

## Drill-in #4 — live runtime sharding

```python
from jax import debug as jd
x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('x','y')))
jd.visualize_array_sharding(x)
```

Prints an ASCII grid of which device owns which tile. For ad-hoc debugging
of a specific variable; not part of the static summary.

## Drill-in #5 — deeper XLA dump (advanced)

If you suspect the SPMD partitioner itself is making bad choices, re-run
with the partitioner stages exposed:

```bash
XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_pass_re=spmd-partitioner|sharding-propagation"
TF_CPP_VMODULE=spmd_partitioner=1
```

This produces before/after HLO per pass and puts SPMD decision lines on
stderr. Expect 3-5× more artifact size — turn off when you're done.

## Common diagnoses and their fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Big `all-gather` then `reshape` then another `all-gather` | Two consecutive reshardings | Merge to one by adding the intermediate sharding |
| `collective-permute` between explicit device IDs | Wrong axis name in a NamedSharding | Check `P('x','y')` vs `P('y','x')` in the problematic call |
| Remat warning with a `copy` of a huge array | `_finalize()`-style cross-mesh transition | Split into two with_sharding_constraint calls |
| `all-reduce` inside a fori_loop | Loop body replicates something per iter | Hoist the reduce out, or pre-batch across loop dim |

## Known LORRAX sharding patterns

(from `sources/lorrax/docs/PROFILING_SUGGESTIONS.md` — keep up-to-date)

| Pattern | Location | Typical fix |
|---|---|---|
| `b_XY` → `b_X` → `m_X,n_Y` transition | `load_wfns.py _finalize()` | two-step reshard via `P(None,'x',None,None)` |
| Serial fori_loop over q with `rep_shard` | `w_isdf.py solve_body` | batch across q with 2D parallelism |
| FFT + reshape + accumulate inside loop | `w_isdf.py _chi_kernel` | preserve sharding via `with_sharding_constraint` after the reshape |

## When the summary isn't enough

| Question | Tool |
|---|---|
| "Which op produced the bad sharding?" | grep the optimized HLO for the collective, walk upward |
| "Did the partitioner see my hint?" | `--xla_dump_hlo_pass_re=spmd-partitioner` — compare before/after |
| "Is my array really sharded at runtime?" | `jax.debug.visualize_array_sharding(x)` |
| "Which kernel is stalled on the collective?" | xprof Trace Viewer |
