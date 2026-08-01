# HLO_HOWTO — verifying sharding from compiled programs

Sharding is invisible at LORRAX call sites; whether a gather happened is a
property of the optimized HLO, not the Python. This page is the loop for
proving semantic claims ("no rank ever holds an N_mu^2 tile") from dumps.

## Get a dump on Frontera

Always in-container, never on a login node.

Single-process probe (minutes, no MPI — the preferred first check):

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_dump_to=$PWD/xla_dump" \
    python3 tools/probe_w_densifier_hlo.py        # in the repo; the canonical example
```

That probe (repo `tools/probe_w_densifier_hlo.py`, 96 lines) is the
pattern to copy: build a 2x2 mesh from 4 host devices, apply the
production `PartitionSpec`, lower the routine under test, scan the
optimized HLO for gather-class ops, and check numerics against the eager
reference. Exit code is the verdict.

Production job: add `--xla_dump_to` to `XLA_FLAGS` in a copy of
`config/frontera/templates/gw_dev.sbatch`. Dumps are thousands of files;
point them at `/scratch2`, never commit them (`.gitignore` blocks
`xla_dump*/`).

Cache-cold rule (INVARIANTS row 4): set `ISDF_JAX_CACHE_DIR=""` or any
collective/layout table silently under-reports — cache-hit modules never
re-dump HLO.

## Read the dump

```bash
python3 tools/hlo/analyze_hlo_dump.py <dir-containing-xla_dump>            # tables
python3 tools/hlo/analyze_hlo_dump.py <dump> --forbid all-gather,all-to-all  # gate, exit 2 on hit
```

Outputs: `hlo_summary.md` (memory, collectives, layout boundaries, remat,
retraces, custom calls — each row with `source_file:line`) plus four
`*_details.txt` companions.

Grep vocabulary for manual checks on `*after_optimizations.txt`:

| Pattern | Meaning |
|---|---|
| `all-gather(`, `all-to-all(` | gather-class — forbidden on N_mu^2-class operands |
| `reduce-scatter(`, `all-reduce(` | expected reduction traffic; check operand sizes |
| `transpose(`, `copy(`, `bitcast(` | layout churn at sharding/layout boundaries (FFT-FFI drove flat-k transposes 6 -> 0) |
| `custom_call_target=` | FFI/vendor calls — confirms a gate actually engaged |
| `sharding={` | annotated shardings, including on the root instruction |

## Assert shardings instead of inspecting

For new code, force the contract rather than checking after the fact:
`jax.jit(f, out_shardings=NamedSharding(mesh, spec))` refuses at compile
time if the layout cannot be produced, and
`jax.lax.with_sharding_constraint` pins interiors (this, not a fused jit,
is what fixed `check_hermitian` — CLAIMS row 9). To inspect from Python:
`jitted.lower(*args).compile()` exposes input/output shardings on the
compiled object.

## Retraces

`jax.config.update("jax_explain_cache_misses", True)` logs every tracing
cache miss with the argument that changed — the first tool to reach for
when a k-loop recompiles per iteration. The analyzer's retrace table gives
the same information post hoc from a dump.
