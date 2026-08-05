# WFN rchunk construction profile

## Summary

Checked the newest `lorrax_D` WFN/rchunk path for whether constructing
`psi_n_XYk(rchunk)` remains rank-local after each rank has its host-resident
`psi(G-flat)` tile.  The source intends this to be local, but the available
4-GPU D-usage HLO profile shows a large unwanted communication point inside the
`fetch_psi_rchunk -> to_rchunk` fused kernel.

The key finding is that XLA all-gathers the band-sharded FFT box around
`common/wfn_transforms.py:109`, turning local
`c128[4,4,9,24,24,80]` shards into full `c128[16,4,9,24,24,80]` buffers.
That is exactly the kind of communication the path was supposed to avoid.

## Sources and profile artifacts

| Item | Value |
|---|---|
| Current source inspected | `sources/lorrax_D`, branch `agent/cusolvermp-ffi-profile`, HEAD `d14b50c` |
| Current local source state | `src/common/isdf_fitting.py` modified in worktree; not touched by this profile |
| Profile mined | `runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13/profile` |
| Profile code level | D usage stack at `c21d855`; one commit before `d14b50c` direct-numpy replicated `device_put` cleanup |
| Allocation status | `52893966` still pending, so no fresh 4-GPU rerun completed in this pass |

The `d14b50c` change removes a separate metadata-staging broadcast hazard in
`PsiGStore._populate_from_loader`; it does not change the `to_rchunk` gather,
transpose, or local FFT code path implicated by the HLO collectives below.

## Path inspected

| Stage | Code | Expected communication behavior |
|---|---|---|
| Populate per-rank host tile | `src/common/psi_G_store.py:163-229` | `loader.load` may be collective; `shard_to_host` copies local shards only |
| Pull tile inside JIT | `src/common/psi_G_store.py:327-340` | `shard_map + io_callback` returns local `(nk, bpd, ns, ngkmax)` per `(x,y)` rank |
| Transform to rchunk | `src/common/wfn_transforms.py:338-400` | Should preserve `P(None, ('x','y'), None, None)` band sharding |
| Local FFT helper | `src/common/wfn_transforms.py:161-175`, `src/common/fft_helpers.py:304-327` | Uses `shard_map` local FFT so FFT axes are replicated and batch/band sharding should remain local |

## Measurements

From `logs/profile_launch.log` in the D-usage profile:

| Section | Calls | Total time |
|---|---:|---:|
| `psi_G_store.populate.loader_load` | 5 | 0.839-0.858 s per process summary block |
| `psi_G_store.populate.shard_to_host` | 5 | 0.006-0.011 s |
| `zeta_fit.chunk.fit_one_rchunk` | 8 | 4.624-6.800 s |

So the host-tile readback itself is not the expensive part.  The expensive
part appears in the fused per-rchunk JIT.

From `profile/collectives_details.txt`:

| Module | Collective | Bytes | Source |
|---|---|---:|---|
| `module_0210.jit__kernel` | `all-gather-start` | 506.25 MiB | `wfn_transforms.py:109` |
| `module_0247.jit__kernel` | `all-gather-start` | 506.25 MiB | `wfn_transforms.py:109` |
| `module_0249.jit__kernel` | `all-gather-start` | 506.25 MiB | `wfn_transforms.py:109` |
| `module_0401.jit__kernel` and later variants | `all-gather-start` | 506.25 MiB | `wfn_transforms.py:109` |

Representative HLO shape:

```text
%bitcast = c128[4,4,9,24,24,80] ...
%all-gather-start = (..., c128[16,4,9,24,24,80]) all-gather-start(...),
  replica_groups=[1,4]<=[4], dimensions={0}
```

The line attribution points at the final transpose from `_box_kernel`:

```python
return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))
```

The HLO also shows `SPMDShardToFullShape` immediately after the
`psi_G_store` `io_callback` in some module variants, e.g. local
`c128[9,4,4,1963]` becoming full `c128[9,16,4,1963]`.  This suggests the
manual `shard_map` callback result is being reconstituted to a full logical
shape before downstream SPMD partitioning, rather than keeping the whole
G-flat-to-rchunk pipeline inside one manual per-rank region.

## Interpretation

The newest source is designed so WFN loading and `psi(rchunk)` construction
should be independent per rank, but the actual compiled HLO is not doing that.
The communication is not mainly PHDF5 WFN loading or host copy-out; it is XLA
resharding around the G-flat gather / FFT-box materialization in
`to_rchunk`.

The most likely reason is a mismatch between the manual `shard_map` region
used to pull local tiles and the ordinary `jax.jit` region used by
`to_rchunk`.  `_box_kernel` is written as globally sharded JAX and only the FFT
itself is wrapped in `shard_map`, so the SPMD partitioner is free to insert a
full-shape reconstruction before/around the transpose and local FFT boundary.

## Recommended next edit

Make the whole `to_rchunk` pipeline execute inside one `shard_map`, not just
the FFT:

1. Add a `to_rchunk_local_shard_map` variant in `common/wfn_transforms.py`.
2. Use `in_specs=(P(None, ('x','y'), None, None), P(None, None, None, None), P(None, None))`.
3. Inside the shard body, run `_box_kernel`, `jnp.fft.ifftn`, flatten, dynamic-slice, and Bloch phase on the local band shard.
4. Return `out_specs=P(None, ('x','y'), None, None)`.
5. Re-profile and require zero `wfn_transforms.py:109` `all-gather-start` entries in `collectives_details.txt`.

That is a more direct fix than tuning PHDF5 or `device_put`: it attacks the
observed 506 MiB all-gathers in the rchunk construction itself.

## Status

- Completed source orientation and HLO/profile artifact inspection.
- Implemented an opt-in `to_rchunk_shard_map` path in `sources/lorrax_D` guarded
  by `LORRAX_PSIG_RCHUNK_SHARDMAP=1`.
- Ran a fresh 4-GPU profile on Perlmutter job `52895542`:
  `runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13`.
- The patch is not committed yet because the full test suite still has unrelated
  failures; see the validation notes below.

## Opt-in shard-map trial

Code touched:

| File | Change |
|---|---|
| `src/common/wfn_transforms.py` | Added `to_rchunk_shard_map`, a shard-local variant that runs G-flat gather, local `ifftn`, r-slice, and Bloch phase inside one `shard_map` region. |
| `src/common/psi_G_store.py` | Added `LORRAX_PSIG_RCHUNK_SHARDMAP=1` switch so `fetch_psi_rchunk` can call the new full-shard-map variant. |

Run:

```bash
module load lorrax_D
export SLURM_JOBID=52895542
export LORRAX_NGPU=4
export LORRAX_PSIG_RCHUNK_SHARDMAP=1
cd runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13
lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
  --out profile -m gw.gw_jax -i cohsex.in
```

Results:

| Metric | Prior D-usage profile | Shard-map trial |
|---|---:|---:|
| `run_module:gw.gw_jax` wall | ~150.30 s | ~151.0 s |
| HLO modules | 849 | 860 |
| XLA compiles | 478 | 486 |
| Cache misses | 525 | 527 |
| Largest WFN-rchunk collective | `506.25 MiB`, `wfn_transforms.py:109` | gone from top collectives |
| Top remaining collectives | `506.25 MiB`, WFN FFT-box all-gathers | `187.50 MiB`, `gw/v_q_tile.py:717/718` |
| Numeric `eqp0.dat` rows | baseline | max absolute float diff `0` |

The patch did exactly what we wanted for this issue: `collectives_details.txt`
no longer contains the `wfn_transforms.py:109` 506 MiB all-gathers.  The largest
remaining collectives are now V_q-side gathers in `gw/v_q_tile.py`, not WFN
rchunk construction.

Representative new top collective:

```text
module_0711.jit__kernel all-gather-start 187.50 MiB
src=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/v_q_tile.py:717
```

Timing detail from the shard-map run:

| Section | Time |
|---|---:|
| charge `zeta_fit.chunk.fit_one_rchunk` | 4.769 s |
| `mu1` `zeta_fit.chunk.fit_one_rchunk` | 5.860 s |
| `mu2` `zeta_fit.chunk.fit_one_rchunk` | 5.524 s |
| `mu3` `zeta_fit.chunk.fit_one_rchunk` | 6.435 s |
| `V_q_compute` | 39.207 s |

The opt-in path improves the communication structure without moving end-to-end
wall time much on this MoS2 3x3 case, because the run is still dominated by
HDF5 close/write time and V_q work.

Validation:

| Check | Result |
|---|---|
| `python -m py_compile src/common/wfn_transforms.py src/common/psi_G_store.py` | passed |
| 4-GPU profiled GWJAX run | passed end-to-end |
| `eqp0.dat` numeric comparison vs prior D-usage profile | exact float match; only timestamp differed |
| `JAX_PLATFORM_NAME=cpu uv run python -m pytest -q tests/test_wfn_transforms.py tests/test_rchunk_gflat_pair.py tests/test_psi_g_store.py` | `24 passed` |
| `JAX_PLATFORM_NAME=cpu uv run python -m pytest -q` | `203 passed, 27 skipped, 3 failed` |

The full-suite failures were not in the touched rchunk tests: the COHSEX
regression subprocess still selected GPU and OOMed, and two
`test_v_q_per_q_g_chunked.py` cases still hit the known
`make_v_munu_chunked_kernel(... mesh_xy)` API drift.
