# Parallel helpers — Agent 2 implementation log

## 0. Status

- **Phase**: helper v1 done. Standalone `gflat_to_rchunk` lives in
  `sources/lorrax_B/src/common/wfn_transforms.py`; three CPU bit-identity
  tests pass; HLO slot-count check passes (≤ 3 FFT-box slots).
  **Integration into `_make_fit_one_rchunk_kernel._kernel` is the next
  step — deferred per `restart_agent_2.md` §5 until standalone helper
  is validated.**
- **Branch**: `sources/lorrax_B`, `agent/zeta-bc-scan-shardmap`. Adds
  `gflat_to_rchunk` + 3 tests on top of the existing scaffolding
  (`to_rchunk_inner`, `_slice_local_tile_bc`).
- **Goal hit**: forward symmetric twin of `accumulate_rchunk_to_gflat`,
  one shard_map + scan, FFT-box-class slot count collapses from N to ≤
  2 (synth scale).

## 1. What was built

### `gflat_to_rchunk` (`wfn_transforms.py`)

Mirror of `accumulate_rchunk_to_gflat` (same file, sits just above the
reverse helper). Signature per `parallel_helpers_design.md` §2:

```python
def gflat_to_rchunk(
    psi_G: jax.Array,                                  # (nk, nb_total, ns, ngkmax) c128
    g_index: np.ndarray | jax.Array,                   # (nk, nx, ny, nz) int32
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],
    r0,                                                # int or traced scalar
    r_len: int,                                        # static slab length
    kvecs_frac: np.ndarray | jax.Array | None = None,  # (nk, 3) float64; None ⇒ no phase
    norm: str = "ortho",                               # matches legacy 1/√N convention
    chunk_size: int | None = None,                     # default one-shot (cs = N)
) -> jax.Array:                                        # (nk, nb_total, ns, r_len) c128
```

Body structure: single shard_map over `('x','y')`; per-rank reshape
`(nk, nb_local, ns, ngkmax) → (N=nk·nb_local, ns, ngkmax)`; `lax.scan`
over `n_chunks = ⌈N/cs⌉` chunks; per-iter dynamic_slice → reuse of
`_box_kernel` with singleton-`nb` reshape and per-row
`g_index[k_row]` → `ifftn` → flat-r slice → optional Bloch phase
(`+1` sign, on-slice) → `dynamic_update_slice` into `out_flat`.
Exactly mirrors `accumulate_rchunk_to_gflat`'s body shape; only the
direction-specific ops swap (gather/scatter with the sphere index,
fftn↔ifftn, phase sign).

Symmetry-breakers per design §3, all preserved:
- direction (`ifftn` + `+sign`),
- norm default `"ortho"` (legacy `get_sharded_wfns_rchunk_slice`),
- no donation (writes a fresh slab; donation makes sense only once a
  caller wants to reuse a buffer across rchunks — out of scope).

Cache: `_GFLAT_TO_RCHUNK_CACHE`, shape-keyed plus `g_index_id` (content
hash) and **`kvecs_id` (content hash, see Discoveries §3)**.

### Tests (`tests/test_wfn_transforms.py`)

Three new bit-identity tests against the current bc-loop + concat path
(`_gflat_to_rchunk_reference` runs `to_rchunk` per-bc and
concatenates):

- `test_gflat_to_rchunk_no_phase` — `kvecs_frac=None`,
  `chunk_size=None` (one-shot). Bands split `[(0,4),(4,nb)]`.
- `test_gflat_to_rchunk_with_phase` — `kvecs_frac` random (rng=0),
  bands split into 3 bc.
- `test_gflat_to_rchunk_chunked_matches_oneshot` — sweeps
  `chunk_size ∈ {1, 3, N, N+1}` (covers divisor / non-divisor / no-pad
  / pad). All match the one-shot output to `rtol=1e-10, atol=1e-12`.

All three pass on `JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1`. The full
`test_wfn_transforms.py` suite (18 tests) green.

### `_box_kernel` reuse trace test

20-line standalone numpy reference vs `_box_kernel` with
`(cs, 1, ns, ngkmax)` ψ + `(cs, nx, ny, nz)` per-row g_index. Bit
identical with x64. **No `_box_kernel_flat` variant added** —
existing helper handles this contract cleanly via singleton-nb
reshape.

## 2. HLO validation (synth scale)

Same synth WFN, nb=6, nk=2, ns=2, fft (8,8,8), `r0=2, r_len=12`.

Reference: `@jax.jit` over a 3-bc Python loop (`(0,2),(2,4),(4,6)`) of
`to_rchunk` calls + `jnp.concatenate(axis=1)` — mirrors today's
`_kernel` body shape.

| Metric                                 | Reference (today) | New `gflat_to_rchunk(cs=3)` | Ratio |
|----------------------------------------|-------------------|-----------------------------|-------|
| Total preallocated-temp                | 256 KiB           | 96 KiB                       | 2.7×  |
| FFT-box-class slot count               | 4 × `c128[2,2,2,8,8,8]` | 2 × `c128[3,2,8,8,8]` | 2× collapse |
| Output buffer (`c128[2,6,2,12]`)       | 4.5 KiB           | 4.5 KiB                       | 1.0×  |

Pass criterion (design §6b): FFT-box slot count ≤ 3. **Met** at 2.
Trend confirms the prediction — XLA's scan-internal allocator aliases
the per-iter FFT box across the 4 scan iters.

(The CrI3 80 Ry HLO comparison is the killer test — 58 → ~2 at
production scale. That comes after the integration step.)

## 3. Discoveries / corrections vs design

1. **Cache-key kvecs hash.** Without it, two callers with same kvecs
   *shape* but different *content* silently hit the same compiled fn
   (whose closure contains stale `phx/phy/phz` tables). The
   chunked-vs-oneshot test caught this immediately (different rng
   seed than the with-phase test). Added `kvecs_id =
   hash(kvecs_arr.tobytes())` to the key, mirroring how
   `accumulate_rchunk_to_gflat` already content-hashes `sphere_idx`
   via `sphere_id`.
   - **The reverse helper has the same latent bug for `qvec_frac`**
     (key uses `qvec_shape` only). In production it's masked because
     `qvec_frac` is the run's q-grid (constant). Worth backporting
     for safety — flagged for follow-up; not in this commit.

2. **`g_index` shape clarification.** Design §2 lists `g_index: (nk,
   ngkmax) int32`. Actual contract from `_box_kernel` and
   `loader.box_index()` is `(nk, nx, ny, nz)` (flat-FFT-box indices,
   sentinel `ngkmax` for empty cells). Helper docstring corrected to
   match.

3. **Squeezing the singleton-nb axis.** Used `box.reshape(cs, ns, nx,
   ny, nz)` rather than `jnp.squeeze(box, axis=1)` — semantically
   identical, but reshape is a guaranteed view (no data motion).

## 4. Integration sketch (deferred — next step)

Per design §4. Replace `isdf_fitting.py:1273-1286`:

```python
psi_Y_parts = []
for bc_range in band_chunk_ranges:
    psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(...))
psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)
```

with:

```python
psi_G_full = psi_G_store.psi_G_device_full       # NEW property, lazy
psi_Y_full = gflat_to_rchunk(
    psi_G_full, psi_G_store.g_index,
    mesh=mesh_xy, fft_grid=meta.fft_grid,
    r0=r_start_dyn, r_len=actual_n_rchunk,
    kvecs_frac=psi_G_store.kvecs_frac,
    norm="ortho",
    chunk_size=...)                               # caller picks per memory budget
```

Open prereqs the integration step needs:
- `PsiGStore.psi_G_device_full` property (one-time pull of all
  band-chunks via existing io_callback pattern, returns the
  `(nk, nb_total, ns, ngkmax)` device tensor).
- Decide chunk_size policy at call site (probably budget-driven from
  `cfg.memory.memory_per_device_gb`).
- Then mirror in `c_q_from_psi_sm` callers (separate commit; CCT
  analog is structurally similar but `to_rmu`-flavoured, not
  `to_rchunk`-flavoured).

## 5. Files touched (this commit)

- `sources/lorrax_B/src/common/wfn_transforms.py`:
  - Add `gflat_to_rchunk` to `__all__`.
  - New helper `gflat_to_rchunk` + cache `_GFLAT_TO_RCHUNK_CACHE`,
    placed just above the existing `accumulate_rchunk_to_gflat` so the
    forward/reverse pair sits together.
- `sources/lorrax_B/tests/test_wfn_transforms.py`:
  - New imports, new reference helper `_gflat_to_rchunk_reference`,
    three new tests as listed above.

No changes to `psi_G_store.py`, `isdf_fitting.py`, the planner, or
configs — all integration work is the next-step.

## 6. Next-pass scope (out of this commit, but flagged)

- Integration of `gflat_to_rchunk` into `_make_fit_one_rchunk_kernel._kernel`
  (drops the bc-loop + concat).
- CCT analog for `c_q_from_psi_sm` (`isdf_fitting.py:1661, 1667`).
  Different inner shape (`r_mu` is centroid indices, not flat-r slab),
  same structural defect (Python-loop in jit body).
- Backport content-hash to `accumulate_rchunk_to_gflat`'s qvec_frac
  cache key (latent correctness bug, masked by production callers
  always passing the same qvec_frac per shape).
- Planner term `band_fft_pool` removal — codifies a defect that the
  integration step eliminates. Should land in the same commit as the
  `_kernel` rewrite.
- `solve_zeta` q-batch Python loop (separate Path D-class follow-up;
  see `PATH_D_PICKUP.md` §0).

## 7. End-of-session declaration

**Agent 2 helper v1 done.**
