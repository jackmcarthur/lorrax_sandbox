# FFT Helper Unification And Zeta-Fit Chunking Probe

## Summary

`lorrax_A` had two local-FFT helper families in `src/common/fft_helpers.py`: the production `shard_map` path and an older `custom_partitioning` path still used by AOT/modeling callers. I migrated the legacy helper names onto the production `shard_map` implementation so runtime and modeling now describe the same local 3D FFT.

I also profiled the zeta-fit r-chunk FFT hot path on a normal 2-component spinor example. The migration is useful, but the manual FFT chunking idea did **not** help on the tested non-bispinor shapes: cuFFT scratch was already effectively zero, so chunking only added extra XLA-visible buffers and large runtime overhead.

On top of that earlier migration, I rebased the branch onto `origin/main` (`92cbd83`) and landed a follow-on helper patch meant to support the next zeta/V refactor without changing current production behavior. The new code adds an **opt-in** chunked local FFT implementation, and restructures the wavefunction r-chunk transforms so the Bloch phase is applied only on the kept slab / centroid points rather than across the entire FFT box when the caller does not need the full box.

## Code Changes

| File | Change |
|------|--------|
| `sources/lorrax_A/src/common/fft_helpers.py` | Removed the old `custom_partitioning`-based local FFT implementation. `make_jittable_local_{f,if}ftn_3d` are now legacy aliases to the production `shard_map` helpers. `query_fft_peak_bytes` now compiles the same shard-map 3D FFT path production uses. |
| `sources/lorrax_A/tests/test_fft_helpers.py` | Added coverage for sharded-helper correctness, legacy-alias equivalence, and the new `query_fft_peak_bytes` behavior. |

### 2026-05-11 update: chunk-capable helper substrate

| File | Change |
|------|--------|
| `sources/lorrax_A/src/common/fft_helpers.py` | Added `apply_local_fft(...)` plus `fft_batch_chunks=` plumbing through `make_sharded_{f,if}ftn_3d`, `make_flat_k_fft`, and `query_fft_peak_bytes`. Default remains `1`, so existing callers keep today’s one-shot FFT behavior unless a future refactor opts in. |
| `sources/lorrax_A/src/common/wfn_transforms.py` | Added generic flat-r helpers `extract_flat_rchunk`, `embed_flat_rchunk`, `apply_bloch_phase_flat_points`, and `apply_bloch_phase_flat_rchunk`. `to_rmu(..., kvecs_frac=...)` now phases only the gathered centroid points; `to_rchunk(..., kvecs_frac=...)` now phases only the retained flat-r slab after slicing. Both `to_rmu` and `to_rchunk` also accept `fft_batch_chunks=` for future opt-in use. |
| `sources/lorrax_A/src/file_io/zeta_reader.py` | Added `fft_batch_chunks=` to `read_zeta_G_slab` / `_do_disk_to_G` so the future zeta `rchunk -> G_flat` refactor can reuse the same local FFT chunking mechanism without reopening reader internals. |
| `sources/lorrax_A/src/file_io/zeta_loader.py` | Threads `fft_batch_chunks=` through the higher-level loader surface for the same reason. |
| `sources/lorrax_A/tests/test_wfn_transforms.py` | Added coverage for phased `to_rmu`, phased `to_rchunk`, chunked `to_rchunk`, and the new flat-r helper utilities. |

## Results

### 1. Exact regression FFT shape now models the production path

Probe shape: `(nk=9, band_chunk=40, nspinor=2, fft_grid=15×15×60)`, single A100 rank, `complex128`.

| Checkout | Measurement | Peak / total GB | Notes |
|----------|-------------|-----------------|-------|
| `main` | `query_fft_peak_bytes(...)` | 0.62208 | Old query path over the legacy helper family. |
| `main` | `make_sharded_ifftn_3d` compiled total | 0.31104 | Actual production-style single 3D IFFT plan. |
| `main` | `make_jittable_local_ifftn_3d` compiled total | 0.46656 | Legacy helper compiled as 3 sequential 1D FFTs. |
| `agent/fft-batch-chunks` | `query_fft_peak_bytes(...)` | 0.31104 | Now matches the production shard-map path. |
| `agent/fft-batch-chunks` | `make_jittable_local_ifftn_3d` compiled total | 0.31104 | Legacy name is now just the production implementation. |

Interpretation: the old modeling/query path was not describing the runtime FFT shape faithfully. After the migration, the estimate and the actual production helper agree.

### 2. cuFFT scratch was not the limiting term in the tested zeta-fit FFTs

Representative non-bispinor local IFFT probe:

| Shape | Mode | AOT / runtime peak GB | Wall time |
|-------|------|------------------------|-----------|
| `(9, 80, 2, 24, 24, 80)` | Plain local 3D IFFT | 2.123 | 0.052 s |
| `(9, 80, 2, 24, 24, 80)` | Manual 8-chunk FFT loop | 2.389 | 10.596 s |

Additional exact AOT breakdowns for the tested local shard shapes:

| Case | Plain total GB | Chunk-8 total GB | cuFFT scratch GB |
|------|----------------|------------------|------------------|
| `mos2_local_bpd20` | 0.531 | 0.597 | 0.0 |
| `si10_local_bpd4` | 1.769 | 1.991 | 0.0 |
| Exact regression FFT shape `(9,40,2,15,15,60)` | 0.31104 | not pursued further | 0.0 |

Interpretation: on these normal 2-component-spinor zeta-fit FFT shapes, the expensive part is not hidden cuFFT scratch. The runtime penalty from chunking is real, but the expected memory win is absent.

### 3. End-to-end regression status

The GPU regression test still fails after zeta fitting, but at the same later `write_qp_wfn_h5` shape check seen on untouched `main`:

- `ValueError: write_qp_wfn_h5: U shape (9, 30, 30) inconsistent with (nk=4, nb_active=30).`

That failure is not introduced by this FFT-helper migration.

## Validation

- `uv run python -m pytest -q tests/test_fft_helpers.py`
  - `5 passed`
- `uv run python -m pytest -q tests/test_wfn_transforms.py`
  - `16 passed`
- `uv run python -m pytest -q`
  - `182 passed, 20 skipped, 4 failed`
  - Remaining failures:
    - `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference`
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[fcc-avec1]`
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[skew-avec2]`
    - `tests/test_kmeans_sharded.py::test_pbc_distance_scan_matches_naive_fcc`

The regression failure is the same late `write_qp_wfn_h5` shape mismatch already present on `main`:

- `ValueError: write_qp_wfn_h5: U shape (9, 30, 30) inconsistent with (nk=4, nb_active=30).`

## Status Checklist

- [x] Migrate legacy local-FFT helper names onto the production shard-map implementation.
- [x] Make `query_fft_peak_bytes` measure the same helper family production uses.
- [x] Add unit coverage for the migration.
- [x] Probe zeta-fit FFT chunking on a non-bispinor case.
- [x] Decide whether to land special-case FFT chunking.
- [x] Decision: do **not** land chunking for now.

## Open Questions

- The zeta-fit high-water logging in `common/isdf_fitting.py` is a coarse end-of-loop `nvidia-smi` sample, so it is useful for gross OOM detection but noisy for fine A/B comparisons.
- If manual chunking is revisited later, it should be gated only after demonstrating a non-zero cuFFT scratch term for a specific production FFT shape. I did not find such a case in this probe.
- The regression test’s `write_qp_wfn_h5` mismatch remains open and is a better candidate than FFT chunking for the next end-to-end unblock.
- The new helper substrate is intentionally not wired into the top-level zeta/V workflow yet. The next agent can opt in surgically where the real high-water mark lives, most likely the `rchunk <-> G_flat` zeta path, while leaving every other FFT call site at `fft_batch_chunks=1`.
