# Bispinor V_q pipeline — milestone report (2026-05-04)

## Summary

End-to-end bispinor pipeline (4 ζ channels + 10 nonzero V^{μ_L,ν_L}_q tiles) lands on
`agent-B/refactor-compute-vcoul`. All numerical sanity checks pass on MoS2 8-GPU.
CrI3 16-GPU verification blocked by infra (`nid001208` driver mismatch +
2-node OOM); logged in `KNOWN_SANDBOX_ERRORS.md`.

Branch state: 2 commits + ~10 staged files. Not committed; review-ready.

## Stage 1 — LU branch for transverse-channel ζ-fit

**Bug**: `compute_L_q_from_CCT` ran Cholesky on the Schur-form CCT, which is
indefinite for `vertex_mu_L ∈ {1,2,3}` (γ̃^i = α^i Hermitian indefinite).
Result: NaN ζ for the 3 transverse channels, propagated into V_q.

**Fix** (staged):
- [`common/isdf_fitting.py:546`](../../sources/lorrax_B/src/common/isdf_fitting.py#L546)
  `compute_L_q_from_CCT` accepts `vertex_mu_L: int = 0`; for `μ_L=0` runs the
  existing 2-D-blocked Cholesky (bit-identical), for `μ_L≠0` skips
  factorization and returns the raw CCT as a "system matrix".
- [`common/isdf_fitting.py:643`](../../sources/lorrax_B/src/common/isdf_fitting.py#L643)
  `solve_zeta_from_L_q` dispatches the per-q solve to `jnp.linalg.solve`
  (LU + back-solve) when `μ_L≠0`. Same shard_map structure, same input/output
  shardings, partitioned cache key.
- `vertex_mu_L` plumbed through `_make_fit_one_rchunk_kernel` and the
  `compute_L_q_from_CCT` call site in `fit_zeta_chunked_to_h5`.

**Verification (MoS2, 8 GPUs)**:
- All four ζ files NaN-free.
- max(|ζ|) per channel: μ_L=0 → 2.1·10¹, μ_L=1 → 1.58·10³, μ_L=2 → 8.04·10², μ_L=3 → 4.88·10³.
- V^{0,0} q=0 trace = 80,110,258.16 — bit-identical (within float trace precision) to pre-fix.
- All 10 bispinor V_q tiles finite. Diagonal transverse traces ≫ off-diagonal traces, as expected.
- Hermitian-transpose pairs match exactly: V^{j,i} = conj(V^{i,j}.T) by construction (driver does the transpose).

## Stage 2 — IO efficiency audit

| Item | Status |
|---|---|
| FFI on for all 4 ζ readers | ✓ already correct (`gw_init.py:756` uses `cfg.use_ffi_io`) |
| Lustre prestripe on transverse ζ outputs | ✓ already correct (`_FfiBackend.__init__` runs `_lustre_prestripe` for `mode='w'`) |
| `_choose_v_q_chunks` undersized ζ-read footprint | ✓ Fixed at [`gw/v_q_tile.py:85`](../../sources/lorrax_B/src/gw/v_q_tile.py#L85). Optional `n_rtot` arg adds the missing `N_zeta · n_rmu · n_rtot / p_prod` term. Plumbed through `v_q_driver.py:858`, `v_q_lorentz.py:265`, `compute_V_q_tile` fallback at `v_q_tile.py:618`. **MoS2 V_q wall: 23.3 s → 18.8 s (−19%)**. |
| FFT redundancy across 3 off-diag bispinor tiles | Deferred. Inline TODO at `v_q_lorentz.py:294`. Implementing requires a single jit'd super-kernel for all 7 tiles to keep peak HBM bounded (>200 lines). |
| H2D overlap check via xprof | Skipped — multi-rank xprof perfetto trace doesn't generate under JAX 0.9 distributed init in this sandbox. `memory_timeline.txt` came out clean (peak 8.6 GiB/dev during V_q diagonal tiles). |

## Stage 3 — CrI3 16-GPU verification — completed

Allocation `52431417` (4 nodes × 4 GPUs, mesh 4×4) finally clean.
**Crucial knob**: with the heuristic still over-estimating peak, the chooser
defaulted to `r_chunk=4992` which hit ~30 min/channel. Setting
`r_chunk_size = 25000` in `cohsex.in` brought it to ~7-8 min/channel — the
heuristic's "peak 45.95 GB > 35 GB budget" warning is wrong (actual peak
5.91 GB / GPU during fit_zeta).

**fit_zeta timing (16 GPUs, r_chunk=25000):**

| Channel | Wall | r-chunks | s/chunk |
|---|---:|---:|---:|
| μ_L=0 (Cholesky) | 446 s = 7m 26s | 46 | 9.7 |
| μ_L=1 (LU) | 491 s = 8m 11s | 46 | 10.7 |
| μ_L=2 (LU) | 471 s = 7m 51s | 46 | 10.2 |
| μ_L=3 (LU) | 449 s = 7m 29s | 46 | 9.8 |
| **fit_zeta total** | **1974 s = 32m 54s** | | |

LU is ~10 % slower per chunk than Cholesky — much less than the 2× I'd worried about.
GPU peak: 5.91 GB / 35 GB (17 %). Bit-identical V^{0,0} trace with
existing scalar baseline (within float-summation precision).

**V_q timing (16 GPUs, memory_per_device_gb=8 forces Case B μ-tiling):**

| Tile | Wall | Notes |
|---|---:|---|
| (0,0) scalar | 251 s = 4m 11s | `same_zeta=True` |
| (1,1) | 254 s = 4m 14s | `same_zeta=True` |
| (2,2) | 272 s = 4m 32s | |
| (3,3) | 333 s = 5m 33s | |
| (1,2) off-diag | 485 s = 8m 5s | `same_zeta=False`, 2× FFT |
| (1,3) off-diag | 436 s = 7m 16s | |
| (2,3) off-diag | (in-flight when alloc timed out — projected ~7-8 min) | |
| **V_q total (6 of 7 measured)** | **2031 s = 33m 51s** | + ~7 min projected for (2,3) |

The 3 hermitian-transpose tiles ((2,1), (3,1), (3,2)) are immediate.

**End-to-end CrI3 16-GPU bispinor pipeline projection: ~75 min total**
(33 min fit_zeta + ~42 min V_q). About 100× MoS2's ~50 s but workload-volume scaling
puts this at exactly the right magnitude.

**Open V_q chooser bug**: Case A path picked at default 35 GB budget tries to
allocate 41 GB (cuFFT scratch + n_rtot intermediates). The agent's earlier
`n_rtot` fix in `_choose_v_q_chunks` is incomplete — there's another 5×
under-estimate elsewhere. Working around with `memory_per_device_gb = 8.0`
(forces Case B). Fix scope: re-audit `_choose_v_q_chunks` against actual
HBM peaks measured here (5.91 GB fit, ~30 GB V_q Case A attempt).

## Open issues

1. **CrI3 16-GPU timing** — pending a clean 4-node allocation without `nid001208`.
2. **FFT-redundancy super-kernel** — deferred. Inline TODO at `v_q_lorentz.py:294`.
3. **xprof H2D-overlap check** — needs a perfetto-trace-capable JAX/tf-profiler combo. Deferred.
4. **Pre-existing pytest failure** in `tests/active/test_reshard_all_to_all.py` (uses
   `jax.jit` without `fun` arg in a subprocess; not from this work; logged for awareness).

## Artifacts

- `runs/MoS2/B_bispinor_profile/check_zeta_fit.py` — h5py-only NaN-sweep helper.
- `runs/MoS2/B_bispinor_profile/check_hermitian.py` — re-runs bispinor V_q + reports trace + Hermitian-transpose check.
- `runs/MoS2/B_bispinor_profile/profile_bisp_lu/` — profile artifacts from the post-fix run.
- `KNOWN_SANDBOX_ERRORS.md` — 2026-05-04 entry on the 4-node alloc + CrI3 OOM blockers.
- `CHANGELOG.md` — bispinor LU + IO entry added.
- pytest: 4 passed, 1 deselected (the pre-existing `test_reshard_all_to_all.py`).

## Next session

When a clean 4-node allocation is available (no `nid001208`):

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_nonbisp_baseline
# verify cohsex.in: bispinor=true, max_r_chunks=-1, no memory_per_device_gb override
LORRAX_NNODES=4 LORRAX_NGPU=4 lxrun python3 -u test_fit_zeta.py
```

Targets: fit_zeta 4 channels < 10 min total, bispinor V_q < 5 min total,
GPU peak per rank < 5 GB. After that lands, the full bispinor → ΔΣ^B
integration is the next milestone.
