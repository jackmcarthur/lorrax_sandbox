# Agent H — Full ζ+V_q lifecycle `jax.live_arrays()` map (Round 1)

**Date:** 2026-05-17
**Branch:** `agent/bispinor-ibz` (lorrax_B), commit `652b004` (Round-1 probe extensions on top of Round-0 `5c884ac`).
**Allocations:** JID 53075115 (4 nodes × 4 GPUs hbm80g).
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**Output:** `mem_probe_full_lifecycle_cs707.out`.
**Launch script:** `_launch_mem_probe_full_lifecycle_cs707.sh`.

Round-0 (Agent F, commit `5c884ac`) wired 3 probes inside the r-chunk loop. Round-1 extends with **P0 zeta_fit_start, P1 pre_rchunk_loop, P3 zeta_fit_end, P4 pre_v_q, P5 post_v_q** for the full HBM lifecycle map. Probes share the now-module-level `mem_probe()` helper in `src/common/isdf_fitting.py` — single source of truth across `fit_zeta_to_h5` and `gw_init.prepare_isdf_and_wavefunctions`. Run config: `cs=707` (safe regime), `LORRAX_MAX_RCHUNKS=4`, `LORRAX_FORCE_FULL_BZ=1`, **no `LORRAX_EXIT_AFTER_ZETA`** (V_q must run).

In bispinor mode `fit_zeta_to_h5` is called 4 times (charge μ_L=0 + transverse μ_L=1,2,3), so P0/P1/P3 each fire 4×. P4/P5 fire once (V_q is per-pipeline, not per-channel). Same `live_arrays()` global-shape convention as Agent F: divide by world_size=16 for per-rank on sharded arrays.

`device.memory_stats()` returns `-1` GB throughout — JAX CUDA PJRT on this stack gives `None`. All quantitative findings from `jax.live_arrays()` shape × dtype × itemsize summation (global).

---

## 1. Per-probe live-total summary (charge channel μ_L=0)

| Probe | Where in code | `live_count` | `live_total` (global GB) | Notes |
|---|---|---:|---:|---|
| **P0 zeta_fit_start** | top of `fit_zeta_to_h5` (after slab_io setup) | 54 | **1.48** | charge centroids pre-loaded (1.12 GB ψ_r + 0.32 GB sphere idx + 0.04 misc) |
| (Round-0) rchunk_start chunk=0 | inside chunk loop, after `psi_G_store.begin_rchunk` | 67 | 58.72 | matches P1 (loop just entered) |
| (Round-0) after_fit_one_rchunk chunk=0 | after `fit_one_rchunk` block_until_ready | 68 | 77.31 | +18.59 GB ζ_chunk transient |
| (Round-0) after_accumulate chunk=0 | after `accumulate_rchunk_to_gflat` | 76 | 58.74 | ζ_chunk donated, baseline restored |
| **P1 pre_rchunk_loop** | after `gflat_acc = jit(zeros)()`, before chunk loop | 67 | **58.72** | charge baseline: gflat_acc (52.52) + L_q (1.33) + ψ_l+ψ_r (4.34) + sphere idx (0.49) |
| **P3 zeta_fit_end** | end of `fit_zeta_to_h5`, before `return` | 78 | **6.22** | gflat_acc del'd; centroids + L_q + sphere idx still alive |

Per-channel P0/P1/P3 (across all 4 fit_zeta_to_h5 calls):

| Channel | μ_L | n_rmu | P0 (GB) | P1 (GB) | P3 (GB) |
|---|---:|---:|---:|---:|---:|
| charge γ̃⁰ | 0 | 1520 (pad) | 1.48 | **58.72** | 6.22 |
| transverse γ̃¹ | 1 | 1504 | 3.10 | **59.72** | 7.78 |
| transverse γ̃² | 2 | 1504 | 3.28 | **59.90** | 7.94 |
| transverse γ̃³ | 3 | 1504 | 3.44 | **60.07** | 8.10 |

**P1 grows +0.5 GB across the 4 channels** (58.72 → 60.07). All four channels load the same shapes — the growth is from the sphere-index leak (see §3).

Last 3 probes (after all four ζ-fits complete):

| Probe | Where | `live_count` | `live_total` (GB) | Δ vs prev |
|---|---|---:|---:|---:|
| (final) P3 zeta_fit_end μ_L=3 | end of 4th `fit_zeta_to_h5` | 111 | 8.10 | — |
| **P4 pre_v_q** | in `prepare_isdf_and_wavefunctions`, just before `compute_V_q(...)` | 93 | **3.61** | −4.49 GB |
| **P5 post_v_q** | just after `compute_V_q` returns, `block_until_ready(V_qmunu)` | 94 | **4.94** | **+1.33 GB** |

---

## 2. Per-array detail at the three new "between-stage" boundaries

### P0 charge zeta_fit_start (μ_L=0, live_total 1.48 GB)

Pre-loaded by `prepare_isdf_and_wavefunctions::load_centroids_band_chunked` BEFORE the first `fit_zeta`:

| Shape (global) | dtype | n | global GB | per-rank GB (p=16) | identity |
|---|---|---:|---:|---:|---|
| (36, 1520, 160, 4) | c128 | 1 | 0.56 | 0.035 (μ-sharded) | ψ_r_rmu_Y charge centroids |
| (36, 160, 4, 1520) | c128 | 1 | 0.56 | 0.035 (μ-sharded) | ψ_r_rmuT_X (transposed) |
| (36, 75, 75, 200) | i32 | 2 | 0.32 | 0.32 (REPLICATED) | sphere/phase indices |
| (36, 59990) | c128 | 1 | 0.03 | mixed | qG phase / g-vector |
| (36, 1508) | c128 | 6 | 0.01 | mixed | norms_l/r + diagonals |
| (1508, 3) | i32 | 2 | <0.001 | <0.001 | μ→r coord table |
| misc small | — | ~20 | <0.01 | — | gamma matrices, small q-tables |

**ψ_l (150-band) is NOT present at P0** — it gets allocated inside `fit_zeta_to_h5` step 1 by the slice/divide-by-norms operation. By P1, ψ_l shapes (36, 1520, 150, 4) × 2 + (36, 150, 4, 1520) × 2 appear (= +2.10 GB).

### P3 charge zeta_fit_end (live_total 6.22 GB)

| Shape | dtype | n | global GB | identity |
|---|---|---:|---:|---|
| (36, 1520, 1520) | c128 | 1 | 1.33 | L_q (Cholesky factor) — still alive |
| (36, 1520, 160, 4) | c128 | 2 | 1.12 | ψ_r charge centroids + transposed view |
| (36, 160, 4, 1520) | c128 | 2 | 1.12 | (same, different layout) |
| (36, 1520, 150, 4) | c128 | 2 | 1.05 | ψ_l charge centroids + transposed view |
| (36, 150, 4, 1520) | c128 | 2 | 1.05 | (same, different layout) |
| (36, 75, 75, 200) | i32 | 3 | 0.49 | sphere/phase indices (replicated; now 3 buffers vs P0's 2 — +1) |
| (36, 59990) | i32 | 2 | 0.02 | gflat sphere idx (per-q g-index) — created by accumulate |
| (36, 59990) | c128 | 1 | 0.03 | qG phase |
| (36, 1, 59990) | bool | 1 | <0.01 | gflat valid mask |
| (36, 1508) | c128 | 6 | 0.01 | norms_l/r + diagonals |

**Delta vs P1 (58.72 → 6.22 = −52.50 GB)**: only `gflat_acc` (52.52 GB) is freed at chunk-loop exit (`del gflat_acc` after the post-loop write). Centroids + L_q + sphere indices PERSIST.

### P4 pre_v_q (live_total 3.61 GB)

| Shape | dtype | n | global GB | identity |
|---|---|---:|---:|---|
| (36, 75, 75, 200) | i32 | **8** | **1.30** | sphere/phase indices — **+5 buffers leaked vs P0** |
| (36, 1520, 160, 4) | c128 | 1 | 0.56 | charge ψ_r centroid (one copy each: Y form, X transpose) |
| (36, 160, 4, 1520) | c128 | 1 | 0.56 | charge ψ_r centroid (transposed) |
| (36, 1504, 160, 4) | c128 | 1 | 0.55 | transverse ψ_r centroid (Y form) |
| (36, 160, 4, 1504) | c128 | 1 | 0.55 | transverse ψ_r centroid (transposed) |
| (36, 59990) | i32 | 4 | 0.03 | gflat sphere idx (per-q, 4 channels' worth) |
| (36, 59990) | c128 | 1 | 0.03 | qG phase |
| (36, 1508) c128 / (36, 1504) c128 | c128 | 6+6 | 0.02 | per-channel norms + diagonals |
| (36, 200) | c128 | 4 | <0.01 | per-channel small tables |

**Δ vs final P3 (8.10 → 3.61 = −4.49 GB)**: at scope exit of `fit_zeta`, `gc.collect()` + `jax.clear_caches()` (already wired between channels) drop the transverse channel's L_q (1.30 GB), the transverse channel's ψ_l 150-band × 2 transposed copies (~2 GB), and 1–2 leaked sphere idx buffers. Charge ψ_r and transverse ψ_r survive (held by `transverse_wfn_data` and `psi_rmu_Y` closures in `prepare_isdf_and_wavefunctions`).

### P5 post_v_q (live_total 4.94 GB)

| Shape | dtype | n | global GB | identity | new vs P4? |
|---|---|---:|---:|---|---|
| **(36, 1520, 1520)** | **c128** | **1** | **1.33** | **V_qmunu_CC (charge × charge)** read back via BispinorVqReader | **NEW (+1.33 GB)** |
| (36, 75, 75, 200) | i32 | 8 | 1.30 | sphere indices | unchanged |
| (36, 1520, 160, 4) | c128 | 1 | 0.56 | charge ψ_r centroid | unchanged |
| (36, 160, 4, 1520) | c128 | 1 | 0.56 | charge ψ_r centroid (transposed) | unchanged |
| (36, 1504, 160, 4) | c128 | 1 | 0.55 | transverse ψ_r centroid | unchanged |
| (36, 160, 4, 1504) | c128 | 1 | 0.55 | transverse ψ_r centroid (transposed) | unchanged |
| (36, 59990) i32 / c128 / (36, 1508/1504) | mixed | — | 0.10 | misc | unchanged |

**Δ vs P4 (3.61 → 4.94 = +1.33 GB)** is exactly `V_qmunu_CC` at full BZ shape (36, μ_pad, μ_pad). That is the CC tile read back at the end of compute_V_q (`reader.get_tile(0,0)` in gw_init.py:1055) plus the pad-to-`n_rmu_padded` pad. The TT tiles stay on disk (consumed lazily by Σ_X^B/Σ_H^B). G0 (a (n_q, n_rmu) tensor at < 1 MB) is below the print threshold.

---

## 3. NEW arrays at each NEW probe point — three top findings

### Finding 1 — V_q's PERSISTENT footprint after return is +1.33 GB (V_qmunu_CC, 0.083 GB/rank)

Pre-V_q (P4) = 3.61 GB; post-V_q (P5) = 4.94 GB. The **only** new buffer ≥100 MB is **V_qmunu_CC c128[36, 1520, 1520]** = 1.33 GB global (0.083 GB/rank μ-sharded). The planner currently models NO V_q peak; this is the first quantitative measurement of V_q's contribution to the *post*-V_q persistent baseline. The TT tiles (μ_L,ν_L = 1..3) stay on disk and do NOT appear in live_arrays.

The TRANSIENT peak of V_q (its kernel's BFC scratch) is INVISIBLE to live_arrays — it's allocated inside `compute_V_q_bispinor_g_flat_to_h5`'s jit and freed by jit exit before P5 fires. To bound it: each tile's `pre-read all 36 IBZ ζ̃ slabs (1 batched call)` reads `(36, 1504..1520, 59990) c128` = ~52 GB global / ~3.3 GB/rank into the jit. The per-q kernel adds an in-place V tile, g-chunked (g_chunk=1714 / 35 g-chunks per q). The planner needs a Peak E that includes at minimum: ζ slab batch (~3.3 GB/rank) + V_q output tile (~0.08 GB/rank) + cuFFT plan scratch (unmeasured here).

### Finding 2 — gflat_acc allocation happens INSIDE fit_zeta_to_h5, not before — Peak C baseline confirmed

P0 charge (1.48 GB) vs P1 charge (58.72 GB) = +57.24 GB allocated inside fit_zeta_to_h5 before chunk loop opens. The +57.24 GB is:
- **gflat_acc (36, 1520, 59990) c128 = +52.52 GB** — `jnp.zeros` jit allocation at isdf_fitting.py:2443
- **L_q (36, 1520, 1520) c128 = +1.33 GB** — from `factor_c_q` (Cholesky), step 3
- **ψ_l charge centroids c128 ×4 buffers (Y + X + transposed-X + transposed-Y forms) = +2.10 GB** — created by step-1 slice/`/norms` operation
- **ψ_r additional transposed copy = +1.12 GB** — slice/`/norms` also doubles the ψ_r footprint
- **sphere index buffer #3 (36, 75, 75, 200) i32 = +0.16 GB** — created in step 5 (gflat sphere build)

This **confirms the Peak C const term** baseline of ~58.7 GB / 16 ≈ 3.67 GB/rank, which matches Agent G's `_peak_C_const` decomposition. The Peak D const term is the same baseline (centroids + L_q + sphere idx) MINUS `gflat_acc` (which was allocated specifically for the chunk loop) PLUS the gflat sphere idx that accumulate populates — net ≈ 6.22 GB / 16 ≈ 0.39 GB/rank persistent at chunk-loop close.

### Finding 3 — Sphere-index LEAK: +1 (36, 75, 75, 200) i32 buffer (0.16 GB global, 0.16 GB/rank REPLICATED) per channel

Replicated `(36, 75, 75, 200) i32` buffers grow monotonically across fit_zeta_to_h5 calls. This is **load-bearing per-rank because the sphere idx is REPLICATED, not sharded**:

| Stage | n buffers | Global GB | Per-rank GB |
|---|---:|---:|---:|
| P0 charge | 2 | 0.32 | 0.32 |
| P1 charge | 3 | 0.49 | 0.49 |
| P3 charge | 3 | 0.49 | 0.49 |
| P0 μ_L=1 | 5 | 0.81 | 0.81 |
| P1 μ_L=1 | 6 | 0.97 | 0.97 |
| P3 μ_L=1 | 7 | 1.13 | 1.13 |
| P0 μ_L=2 | 6 | 0.97 | 0.97 |
| P1 μ_L=2 | 7 | 1.13 | 1.13 |
| P3 μ_L=2 | 7 | 1.13 | 1.13 |
| P0 μ_L=3 | 7 | 1.13 | 1.13 |
| P1 μ_L=3 | 8 | 1.30 | 1.30 |
| P3 μ_L=3 | 8 | 1.30 | 1.30 |
| P4 pre_v_q | 8 | 1.30 | 1.30 |
| P5 post_v_q | 8 | 1.30 | 1.30 |

The buffer is the **flat-k FFT sphere/index table** built by `make_flat_k_fft` (constructed once per fit_zeta_to_h5 inside the FFT-helper cache). Each channel's call adds one new replicated copy and the previous channel's copy is NOT freed by `gc.collect()` + `jax.clear_caches()` (the helper holds a strong ref via its module-level cache). By P5, **1.30 GB of replicated indices sit on EVERY rank** (8 buffers × 0.16 GB) — same global-AND-per-rank because they're not sharded.

The planner does not currently account for this. Across 4 channels → ~1.3 GB/rank that is real but invisible to Peak A/B/C/D. Without this leak, P3 final would be 8.10 − (8−2) × 0.16 = 8.10 − 0.96 = 7.14 GB, and P5 would be 4.94 − 0.96 = 3.98 GB. The leak is a known-class issue ("cross-jit-leaked" in Agent G's terminology, but at 0.16 GB/rank/channel it adds up).

---

## 4. What persists from ζ-fit into V_q (P3 → P4 transition)

| Quantity | At final P3 μ_L=3 | At P4 pre_v_q | Status |
|---|---:|---:|---|
| live_total (GB) | 8.10 | 3.61 | −4.49 GB drop |
| Charge L_q (36, 1520, 1520) c128 | 1.33 | absent | freed when charge ζ-fit closure exits |
| Transverse L_q (36, 1504, 1504) c128 | 1.30 | absent | freed at exit of `fit_zeta` |
| Charge ψ_l (150-band) × 4 buffers | 2.10 | absent | freed |
| Transverse ψ_l (150-band) × 4 buffers | 2.07 | absent | freed |
| Charge ψ_r (160-band) × 2 buffers | 1.12 | 1.12 | retained via `psi_rmu_Y`/`psi_rmuT_X` closure |
| Transverse ψ_r (160-band) × 2 buffers | 1.11 | 1.10 | retained via `transverse_wfn_data` dict |
| Sphere idx (36,75,75,200) i32 ×8 | 1.30 | 1.30 | leaked, persist into V_q |
| gflat_acc | absent | absent | already del'd at end of each chunk loop |

The **charge L_q (1.33 GB) is freed between P3-final and P4** — `fit_zeta` closes its local scope so `L_q` falls out of reference. This matches the `gc.collect()` + `jax.clear_caches()` already wired in `fit_zeta`. But note: P4 inherits BOTH centroid sets (charge ψ_r at μ=1520 AND transverse ψ_r at μ=1504), since they are needed for V_q.

**Implication for the planner**: V_q's "persistent baseline" = P4 = 3.61 GB global ≈ 0.23 GB/rank on sharded arrays + 1.30 GB replicated sphere idx that exists on every rank. So Peak E persistent ≈ 1.53 GB/rank before V_q's transient peak kicks in.

---

## 5. Verdict and planner recommendations

1. **Add a Peak E for V_q in `gflat_memory_model.py`**. P4 (pre_v_q) measures persistent baseline = ~3.61 GB global; per-rank ≈ 0.23 GB sharded + ~1.30 GB replicated sphere idx = ~1.53 GB/rank. P5 (post_v_q) adds 1.33 GB / 16 = 0.083 GB/rank for V_qmunu_CC. The cuFFT-plan-scratch + ζ batch-read transient peak (the "tile pre-read 12.78 s" line) is not measured here but bounded above by `n_q × ngkmax × μ_pad × 16 B` ≈ 52 GB global / 3.3 GB/rank for the worst tile. Together: V_q peak ≈ 1.53 + 3.3 + cuFFT-plan ≈ 5+ GB/rank for the CC tile. Comfortable under 70 GB but should be modeled.

2. **The replicated sphere-idx leak (Finding 3) is the cleanest fix-first target**. By μ_L=3 it costs 1.30 GB/rank PER rank (replicated, not sharded). Drop the `make_flat_k_fft` module-level cache between channels — or have `fit_zeta_to_h5` explicitly nuke the FFT helper cache after `gc.collect()`. This frees 0.96 GB/rank by P5.

3. **Confirm Peak C baseline**. P1 across all 4 channels = 58.72…60.07 GB global ≈ 3.67…3.75 GB/rank, matching the existing planner `_peak_C_const` decomposition (gflat_acc + L_q + 4 ψ_l buffers + 4 ψ_r buffers + sphere idx). No new persistent terms missed in Peak C — Agent G's audit holds.

4. **Confirm Peak D donation correctness**. The `after_accumulate chunk=N → rchunk_start chunk=N+1` transition is bit-identical at 58.74 GB (chunks 1,2,3 of charge all show 58.74). `gflat_acc` is correctly donated in-place by `accumulate_rchunk_to_gflat`. No accumulate-side leak.

---

## 6. Notes / caveats

1. **`memory_stats()` still returns -1**. JAX CUDA PJRT on this jax/jaxlib (image `nvcr.io/nvidia/jax:25.04-py3`) returns `None` from `memory_stats()`. All numbers come from `jax.live_arrays()` shape × dtype × itemsize summation, which reports **global** values. Divide by world_size=16 for per-rank on sharded arrays; replicated arrays appear at full bytes on every rank.

2. **`live_arrays()` does NOT see XLA's BFC scratch pool, cuFFT plan-workspace, or transient inside-jit buffers**. What we measured at P4/P5 is the *Python-side persistent residency*. The V_q transient peak (cuFFT scratch + ζ batch-read + per-q in-jit FFT box) is allocated inside `compute_V_q_bispinor_g_flat_to_h5`'s jit and is invisible here — but it sits on top of the P4 baseline.

3. **`LORRAX_MAX_RCHUNKS=4`** breaks each channel's r-chunk loop at 4 (of 53) chunks. P1 measurements are unaffected (allocation done before the loop). P3 and P5 are unaffected (post-loop state). The persistent buffers measured here are the same as a full run; only `gflat_acc`'s content (ζ accumulated over r) differs, and `gflat_acc` is del'd before P3 anyway.

4. **All 4 channels load identical-shaped sphere indices**; the +1 buffer per channel is from the `make_flat_k_fft` cache holding a new copy per (μ, kgrid, sys_dim) cache key — when the transverse channel runs with μ_T = 1504, it creates a *new* cache entry distinct from the charge channel's μ_C = 1520. The leak is NOT 3 × per-channel duplicates of the same buffer; it's distinct entries for charge vs transverse, with the charge entry leaked once per call. A clean cache reset between channels eliminates 4–5 of the 8 buffers.

5. **V_q SPMD remats fire 70+ times** (`Involuntary full rematerialization ... %copy.7 = c128[1,1520,59990]`) inside `gw/v_q_g_flat.py:94`. Cosmetic / perf-tax (not OOM), but worth fixing post-Round-1 for V_q efficiency.
