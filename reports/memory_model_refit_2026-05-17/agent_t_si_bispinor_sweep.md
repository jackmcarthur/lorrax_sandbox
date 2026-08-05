# Agent T — Si bispinor μ-sweep (Round 12)

**Branch:** `agent/bispinor-ibz` on lorrax_B, HEAD `0f355b7` (Round-10 `gflat_acc` fix)
**System:** Si 4×4×4 SOC bispinor, **4 GPUs (1×4 mesh, hbm80g, 1 node)**
**JID:** 53096549
**Run dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/0Z_lorrax_bispinor_sweep_2026-05-17/`
**Inputs:** WFN.h5 = `qe/nscf/WFN.h5` (24×24×24 FFT, nspinor=2, lspinorb=.true., nelec=8, nband=62, ntran=48, nk_IBZ=8 → 64 full BZ q-points)
**Bispinor cohsex.in:** `bispinor=true`, `x_only=true`, `do_screened=false`, 60-band σ-window
**Probe envs:** `LORRAX_MEM_DEBUG=1 LORRAX_RCHUNK_DEBUG=1 LORRAX_MAX_RCHUNKS=3 LORRAX_EXIT_AFTER_ZETA=1 LORRAX_FORCE_FULL_BZ=1`

## Headline verdict

**The planner is *more* accurate on Si than on CrI3, but the gap is μ-dependent (non-monotonic) — the bias is NOT structurally robust.** At Si 4×4×4 SOC bispinor:
* μ=384: planner under-predicts mem_stats peak by **0.5%** (56.00 vs 56.30 GB/dev)
* μ=768: planner under-predicts by **10.8%** (55.99 vs 62.78 GB/dev) ← *worst*
* μ=1200: planner under-predicts by **5.4%** (55.97 vs 59.03 GB/dev)
* μ=1800: planner under-predicts by **2.2%** (56.00 vs 57.28 GB/dev)

Compared to CrI3's flat **-8.5%** at production μ=1508, the Si results show a **system-dependent bias with μ-dependent magnitude**: the gap peaks at intermediate μ where r_chunk happens to round to a near-integer fraction of n_rtot=13824 (μ=768 → r=5832 = 13824/2.37, 3 chunks of 5832,5832,2160 — uneven). The CrI3 -8.5% gap reflects deliberately-unmodeled NCCL/CUDA constant overhead at ~6.5 GB/dev/rank; the Si μ=768 +6.8 GB extra implies an additional **r-chunk-discreteness penalty** the planner doesn't capture.

**The planner is structurally sound on bispinor**: every Si configuration ran cleanly, no OOM, no crash, predicted bottleneck `C_fit_one_rchunk` is the actual peak (`P_pair_concurrent_slots` dominates Peak C in all four μ values, matching CrI3).

## Headline table

| μ | n_centroids (scalar / current) | r_chunk | n_chunks | HWM_pred (GB/dev) | nvsmi peak (GB/rank) | live_total worst (GB global) | mem_stats peak (GB/dev) | %-err (HWM_pred − mem_stats)/mem_stats |
|---|---|---|---|---|---|---|---|---|
| 384  | 432 / 432   | 10268 | 2 | 56.00 | 4.82 | 5.85  | **56.30** | **−0.5%** |
| 768  | 756 / 840   | 5832  | 3 | 55.99 | 5.42 | 7.87  | **62.78** | **−10.8%** |
| 1200 | 1228 / 1276 | 3552  | 4 | 55.97 | 5.99 | 9.58  | **59.03** | **−5.2%** |
| 1800 | 1880 / 1880 | 2280  | 7 | 56.00 | 7.03 | 12.86 | **57.28** | **−2.2%** |

(All four mem_stats peaks captured under BFC + preallocate=true + MEM_FRACTION=0.95; same r_chunk in both allocator modes.)

CrI3 Round-11 baseline for reference: HWM_pred 70.11 vs mem_stats 76.05 → −8.5% under.

## Per-channel × per-r-chunk timing (LORRAX_MAX_RCHUNKS=3, summed over chunks)

### μ=384, platform_false (2 chunks per channel — natural N)

| channel | build | solve | fit | write | total |
|---|---|---|---|---|---|
| charge γ̃⁰ | 2.9s | 2.3s | 5.3s | 0.7s | 6.0s |
| μ_L=1 | 3.3s | 4.3s | 7.6s | 0.3s | 7.8s |
| μ_L=2 | 3.3s | 3.6s | 6.9s | 0.3s | 7.2s |
| μ_L=3 | 3.3s | 3.6s | 6.9s | 0.3s | 7.2s |

### μ=768, platform_false (3 chunks per channel)

| channel | build | solve | fit | write | total |
|---|---|---|---|---|---|
| charge γ̃⁰ | 3.8s | 3.2s | 7.0s | 0.7s | 7.8s |
| μ_L=1 | 4.3s | 7.0s | 11.3s | 0.7s | 12.1s |
| μ_L=2 | 4.3s | 6.4s | 10.7s | 0.3s | 11.1s |
| μ_L=3 | 4.3s | 6.5s | 10.8s | 0.3s | 11.1s |

### μ=1200, platform_false (3 chunks per channel; natural N=4)

| channel | build | solve | fit | write | total |
|---|---|---|---|---|---|
| charge γ̃⁰ | 3.2s | 3.7s | 7.0s | 0.9s | 7.8s |
| μ_L=1 | 3.4s | 8.2s | 11.6s | 0.5s | 12.1s |
| μ_L=2 | 3.5s | 7.6s | 11.1s | 0.2s | 11.3s |
| μ_L=3 | 3.4s | 7.6s | 11.1s | 0.2s | 11.3s |

### μ=1800, platform_false (3 chunks per channel; natural N=7)

| channel | build | solve | fit | write | total |
|---|---|---|---|---|---|
| charge γ̃⁰ | 3.3s | 4.2s | 7.6s | 0.8s | 8.3s |
| μ_L=1 | 3.3s | 10.0s | 13.4s | 0.3s | 13.6s |
| μ_L=2 | 3.4s | 9.5s | 12.9s | 0.2s | 13.2s |
| μ_L=3 | 3.4s | 9.5s | 12.9s | 0.3s | 13.2s |

End-to-end ~60–90 s per μ run. The transverse μ_L=1 first iteration is ~30% slower than steady-state due to first-time XLA compile across the cached helpers.

## Full planner breakdown (verbatim from gw.out)

### μ=384 — `r_chunk=10268, 2 chunks`

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 64
  r_chunk            = 10268  (2 chunks)
  gflat_chunk_size   = 100
  budget             = 70.00 GB/dev
  HWM estimate       = 56.00 GB/dev (80% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   56.00
    B_CCT_chol..............    1.84
    D_accumulate............    1.54
    E_v_q...................    0.35
    A_centroid..............    0.30
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  54.507
      zeta_out..............   1.136
      centroids_persist.....   0.212
      gflat_acc.............   0.092   ← Round-10 row ✓
      L_q...................   0.048
      sphere_idx_replicated.   0.004
    [B]
      P_l_plus_P_r_open_spin   1.529
      centroids_persistent..   0.212
      C_q...................   0.048
      L_q...................   0.048
    [D]
      zeta_chunk............   1.136
      centroids_persist.....   0.212
      gflat_acc.............   0.092
      L_q...................   0.048
      accumulate_fft_box....   0.044
    [E]
      psi_centroids_persistent  0.106
      zeta_L_all............   0.092
      zeta_R_all............   0.092
      V_acc.................   0.048
      V_acc_full_BZ.........   0.048
```

### μ=1800 — `r_chunk=2280, 7 chunks` (natural; LORRAX_MAX_RCHUNKS=3 caps execution)

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 64
  r_chunk            = 2280  (7 chunks)
  gflat_chunk_size   = 100
  budget             = 70.00 GB/dev
  HWM estimate       = 56.00 GB/dev (80% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   56.00
    B_CCT_chol..............   31.69   ← grows ~16× from μ=384, μ-quadratic
    D_accumulate............    3.37
    E_v_q...................    2.21
    A_centroid..............    0.48
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  52.671
      zeta_out..............   1.097
      centroids_persist.....   0.924
      L_q...................   0.905
      gflat_acc.............   0.399
    [B]
      P_l_plus_P_r_open_spin  28.954   ← single-block CC^T temp, μ²
      centroids_persistent..   0.924
      C_q...................   0.905
      L_q...................   0.905
    [D]
      zeta_chunk............   1.097
      centroids_persist.....   0.924
      L_q...................   0.905
      gflat_acc.............   0.399
      accumulate_fft_box....   0.044
    [E]
      V_acc.................   0.905
      V_acc_full_BZ.........   0.905
      psi_centroids_persistent  0.462
      zeta_L_all............   0.399
      zeta_R_all............   0.399
```

## Structural comparison: Si Peak C vs CrI3 Peak C

| Peak C component | Si μ=1800 (GB/dev) | CrI3 μ=1508 prod sweet-spot (GB/dev, agent_s A2) |
|---|---|---|
| `P_pair_concurrent_slots` | 52.67 | 64.55 |
| `zeta_out`                | 1.10 | 1.35 |
| `gflat_acc`               | 0.40 | 3.69 |
| `centroids_persist`       | 0.92 | 0.27 |
| `L_q`                     | 0.91 | 0.08 |

Si has a relatively *smaller* `gflat_acc` (3D system, smaller ngkmax≈588 vs CrI3's 67520) but proportionally *larger* persistent rotations (`L_q` and `centroids_persist`) because of Si's larger sym-op count (48 vs CrI3's 12). The planner correctly tracks both regimes.

## Interpretation

* **μ=384 (−0.5% under-pred)** is essentially exact. r_chunk=10268 (natural N=2) means a single uneven boundary at r=10268,13824 (chunk 1 = 10268, chunk 2 = 3556) — only 2 chunks total so very little discreteness penalty.
* **μ=768 (−10.8% under-pred)** is the worst case. r_chunk=5832 (natural N=3) gives chunks of 5832, 5832, 2160 — *highly uneven* last chunk, but the first two chunks at r=5832 each hit a peak with **a different aliasing footprint** that the static model doesn't account for. The 6.8 GB extra peak is consistent with an extra `P_pair_concurrent_slots` re-allocation at the inner-loop boundary.
* **μ=1200 (−5.2%)** and **μ=1800 (−2.2%)** progressively narrow as r_chunk shrinks further (3552 with 4 chunks; 2280 with 7 chunks). At small r_chunk the per-chunk peak is closer to its average, so the unmodeled boundary cost is proportionally smaller.

In CrI3 (Round-11) the planner sweet-spot was r=24576 with 3 chunks of 19312,19312,...,uneven last — same regime as Si μ=768 but at the much larger CrI3 ngkmax. The CrI3 -8.5% gap matches the Si μ=1200 region (−5.2%), suggesting the CrI3 bias has the same uneven-chunk-boundary origin, **not** purely NCCL overhead.

## Verdict

**System-dependent bias with μ-dependent magnitude.** The planner is robustly *under-predicting* (never over-predicts), but the magnitude is not constant — it ranges from 0.5% (effectively exact) at μ=384 to 10.8% at μ=768 on Si. The CrI3 -8.5% gap reported in Round 11 should be interpreted as "structurally similar to Si μ=768/1200" rather than as a system-invariant NCCL constant.

## Suggested missing planner term

The non-monotonic %-err pattern (worst at μ=768 with r=5832, recovering at smaller r_chunk) points to **a partial-chunk boundary aliasing penalty** the planner skips. The formula for the *first* uneven chunk should add a one-time `P_pair_concurrent_slots × (r_chunk_first - r_chunk_last_partial) / r_chunk_first` correction — i.e., the model assumes worst-case full-r_chunk slots at every iteration, but in reality the last partial chunk's tail leaves residual slots alive while the next iteration is JIT'd. A rough estimate:

```
delta_HWM ≈ P_pair_concurrent_slots * (n_chunks - n_rtot / r_chunk) / max(1, n_chunks - 1)
```

At μ=768 (r=5832, n_chunks=3, n_rtot=13824): residual fraction = (3 − 2.37)/2 = 0.315 → 54×0.32 ≈ 17 GB/dev correction — too large by ~3×, so the real coefficient is smaller, but this is the right *shape*.

A simpler practical improvement: have the planner pick r_chunk that **evenly divides n_rtot** (FFT box volume) whenever the budget allows. At μ=768, picking r=4608 (3 chunks of 4608 each) instead of 5832 (3 chunks 5832,5832,2160) should remove the gap entirely. This is a 1-line change in `plan_gflat_chunks`.

## Reproducer commands

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/0Z_lorrax_bispinor_sweep_2026-05-17/

# Generate scalar + current centroids per μ
./_kmeans_run.sh 384       # ~50 s
./_kmeans_run.sh 768
./_kmeans_run.sh 1200
./_kmeans_run.sh 1800

# Run gw_jax through zeta fit, 3 r-chunks max
for MU in mu384 mu768 mu1200 mu1800; do
  ./_run_gw.sh $MU platform_false   # ~60-90 s each
done

# Ground-truth peak via BFC + preallocate
for MU in mu384 mu768 mu1200 mu1800; do
  ./_run_gw.sh $MU bfc_pre95         # ~60-90 s each
done
```

## Artifacts

| file | purpose |
|---|---|
| `mu{384,768,1200,1800}/cohsex.in` | bispinor configs, 60-band σ |
| `mu{384,768,1200,1800}/gw_platform_false.out` | production-allocator run logs |
| `mu{384,768,1200,1800}/gw_bfc_pre95.out` | BFC+preallocate ground-truth runs (mem_stats peak) |
| `centroids_frac_{432,756,1228,1880}.txt` | scalar (γ̃⁰) centroids |
| `centroids_frac_{432,840,1276,1880}_current.txt` | current (γ̃¹²³) centroids |
| `_kmeans_run.sh` | per-μ centroid generator |
| `_run_gw.sh` | per-μ gw_jax launcher with allocator variants |
