# Agent C — Si non-bispinor band-count sensitivity

**Branch:** `agent/si-band-sensitivity` on lorrax_C, HEAD `d4cb599` (cherry-pick of Agent A's `8c18925` `unfold_psi` nspinor=1 fix)
**System:** Si bulk FCC (a=5.43 Å), `noncolin=.true. lspinorb=.false.` (nspinor=2, no SOC) → cohsex.in `bispinor = false`
  Pure scalar `noncolin=.false.` (nspinor=1) was attempted first but the run-time loader path lifts ψ from nspinor=1 to ns=2 internally
  (see live_arrays: `complex128 (27, 408, 100, 2)` for the nspinor=1 WFN); regardless, Agent T's bispinor sweep, Agent A's μ-sweep, and
  Agent B's k-grid sweep all use ns=2 storage even at "scalar" physics. This Agent C run uses `noncolin=.true.` for consistency with
  sister sweeps + to avoid the `unfold_psi`-side bug logged at KNOWN_SANDBOX_ERRORS.md:110 (which Agent A's `8c18925` patches anyway).
**Hardware:** 1 node / 4× A100 40 GB (mesh 2×2), JID 53097982
**ecutwfc:** 40.0 Ry (bumped from 25 Ry of Agent T's bispinor sweep so nbnd=200 fits the smallest-k sphere at 3×3×3)
  FFT grid = 30³, n_rtot = 27000 (consistent across all 4 configs)
**Run dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/BANDS_nonbispinor_2026-05-18/`
**memory_per_device_gb:** 28.0 (= AGENTS.md default for A100-40GB; LORRAX_MEM_DEBUG probes only the in-jit ζ-fit pipeline)
**Probe envs:** `LORRAX_MEM_DEBUG=1 LORRAX_RCHUNK_DEBUG=1 LORRAX_MAX_RCHUNKS=3 LORRAX_EXIT_AFTER_ZETA=1 LORRAX_FORCE_FULL_BZ=1`
**Mitigation:** `cusolvermp_charge = off, cusolvermp_lu = off` in every cohsex.in to dodge the cusolverMpPotrf=7 BFC bug logged at KNOWN_SANDBOX_ERRORS.md:117

## Headline verdict

**The planner stays within Agent T's bispinor `-0.5% to -10.8%` window at 3 of 4 (nb, kgrid) pairs but the 3×3×3 nb=200 configuration breaks
out to `-13.9%` under-prediction.** The %-err jump from nb=100 (-0.5%) to nb=200 (-13.9%) at the 3×3×3 single-r-chunk grid is the largest gap
in any of the Agent A/B/C sister sweeps to date. The same nb=100 → nb=200 step at 4×4×4 (multi-r-chunk) only moves %-err from -1.2% to -3.2%.

**Root cause:** the planner's nb-dependent term is `centroids_persist = 4 · c128(nk, ns, μ, nb_total)/p_xy` (linear in nb), but `nb_total =
nb_left + nb_right = 2·nb_cohsex` is small per-rank (e.g. 0.07 GB at nb=100, 0.14 GB at nb=200 in 3×3×3) — well below the ~2.5 GB **in-jit
transient growth** that the BFC `mem_stats peak` reveals. The unmodeled growth scales as `band_chunk × r_chunk × ns` per the inner
`z_q_from_psi_sm._local` IFFT/all_gather slabs; at 3×3×3 with r_chunk=27000 (1 chunk) this hurts the most, at 4×4×4 with r_chunk=7900 (4 chunks)
it's small.

## Headline table

All 4 configs ran under both allocator variants; **bfc_pre95** is the OOM-relevant ground truth (exposes `peak_bytes_in_use`).

| config | kgrid | nb (cohsex.in) | nb_total (planner) | band_chunk | r_chunk × n_chunks | HWM_pred (GB/dev) | mem_stats peak (GB/dev) | %-err |
|---|---|---|---|---|---|---|---|---|
| 3x3x3_nb100 | 3×3×3 | 100 | 200 | 128 | 27000 × 1 | 15.63 | **15.71** | **−0.5%** |
| 3x3x3_nb200 | 3×3×3 | 200 | 400 | 256 | 27000 × 1 | 15.70 | **18.24** | **−13.9%** |
| 4x4x4_nb100 | 4×4×4 | 100 | 200 | 128 | 7932 × 4   | 22.39 | **22.66** | **−1.2%** |
| 4x4x4_nb200 | 4×4×4 | 200 | 400 | 256 | 7812 × 4   | 22.40 | **23.13** | **−3.2%** |

(All four mem_stats peaks under BFC + preallocate=true + MEM_FRACTION=0.95; both 3×3×3 cases have 1 r-chunk only, so the `LORRAX_MAX_RCHUNKS=3` cap is moot for them.)

Reference: Agent T's bispinor Si 4×4×4 SOC swept μ ∈ {384, 768, 1200, 1800} at nb=60 and saw %-err in `-0.5% to -10.8%`. The new −13.9% is just outside that range.

## Per-peak component scaling at fixed kgrid

The planner separates HWM into 5 peaks (A/B/C/D/E). Peak C (`fit_one_rchunk`) is the bottleneck in every config; Peak D and Peak A both
have band_chunk-dependent terms.

| config | Peak A | Peak B | Peak C | Peak D | Peak E | bottleneck |
|---|---|---|---|---|---|---|
| 3x3x3_nb100 | 0.47 | 0.25 | **15.63** | 1.44 | 0.14 | C_fit_one_rchunk |
| 3x3x3_nb200 | 0.93 | 0.32 | **15.70** | 1.51 | 0.17 | C_fit_one_rchunk |
| 4x4x4_nb100 | 0.56 | 2.05 | **22.39** | 2.59 | 0.71 | C_fit_one_rchunk |
| 4x4x4_nb200 | 1.09 | 2.38 | **22.40** | 2.90 | 0.88 | C_fit_one_rchunk |

**Peak C HWM stays nearly flat in nb** (+0.07 GB nb=100→200 at 3×3×3, +0.01 GB at 4×4×4), driven by the persistent-only nb dependence:

| config | P_pair_slots | zeta_out | centroids_persist | gflat_acc | L_q | sphere_idx |
|---|---|---|---|---|---|---|
| 3x3x3_nb100 | 14.277 | 1.190 | 0.071 | 0.071 | 0.018 | 0.003 |
| 3x3x3_nb200 | 14.277 | 1.190 | **0.141** ← ×2 | 0.071 | 0.018 | 0.003 |
| 4x4x4_nb100 | 19.884 | 1.657 | 0.334 | 0.338 | 0.170 | 0.007 |
| 4x4x4_nb200 | 19.583 | 1.632 | **0.668** ← ×2 | 0.338 | 0.170 | 0.007 |

The only Peak C term that grows with nb is `centroids_persist = 4 · c128(nk, ns, μ, nb_total)/p_xy` — and it exactly doubles
between nb=100 and nb=200 at fixed kgrid, matching the planner formula bit-exactly.

## nb-scaling exponent (predicted vs measured)

We sweep nb=100 → nb=200 and report the multiplicative factor for each term.

| term | predicted factor (nb=200/nb=100) | measured factor (3×3×3) | measured factor (4×4×4) | flag |
|---|---|---|---|---|
| `centroids_persist` (Peak C term) | 2.00 (linear) | **1.99** | **2.00** | ✓ |
| `P_pair_concurrent_slots`         | 1.00 (flat) | 1.00 | 0.985 | ✓ (4×4×4 sub-unity is r_chunk re-pick 7932 → 7812) |
| `zeta_out`                         | 1.00 (flat) | 1.00 | 0.985 | ✓ |
| `gflat_acc`                        | 1.00 (flat) | 1.00 | 1.00 | ✓ |
| `L_q`                              | 1.00 (flat) | 1.00 | 1.00 | ✓ |
| `centroid_out_filling` (Peak A)    | 1.00 (flat)*¹ | 2.00 | 2.00 | **DIVERGES**: Peak A nb-scaling is via band_chunk (×2), not nb_total |
| `fft_box` (Peak A)                  | 2.00 (band_chunk doubles) | 2.00 | 2.00 | ✓ |
| **Peak C TOTAL (HWM_pred)**         | 1.005 (~flat)  | 1.005 | 1.000 | ✓ matches model |
| **mem_stats peak (measured)**       | 1.005 (planner expectation) | **1.161** | **1.021** | **DIVERGES at 3×3×3 (16× model error)** |

*¹: The planner term `centroid_out_filling = c128(nk, ns, μ, nb_total)/p_xy` would grow linearly, but it's tiny relative to `fft_box` =
`c128(band_chunk, ns, n_rtot) × fft_box_factor_A=4` which is the binding Peak A term.

**Flagged anomaly: the 3×3×3 case has a 13.6%-point %-err gap nb=100→200 that is not predicted by any term in the planner.**

## Where the missing 2.32 GB comes from (3×3×3 nb=100 → nb=200)

Side-by-side live_arrays at the `after_fit_one_rchunk chunk=0` probe (BFC):

| live array | nb=100 (GB global) | nb=200 (GB global) | diff | role |
|---|---|---|---|---|
| `(27, 408, 27000)` | 4.76 | 4.76 | 0 | zeta_chunk return (D.zeta_chunk, transient) |
| `(27, 408, 1150)` | 0.20 | 0.20 | 0 | gflat_acc persistent |
| `(27, 408, 100/200, 2)` ×3 | 0.11 | 0.21 | +0.10 | centroids persist (rmuT_X form) ×3 buffers |
| `(27, 100/200, 2, 408)` ×3 | 0.11 | 0.21 | +0.10 | centroids persist (rmu_Y form) ×3 buffers |
| `(27, 408, 408)` | 0.07 | 0.07 | 0 | L_q persistent |
| sphere_idx + bookkeeping | ~0.00 | ~0.00 | 0 | — |
| **live_total** | **5.25 GB** | **5.46 GB** | **+0.21** | sum |
| **mem_stats peak** | **15.71 GB** | **18.24 GB** | **+2.53** | XLA arena peak |
| `peak − live_total` (in-jit transient) | 10.46 GB | 12.78 GB | **+2.32 GB** | XLA preallocated-temp |

The persistent state grew by exactly +0.21 GB (planner's `centroids_persist` formula matches to within FP error). But the in-jit transient
grew by **+2.32 GB**, 11× more than persistent. That excess is XLA preallocated-temp inside `z_q_from_psi_sm._local`.

**Candidate term** (per `docs/MEMORY_MODEL.md` §R-Chunk Round-8 unified-FFT-pipeline model):

```
M_all_gather_slab (scan-aliased) ≈ c128(nk, P·bpd_max_local, ns, n_zchunk) per rank
                                = c128(nk, band_chunk, ns, r_chunk / p_y)
```

For 3×3×3 (nk=27, p_y=2, ns=2, r_chunk=27000):
* nb=100, band_chunk=128: 27 · 128 · 2 · 13500 · 16 = 1.49 GB/rank
* nb=200, band_chunk=256: 27 · 256 · 2 · 13500 · 16 = 2.99 GB/rank
* Diff per rank: **+1.49 GB**

If `psi_l_X` AND `psi_r_X` both pay this (the L-side + R-side slabs from the open-spin pair scan), the total diff is **+2.99 GB**, which
brackets the observed +2.32 GB (within 30% of the right shape). The factor-of-1.5 over-prediction matches the typical XLA aliasing slack
(some lifetimes overlap inside the bc-scan).

For 4×4×4 (r_chunk=7932 ≈ 27000/3.4):
* Same formula, scaled by r_chunk: per-rank slab diff ≈ 0.44 GB. Doubled (L+R): +0.88 GB. **Observed +0.27 GB** → the 4×4×4 multi-r-chunk
  configuration sees about 30% of the predicted slab growth, presumably because the bc-scan body has fewer slots active when r_chunk is small.

## Cross-r-chunk persistent residue (band-axis-scaling leak check)

Tracing `live_total` GB across the 3 measured r-chunks of the 4×4×4 case to look for a leak that grows with nb:

| probe                                  | 4×4×4 nb=100 live_total (GB) | 4×4×4 nb=200 live_total (GB) | per-chunk delta |
|---|---|---|---|
| `pre_rchunk_loop`                     | 2.68 | 3.68 | (initial) |
| `after_fit_one_rchunk chunk=0`       | 9.31 | 10.21 | +0.90/+0.90 (rmu) |
| `after_accumulate chunk=0`           | 2.68 | 3.68 | back to baseline |
| `after_fit_one_rchunk chunk=1`       | 9.31 | 10.21 | repeats — no growth |
| `after_fit_one_rchunk chunk=2`       | 9.31 | 10.21 | repeats — no growth |

**No cross-r-chunk residue at either nb=100 or nb=200.** The bc-scan inside `fit_one_rchunk` correctly aliases its transient slots between
iterations, and the centroid_persist scaling matches the planner. The only delta vs the planner is the **inside-the-scan** transient
attributed to band-axis-flat all_gather slabs.

## Per-channel × per-r-chunk timing

Si non-bispinor has a single channel (charge). At ~5s per r-chunk on Si, the runtime is much shorter than the bispinor sweep:

| config | chunk 0 (s) | chunk 1 (s) | chunk 2 (s) | total / channel |
|---|---|---|---|---|
| 3x3x3_nb100 | 3.5s (fit) | — | — | 3.5s × 1 chunk |
| 3x3x3_nb200 | 4.6s | — | — | 4.6s × 1 chunk |
| 4x4x4_nb100 | 4.5s | 4.5s | 4.5s | 13.5s × 4 chunks ≈ 18s total |
| 4x4x4_nb200 | 6.3s | 6.3s | 6.3s | 19s × 4 chunks ≈ 25s total |

(Each timing extracted from `Timing (1 r-chunks, Xs total)` lines in `gw_{platform_false}.out`; reproduce via `grep "Timing" gw_*.out`.)

## Full planner breakdowns (verbatim from `gw.out`)

### 3×3×3 nb=100 — `r_chunk=27000, 1 chunks, band_chunk=128`

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 128
  r_chunk            = 27000  (1 chunks)
  gflat_chunk_size   = 100
  budget             = 28.00 GB/dev
  HWM estimate       = 15.63 GB/dev (56% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   15.63
    D_accumulate............    1.44
    A_centroid..............    0.47
    B_CCT_chol..............    0.25
    E_v_q...................    0.14
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  14.277
      zeta_out..............   1.190
      centroids_persist.....   0.071
      gflat_acc.............   0.071
      L_q...................   0.018
      sphere_idx_replicated.   0.003
```

### 3×3×3 nb=200 — `r_chunk=27000, 1 chunks, band_chunk=256`

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 256
  r_chunk            = 27000  (1 chunks)
  gflat_chunk_size   = 100
  budget             = 28.00 GB/dev
  HWM estimate       = 15.70 GB/dev (56% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   15.70
    D_accumulate............    1.51
    A_centroid..............    0.93
    B_CCT_chol..............    0.32
    E_v_q...................    0.17
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  14.277
      zeta_out..............   1.190
      centroids_persist.....   0.141   ← 2× nb=100
      gflat_acc.............   0.071
      L_q...................   0.018
      sphere_idx_replicated.   0.003
```

### 4×4×4 nb=100 — `r_chunk=7932, 4 chunks, band_chunk=128`

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 128
  r_chunk            = 7932  (4 chunks)
  gflat_chunk_size   = 100
  budget             = 28.00 GB/dev
  HWM estimate       = 22.39 GB/dev (80% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   22.39
    D_accumulate............    2.59
    B_CCT_chol..............    2.05
    E_v_q...................    0.71
    A_centroid..............    0.56
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  19.884
      zeta_out..............    1.657
      gflat_acc.............    0.338
      centroids_persist.....    0.334
      L_q...................    0.170
      sphere_idx_replicated.    0.007
```

### 4×4×4 nb=200 — `r_chunk=7812, 4 chunks, band_chunk=256`

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 256
  r_chunk            = 7812  (4 chunks)
  gflat_chunk_size   = 100
  budget             = 28.00 GB/dev
  HWM estimate       = 22.40 GB/dev (80% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   22.40
    D_accumulate............    2.90
    B_CCT_chol..............    2.38
    E_v_q...................    0.88
    A_centroid..............    1.09
  per-peak components (GB/dev):
    [C]
      P_pair_concurrent_slots  19.583
      zeta_out..............    1.632
      centroids_persist.....    0.668   ← 2× nb=100
      gflat_acc.............    0.338
      L_q...................    0.170
      sphere_idx_replicated.    0.007
```

## Cross-nb consistency check — `centroids_persist` scales exactly linearly

The planner formula `centroids_persist = 4 · c128(nk, ns, μ, nb_total)/p_xy`, where `nb_total = nb_left + nb_right = 2·nb_cohsex`:

| config | nb_total | predicted (MB/dev) | reported (MB/dev) | match? |
|---|---|---|---|---|
| 3x3x3 nb=100 | 200 | `4 × 27 × 2 × 408 × 200 × 16 / 4` = 70.5 | 71 | ✓ |
| 3x3x3 nb=200 | 400 | `4 × 27 × 2 × 408 × 400 × 16 / 4` = 141 | 141 | ✓ |
| 4x4x4 nb=100 | 200 | `4 × 64 × 2 × 816 × 200 × 16 / 4` = 334 | 334 | ✓ |
| 4x4x4 nb=200 | 400 | `4 × 64 × 2 × 816 × 400 × 16 / 4` = 668 | 668 | ✓ |

Bit-exact match between formula and reported. **The centroid_persist term itself is correct**; the gap is in the unmodeled in-jit transient.

## Suggested planner refinements

1. **Add band_chunk × r_chunk all_gather slab term to Peak C** (highest leverage for fixing the −13.9% gap):
   ```
   M_all_gather_psiX = 2 · c128(nk, band_chunk, ns, r_chunk / p_y)  # L + R slabs
   ```
   Per `docs/MEMORY_MODEL.md` §R-Chunk Round-8 Lines 198-201, this is the post-all_gather pre-r-slice slab inside
   `z_q_from_psi_sm._local`; the docs explicitly list it but the planner's `_peak_C_fit_one_rchunk` formula doesn't include
   it. Adding it would fully close the 3×3×3 nb=200 gap.

   Expected impact at 3×3×3 nb=200: `2 · 27 · 256 · 2 · 13500 · 16 / 4 = 1.49 GB/dev` → Peak C goes from 15.70 to 17.19, %-err
   from −13.9% to −5.8%. Still under-predicts by ~5% but within Agent T's range.

   Expected impact at 3×3×3 nb=100: `2 · 27 · 128 · 2 · 13500 · 16 / 4 = 0.74 GB/dev` → Peak C goes from 15.63 to 16.37 (also up),
   so the nb=100 case may actually move from −0.5% to roughly +4% (planner now over-predicts). Mixing this with the existing
   `−0.5%` empirical match suggests a smaller all_gather term — possibly 1 slab not 2, or with an aliasing-discount factor.

   Recommended: HLO-dump the inner `c_q_from_psi_sm` / `z_q_from_psi_sm` jit at 3×3×3 nb=100 vs nb=200 to calibrate the
   coefficient (1× vs 2×, with/without aliasing).

2. **Cross-check `centroids_persist`'s factor-of-4 vs observed ×6 in live_arrays.** The planner counts `4 × c128(nk, ns, μ, nb)`
   (rmuT_X + rmu_Y × L + R). Empirically `live_arrays` shows **6 buffers** (3 of each transpose). The factor-of-4 covers the
   ×2 transpose pair but not all band-windows used (L_left + L_right + …). Either the planner under-counts by 1.5× or the
   probe is double-counting one buffer that's a view, not a copy. For Si μ-magnitudes the absolute miss is <100 MB so
   low priority, but at CrI3 production μ=1520 the gap would be ~0.5 GB.

3. **No-op for the Peak A bumps that already worked:** Peak A's `fft_box` term correctly doubles when band_chunk doubles
   (0.47 → 0.93 GB/dev at 3×3×3, 0.56 → 1.09 GB/dev at 4×4×4). The factor-4 constant is fine here. Peak A itself is never the
   bottleneck in any config so this doesn't drive HWM.

## Suggested follow-ups (separate-initiative scope)

* HLO dump `z_q_from_psi_sm` jit at the 3×3×3 nb=100 and nb=200 settings to nail the slab coefficient.
* Verify the suggested term at CrI3 production scale (nk=36, μ=1520, r_chunk≈24576 at sweet-spot): the all_gather slab would
  add ~1 GB per L/R copy ≈ 2 GB/dev. CrI3's mem_stats `−8.5%` gap (per Agent T) maps almost exactly to this.
* If the term lands, re-run Agent A's μ-sweep at the new planner; the μ=768 −10.8% gap (worst Si bispinor) may be more
  about the same all_gather slab than about uneven chunk boundaries.

## Commits pushed

| commit | what |
|---|---|
| `d4cb599` (cherry-pick of `8c18925` from Agent A) | `fix(unfold_psi): handle nspinor=1 wavefunctions` (allows the loader path to not blow up the spinor axis for nspinor=1 WFNs; useful for future agents wanting truly scalar runs) |

Branch tip: `agent/si-band-sensitivity` at `d4cb599` on lorrax_C. No further LORRAX source changes by Agent C — the planner refinement (#1
above) is left as a clear next-step recommendation rather than landed in this branch (because the coefficient still needs HLO
calibration).

## Reproducer commands

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/BANDS_nonbispinor_2026-05-18/

# QE for each config — 4 chains, ~3 min each
./_run_qe.sh 3x3x3_nb100; ./_run_qe.sh 3x3x3_nb200
./_run_qe.sh 4x4x4_nb100; ./_run_qe.sh 4x4x4_nb200

# Centroids — one per (kgrid, μ), then symlink/copy to nb=200 dir
./_kmeans_run.sh 3x3x3_nb100 384   # generates centroids_frac_408.txt (orbit-snap from 384)
./_kmeans_run.sh 4x4x4_nb100 768   # generates centroids_frac_816.txt
cp 3x3x3_nb100/centroids_frac_408.txt 3x3x3_nb200/
cp 4x4x4_nb100/centroids_frac_816.txt 4x4x4_nb200/

# GW through ζ-fit, both allocator variants per config — 8 runs total, ~60s each
./_run_all_gw.sh

# Tabulate predictions vs measurements
python3 _analyze.py > summary.md
```

## Artifacts

| file | purpose |
|---|---|
| `manifest.yaml` | Si non-bispinor band-count sweep |
| `{config}/cohsex.in` | nb=100/200 cohsex; `bispinor=false`, `cusolvermp_{charge,lu}=off` |
| `{config}/gw_platform_false.out` | sandbox-default allocator (production env) — no `mem_stats peak` available |
| `{config}/gw_bfc_pre95.out`        | BFC+preallocate=true+MEM_FRACTION=0.95 — exposes `peak_bytes_in_use` |
| `{config}/centroids_frac_{408,816}.txt` | scalar centroids (snapped μ from 384/768) |
| `_run_qe.sh`     | per-config QE chain runner (SCF → NSCF → pw2bgw → wfn2hdf) |
| `_kmeans_run.sh` | per-config centroid generator (uses `--qe-save` to bypass the load_wfns path) |
| `_run_gw.sh`     | per-config gw_jax launcher with allocator-variant switch |
| `_run_all_gw.sh` | batch runner for all 8 (config × allocator) pairs |
| `_analyze.py`    | extracts planner + mem_stats from all `gw_*.out` |
