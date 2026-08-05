# Agent G — Per-array census of `jax.live_arrays()` at every probe point

**Date:** 2026-05-17
**Source files:**
- `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/mem_probe_cs707.out`
- `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/mem_probe_cs1414.out`
- Probe code: `src/common/isdf_fitting.py:2354-2390` (commit `5c884ac`).
- Planner: `src/gw/gflat_memory_model.py` (4 `_peak_*` functions).

**Meta vars (CrI3 6×6 80Ry bispinor production):**
`nk=nq=nq_disk=36, ns=2 (charge) / 4 (transverse 4-density), nb_l=150, nb_r=160, mu_charge_pad=1520, mu_transv_pad=1504+pad-to-mesh, ngkmax=59990, n_rtot=1,125,000, fft_grid=(75,75,200), p_x=p_y=4, p_xy=16.`

NOTE: `live_arrays()` reports **global** logical shapes. For arrays sharded over `('x','y')` the **per-rank** byte count is the global value / 16. Replicated arrays (sphere indices, small tables) appear at full bytes on every rank.

NOTE: `nb_l=150` (bispinor band count) and `nb_r=160` (right-band count) differ — the (·,1520,150,·) shapes are ψ_l, the (·,1520,160,·) shapes are ψ_r. The 4 in trailing axis = `ns=4 four-density` (or it is `ns=2 spinor × 2 left/right` — same number, used here for bispinor four-channel ζ).

---

## 1. Charge channel (cs=707, lines 258–290 / cs=1414 lines 258–280) — IDENTICAL

cs=707 and cs=1414 probes are **bit-identical** at chunk=0 (confirmed by agent_f; verified again here). Both configs produce same `live_total`, same shapes, same counts. The cs=707 vs cs=1414 difference lives **only** inside the accumulate-jit scratch, never in the Python-tracked live set. → A single charge-channel column below covers both configs at chunk=0.

### Probe A: `rchunk_start chunk=0`   (cs=707/1414 charge, line 258)

`live_count=67, live_total=58.72 GB (global)`

| # | Shape (global) | dtype | n | global GB | per-rank GB (p=16) | meta-var formula | lifetime class | planner peak | counted? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | (36,1520,59990) | c128 | 1 | 52.52 | 3.28 (μ-sharded) | `nq_disk × mu_pad_charge × ngkmax × 16` | persistent_throughout_zeta_fit | D | YES (`D.gflat_acc`) |
| 2 | (36,1520,1520) | c128 | 1 |  1.33 | 0.083 (μ-sharded) | `nq × mu_pad × mu_pad × 16` | persistent_throughout_zeta_fit | C, D | YES (`C.L_q`, `D.L_q`) |
| 3 | (36,1520,160,4) | c128 | 2 |  1.12 | 0.070 (μ-sharded) | `2 × nq × mu_pad × nb_r × ns × 16` | persistent_throughout_zeta_fit (charge ψ_r centroids + transpose) | C, D | partially — `D.centroids_persist` uses `2*nk*ns*mu*nb_total` (counts once, ignores ×2 transpose copy and uses nb=150 not 160) |
| 4 | (36,160,4,1520) | c128 | 2 |  1.12 | 0.070 (μ-sharded) | same as #3 (transposed view, separate physical buffer) | persistent_throughout_zeta_fit | C, D | NO (transpose copy not in planner) |
| 5 | (36,1520,150,4) | c128 | 2 |  1.05 | 0.066 (μ-sharded) | `2 × nq × mu_pad × nb_l × ns × 16` (ψ_l centroids + transpose) | persistent_throughout_zeta_fit | C, D | partially — same comment as #3 |
| 6 | (36,150,4,1520) | c128 | 2 |  1.05 | 0.066 (μ-sharded) | same as #5 (transposed view) | persistent_throughout_zeta_fit | C, D | NO |
| 7 | (36,75,75,200) | i32  | 3 |  0.49 | 0.49 (REPLICATED) | `3 × nq × fft_grid × 4` — sphere/phase indices | persistent_throughout_zeta_fit | none | **NO — UNACCOUNTED** |
| 8 | (36,59990) | c128 | 1 |  0.034 | 0.002 (μ-sharded?)/0.034 (rep) | `nq × ngkmax × 16` — qG phase / g-vector | persistent | none | small, ignored |
| 9 | (36,1508) | c128 | 6 |  0.005 | small | `6 × nq × n_mu_unpad × 16` — norms_l/r + diagonals | persistent | none | small, ignored |
| 10 | (1508,3) | i32 | 2 | <0.001 | <0.001 | `2 × n_mu_unpad × 3 × 4` — μ→r coord table | persistent | none | small, ignored |

### Probe B: `after_fit_one_rchunk chunk=0`   (cs=707/1414 charge, line 270)

`live_count=68, live_total=77.31 GB`  →  **+18.59 GB vs A**

| Delta | Shape | dtype | n | global GB | per-rank GB | formula | lifetime class | planner peak | counted? |
|---|---|---|---|---|---|---|---|---|---|
| +1 NEW | (36,1520,21232) | c128 | 1 | 18.59 | 1.16 (μ-sharded) | `nq_disk × mu_pad × r_chunk × 16` | fit_one_rchunk output, alive until accumulate consumes | D | YES (`D.zeta_chunk`) |

**All other arrays unchanged from probe A.** Specifically, centroids (#3–#6) and L_q (#2) are **NOT freed** between fit_one_rchunk and accumulate — confirmed at runtime.

### Probe C: `after_accumulate chunk=0`   (cs=707/1414, line 281)

`live_count=76, live_total=58.74 GB`  →  back to ~A baseline, +1 small new entry (sphere idx for gflat).

| Delta | Shape | dtype | n | global GB | per-rank GB | formula | lifetime class | planner peak | counted? |
|---|---|---|---|---|---|---|---|---|---|
| −1 | (36,1520,21232) | c128 | 1 | −18.59 | — | zeta_chunk donated/freed | — | D | YES (transient correctly drops out) |
| +2 NEW small | (36,59990) | i32 | 2 | 0.017 | 0.017 | gflat sphere idx (per-q g-index) | persistent_within_zeta_fit after accumulate touches it | D | NO (small, ignored) |
| live_count jump 68→76 | — | — | +8 | — | — | small donated buffer fragments (8 fragments) accumulate (likely small constants in shard_map closure) | cross_jit_leaked | D | small, OK to ignore |

The `live_count` jumps from 68 to 76 with **only 17 MB of new bytes**, so 8 small constants leaked from the accumulate jit but at negligible bytes.

---

## 2. Charge channel chunk=1 — clean steady state

cs=707 chunk=1 (lines 295–328): identical pattern to chunk=0. `rchunk_start` = 58.74 GB (== after_accumulate chunk=0). No new arrays appear between chunks. The 2 sphere idx leftovers from accumulate are stable. **Conclusion: no inter-chunk leak inside one channel.**

---

## 3. Transverse channel #1 (μ=1504, cs=707 starting at line 584)

`live_count=96 → 105` (vs 76 at end of charge channel).
`live_total=59.72 GB` at rchunk_start chunk=0 of transverse.

**New top-10 entries vs end-of-charge:**

| Shape | dtype | n | global GB | per-rank GB | formula | lifetime class | planner peak | counted? |
|---|---|---|---|---|---|---|---|---|
| (36,1504,59990) | c128 | 1 | 51.97 | 3.25 (μ-sharded) | `nq_disk × mu_pad_transv × ngkmax × 16` | persistent for THIS channel | D | YES (`D.gflat_acc`, but mu argument is the charge value 1520) |
| (36,1504,1504) | c128 | 1 |  1.30 | 0.081 (μ-sharded) | `nq × mu_transv × mu_transv × 16` | persistent for this channel | C, D | YES (L_q rebuilt for transv) |
| (36,1504,160,4) | c128 | 2 |  1.11 | 0.069 | same as charge #3 but μ_transv | persistent | C, D | partial |
| (36,160,4,1504) | c128 | 2 |  1.11 | 0.069 | transpose view | persistent | C, D | NO |
| (36,1504,150,4) | c128 | 2 |  1.04 | 0.065 | ψ_l transv | persistent | C, D | partial |
| (36,150,4,1504) | c128 | 2 |  1.04 | 0.065 | transpose view | persistent | C, D | NO |
| (36,75,75,200) | i32 | **6** | **0.97** | **0.97 (rep)** | `6 × nq × fft_grid × 4` — 3 sphere idx leaked from charge + 3 NEW for transv | **cross_jit_leaked between channels** | **none** | **NO — UNACCOUNTED & GROWING** |
| (36,1520,160,4) | c128 | 1 | 0.56 | 0.035 | leftover **charge** ψ_r centroid (one copy) | cross_jit_leaked between channels | none | **NO — UNACCOUNTED** |
| (36,160,4,1520) | c128 | 1 | 0.56 | 0.035 | leftover charge transpose | cross_jit_leaked | none | **NO** |

### Transverse `after_fit_one_rchunk chunk=0` (line 596)

Adds `(36,1504,21232) c128 = 18.39 GB` zeta_chunk transverse — analogous to charge.

### Transverse `after_accumulate chunk=0` (line 607)

Adds `(36,59990) i32 × 4 = 0.034 GB` (gflat sphere idx — now **4 copies**, was 2 after charge). Another small per-channel leak.

---

## 4. Across channels — leak growth (the new finding)

Comparing the start of each subsequent channel in cs=707:

| Channel | line | live_count | live_total (GB) | sphere(36,75,75,200) i32 count | small gflat sphere(36,59990) i32 count |
|---|---|---|---|---|---|
| charge ch=0 start | 258 |  67 | 58.72 | 3 | 0 |
| transv1 ch=0 start | 584 |  96 | 59.72 | 6 | 0 |
| transv2 ch=0 start | 906 | 106 | 59.90 | 7 | 4 |
| transv3 ch=0 start | 1228 | 108 | 60.07 | 8 | 4 |

**Each channel boundary leaks 1–3 additional `(36,75,75,200) i32` sphere indices** (≈ 162 MB each), plus charge ψ centroids (`(36,1520,160,4) c128 = 0.56 GB` and its transpose, never freed when transverse channel runs).

The leak between charge → transv1 includes the 2× 0.56 GB charge ψ_r centroid leftovers (1.12 GB global, 70 MB/rank). **These persist for the rest of the run.**

By transv3, the **between-channel persistent footprint is +1.35 GB (rep sphere idx) + 1.12 GB (μ-sharded charge ψ leftover) above the charge-only baseline** — small per-rank (~84 MB/rank on the sphere idx + ~70 MB/rank on the ψ leftover) but it goes onto every channel's Peak D base.

---

## 5. Per-lifetime tally

### A. `persistent_throughout_zeta_fit` (alive at all 3 probe points in one channel)
- gflat_acc — counted in D
- L_q — counted in C and D
- ψ_l centroids ×2 (rmuT_X + Y transpose) — counted as `centroids_persist` once; the second buffer (transpose) is **NOT counted**
- ψ_r centroids ×2 — same; **transpose NOT counted**
- sphere/phase indices `(36,75,75,200) i32 × 3` — **UNCOUNTED, 0.49 GB global / 0.49 GB per-rank (replicated)**

### B. `persistent_within_rchunk` only
None besides A's set: nothing dies at rchunk_start that was alive at after_accumulate.

### C. `fit_one_rchunk_transient` (alive at after_fit, gone at after_accumulate)
- zeta_chunk `(36,1520,21232) c128` — counted in D as `D.zeta_chunk`

### D. `accumulate_transient` (allocated and freed inside accumulate jit — not visible in live_arrays)
- `accumulate_fft_box` (~25.45 GB/rank at cs=1414) — counted via `D.accumulate_fft_box` with `factor_D=2.0`
- **cuFFT plan-scratch** (~23.7 GB ask at cs=1414) — **NOT modeled**. This is what kills cs=1414.

### E. `cross_jit_leaked` (alive at probes where it shouldn't be)
- Inter-channel: sphere `(36,75,75,200) i32` grows from 3→6→7→8 copies across channels (+0.16 GB/channel global, replicated so all 0.16 GB/rank)
- Inter-channel: charge ψ_r centroid copies leftover when transverse runs: `(36,1520,160,4)` and transpose (0.56 GB ×2 global = 1.12 GB, 70 MB/rank)
- Inter-chunk inside one channel: ~8 small constants (~17 MB total) per channel — negligible

---

## 6. Currently UNACCOUNTED ≥100 MB arrays — ranked

| Rank | Array | Lifetime | Global / per-rank | Where it should go in planner |
|---|---|---|---|---|
| 1 | **ψ centroid TRANSPOSE copies** — `(36,1520,160,4) c128 × 2` + `(36,1520,150,4) c128 × 2` (4 transpose buffers in addition to the 4 rmuT_X buffers) | persistent_throughout_zeta_fit | 4.34 GB / 0.27 GB | Double the `centroids_persist` coefficient in `_peak_C` and `_peak_D` (currently `2 × …`, should be `4 × …`). Or factor out a `n_centroid_copies=4` literal. Use the actual band counts: ψ_l uses nb_l, ψ_r uses nb_r — **planner uses single `nb_total` which is the LEFT count**. |
| 2 | **FFT sphere indices `(36,75,75,200) i32`** — replicated, count grows from 3 (1st channel) up to 8 (4th channel) | persistent | 0.49→1.30 GB global; **0.49→1.30 GB per-rank** (replicated, NOT divided by 16) | NEW term in Peak C and D: `n_sphere_copies × nq × fft_grid_x × fft_grid_y × fft_grid_z × 4`. With observed leak: `n_sphere_copies = 3·channel_idx` (worst-case 8 by 4th channel). |
| 3 | **Leftover charge ψ centroids during transverse channels** — `(36,1520,160,4) c128 + transpose` | cross_jit_leaked between channels | 1.12 GB / 0.07 GB | EITHER fix the leak (delete charge centroids before transverse starts) OR add `D.cross_channel_leftover_psi = 2 × nq × ns × mu_pad_charge × nb_r × ns × 16 / p_xy` to all transverse-channel Peak D estimates. |
| 4 | **cuFFT plan-scratch growth with `gflat_chunk_size`** — invisible to live_arrays; observed 23.7 GB ask at cs=1414 vs `factor_D × box = 2 × 25.45 = 50.9 GB` already counted, so the *additional* unaccounted plan-workspace is the gap that fails to fit when fewer than 80−50.9−5.3 = 23.8 GB remain | accumulate_transient | up to 23.7 GB per-rank at cs=1414 | NEW term `D.cufft_plan_overhead = cufft_overhead_factor × gflat_chunk_size × n_rtot × 16` with empirical `cufft_overhead_factor ≈ 1.0` (matches the 25.45 GB observed plan ask vs the 25.45 GB box). |
| 5 | **Replicated sphere `(36,75,75,200) i32`** itself, even at minimum count 3 — current planner has zero sphere-index accounting | persistent_throughout_zeta_fit | 0.49 GB / **0.49 GB per-rank** (replicated!) | Add to Peak A/B/C/D persistent: `n_q × prod(fft_grid) × 4`. Because it's replicated, the /p_xy is absent — this is a sneaky 0.5 GB/rank that gets bigger at finer grids. |

---

## 7. Currently-counted vs reality at the production CrI3 80Ry config

Computing each peak with the corrected centroid coefficient (×4 not ×2) and adding the sphere-idx, charge-leftover, and cufft-plan terms (per-rank, after /p_xy where μ-sharded):

| Peak | Current planner (per-rank, GB) | Corrected (per-rank, GB) | Delta | Drives bottleneck? |
|---|---|---|---|---|
| A — centroid_load (pre-loop)        | ~0.5–1 | ~1 (sphere idx +0.5) | +0.5 | no |
| B — CCT+Chol (pre-loop)             | ~5 (P_l+P_r open-spin term dominates) | ~5.5 | +0.5 | no |
| C — fit_one_rchunk                  | 4.0 (centroids 0.27 + L_q 0.08 + 3 × P-pair) | 4.5 (×2 centroid bug fix +0.27 + sphere +0.49) | +0.8 | YES at production picks |
| D — accumulate at cs=707            | 4.7 (gflat 3.28 + zeta 1.16 + box 0.36 + ε) | 5.2 (+0.27 centroid + 0.49 sphere) | +0.8 | no (cs=707 well under budget) |
| D — accumulate at cs=1414           | 56.0 (per agent_f) | 56.5 (centroid+sphere fixes) — **+up to 23.7 GB cuFFT plan** | up to +24 | **YES — cause of OOM** |

So the corrected Peak D at cs=1414 should be **~80 GB/rank, not 56** — which matches the observed cuFFT plan-creation failure. The single missing modeled term that *explains the OOM* is the cuFFT plan-scratch (gap #4 above), with the centroid-transpose and sphere-idx terms being smaller systematic corrections (gaps #1, #2, #3).

---

## 8. Summary — corrected Peak A/B/C/D formulae

```
Peak A (per-rank, GB):
  centroid_out_filling:     nk*ns*mu*nb_per_load*16 / p_xy / 1e9
  phase_table:              nk*n_rtot*16 / 1e9                  # replicated
  fft_box:                  band_chunk*ns*n_rtot*16*fft_box_factor_A / 1e9
+ sphere_idx_replicated:    3*nq*prod(fft_grid)*4 / 1e9          # NEW

Peak B (per-rank, GB):
  centroids_persistent:     4*nk*ns*mu*nb_used*16 / p_xy / 1e9   # was 2, transpose copies make it 4
  P_l_plus_P_r_open_spin:   2*nk*ns*ns*mu*mu*16 / p_xy / 1e9
  C_q, L_q
+ sphere_idx_replicated:    3*nq*prod(fft_grid)*4 / 1e9          # NEW

Peak C (per-rank, GB):
  centroids_persist:        4*nk*ns*mu*nb_used*16 / p_xy / 1e9   # was 2
  L_q:                      nq*mu*mu*16 / p_xy / 1e9
  P_pair_concurrent_slots:  slots * nk*ns*ns*mu*r_chunk*16 / p_xy / 1e9
  zeta_out:                 nq*mu*r_chunk*16 / p / 1e9
+ sphere_idx_replicated:    3*nq*prod(fft_grid)*4 / 1e9          # NEW

Peak D (per-rank, GB):
  centroids_persist:        4*nk*ns*mu*nb_used*16 / p_xy / 1e9   # was 2
  L_q:                      nq*mu*mu*16 / p_xy / 1e9
  gflat_acc:                nq_disk*mu*ngkmax*16 / p_xy / 1e9
  zeta_chunk:               nq_disk*mu*r_chunk*16 / p_xy / 1e9
  accumulate_fft_box:       gflat_chunk_size*n_rtot*16*fft_box_factor_D / 1e9
+ cufft_plan_workspace:     cufft_factor*gflat_chunk_size*n_rtot*16 / 1e9   # NEW; cufft_factor≈1.0
+ sphere_idx_replicated:    n_sphere_copies(channel)*nq*prod(fft_grid)*4 / 1e9   # NEW; ≥3, leaks by +1..3/channel
+ leftover_psi_charge:      2*nq*ns*mu_pad_charge*nb_r*ns*16 / p_xy / 1e9  # only transverse channels; or fix the leak
```

**Most-likely "single fix that closes the cs=1414 OOM":** add the cuFFT plan-workspace term (#4). The other gaps (transpose copies, sphere idx, channel leftover ψ) are real but each ≤1 GB/rank — they bias the planner toward over-aggression but don't on their own explain the 24 GB cuFFT failure.

---

## 9. Caveats / unresolved decompositions

1. The "x 2" multiplicity on `(36,1520,160,4)` and `(36,160,4,1520)` is read literally from `live_arrays()`. I interpret one of each pair as the rmuT_X-ordered centroid and the other as a transpose view, but I have not opened `wfn_transforms.py` to confirm — this should be verified before patching the planner formula.
2. The `(36,1508) c128 × 6` block is interpreted as 6 small (n_q × n_mu_unpad) tables — likely some combination of norms_l/r, diag(L_q), and quotient factors. ≤6 MB total, ignored.
3. The "sphere x N" count grows monotonically and is **replicated** (per probe `live_arrays()` reporting **global** shape but the index tensor itself is per-rank for replicated arrays — every rank holds 0.49 GB worth, not 0.49/16). This is the largest single per-rank surprise.
4. Channel-boundary cross-jit leaks (the ψ_r charge centroid leftovers visible during transverse) are likely fixable by an explicit `del psi_l_rmuT_X_fit, psi_r_rmuT_X_fit; jax.block_until_ready(...)` in the channel-switch glue — recommend the downstream refit agent file an issue against `isdf_fitting.py` for this rather than building the leak into the planner.

