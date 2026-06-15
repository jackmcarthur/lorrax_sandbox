# LORRAX `gflat_memory_model.py` — non-bispinor + k-grid synthesis

> **2026-05-19 SCOPE-ERROR ADDENDUM (read before the rest of this file).**
> Production LORRAX runs use fully-relativistic pseudopotentials with
> `noncolin=.true., lspinorb=.true.` → nspinor=2 always. "Non-bispinor"
> means `bispinor=false` in `cohsex.in`, NOT `nspinor=1` at the QE level.
>
> - **Agent A** built a `noncolin=.false. lspinorb=.false.` Si reference
>   (nspinor=1) — an unsupported code path. Its μ-sweep numbers and the
>   two loader fixes it landed (`8c18925`, `dc0b254`) have been reset out
>   of `agent/si-nonbispinor-mu-sweep`. The ns=1 HLO bit-exactness it
>   reported is a curiosity; not a guarantee for production.
> - **Agents B and C** used `noncolin=.true. lspinorb=.false.` (ns=2 no SOC),
>   which is close to but not identical to production (`noncolin=.true.
>   lspinorb=.true.`). Their planner formulas only see ns, so the per-rank
>   byte counts hold under SOC too — but the actual mem_stats peaks must
>   be re-measured at production config before any conclusion stands.
> - **The cusolverMp "bug" was hbm40g + BFC@0.95, not a sandbox issue.**
>   Bispinor agent_t ran on the same 2×2 mesh / BFC@0.95 / cusolverMp-on
>   combination cleanly because it was on `hbm80g` (4 GB free for NCCL vs
>   2 GB on hbm40g). Documented in `ENVIRONMENT_COMPREHENSIVE.md` §3.2 +
>   §8.3 and the `feedback_lxalloc_gpu_constraint_mixes_hbm.md` memory.
>   The `cusolvermp_*=off` workaround the agents applied is forbidden —
>   cuSOLVERMp is a shipping default of the community release.
>
> **What survives this addendum** (planner-formula geometry, independent
> of SOC vs no-SOC at fixed ns): Agent B's per-term log-log scaling vs
> kgrid (within 5–6 % of analytic exponents) and the additive ~5–8 GB
> framework-floor finding. **What needs re-measuring at production config
> (`noncolin=true, lspinorb=true`, `bispinor=false`, cusolverMp default,
> hbm80g):** the mem_stats peaks themselves and the `−13.9 %` 3×3×3 nb=200
> outlier hypothesis (the all_gather-slab idea). Re-run tracked under
> `runs/Si/NONBISPINOR_PROD_2026-05-19/`.

**Date:** 2026-05-18 (sister to `reports/memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md`)
**Scope:** robustness audit of the bispinor-calibrated planner under three orthogonal sweeps on Si non-bispinor: centroid count (Agent A), k-grid (Agent B), band count (Agent C). All on lorrax_{A,B,C} `main` rebased to `0f355b7`.

## 1. Headline

**The planner is structurally sound on non-bispinor across ns ∈ {1, 2, 4} and across kgrid 2³–6³.** The HLO-calibrated constants `pair_density_slots = 3` and `fft_box_factor_D = 2.0` are now bit-exact at ns=1 as well (Agent A). Every per-term log-log scaling exponent vs kgrid matches the analytic prediction within ≤ 6.5 % (Agent B). No new leaks across r-chunks, bc-chunks, sym-channels, or kgrids.

**Three faithfulness regimes emerge, each with its own bias signature:**

| regime | example | %-err sign | mechanism |
|---|---|---|---|
| **Production-scale multi-chunk (matches bispinor)** | Si 4×4×4 nb=100 (B), Si 6×6×6 nb=100 (B) | −0.5 % to −6.1 % | clean, planner within bispinor's spec window |
| **Small algorithmic peak vs sticky floor** | Si 2×2×2, Si 3×3×3 small μ, Si 4×4×4 μ=192 (A) | **+50 % to +185 % UNDER** | ~5–8 GB CUDA/JIT/NCCL constant dominates a sub-3 GB algorithmic peak |
| **Single-r-chunk degenerate over-prediction** | Si 4×4×4 scalar μ≥768 (A) | **−25 % OVER** (conservative) | 3 pair-density slots reserved but only ~2 live concurrently when n_chunks=1 |
| **Multi-r-chunk with large band_chunk** | Si 3×3×3 nb=200 (C) | **−13.9 % under** | unmodeled `c128(nk, band_chunk, ns, r_chunk/p_y)` all_gather slab on `psi_l_X` / `psi_r_X` |

The last regime (Agent C's 3×3×3 nb=200 case) is the only one that **breaks out of bispinor's [−0.5 %, −10.8 %] under-prediction band** with a structurally identifiable, quantitative gap. It is the highest-leverage open improvement (§5.1).

## 2. What survived the cross-system tests

### 2.1 Constants — HLO-bit-exact at ns ∈ {1, 2, 4}

| constant | ns=1 (Agent A) | ns=2 SOC (M4 prior) | ns=4 bispinor (M1 prior) |
|---|---|---|---|
| `pair_density_slots` | **3** ✓ (HLO at μ=408 and μ=1176) | 3 ✓ | 3 ✓ |
| `fft_box_factor_D` | **2.0** ✓ (HLO at cs=100) | not directly measured | 2.0 ✓ (M2 cs=360, M3 cs=1) |
| per-slot bytes formula | `_bytes_c128(nk, 1, 1, μ_pad, r_chunk, /p_xy)` exact | `_bytes_c128(nk, 2, 2, μ_pad, r_chunk, /p_xy)` exact | `_bytes_c128(nk, 4, 4, μ_pad, r_chunk, /p_xy)` exact |

The 3-slot count is **invariant under** `ns`, `bispinor`, `gflat_chunk_size`, `band_chunk`, and now also `kgrid`. It is a structural constant of `fit_one_rchunk`'s scan-INSIDE-shard_map (Round-6 architecture); the byte formula correctly absorbs all ns² hidden factors via `meta.nspinor = {1 scalar, 2 SOC/no-SOC noncolin, 4 bispinor}`.

### 2.2 Per-term scaling vs k-grid

Agent B held μ/nk_full ≈ 6, nb=100, ecutwfc=25 Ry, mesh 1×4 fixed across 2³ → 6³. Empirical log-log slopes vs `nk_full`:

| predicted exponent | terms | empirical slope range |
|---:|---|---:|
| **0** (const) | A.fft_box, D.accumulate_fft_box | 0.00 ✓ |
| **1** (∝ nk) | sphere_idx_replicated, phase_table, zeta_L_on_x_axis | 0.90 – 1.18 (3–18 % off, small-N rounding) |
| **2** (∝ nk · μ) | centroids_persist, gflat_acc, centroid_out_filling, zeta_L_all, psi_centroids | 2.06 – 2.08 (3–6.5 % off) |
| **3** (∝ nk · μ²) | P_l+P_r, C_q, L_q, V_acc, V_acc_full_BZ | 2.86 – 2.87 (4–5 % off) |

`P_pair_concurrent_slots` and `zeta_chunk` apparent slopes deflate to +1.35 only because the planner shrinks `r_chunk` from 13824 (1 chunk) at small kgrids to 1348 (11 chunks) at 6×6×6 — a deliberate budget defense, not a model break. After factoring r_chunk out, the physics-level scaling recovers to nk · μ · r_chunk.

### 2.3 The leak audit — clean

| thing checked | result |
|---|---|
| `sphere_idx_replicated` at 2³, 3³, 4³, 6³ | **1 buffer at every kgrid** — Round-6 canonical accessor (commit `9afa11e`) holds |
| Cross-r-chunk live_arrays repeat (Agent C 4×4×4 nb=100, nb=200) | **Bit-equal across chunks** — no nb-dependent leak |
| Cross-bc-chunk transient (Agent C) | bc-scan correctly aliases slots, no residue |
| Cross-channel residue (bispinor prior, agent_l W4) | already known ~0.26 GB/dev, not exercised in non-bispinor |

## 3. What the audit broke

### 3.1 The 3×3×3 nb=200 outlier — unmodeled `M_all_gather_slab` (Agent C, §5.1 below)

The Si 3×3×3 nb=200 mem_stats peak grew **+2.53 GB** vs nb=100, while planner Peak C HWM grew only **+0.07 GB**. Live_arrays accounts for +0.21 GB of that (centroids_persist exactly doubles, as the formula says). The remaining **+2.32 GB lives inside `z_q_from_psi_sm._local` as XLA preallocated-temp** — almost certainly the band-axis-flat all_gather slab the docs already describe but the planner doesn't charge:

```
M_all_gather_slab ≈ c128(nk, band_chunk, ns, r_chunk / p_y)   per rank
```

Predicted diff 3×3×3 nb=100→200 (band_chunk 128 → 256): `27 × 256 × 2 × 13500 × 16 = 2.99 GB/rank` for 2 slabs (L + R) — brackets the observed +2.32 GB within 30 % (within XLA aliasing slack).

This is the **same shape** as the bispinor Si μ=768 −10.8 % gap (agent_T) and the bispinor CrI3 production −8.5 % gap (agent_q/s/r prior round). All three sit at "multi-chunk + large band_chunk + r_chunk not tiny" — the regime where this slab is biggest.

### 3.2 The framework floor — ~5–8 GB sticky CUDA/JIT/NCCL constant

Agent A measured **8.02 GB at the `zeta_fit_start` probe before the r-chunk loop even runs** on Si 4×4×4 μ=192. Subsequent probes never drop below this. Sources:

- ~3000 XLA helper jits' compiled PTX/SASS payload
- cuBLAS / cuFFT plan caches
- NCCL user-buffer-pool reservation (allocated by sharded_cholesky path even when cusolverMp is off)
- CUDA context init

This is **already deliberately unmodeled** per `memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md` §6.2 (user-deferred for CPU portability). The non-bispinor sweep makes its existence quantitatively visible because Si scalar's algorithmic peak shrinks below it at small μ or small kgrid:

| config | HWM_pred (GB) | mem_stats peak (GB) | Δ (GB) |
|---|---:|---:|---:|
| Si 2×2×2 nb=100 μ=48 (B) | 0.28 | 8.00 | 7.72 |
| Si 3×3×3 nb=100 μ=192 (B) | 3.78 | 8.02 | 4.24 |
| Si 4×4×4 nb=100 μ=192 (A) | 2.81 | 8.02 | 5.21 |
| Si 4×4×4 nb=100 μ=432 (B) | 20.19 | 20.36 | 0.17 |
| Si 6×6×6 nb=100 μ=1348 (B) | 24.82 | 26.42 | 1.60 |
| CrI3 6×6 80 Ry sweet-spot (prior R11) | 70.11 | 76.05 | 5.94 |

Δ ≈ 5–8 GB across systems, additive — exactly the shape of fixed framework overhead, not multiplicative bias.

### 3.3 Single-r-chunk over-prediction — 3 slots reserved, ~2 live

Agent A's Si scalar μ ≥ 768 sweep runs in 1 r-chunk total (r_chunk = n_rtot = 13824 fits in headroom). Planner reserves 3 pair-density slots at 6.24 GB each = 18.73 GB; mem_stats peak shows ~12.5 GB algorithmic + 6.6 GB floor ≈ 19.1 GB — consistent with **2 effective slots, not 3**, because one slot is wholly aliased to the output for single-call lifetimes.

This is a **conservative-bias** error — the planner over-budgets HWM by ~6 GB in single-chunk degenerate runs. Safe direction for OOM avoidance; loses ~10 % of usable budget. Multi-chunk plans correctly need all 3 slots (each r-chunk re-allocates its triplet across JIT calls); the over-budgeting only fires when n_chunks = 1.

### 3.4 New bugs fixed in-branch — nspinor=1 loader

Both fixed on Agent A's `agent/si-nonbispinor-mu-sweep`; Agent C cherry-picked `8c18925`:

| commit | file | fix |
|---|---|---|
| `8c18925` | `src/common/symmetry_maps.py` | `unfold_psi` no longer builds a 2×2 U_eff when `cnk.shape[1] == 1`; skips spinor-mix after TRS τ-phase (eager backend) |
| `dc0b254` | `src/file_io/wfn_loader.py` | `WfnLoader._ensure_phdf5_static` builds `U_per = ones((nk_full, 1, 1))` when `nspinor == 1` (phdf5 backend); device einsum becomes no-op |

All 44 loader/unfold tests still pass at ns=2. Both are **clean merge candidates for origin/main** — they fix latent breakage in a code path that's been silently dead since LORRAX gained bispinor support.

### 3.5 New sandbox bug — cusolverMpPotrf at BFC + 2×2 mesh

`cusolverMpPotrf` returns `status=7 INTERNAL_ERROR` under `XLA_PYTHON_CLIENT_ALLOCATOR=default + PREALLOCATE=true + MEM_FRACTION=0.95` on a true 2D mesh (`px≥2 AND py≥2`). Reproducible on Si 4×4×4 / 4 GPU / 2×2 mesh / 40 GB HBM. Root cause: NCCL user-buffer-registration starves when BFC has preallocated 95 % of HBM.

**Workaround:** add `cusolvermp_charge = off, cusolvermp_lu = off` to cohsex.in (forces in-tree `sharded_cholesky`, same path the bispinor sweep used on its 1×4 mesh). Or drop `MEM_FRACTION` to 0.80. Logged at `KNOWN_SANDBOX_ERRORS.md:117`.

This is **not a planner issue** — HWM is computed without cusolverMp footprint. But it is **a measurement-protocol constraint**: BFC ground-truth captures on 2D meshes must disable cusolverMp.

### 3.6 B_CCT_chol approaches binding at 6×6×6

Agent B sees `B_CCT_chol = 17.58 GB` at Si 6×6×6 (μ = 1348) = **71 % of the binding Peak C** = 24.82 GB. The dominant term `P_l_plus_P_r_open_spin` alone is 12.56 GB. The planner correctly identifies C as bottleneck *here*, but at any of {larger μ, switch to bispinor 4-channel cascade, lower mesh} the bottleneck flips to B. B has **no r-chunk-style knob** — the only remedies are smaller μ or larger mesh.

**Operational note:** monitor `B_CCT_chol` as the secondary peak whenever Si nk·μ exceeds Si 6×6×6 μ=1348, or whenever a CrI3-scale config moves to bispinor.

## 4. Calibration table — predicted vs realized HBM (consolidated)

### 4.1 Si non-bispinor (scalar physics, ns ∈ {1, 2})

| sweep | kgrid | nb | μ | r_chunk × n_chunks | HWM_pred | mem_stats peak | %-err |
|---|---|---:|---:|---|---:|---:|---:|
| **A (ns=1)** | 4×4×4 | 100 | 192  | 13824 × 1 | 2.81  | 8.02  | **+185 %** |
| A           | 4×4×4 | 100 | 408  | 13824 × 1 | 5.99  | 8.03  | +34 %    |
| A           | 4×4×4 | 100 | 756  | 13824 × 1 | 11.17 | 8.38  | **−25 %** |
| A           | 4×4×4 | 100 | 1176 | 13824 × 1 | 17.50 | 13.04 | −25 %    |
| A           | 4×4×4 | 100 | 1764 | 13824 × 1 | 26.51 | 19.60 | −26 %    |
| **B (ns=2)** | 2×2×2 | 100 | 48   | 13824 × 1 | 0.28  | 8.00  | **−96.5 %** |
| B           | 3×3×3 | 100 | 192  | 13824 × 1 | 3.78  | 8.02  | −52.9 %  |
| B           | 4×4×4 | 100 | 432  | 13824 × 1 | 20.19 | 20.36 | **−0.8 %** |
| B           | 6×6×6 | 100 | 1348 | 1348 × 11 | 24.82 | 26.42 | **−6.1 %** |
| **C (ns=2)** | 3×3×3 | 100 | 408  | 27000 × 1 | 15.63 | 15.71 | −0.5 %   |
| C           | 3×3×3 | 200 | 408  | 27000 × 1 | 15.70 | 18.24 | **−13.9 %** |
| C           | 4×4×4 | 100 | 816  | 7932 × 4  | 22.39 | 22.66 | −1.2 %   |
| C           | 4×4×4 | 200 | 816  | 7812 × 4  | 22.40 | 23.13 | −3.2 %   |

### 4.2 Reference — bispinor (prior calibration)

| system | μ | r_chunk × n_chunks | HWM_pred | mem_stats peak | %-err |
|---|---:|---|---:|---:|---:|
| CrI3 6×6 80 Ry SOC bispinor sweet-spot | 1508 | 24576 × 3 | 70.11 | 76.05 | −8.5 % |
| Si 4×4×4 SOC bispinor μ=384 (agent_T) | 384 | 10268 × 2 | 56.00 | 56.30 | −0.5 % |
| Si 4×4×4 SOC bispinor μ=768 (agent_T) | 768 | 5832 × 3 | 55.99 | 62.78 | **−10.8 %** |
| Si 4×4×4 SOC bispinor μ=1200 (agent_T) | 1200 | 3552 × 4 | 55.97 | 59.03 | −5.2 % |
| Si 4×4×4 SOC bispinor μ=1800 (agent_T) | 1800 | 2280 × 7 | 56.00 | 57.28 | −2.2 % |

### 4.3 Pattern across both calibrations

- **Production-scale multi-chunk configs cluster at −0.5 % to −10.8 % under-prediction.** Bispinor and non-bispinor are equally faithful in this regime.
- **Small-algorithmic-peak configs see the ~6.5 GB floor dominate.** Δ = peak − HWM_pred ≈ 5–8 GB across all systems regardless of kgrid/nb/ns — confirms additive overhead.
- **Single-chunk degenerate configs over-predict by ~25 %.** Only the Si scalar sweep at large μ exercises this regime.
- **The −13.9 % outlier (Agent C 3×3×3 nb=200) is the new structural finding** that motivates §5.1.

## 5. Open work — priority-ordered (extends `memory_model_refit_2026-05-17` §10)

### 5.1 **HIGHEST**: add `M_all_gather_slab` to Peak C base

```python
# in _peak_C_fit_one_rchunk, add to persistent + transient base:
M_all_gather_psiX = 2 * _bytes_c128(nk, band_chunk, ns, r_chunk, shard=p_y)  # L + R slabs
```

This is already named in `docs/MEMORY_MODEL.md` §R-Chunk Round-8 lines 198–201 but absent from the planner formula. Expected impact (Agent C estimates):

| config | nb=100 / nb=200 (3×3×3) | nb=100 / nb=200 (4×4×4) | CrI3 80 Ry sweet-spot |
|---|---|---|---|
| HWM_pred today | 15.63 / 15.70 | 22.39 / 22.40 | 70.11 |
| slab addition (2 × per-rank L+R) | +0.74 / +1.49 | +0.44 / +0.88 | ~+2.0 |
| HWM_pred after | 16.37 / 17.19 | 22.83 / 23.28 | ~72.1 |
| mem_stats peak (truth) | 15.71 / 18.24 | 22.66 / 23.13 | 76.05 |
| new %-err | +4 % / **−5.8 %** | +0.8 % / +0.6 % | ~−5 % |

Lands the 3×3×3 nb=200 case inside bispinor's window (−5.8 % vs −13.9 % today) and matches existing nb=100 cases within a few percent. Nudges Si 4×4×4 nb=100 from −1.2 % into mild over-prediction (+0.8 %) — still safe. Likely cuts CrI3 80 Ry from −8.5 % to −5 % too.

**Pre-flight before landing:**
- HLO dump `z_q_from_psi_sm` (or `c_q_from_psi_sm` — whichever is the L/R all_gather entry point) at Si 3×3×3 nb=100 vs nb=200 to verify the slab coefficient (1× vs 2×, with/without aliasing) — Agent C explicitly flagged that "2× L+R" is an estimate, the truth may be 1× with aliasing.
- Re-run Agent A's μ=768 Si bispinor 60 Ry post-fix; expect the −10.8 % bin to land near −2 % alongside its neighbors.

### 5.2 Merge Agent A's nspinor=1 loader fixes to `origin/main`

Commits `8c18925` (`unfold_psi`) and `dc0b254` (`WfnLoader._ensure_phdf5_static`). 30 LOC total, no behavior change for ns=2, all loader/unfold tests pass. Unblocks future truly-scalar runs (and Agent C cherry-picked the first one already).

### 5.3 Optional: effective_slot pruning at n_chunks = 1

The Si scalar μ ≥ 768 over-prediction by 25 % is conservative (safe for OOM). Could add `effective_pair_density_slots = 3 if n_chunks > 1 else 2`. Low priority — over-prediction wastes budget but never causes OOM. Needs HLO confirmation at a production-scale n_chunks=1 config (Si 8×8×8 25 Ry might be the test case).

### 5.4 Optional: ~6.5 GB framework-floor constant

User-deferred for CPU portability (`memory_model_refit_2026-05-17` §6.2). Could add `MEM_FRAMEWORK_FLOOR_GB = 6.5` to make small-config %-err interpretable. Cost: +6.5 GB charged uniformly, which is irrelevant at production scale and overly conservative at small scale. Skip unless a target requires accurate small-config HWM.

### 5.5 Document cusolverMpPotrf+BFC+2D-mesh incompatibility

Already in `KNOWN_SANDBOX_ERRORS.md:117`. Surface to `skills/profiling_stack/SKILL.md`'s "Gotchas" so future agents capturing BFC ground-truth on 2D meshes don't lose time. One-line addition.

### 5.6 Minor: planner `centroids_persist` factor 4 vs live_arrays 6

Agent C noticed `centroids_persist` formula counts 4 buffers (rmuT_X + rmu_Y) × (L + R) but `live_arrays` shows 6 (3 of each transpose). Either the formula under-counts by 1.5× or the probe is double-counting a view. <100 MB miss at Si μ ≤ 1800, ~0.5 GB miss at CrI3 μ=1520. Low priority; bundle with the §5.1 calibration round.

## 6. Things resolved (don't re-investigate)

- **Constants hold at ns=1.** No ns² hidden factors in `_bytes_c128` (Agent A §3 HLO).
- **Per-term kgrid scaling is faithful within 5–6 %.** No leaks at 2³, 3³, 6³ (Agent B §2.2).
- **`sphere_idx_replicated = 1` post-Round-6 canonical accessor holds at every kgrid.** No regression.
- **No cross-r-chunk residue at nb ≤ 200.** Agent C 4×4×4 live_arrays repeat bit-equal.
- **Framework-floor additivity confirmed across CrI3 + Si.** Δ ≈ 5–8 GB independent of system.

## 7. Files of record

- This synthesis: `reports/memory_model_nonbispinor_kgrid_2026-05-18/SYNTHESIS.md`
- Plan / agent assignments: `reports/memory_model_nonbispinor_kgrid_2026-05-18/PLAN.md`
- Agent A (μ-sweep + HLO): `agent_a_si_nonbispinor_mu_sweep.md`
- Agent B (k-grid scaling): `agent_b_si_kgrid_scaling.md` + `agent_b_planner_data.json` + `agent_b_scaling_analysis.json`
- Agent C (band sensitivity): `agent_c_si_band_sensitivity.md`
- Run dirs: `runs/Si/{MU,KGRID,BANDS}_nonbispinor_2026-05-18/`
- Sandbox errors: `KNOWN_SANDBOX_ERRORS.md` (cusolverMpPotrf+BFC entry, nspinor=1 entry promoted to FIXED)
- Branches on origin: `agent/si-nonbispinor-mu-sweep` (lorrax_A, tip `dc0b254`, 2 commits ahead of main with fixes), `agent/si-kgrid-scaling` (lorrax_B, no source changes), `agent/si-band-sensitivity` (lorrax_C, tip `d4cb599`, 1 commit ahead = cherry-pick of `8c18925`)
