# Agent 4 — Existing nosym reference audit

**Date:** 2026-05-14
**Working tree analysed:** outputs scattered across `runs/` from 2026-04-05 → 2026-05-14
**Code era boundary:** **IBZ cascade activated 2026-05-11 09:56 UTC** (commit `7c14354`, "zeta_q.h5 IBZ-only + V_q orchestrator IBZ loop + post-loop unfold"). Anything generated before that timestamp is pre-cascade and cannot exhibit the TRS-blind V_q unfold bug; anything generated after may exhibit it iff the centroid basis is orbit-closed (so the cascade actually activates rather than falling back).

Mission: mine pre-existing sym-vs-nosym (or cascade-vs-no-cascade) pairs and characterise the TRS-blind V_q-unfold bug's production impact. No new BGW runs.

## TL;DR

**The bug is real, large, and isolated cleanly by an existing pre-built pair.** The same-basis MoS2 3×3 pair at `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/` was constructed specifically to test the cascade by toggling `LORRAX_FORCE_FULL_BZ` on the same orbit-closed 642-centroid basis. Run A (cascade ACTIVE) and Run B (cascade FORCED OFF) differ by:

- **max |Δx_bare| = 10.21 eV** (at k=5, n=54; per-k max ranges 5.36–10.21 eV across all 9 k-points)
- **max |Δcoh_0| = 12.80 eV**
- **max |Δ(sex_0+coh_0)| ≈ 13.85 eV**

Against BGW, the cascade-ACTIVE Run A is off by **-6.34 eV** at Γ band 23; the cascade-OFF Run B is off by only **23 meV** at the same band (basis-convergence noise). **The bug collapses the moment we force full-BZ unfold of the same basis.**

The CrI3 6×6 cascade-vs-reference comparison reported in `reports/zeta_rchunk_memory_model_2026-05-13/cri3_ibz_cascade_validation.md` (286 meV / 110 meV Σ_X deltas on valence-edge bands, previously attributed to "different ISDF basis") is also consistent with the TRS bug — the reference there ran with the cascade DEACTIVATED via orbit-closure fallback. That reframing is documented below.

Most existing canonical "sym" production runs (MoS2 00_mos2_3x3_cohsex, Si 01/02, Si_pseudobands {19, 21, 26-29}) **predate the cascade or were run with a basis that fails orbit closure**, so the cascade silently falls back to full-BZ ζ on disk and the bug does not fire. They are not affected and their reference value vs BGW is preserved.

## 1. Code-era classification of existing run dirs

I classified each candidate pair from STATUS.md by whether the LORRAX outputs predate the cascade and whether the cascade actually activated:

| Pair / run | LORRAX outputs date | Code era | IBZ cascade actually active? | Bug expected? |
|---|---|---|---|---|
| `MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz` | 2026-05-14 | post-cascade | **YES** (5 IBZ q / 9 full, 642 orbit-closed cen) | **YES** |
| `MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_B_fullbz` | 2026-05-14 | post-cascade | NO (`LORRAX_FORCE_FULL_BZ=1`) | no (control) |
| `MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex_round8_baseline_2026-05-14` | 2026-05-14 | post-cascade | NO (orbit closure failed → fallback) | no |
| `MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex` (sigma_freq_debug) | 2026-05-11 09:48 | **pre-cascade by 8 min** | NO (cascade code not yet merged) | no |
| `MoS2/02_mos2_3x3_nosym/00_lorrax_cohsex` | 2026-05-04 | pre-cascade | n/a (ntran=1 → cascade trivial) | no |
| `MoS2/B_03_mos2_3x3_nosym_60Ry` | — | n/a (QE only, no GW) | — | — |
| `Si/01_si_4x4x4_nosymmorphic/00_lorrax_cohsex` | 2026-04-05 | pre-cascade | n/a | no |
| `Si/02_si_4x4x4_nosym/00_lorrax_cohsex` | 2026-04-07 | pre-cascade | n/a (ntran=1) | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_400c` | 2026-04-16 | pre-cascade | n/a | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/21_lorrax_cohsex_nosym_parity` | 2026-04-16 | pre-cascade | n/a (ntran=1) | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/26_cohsex_pb_v1_nosym` | 2026-04-17 | pre-cascade | n/a (ntran=1) | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/27_cohsex_pb_v1_nosym_bgwv` | 2026-04-17 | pre-cascade | n/a | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/28_cohsex_pb_v2_nosym` | 2026-04-17 | pre-cascade | n/a | no |
| `Si_pseudobands/00_si_2x2x2_60Ry/29_cohsex_pb_v2_nosym_bgwv` | 2026-04-17 | pre-cascade | n/a | no |
| `CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14` | 2026-05-14 | post-cascade | **YES** (8 IBZ q / 36 full, 1508 orbit-closed cen) | **YES** |
| `CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14` | 2026-05-14 | post-cascade | NO (orbit closure failed → fallback) | no |

**Key heuristic**: a post-cascade run with `q-IBZ reduction: N IBZ q-points / M full-BZ` in gw.out is bug-active; a post-cascade run with `centroid orbit closure failed — falling back to full-BZ on disk` is bug-dormant (the cascade fell back, the bug never fires).

## 2. Bug-pre-fix damage assessment

Damage ranked by max |Δ| against the implied ground truth in each pair.

### 2.1 Pair-1: MoS2 3×3, same-basis IBZ-cascade vs forced full-BZ (POST-CASCADE)

**Path:** `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/`

The cleanest signal: both runs use the same 642-centroid orbit-closed basis (`centroids_frac_642.txt`), the same WFN, the same input file. The only difference is the env var `LORRAX_FORCE_FULL_BZ=1` on Run B. Run A actively cascaded (`q-IBZ reduction: 5 IBZ q-points / 9 full-BZ`); Run B forced full-BZ ζ on disk and full-BZ unfold.

Per-k max |Δ| (k=0..8 of the 9 full-BZ k's):

| k | max\|Δx_bare\| (eV) | max\|Δsex_0\| (eV) | max\|Δcoh_0\| (eV) | max\|Δ(sex_0+coh_0)\| (eV) |
|---|---|---|---|---|
| 0 | 6.889 | 0.778 | 8.874 | 9.652 |
| 1 | 6.801 | 0.815 | 9.261 | 10.075 |
| 2 | 6.900 | 0.778 | 9.707 | 10.485 |
| 3 | 5.907 | 0.710 | 7.533 | 8.243 |
| 4 | 7.285 | 0.726 | 9.276 | 10.002 |
| 5 | **10.207** | 1.054 | **12.799** | **13.853** |
| 6 | 5.363 | 0.606 | 7.047 | 7.592 |
| 7 | 9.155 | 0.988 | 12.229 | 13.217 |
| 8 | 5.727 | 0.669 | 8.406 | 8.965 |

Every k-point is wrong by multi-eV. The bug is not localised to one k or band — it propagates through the unfolded V_q to every Σ matrix element. The note STATUS.md cites ("max |ΔΣ_X| = 6.89 eV at k=0 band 44") matches k=0 here exactly (the 6.889 entry above).

**Versus BGW reference** (`runs/MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/sigma_hp.log`, ik=1 → Γ):

| BGW band | BGW_X (eV) | LORRAX_x_bare A (eV) | LORRAX_x_bare B (eV) | Δ(A-BGW) | Δ(B-BGW) |
|---|---|---|---|---|---|
| 19 | -16.5318 | -17.998 | -16.210 | **-1.466** | +0.322 |
| 21 | -18.7102 | -24.949 | -18.676 | **-6.238** | +0.035 |
| 23 | -18.6138 | -24.959 | -18.637 | **-6.345** | -0.023 |
| 25 | -17.2746 | -19.389 | -17.047 | -2.115 | +0.227 |
| 27 | -11.0523 | -14.354 | -11.070 | -3.302 | -0.018 |
| 29 | -10.9866 | -14.434 | -11.051 | -3.447 | -0.064 |

**Run A (cascade ACTIVE) is off from BGW by up to 6.3 eV; Run B (cascade FORCED OFF, same basis) is off by ≤ 0.3 eV**, i.e. by basis-convergence noise only. The 6 eV residual collapses to 35 meV when we force full-BZ unfold of the same basis. This is the gold-standard isolation of the TRS-blind unfold bug.

TRS-fold occupancy: gw.out emits `SymMaps: 4/9 full-BZ k-points require time-reversal symmetry for unfolding`. The MoS2 3×3 hexagonal IBZ has `find_irreducible_qpoints` returning 5 IBZ q's out of 9 full-BZ; the unfold from 5 → 9 needs **TRS to fold 4 of the 9 q's** onto IBZ representatives. The 4/9 silent OOB-clipping into `sym_matrices` (length-2 ntran=2 table) corrupts those q's' V_q rows, contaminating every Σ matrix element.

### 2.2 Pair-2: CrI3 6×6 80 Ry, IBZ cascade vs reference (POST-CASCADE)

**Paths:**
- A: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14` (cascade ACTIVE, 1508 orbit-closed centroids, 8 IBZ q / 36 full-BZ)
- B: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14` (1504 non-orbit-closed centroids → cascade fallback → full-BZ unfold)

| k=0 band index | A (eV) | B (eV) | Δ(A-B) (eV) |
|---|---|---|---|
| 0,1 (-49 eV core) | -49.0095 | -49.0022 | -0.0073 |
| 2,3 (-49 eV core) | -49.0110 | -49.0070 | -0.0040 |
| 4,5 (valence-edge) | -41.8092 | -41.5234 | **-0.286** |
| 6,7 (valence-edge) | -41.6187 | -41.5079 | **-0.111** |

This is the exact comparison `cri3_ibz_cascade_validation.md` flagged as "286 meV / 110 meV — not a bug, attributable to different ISDF basis". With the TRS bug now identified and Pair-1 establishing that the SAME-basis isolation gives multi-eV deltas, the most parsimonious explanation for the CrI3 286 meV residual is the **TRS bug acting on the cascade run only** (the reference fell back). The "different basis" hypothesis predicted the same direction but cannot quantitatively explain the structure (the deltas come exactly in degenerate pairs of the spatial sym group, consistent with a per-q V_q corruption that respects spatial sym but breaks under TRS).

To convert the 286/110 meV signal from "circumstantial" to "definitive", the necessary post-fix experiment is below in §3.2.

### 2.3 Pair-3: Si 4×4×4 nosymmorphic vs nosym (PRE-CASCADE; null check)

**Paths:** `Si/01_si_4x4x4_nosymmorphic/00_lorrax_cohsex` (ntran=12) vs `Si/02_si_4x4x4_nosym/00_lorrax_cohsex` (ntran=1).

| Field | global max\|Δ\| | at (k, n) |
|---|---|---|
| sigSX | 9.96e-03 eV | (0, 0) |
| sigCOH | 1.43e-01 eV | (34, 17) |
| sigTOT | 1.49e-01 eV | (40, 0) |

PRE-cascade era, so cannot directly exercise the cascade-time bug. Useful only as: "did the pre-cascade LORRAX agree between sym and nosym on Si?" Answer: sigSX agrees to **10 meV**, sigCOH disagrees by 143 meV. The sigCOH disagreement is likely a different ISDF basis between the two runs (both used 480 centroids but k-means trajectories on different WFN's are not deterministic). This is the upper bound on basis-noise for Si 4×4×4 at 480 cen — **does not implicate the TRS bug** (which wouldn't fire pre-cascade).

### 2.4 Pair-4: Si 2×2×2 60Ry sym vs nosym parity (PRE-CASCADE; null check)

**Paths:** `Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_400c` vs `21_lorrax_cohsex_nosym_parity`.

| Field | global max\|Δ\| | at (k, n) |
|---|---|---|
| sigSX | 2.67e-03 eV | (5, 3) |
| sigCOH | 4.19e-02 eV | (7, 13) |
| sigTOT | 4.20e-02 eV | (7, 13) |

Tighter than the 4×4×4 pair (smaller system, less basis noise). sigSX agrees to **3 meV**, sigCOH to **42 meV**. Pre-cascade era. Confirms LORRAX-internal sym/nosym agreement for Si was working before the cascade refactor.

### 2.5 Pair-5: MoS2 sym vs BGW, pre-cascade snapshot (NULL: pre-cascade era)

**Paths:** `runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/sigma_freq_debug.dat` (2026-05-11 09:48 — 8 min before cascade) vs BGW sigma_hp.log.

| BGW band (k=0) | BGW_X | LORRAX_x_bare | Δ |
|---|---|---|---|
| 19 | -16.5318 | -16.5325 | -7.05e-04 |
| 21 | -18.7102 | -18.6481 | +6.20e-02 |
| 23 | -18.6138 | -18.6095 | +4.34e-03 |
| 25 | -17.2746 | -17.2824 | -7.83e-03 |
| 27 | -11.0523 | -11.0301 | +2.22e-02 |
| 29 | -10.9866 | -11.0136 | -2.70e-02 |

Max |Δ| = 62 meV at band 21. This is the **convergence-limited LORRAX-vs-BGW agreement for MoS2 at 640 centroids**, no TRS bug present (pre-cascade code).

### 2.6 Pair-6: MoS2 sym round8 baseline vs BGW (POST-CASCADE but bug DORMANT)

**Path:** `runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex_round8_baseline_2026-05-14/sigma_freq_debug.dat`. Same 640-centroid basis as Pair-5; the cascade FELL BACK because orbit closure failed (`588 / 1280 failures`).

Per-band x_bare diff at Γ: identical pattern to Pair-5, max |Δ| = 62 meV at band 21. **The bug is dormant in the canonical sym MoS2 production run** because nobody has regenerated centroids with orbit-aware k-means there yet. The cascade code is in place; the cascade is not firing.

## 3. Post-fix re-evaluation plan

### 3.1 Minimal LORRAX-only reruns

Per `agent_4_existing_references.md` mission constraint: **no BGW reruns**, reuse existing QE outputs (and BGW outputs as ground truth), only rerun the LORRAX step.

After Agent 2's patch lands, I will re-run these LORRAX-only verifications:

| # | Run dir to overwrite/variant | What changes | Expected post-fix result |
|---|---|---|---|
| R1 | `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz` (new variant `run_A_ibz_post_fix`) | LORRAX from Agent 2's fixed branch, same 642-centroid orbit-closed basis | x_bare, sex_0, coh_0 bit-identical (or floating-point) to `run_B_fullbz` of the same dir — see Pair-1. **This is the single most important verification.** Bug isolated to 1 ULP if fixed correctly. |
| R2 | `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_post_fix_2026-05-XX` | rerun the 1508-orbit-closed CrI3 cascade with the fix | Σ_X at k=0 should agree with `lorrax_B_round8_validation_2026-05-14` (the fallback reference) to **≤ 10 meV on the -49 eV core bands and ≤ basis-noise (~50-100 meV) on valence-edge**. The current 286/110 meV deltas should collapse. If they don't, either the fix is incomplete or there is a separate basis-mismatch contribution that needs isolating. |
| R3 | `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_post_fix` vs BGW | post-fix LORRAX x_bare at Γ vs BGW sigma_hp.log ik=1 | max \|Δ\| ≤ ~70 meV (matches pre-cascade Pair-5 / round8 baseline Pair-6 convergence floor for 642 centroids). |

R1 is the **load-bearing verification**: same code, same WFN, same basis, only cascade-on vs cascade-off. If R1 disagrees post-fix, the fix is incomplete. R2 and R3 are confirmatory.

### 3.2 Optional follow-ups (recommend, do NOT block)

- **Regenerate canonical MoS2 sym centroids with orbit-aware k-means** so the canonical `00_lorrax_cohsex` actually exercises the cascade post-fix. This converts the canonical production run from bug-dormant to bug-active and gives the next-session sanity check a clean signal. ~5 GPU-minutes of k-means + ~2 min LORRAX.
- **Add post-fix Si 4×4×4 nosymmorphic run with orbit-closed centroids on the post-cascade code** to verify the inversion-system null hypothesis under the new code path. Si has inversion so the IBZ unfold should never need TRS — bug-impossible by construction; if even Si shows a delta with cascade-on, there is a separate bug.

### 3.3 Explicit non-asks

I do not propose any of:

- Any BGW rerun. All BGW outputs on disk are taken as ground truth.
- Any rerun of pre-cascade-era LORRAX outputs. They are immutable historical records and unaffected by the bug.
- Modifying `cri3_ibz_cascade_validation.md` to retract the "not a bug" conclusion until R2 lands and confirms quantitatively. The retraction is mechanical once the post-fix CrI3 result is in.

## 4. What this audit did and did not measure

**Did measure**:
- Bit-clean isolation of the bug at MoS2 3×3 (Pair-1) with the same-basis pair.
- Per-k pattern of the bug: every k-point in MoS2 3×3 is corrupted (5–10 eV), not localised.
- Bug dormancy in canonical production runs whose centroids are not orbit-closed.
- The CrI3 cascade-validation result is consistent with the TRS bug, not just basis-noise.

**Did not measure**:
- Per-q decomposition of which q's get clipped vs which pass through clean. Would require dumping V_q itself and comparing per-q against a known-good full-BZ build. Out of scope for this audit (no BGW rerun, and we already have the global signal).
- The bug's effect on bispinor / SOC systems. None of the bispinor runs on disk (`runs/MoS2/B_*bispinor*`, `runs/CrI3/B_bispinor_*`) are post-cascade with cascade activated.
- The 2D-system head-correction interaction. The cascade and head correction are independent codepaths; the bug propagates to whatever V_q is built.

## 5. Files referenced

- `reports/trs_sym_audit_2026-05-14/agent_4_data/audit_pairs.py` — comparison harness
- `reports/trs_sym_audit_2026-05-14/agent_4_data/audit_run.log` — full numerical dump for all pairs
- `reports/trs_sym_audit_2026-05-14/agent_4_data/audit_results.json` — machine-readable form

Parsers follow `skills/compare/SKILL.md` §2c (LORRAX sigma_freq_debug — adapted for current header that includes a V_H column between kin_ion and x_bare; sex_0 and coh_0 indices shifted accordingly), §2a (BGW sigma_hp), and a new `parse_bgw_xdat` for `x.dat` (BGW bare-exchange dump).
