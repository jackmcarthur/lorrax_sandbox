# Post-fix verification — R1, R2, R3

**Date:** 2026-05-14
**Branch / HEAD:** `agent/trs-aware-sym-fix` @ `9e644e9` (`sources/lorrax_B`)
**Allocation:** SLURM JID 52953227 (Perlmutter, 4× A100 80GB)
**Patch under test:** TRS-aware V_q IBZ→full-BZ unfold (scalar charge channel).
**Compute used:** ~15 GPU-min total (R1 ~3 min, R2 ~12 min).

## TL;DR

| Round | Test | Gate | Measured | Verdict |
|---|---|---|---|---|
| **R1** | MoS2 3×3 IBZ-cascade vs forced full-BZ, same 642-cen basis | max\|Δx_bare\| ≤ 1e-10 eV | **0.000e+00 eV** | **PASS** (bit-equal at FP precision) |
| **R2** | CrI3 6×6 IBZ-cascade postfix vs cascade-prefix, same 1508-cen basis | n/a (auxiliary) | 0 eV | **PASS** (fix is a no-op for CrI3 — see §R2) |
| **R3** | MoS2 3×3 IBZ-cascade postfix Σ_X at Γ vs BGW | ≤ 70 meV | **322 meV at band 19** | **FAIL** (gate too tight; see §R3) |

The patch is correct. R1 — the load-bearing verification — passes by bit-equality with zero floating-point residual. R3 fails the user-specified 70 meV gate but matches the pre-existing `run_B_fullbz` vs BGW residual exactly (the 642-centroid basis has a ~320 meV convergence floor vs BGW at MoS2 Γ band 19, documented in Agent 4 §2.1 Pair-1 Run B column). R2 reveals that CrI3 doesn't exercise the TRS code path at all (inversion sym), so it cannot be used to verify the fix — but it does reframe the CrI3 286/110 meV deltas from `cri3_ibz_cascade_validation.md` as pure basis-shift (1508 vs 1504 centroids), NOT a TRS bug. Agent 4 §2.2's hypothesis was incorrect.

## R1 — MoS2 3×3 same-basis bit-equality (load-bearing)

**Runs:**

- `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/` (NEW, cascade-ACTIVE)
- `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_B_fullbz/` (existing, cascade-FORCED-OFF)
- Both use the same WFN, same 642 orbit-closed centroids, same inputs. Only difference: the postfix run lets the cascade fire (5 IBZ q's → unfold to 9 full BZ via TRS-aware sym table). The fullbz run sets `LORRAX_FORCE_FULL_BZ=1` and builds ζ on the full 9-q grid directly.

**Per-k max |Δ| (all 720 (k, n) records, 9 k-points × 80 bands, parser per SKILL.md §2c adapted for the current V_H/x_head/sex_head/coh_head column layout):**

```
  k    |Δx_bare|     |Δsex_0|     |Δcoh_0|   |Δ(sex_0+coh_0)|
  0    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  1    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  2    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  3    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  4    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  5    0.000e+00    0.000e+00    0.000e+00         0.000e+00   ← pre-fix max |Δx| was 10.207 eV here
  6    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  7    0.000e+00    0.000e+00    0.000e+00         0.000e+00
  8    0.000e+00    0.000e+00    0.000e+00         0.000e+00
```

**Global max |Δ| = 0.000e+00 eV** for x_bare, sex_0, coh_0, and sex_0+coh_0 across all 720 records.

**R1 GATE (≤ 1e-10 eV): PASS.** Bit-equal at FP precision — every byte of every output column matches between the IBZ-cascade and forced-full-BZ paths.

The 10.207 eV pre-fix delta at k=5 (Agent 4 §2.1 Pair-1) collapses to literal zero. This is the definitive evidence that the TRS-aware sym table extension and the `n_sym_spatial` index gate in `_unfold_v_q_ibz_to_full` correctly handle the 4-out-of-9 TRS-folded q's in MoS2's hexagonal IBZ.

## R2 — CrI3 6×6 cascade (auxiliary, finding-as-side-effect)

**Runs:**

- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_postfix_2026-05-14/` (NEW)
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14/` (pre-existing, same 1508-cen basis, pre-fix branch)
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14/` (pre-existing, 1504-cen basis, cascade fell back)

Both CrI3 cascade runs trigger the cascade: gw.out shows `q-IBZ reduction: 8 IBZ q-points / 36 full-BZ`. The cascade-postfix run completed all V_q work and emitted the bare Σ_X diagonal, then crashed in `write_qp_wfn_h5` (`U shape (36, 150, 150) inconsistent with (nk=8, nb_active=150)` — pre-existing qp_wfn bug, unrelated to this patch; same crash as the pre-fix run).

**Bare Σ_X diagonal (eV), k=0, 8 bands (from gw.out — sigma_freq_debug.dat not written due to the qp_wfn crash):**

```
                       core (b0..3)                          valence-edge (b4..7)
cascade_postfix 1508: -49.0095 -49.0095 -49.0110 -49.0110  -41.8092 -41.8092 -41.6187 -41.6187
cascade_prefix  1508: -49.0095 -49.0095 -49.0110 -49.0110  -41.8092 -41.8092 -41.6187 -41.6187   ← identical
round8 ref      1504: -49.0022 -49.0022 -49.0070 -49.0070  -41.5234 -41.5234 -41.5079 -41.5079
```

**Cascade-postfix vs cascade-prefix (same 1508-cen basis): max |Δ| = 0.000 meV (per-band: all zeros).**

**Cascade-postfix vs round8 reference (1508 vs 1504 cen): max |Δ| = 285.8 meV (per-band: [-7.3, -7.3, -4.0, -4.0, -285.8, -285.8, -110.8, -110.8] meV).**

**R2 finding (this is the new physics result, not a verification pass/fail):**

The cascade-postfix and cascade-prefix bare Σ_X are bit-equal at the 4-decimal-place output precision. This means **the TRS fix does not change any CrI3 number** — because CrI3 (P-31m, ntran=6) has spatial inversion symmetry, the q-folding IBZ→full BZ never requires the TRS row of the augmented sym table. The TRS-augmented operations sit at indices `≥ ntran` in `sym_mats_k`, and `find_irreducible_qpoints` on CrI3 returns sym indices all in `[0, ntran)` because every Star(q) representative is reachable by a spatial op alone.

Consequence: **Agent 4's hypothesis that the 286/110 meV CrI3 cascade-vs-reference deltas in `reports/zeta_rchunk_memory_model_2026-05-13/cri3_ibz_cascade_validation.md` were due to the TRS bug is incorrect.** Those deltas are pure basis-shift between the 1508-centroid orbit-closed set (cascade-active) and the 1504-centroid non-orbit-closed set (cascade fallback). The basis-shift on valence-edge bands per ~0.27% change in centroid count is ~280 meV — substantially larger than the "≤10-20 meV" hand-wave in Agent 4 §3.1 (R2 row).

The original `cri3_ibz_cascade_validation.md` conclusion ("not a bug, attributable to different ISDF basis") is **correct** and should be reinstated; the retraction proposed in Agent 4 §3.3 should NOT be applied.

The strict 10 meV gate in the user's R2 expectation cannot be evaluated against this pair, because the bases differ. An apples-to-apples evaluation would require rerunning the round8 reference (cascade-OFF) on the 1508-centroid orbit-closed file — but since the postfix and prefix cascade runs already produce bit-equal output for CrI3, this rerun adds no information about the patch's correctness. Skipped.

## R3 — MoS2 Γ x_bare vs BGW

**Runs:**

- LORRAX: `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/sigma_freq_debug.dat`, k=0 (Γ)
- BGW reference: `runs/MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/sigma_hp.log`, ik=1 (Γ)
- Parser: SKILL.md §2a (`parse_sigma_hp`). The user asked for "bands 1..50"; BGW only computed bands 19–30 around the gap, so the table below covers what's actually in the BGW output.

| BGW band | BGW X (eV) | LORRAX x_bare (eV) | Δ (LORRAX–BGW, meV) |
|---|---|---|---|
| 19 | -16.532 | -16.210 | **+321.7** |
| 20 | -16.532 | -16.210 | +321.7 |
| 21 | -18.710 | -18.676 | +34.6 |
| 22 | -18.710 | -18.676 | +34.6 |
| 23 | -18.614 | -18.637 | −22.8 |
| 24 | -18.614 | -18.637 | −22.8 |
| 25 | -17.275 | -17.047 | +227.3 |
| 26 | -17.275 | -17.047 | +227.3 |
| 27 | -11.052 | -11.070 | −17.7 |
| 28 | -11.052 | -11.070 | −17.7 |
| 29 | -10.987 | -11.051 | −64.5 |
| 30 | -10.987 | -11.051 | −64.5 |

**Global max |Δ| = 321.7 meV at band 19. MAE = 114.8 meV.**

**R3 GATE (≤ 70 meV): FAIL** (max 322 meV).

**Interpretation:** these numbers are **bit-identical** to Agent 4's Pair-1 Run B vs BGW column (§2.1 Table at lines 76–83 of `agent_4_reference_audit.md`: Run B band 19 = +0.322, band 21 = +0.035, band 23 = -0.023, band 25 = +0.227, band 27 = -0.018, band 29 = -0.064 — matches every value exactly). This is the **convergence floor of the 642-centroid orbit-closed basis vs BGW at MoS2 Γ**, not a residual TRS effect.

The 70 meV gate the user cited from Agent 4 §3.1 ("matches pre-cascade Pair-5 / round8 baseline Pair-6 convergence floor for 640 cen") corresponds to the *640-centroid non-orbit-closed* basis (`runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/`). The Pair-1 same-basis run uses the *642-centroid orbit-closed* basis — a different ISDF basis that happens to converge less well at band 19 (322 meV) but better at band 27 (18 meV vs Pair-5's 22 meV). The 70 meV gate is reasonable as a back-of-envelope but not predicted at the per-basis level.

Because R1 confirms the postfix run produces bit-identical output to Run B, and R3's residuals are bit-identical to the documented Run B–vs–BGW table, **R3 is "FAIL by the stated gate, PASS by the underlying physics":** the patch makes the IBZ-cascade path numerically indistinguishable from the cascade-off path, and both have the same basis-convergence residual against BGW. To pass a 70 meV gate, regenerate centroids — but that's a basis-quality problem, not a patch correctness problem.

## Manifests created

- `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/manifest.yaml`
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_postfix_2026-05-14/manifest.yaml`

## Comparison scripts (parsers per SKILL.md)

- `reports/trs_sym_audit_2026-05-14/postfix_compare_R1.py` — adapts SKILL.md §2c parser for current V_H/x_head/sex_head/coh_head columns
- `reports/trs_sym_audit_2026-05-14/postfix_compare_R2.py` — parses the "Bare Σ_X diagonal" line from gw.out (qp_wfn crash precluded sigma_freq_debug)
- `reports/trs_sym_audit_2026-05-14/postfix_compare_R3.py` — uses SKILL.md §2a `parse_sigma_hp`

## Followups (not blocking)

1. **qp_wfn write crash** for CrI3 6×6 with `bispinor=false`, `nval=70, ncond=80`, runs on 4 nodes × 4 GPUs → 16 ranks, but `U` array gets shape `(36, 150, 150)` while `nk=8` (IBZ) is expected. Pre-existing bug in `file_io/qp_wfn.py:137`. Not on the critical path for the patch.
2. **CrI3 286/110 meV basis-shift mystery** — the round8 reference (1504 cen) vs cascade (1508 cen) shows ~290 meV on valence-edge bands. This is too large for a 0.27% basis-count change in a converged regime, suggesting either: (a) the 1504 and 1508 centroid sets are k-means-distinct (different seed trajectories produce qualitatively different bases), (b) one of the two runs is not at convergence. Worth a brief k-means audit, but not for this patch verification.
3. **Throwaway debug code** in the working tree (uncommitted as of HEAD = 9e644e9): `LORRAX_FORCE_FULL_BZ` env-var path in `src/gw/v_q_g_flat.py` and `src/gw/gw_init.py`, and a documentation comment in `src/common/symmetry_maps.py`. These were used to construct the original `run_B_fullbz` (and the R1 postfix run leaves them untouched). Should be either cleaned up or promoted to a CLI flag before this branch merges.
