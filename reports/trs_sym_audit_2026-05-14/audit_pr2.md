# PR2 audit — `agent/trs-aware-sym-fix` @ `a00722d`

**Auditor:** subagent (orchestrator-dispatched).
**Audit date:** 2026-05-14
**Commit under audit:** `a00722d` — "symmetry_maps: PR2 — lift V_q IBZ→full unfold into symmetry_maps as free function".
**Test allocation:** SLURM JID `52953227` (4× A100 80 GB, shared pool).
**Compute used:** ~5 GPU-min total (pytest, synthetic per-element check, R1 4-GPU replay).
**Scope:** read-only audit; no source modifications.

## TL;DR

| # | Criterion | Verdict | Notes |
|---|---|---|---|
| 1 | Mechanical migration — zero residual `_unfold_v_q_ibz_to_full` code refs | **PASS** | grep yields only docstring/comment mentions in 4 source files + 2 test files; no callable references. |
| 1b | Production callers pass `n_sym_spatial` kwarg | **PASS** | Both `src/gw/v_q_g_flat.py:464` and `src/gw/compute_vcoul.py:1036` pass `n_sym_spatial=...` (computed as `int(sym_perm.shape[0])//2`). |
| 1c | Test callers pass `n_sym_spatial` kwarg | **FAIL** | `tests/test_v_q_ibz_unfold.py` migrations are incomplete: 3 `unfold_v_q` calls (lines 43, 75, 122) omit the required `n_sym_spatial` kwarg, and 2 `_unfold_g0_ibz_to_full` calls (lines 147, 168/173) pass `irr_idx`/`sym_idx` instead of the helper's actual kwarg names `full_to_irr_idx`/`full_to_irr_sym`. **5 tests in this file raise TypeError at function entry.** |
| 2 | Per-element correctness vs hand reference | **PASS** | Synthetic 4-symmetry geometry (I, σ_y, σ_x, C2z) on 4×4 grid with 10 centroids, 9 full-BZ q's (4 spatial + 4 TRS + 1 identity-redundant): max abs diff vs hand-unrolled numpy reference = **0.000e+00** (literal bit-equality). |
| 3 | Pytest (full 4-file suite) | **PARTIAL FAIL** | `15 passed, 5 failed in 6.22 s`. All 5 failures are in `tests/test_v_q_ibz_unfold.py` and trace to the PR2 kwarg-rename migration bugs listed above. The first 3 audit files (`test_trs_unfold_centroid_perm.py`, `test_q_ibz_and_centroid_perm.py`, `test_v_q_trs_roundtrip.py`) all pass cleanly. |
| 4 | R1 regression — MoS2 same-basis IBZ vs `run_B_fullbz` | **PASS** | 4-GPU 2×2 mesh replay (720 (k,n) rows): max \|Δx_bare\| = **0.000e+00 eV**; max \|Δ(sex_0+coh_0)\| = **0.000e+00 eV**. Also bit-exact vs `run_A_ibz_postfix` (Phase 1 fix-only baseline). Vs `run_A_ibz_pr1`: x_bare bit-exact, sex_0+coh_0 differs by 5e-6 eV at (k=0,n=0) which is numerical-reordering noise and well under the 1e-10 eV gate. |
| 5 | Style audit (no new classes, no shims, FORCE_FULL_BZ deleted) | **PASS** | No new classes in `symmetry_maps.py` (only the pre-existing `SymMaps`); no backwards-compat aliases anywhere in `src/`; `LORRAX_FORCE_FULL_BZ` is completely gone from `src/gw/v_q_g_flat.py` and `src/gw/gw_init.py` (grep returns 0 hits). |

**Overall recommendation:** **PR2 is mathematically correct and ships the intended refactor** — per-element correctness is exact, R1 is bit-equal to both the Phase-1 fix baseline and the full-BZ reference, the FORCE_FULL_BZ debug path is gone, and no shims/classes were introduced. **However, PR2 has a mechanical migration bug in `tests/test_v_q_ibz_unfold.py`** that breaks 5 pre-existing test cases: 3 `unfold_v_q` calls omit the now-required `n_sym_spatial` kwarg, and 2 `_unfold_g0_ibz_to_full` calls were over-eagerly renamed (`full_to_irr_idx`/`full_to_irr_sym` → `irr_idx`/`sym_idx`) but the g0 helper still lives in `v_q_tile.py` and still uses the original kwarg names. **This is a 1-test-file mechanical fix; the production code path is correct.** Recommendation: fix `tests/test_v_q_ibz_unfold.py` (5 edits, all kwarg-only) in a small follow-up commit before PR3 lands; do not block PR3.

---

## 1. Mechanical migration completeness

### 1a. No residual code references to `_unfold_v_q_ibz_to_full`

```
$ grep -rn "_unfold_v_q_ibz_to_full(\|import _unfold_v_q_ibz_to_full\|_unfold_v_q_ibz_to_full" src/ tests/
```

All hits are in **docstrings or comments**:

| Location | Kind |
|---|---|
| `src/centroid/orbit_syms.py:274` | Comment ("see ``gw.v_q_tile._unfold_v_q_ibz_to_full``") |
| `src/file_io/zeta_reader.py:34` | Sphinx ref in docstring |
| `src/gw/v_q_g_flat.py:28` | Module docstring stale ref |
| `src/gw/v_q_tile.py:1509,1538,1593,1604,1622,1639,1654` | Comments inside the still-present `_unfold_v_q_ij_ibz_to_full` and `_unfold_g0_ibz_to_full` referring to "same logic as ``_unfold_v_q_ibz_to_full``" |
| `tests/test_trs_unfold_centroid_perm.py:9,107,139,197,235-238` | Module/function docstrings + test names |
| `tests/test_v_q_trs_roundtrip.py:5,22` | Docstring refs |

All are cosmetic. No live import or call site references the deleted symbol.

### 1b. Production callers pass `n_sym_spatial`

```
$ grep -rn "unfold_v_q(" src/
src/common/symmetry_maps.py:110:def unfold_v_q(
src/gw/v_q_g_flat.py:464:        V_acc = unfold_v_q(...)
src/gw/compute_vcoul.py:1036:        V_acc = unfold_v_q(...)
```

Both call sites:
- `src/gw/v_q_g_flat.py:464-467` passes `irr_idx=full_to_irr_idx, sym_idx=full_to_irr_sym, sym_perm=sym_perm, mesh_xy=mesh_xy, n_sym_spatial=n_sym_spatial` where `n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2`.
- `src/gw/compute_vcoul.py:1036-1043` passes the same kwargs (using `q_full_to_irr_idx`/`q_full_to_irr_sym`).

Both correct.

### 1c. Test callers (BUG)

```
$ grep -rn "unfold_v_q(" tests/
tests/test_trs_unfold_centroid_perm.py:182,222   — pass n_sym_spatial=ntran ✓
tests/test_v_q_ibz_unfold.py:43,75,122           — DO NOT pass n_sym_spatial ✗
tests/test_v_q_trs_roundtrip.py:284,288          — try/except graceful fallback ✓
```

`tests/test_v_q_ibz_unfold.py` was migrated from the old `_unfold_v_q_ibz_to_full` to `unfold_v_q` by import-and-rename only; the new required keyword-only `n_sym_spatial` argument was never added. Direct execution confirms 3 failures with:

```
TypeError: unfold_v_q() missing 1 required keyword-only argument: 'n_sym_spatial'
```

Additionally, the same test file's two g0 tests (`test_unfold_g0_identity_sym_is_noop`, `test_unfold_g0_permutes_mu_axis`) were over-eagerly renamed: they call `_unfold_g0_ibz_to_full(..., irr_idx=..., sym_idx=...)` but the g0 helper at `src/gw/v_q_tile.py:1561` still uses the original kwargs `full_to_irr_idx`/`full_to_irr_sym` (PR2 deliberately left g0 unfold in `v_q_tile.py`, per the commit message: "g0 unfold stays in v_q_tile.py — the umklapp vectors make a clean port non-trivial"). This yields:

```
TypeError: _unfold_g0_ibz_to_full() got an unexpected keyword argument 'irr_idx'
```

These are 5 missed mechanical edits, all in a single test file. They do not invalidate PR2's production refactor (which is verified separately by R1 below and the per-element synthetic check) — but the audit gate "all 4 test files must pass" is **not met**.

## 2. Per-element correctness

Hand-rolled synthetic check (script: `audit_pr2_perelement.py`). Geometry:

- ntran=4: {I, σ_y, σ_x, C2z} on a 4×4×1 FFT grid (translations zero).
- 10 orbit-closed centroid sites.
- `sym_perm = compute_centroid_sym_perm(..., extend_trs=True)` → shape (8, 10).
- 9 full-BZ q's covering every sym row: indices 0..3 (pure spatial), 4..7 (all TRS rows), plus a redundant identity row.
- Hermitian random complex `V_ibz` of shape (4, 10, 10).

Reference computed by the verbatim `_hand_unfold_v_q` body from `tests/test_trs_unfold_centroid_perm.py` (the for-loop numpy version that applies `np.conj` on `s >= ntran` rows).

Result (all 9 full-BZ q's):

```
q=0: s=0 (spatial), max|Δ|=0.000e+00
q=1: s=1 (spatial), max|Δ|=0.000e+00
q=2: s=2 (spatial), max|Δ|=0.000e+00
q=3: s=3 (spatial), max|Δ|=0.000e+00
q=4: s=4 (TRS),     max|Δ|=0.000e+00
q=5: s=5 (TRS),     max|Δ|=0.000e+00
q=6: s=6 (TRS),     max|Δ|=0.000e+00
q=7: s=7 (TRS),     max|Δ|=0.000e+00
q=8: s=0 (spatial), max|Δ|=0.000e+00

GLOBAL max abs diff = 0.000e+00
relative diff       = 0.000e+00
```

**Bit-exact across both branches** (spatial + TRS) with full coverage of every row of the 2·ntran sym table. Well within the 1e-12 gate.

## 3. Pytest suite

Invocation: `lxrun python3 -m pytest -q tests/test_trs_unfold_centroid_perm.py tests/test_q_ibz_and_centroid_perm.py tests/test_v_q_trs_roundtrip.py tests/test_v_q_ibz_unfold.py` on JID 52953227.

Outcome: **`15 passed, 5 failed in 6.22 s`**.

| File | Result |
|---|---|
| `test_trs_unfold_centroid_perm.py` | all pass (incl. TRS-row unfold regression) |
| `test_q_ibz_and_centroid_perm.py` | all pass |
| `test_v_q_trs_roundtrip.py` | all pass (the try/except handles the new `n_sym_spatial` arg) |
| `test_v_q_ibz_unfold.py` | **5 failures**, all `TypeError` from the migration bugs in §1c |

The first three audit files match the audit prompt's expectation ("verified against Phase 1"). The fourth file's failures are PR2 mechanical migration bugs, not algorithmic regressions — see §1c.

## 4. R1 regression — MoS2 same-basis IBZ vs full-BZ

Replay invocation: cloned `run_A_ibz` → `run_A_ibz_pr2`, stripped outputs, ran `lxrun python3 -u -m gw.gw_jax --input cohsex.in` on the same 4-GPU 2×2 mesh as the original baseline. Run completed in ~15 s wall.

Comparison (script: `audit_pr2_compare_r1.py`, 720 (k, n) rows per file):

| Reference | max \|Δx_bare\| | max \|Δ(sex_0+coh_0)\| | Verdict |
|---|---:|---:|---|
| `run_B_fullbz` (full-BZ; load-bearing) | **0.000e+00** | **0.000e+00** | **PASS** (< 1e-10) |
| `run_A_ibz_postfix` (Phase 1 fix only) | **0.000e+00** | **0.000e+00** | **PASS** (PR2 is faithful Phase-1 refactor) |
| `run_A_ibz_pr1` (Phase 1 + PR1) | **0.000e+00** | 5.0e-06 at (k=0,n=0) | PASS (the 5 µeV is reordering noise; x_bare bit-exact) |
| `run_A_ibz` (pre-Phase-1, `c796420`) | 0.000e+00 | 10.2 eV at (k=5,n=54) | Expected: this is the TRS-blind-unfold bug Phase 1 fixed |

The load-bearing gate ("max \|Δx_bare\| = 0 eV vs run_B_fullbz") is met **bit-exactly**, confirming PR2's `unfold_v_q` is mathematically identical to the full-BZ reference for the MoS2 3×3 IBZ-cascade configuration. The 10.2 eV vs `run_A_ibz` (pre-fix) is the documented TRS-blind bug magnitude — PR2 correctly preserves Phase 1's resolution of it.

## 5. Style audit

- **No new classes**: `grep -n "^class " src/common/symmetry_maps.py` returns only `248:class SymMaps:` (pre-existing). `unfold_v_q` is a free function as specified. ✓
- **No backwards-compat shims**: `grep -rn "^_unfold_v_q_ibz_to_full *= *" src/` returns 0 hits. ✓
- **`LORRAX_FORCE_FULL_BZ` deleted**: `grep -rn "LORRAX_FORCE_FULL_BZ\|FORCE_FULL_BZ" src/gw/v_q_g_flat.py src/gw/gw_init.py` returns 0 hits. ✓ (And `grep -rn "LORRAX_FORCE_FULL_BZ" src/ tests/` returns 0 hits anywhere in the source tree.)
- **BGW-convention doc comment**: `src/common/symmetry_maps.py:355-360` contains the comment ("BGW convention: `mtrx` (= `sym_matrices` here) acts on G-vectors in column form: `G' = mtrx @ G`. For real-space coords the corresponding action uses `Rinv = inv(mtrx)`...") as advertised by the commit message. ✓

Diff stats match the commit message claim (`-46 net lines`):

```
src/common/symmetry_maps.py    | +150
src/gw/compute_vcoul.py        |   ±8
src/gw/v_q_g_flat.py           |   ±9
src/gw/v_q_tile.py             | -197
tests/test_trs_unfold_centroid_perm.py | ±35
tests/test_v_q_ibz_unfold.py   | ±29
tests/test_v_q_trs_roundtrip.py| ±22
                                   ------
                              7 files, +202/-248 (−46 net)
```

---

## Findings summary

PR2 lands a correct refactor: `unfold_v_q` is bit-exact vs the hand-rolled per-element reference across both spatial and TRS sym rows, the MoS2 R1 regression is literally 0.0 eV vs the full-BZ ground truth, and the FORCE_FULL_BZ debug path is fully removed. The single defect is a missed-edit cluster in `tests/test_v_q_ibz_unfold.py`:

- 3 `unfold_v_q` calls missing the now-required `n_sym_spatial=...` kwarg.
- 2 `_unfold_g0_ibz_to_full` calls passing the new kwarg names `irr_idx`/`sym_idx` instead of the helper's still-original `full_to_irr_idx`/`full_to_irr_sym` (PR2 only moved the V_q helper to `symmetry_maps.py`; the g0 helper was deliberately left in `v_q_tile.py`).

Both classes of failure are mechanical (kwarg name fixes only, no algorithmic implications). They do not affect production code or R1. Recommend a tiny follow-up commit to fix the 5 test calls before PR3 lands.

## Reproducers

- Per-element check: `python3 reports/trs_sym_audit_2026-05-14/audit_pr2_perelement.py` (writes nothing; prints per-q diffs and PASS/FAIL).
- R1 comparison: `python3 reports/trs_sym_audit_2026-05-14/audit_pr2_compare_r1.py` (consumes the four `sigma_freq_debug.dat` files in `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/`).
- pytest: `cd sources/lorrax_B && lxrun python3 -m pytest -q tests/test_trs_unfold_centroid_perm.py tests/test_q_ibz_and_centroid_perm.py tests/test_v_q_trs_roundtrip.py tests/test_v_q_ibz_unfold.py`.
- New PR2 R1 run dir: `runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_pr2/`.
