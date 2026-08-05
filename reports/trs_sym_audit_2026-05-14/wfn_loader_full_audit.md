# Fresh-eyes audit: wfn_loader.py + symmetry_maps.py Phase 2 changes

**Date**: 2026-05-14
**Scope**: PR1 (`5da9ec7`) + PR2 (`a00722d` + audit-fix `796c043`) + PR3 (`8504994`) — i.e. `git diff 9e644e9..a45f039 -- src/file_io/wfn_loader.py src/common/symmetry_maps.py`. Cleanup commit `69ab42c` and stale-API fix `a45f039` do NOT touch these two files (verified via `git show --stat`).
**Auditor**: read-only static review, no pytest, no runtime.
**Question**: did Phase 2 introduce a regression in the **non-TRS-row** codepath that would break <1 meV sym-vs-nosym Σ_X on inversion-symmetric materials?

---

## TL;DR verdict

**SHIP** — high confidence no regression on non-TRS rows.

For every non-TRS sym row (`sym_idx < ntran`), the eager-backend ψ-unfold, the phdf5 static-table build, and the V_q IBZ→full-BZ unfold all produce **byte-identical** numerical operations to the pre-Phase-2 baseline at commit `9e644e9`. PR1's index-table consolidation is a pure rename (`irk_to_k_map → irr_idx_k`, `irk_sym_map → sym_idx_k`) plus an eager call into a refactored `find_irreducible_bz_points` whose algorithm matches the deleted lazy `find_irreducible_qpoints` line-for-line. PR2 lifted `_unfold_v_q_ibz_to_full` body into `common.symmetry_maps.unfold_v_q` unchanged (the only narrowing — making `n_sym_spatial` a required kwarg — does not affect production callers, who always pass it). PR3 confines its real semantic change (the iσ_y · conj(U) TRS-row rule) to the `is_trs=True` branch of `unfold_psi` and to the `tr_mask=True` rows of the phdf5 static-table build; the `is_trs=False` / `tr_mask=False` code path is unchanged.

The sym-vs-nosym gate failure (if any) is NOT in this diff.

---

## 1. Per-hunk classification

| # | File | Line range (HEAD) | Classification | Notes |
|---|------|-------------------|----------------|-------|
| 1 | symmetry_maps.py | 9-16 | Pure import addition | adds `functools.partial`, `jax`, `Mesh`/`NamedSharding`/`P` — needed by lifted `unfold_v_q` |
| 2 | symmetry_maps.py | 18-107 | New helper (`find_irreducible_bz_points`) | Algorithm body matches deleted `find_irreducible_qpoints` (lines 346-459 of baseline) line-for-line for `irr_kgrid_int=None`; lex-min canonical orbit, first-occurrence IBZ list, second-pass per-q sym lookup with `matches[0]` tiebreak. The `irr_kgrid_int is not None` branch is a k-side variant **not used in production** (k-side still uses unchanged `find_symmetry_ops_simple`). Verified test fixture in tests/test_q_ibz_and_centroid_perm.py wires the same outputs. |
| 3 | symmetry_maps.py | 110-247 | Algorithm refactor (`unfold_v_q`) | Identical body to baseline `_unfold_v_q_ibz_to_full` in `gw/v_q_tile.py` of Phase 1 commit `9e644e9`: same trivial-IBZ short-circuit (lines 177-182), same `inv_perm` argsort + pad-to-padded-extent (lines 206-218), same `take_along_axis(mode='promise_in_bounds')` double-permute (lines 234-239), same `where(trs_mask, conj(V_full), V_full)` (lines 243-244). Only narrowing: `n_sym_spatial` was `Optional[int] = None` in baseline; now required. Both production callers (`compute_vcoul.py:1036`, `v_q_g_flat.py:464`) always pass it, so no behavioural change. |
| 4 | symmetry_maps.py | 250-361 | Real semantic change (new `unfold_psi`) | `_I_SIGMA_Y` constant + `unfold_psi` free function. Body is the equivalent of baseline `wfn_loader._eager_build`'s inline lines 839-849, restructured into a single helper. For `is_trs=False` branch (line 353-356): tau lookup + rotated G + phase + `cnk * phase[None,None,:]` + spinor einsum — all bit-identical operations to baseline. For `is_trs=True` branch (line 343-352): the new iσ_y · conj(U) rule — this IS the bug fix, only fires on TRS rows. |
| 5 | symmetry_maps.py | 364-545 | Pure rename + field-structure | Class docstring updated to describe `irr_idx_k`/`sym_idx_k`/`irr_idx_q`/`sym_idx_q`. Trivial-identity init (ntran ≤ 1) renames assignments at lines 405-406 (was 51-52 of baseline). Spatial-symmetry init: same `sym_matrices`/`sym_mats_k`/`translations` build (lines 477-485), same TRS-concat (lines 490-491), same `kpoint_map` + `unfolded_kpts` (line 494), same kpoint_map_ibz_ids bounds check (495-503). |
| 6 | symmetry_maps.py | 505 | Pure rename | `self.irk_to_k_map, self.irk_sym_map = find_symmetry_ops_simple(...)` → `self.irr_idx_k, self.sym_idx_k = ...`. `find_symmetry_ops_simple` body unchanged (lines 703-726 vs baseline 309-326), so the assigned values are identical. |
| 7 | symmetry_maps.py | 513-519 | Pure rename | `kirr_fullids` build loop uses `self.irr_idx_k` (was `self.irk_to_k_map`). Logic unchanged. |
| 8 | symmetry_maps.py | 530-541 | Real semantic change (U_spinor shape shrink) | `self.U_spinor = self.get_spinor_rotations(wfn, self.R_cart[:wfn.ntran])` — now shape `(ntran, 2, 2)` (was `(2·ntran, 2, 2)`). `R_cart` itself still length-`2·ntran` (line 540 comment). Verified `sym.U_spinor` has no external consumers outside `wfn_loader.py` (grep across `src/`). |
| 9 | symmetry_maps.py | 568-583 | Algorithm refactor (eager q-IBZ) | Calls `find_irreducible_bz_points(self.kvecs_asints, self.sym_mats_k, irr_kgrid_int=None)`; assigns `self.irr_idx_q`/`self.sym_idx_q`/`self.q_irr_kgrid_int`. Then derives `self.q_irr_full_idx` as `sort(unique(irr_idx_q)[1])` — equals baseline's `first_idx.astype(int32)` for the same canonical orbit ordering (the IBZ list is built by `unique(canon_keys)[first_idx]` so first-occurrence-of-each-irr in the full-grid equals `first_idx`). |
| 10 | symmetry_maps.py | 703-726 | Body unchanged + diagnostic removal | `find_symmetry_ops_simple` body byte-identical to baseline lines 309-326. Only difference: removed post-loop warning about TRS-row unfold being unsafe (PR3 fixes it, warning obsolete). No behavioural change to the returned `(irk_to_k_map, irk_sym_map)` tuple. |
| 11 | wfn_loader.py | 306-307 | Pure rename | `sym.irk_sym_map[nk]` → `sym.sym_idx_k[nk]`; `sym.irk_to_k_map[nk]` → `sym.irr_idx_k[nk]`. Same data. |
| 12 | wfn_loader.py | 330 | Pure rename | `sym.irk_to_k_map` → `sym.irr_idx_k` in `ngk_valid`. |
| 13 | wfn_loader.py | 446-447 | Pure rename | phdf5 static-table renames. |
| 14 | wfn_loader.py | 449-466 | Real semantic change (U_per construction for TRS rows) | For spatial rows (`s < ntran`): `s_spatial_idx = s % n_tran == s`, so `U_per_spatial[k] = sym.U_spinor[s]`. `tr_mask[k] = False`, `np.where(False, U_per_trs, U_per_spatial) = U_per_spatial`. **Identical to baseline.** For TRS rows: applies new iσ_y · conj(U) rule (the fix). |
| 15 | wfn_loader.py | 478-492 | Real semantic change (phase loop unified) | Removed `if s >= n_tran: continue` skip. Now for ALL rows: `s_spatial = s - n_tran if s >= n_tran else s`, `tau = sym.translations[s_spatial]`. Then identical phase math: `phase = exp(-1j * (sym.sym_mats_k[s] @ g_bar) @ tau)`. **For `s < ntran`: `s_spatial = s`, so τ and the formula are identical to baseline.** For TRS rows: the `sym_mats_k[s] = -S_spatial` automatically gives `exp(+i (S_spatial · g_bar) · τ) = conj(spatial-phase)` (the PR3 design rule). |
| 16 | wfn_loader.py | 847, 852-853 | Pure rename + algorithm refactor (eager unfold) | `sym.irk_sym_map → sym.sym_idx_k`, `sym.irk_to_k_map → sym.irr_idx_k`. Body refactored to call `unfold_psi(...)` instead of inline τ-phase + spinor-rotate. For `sym_idx < ntran`: traced through unfold_psi line 322-360 — same `S_full = sym_mats_k[sym_idx]`, same `tau = translations[s_spatial=sym_idx]`, same `rotated = (S_full @ g_bar.T).T`, same `phase = exp(-1j * rotated · tau)`, same `cnk *= phase[None,None,:]`, same `U_eff = U_spinor_spatial[s_spatial=sym_idx] = sym.U_spinor[sym_idx]`, same `einsum("jk,nkl->njl", U_eff, cnk)`. **Bit-identical**. |
| 17 | wfn_loader.py | 860 | Hoisted slice | `g_bar = self._gvecs_raw[start:end]` is now computed unconditionally (was inside the τ guard in baseline). Pure host-side numpy slice; no floating-point operation. No bit difference. |
| 18 | wfn_loader.py | 968-1022 | Body unchanged | `_phdf5_unfold_kernel` body byte-identical to baseline (verified via `diff` of the two slices — no output). For spatial rows: `tr_mask_per_k[k] = False`, `where(False, conj(cnk), cnk) = cnk` (no conj). Then multiply by phase, apply U. Identical to baseline. |

**Every hunk classifies cleanly into rename / algorithm-refactor-with-preserved-formula / real-semantic-change-confined-to-TRS-branch. No "I don't understand why this changed" remnants.**

---

## 2. Bit-equality claim for non-TRS rows

### Eager path (`wfn_loader._eager_build`, `sym_idx < ntran`)

**Pre-Phase-2** (`9e644e9` lines 823-850, with `sym_idx < ntran` selected):

```python
sym_krep = sym.sym_mats_k[sym_idx]
tau = self.translations[sym_idx]
if np.any(np.abs(tau) > 1e-12):
    g_bar = self._gvecs_raw[start:end]
    rotated = (sym_krep @ g_bar.T).T
    phase = np.exp(-1j * rotated.astype(np.float64) @ tau)
    cnk = cnk * phase[None, None, :]
cnk = np.einsum("jk,nkl->njl", U_per[sym_idx], cnk)   # U_per = sym.U_spinor (length 2·ntran), for sym_idx < ntran this IS the spatial spinor
```

**Post-PR3** (`a45f039` line 861-869, with `sym_idx < ntran` selected, traced into `unfold_psi`):

```python
# unfold_psi line 322-360 with is_trs = False, s_spatial = sym_idx
S_full = sym_mats_k[sym_idx]                          # == sym_krep above
tau = translations[s_spatial = sym_idx]                # same value
rotated = (S_full @ g_bar.T).T.astype(float64)         # same array
phase = np.exp(-1j * (rotated @ tau))                  # same exponent
cnk = cnk * phase[None, None, :]                       # same
U_eff = U_spinor_spatial[s_spatial = sym_idx]          # == sym.U_spinor[sym_idx]; for sym_idx < ntran, the PRE-PR3 sym.U_spinor[sym_idx] was the SAME spatial value (the bug was in the rows ≥ ntran, never the < ntran rows). U_spinor's first ntran entries are unchanged by the PR3 shape shrink.
cnk = np.einsum("jk,nkl->njl", U_eff, cnk)             # same
```

**Result**: bit-identical at IEEE-754 level for `sym_idx < ntran`. No reordering of fp ops, no extra cast, no different intermediate dtypes. ✓

One incidental difference: the new code unconditionally builds `g_bar = self._gvecs_raw[start:end]` at line 860 (outside the τ guard), where baseline built it inside the `if np.any(np.abs(tau) > 1e-12)` branch. This is a memory-bandwidth no-op — the slice exists in either case if τ is nonzero, and is computed-but-unused if τ is zero. No floating-point implications.

### Phdf5 path (`wfn_loader._ensure_phdf5_static`, `sym_idx_per_full < ntran`)

**Pre-Phase-2** (`9e644e9` lines 446-469):

```python
U_per = sym.U_spinor[sym_idx_per_full]      # length-2·ntran table; for s<ntran returns spatial spinor
# phase loop:
for nk in range(nk_full):
    s = sym_idx_per_full[nk]
    if s >= n_tran: continue                # SKIP TRS rows (phase stays 1.0)
    tau = sym.translations[s]
    if not np.any(np.abs(tau) > 1e-12): continue
    g_bar = self._gvecs_raw[start:start+ngk_k]
    rotated = (sym.sym_mats_k[s] @ g_bar.T).T
    phase[nk, :ngk_k] = np.exp(-1j * rotated.astype(float64) @ tau)
```

**Post-PR3** (`a45f039` lines 459-492):

```python
U_spatial = sym.U_spinor                            # NOW length-ntran
s_spatial_idx = sym_idx_per_full % n_tran           # for s<ntran: s_spatial_idx == s
U_per_spatial = U_spatial[s_spatial_idx]            # for s<ntran: identical to pre-Phase-2 U_per
U_per_trs = np.einsum('ij,kjl->kil', iσ_y, conj(U_per_spatial))
U_per = np.where(tr_mask[:,None,None], U_per_trs, U_per_spatial)
                                                    # for s<ntran: tr_mask[k]=False → U_per[k] = U_per_spatial[k] = pre-Phase-2 value

# phase loop (unified):
for nk in range(nk_full):
    s = sym_idx_per_full[nk]
    s_spatial = s - n_tran if s >= n_tran else s    # for s<ntran: s_spatial == s
    tau = sym.translations[s_spatial]               # for s<ntran: identical to baseline tau
    if not np.any(np.abs(tau) > 1e-12): continue
    g_bar = self._gvecs_raw[start:start+ngk_k]
    rotated = (sym.sym_mats_k[s] @ g_bar.T).T      # for s<ntran: identical sym_mats_k row
    phase[nk, :ngk_k] = np.exp(-1j * rotated.astype(float64) @ tau)
                                                    # for s<ntran: identical
```

**Result**: bit-identical at IEEE-754 level for `sym_idx_per_full < ntran`. The shape change `sym.U_spinor: (2·ntran,...) → (ntran,...)` is invisible to the spatial branch because `U_per_spatial[s_spatial_idx]` indexes the same first-ntran rows in both versions, and those rows were never the buggy half of the pre-PR3 array. ✓

The `_phdf5_unfold_kernel` body (lines 988-1022 of HEAD) is unchanged at byte level from baseline (lines 968-1004 of `9e644e9`). For spatial rows, `where(tr_mask=False, conj(cnk), cnk) = cnk`, so the conj op is a no-op. The phase multiply and spinor einsum use the same identical inputs as baseline.

### V_q unfold (`unfold_v_q`, `n_sym_spatial=ntran`, all sym_idx values)

**Pre-Phase-2** (`9e644e9` `gw/v_q_tile._unfold_v_q_ibz_to_full`) and **post-PR2** (`a45f039` `common.symmetry_maps.unfold_v_q`):

Side-by-side traced — every algorithmic line matches:
- Trivial-IBZ short-circuit (`return V_q_ibz` if identity)
- TRS hard-fail check `max_sym >= n_sym_perm`
- TRS consistency check `2*n_sym_spatial == n_sym_perm` (post-PR2: only enforced when TRS actually used)
- `inv_perm = argsort(sym_perm, axis=-1)` + pad to `n_rmu_padded`
- jitted inner: `perm_q = inv_perm_j[sym_j]`, `V_at_irr = V_ibz[idx_j]`, two `take_along_axis(mode='promise_in_bounds')` calls, `where(trs_mask, conj(V_full), V_full)`

The TRS branch fires only at `sym_idx >= n_sym_spatial` rows — same conj-Hermitian rule both before and after. For non-TRS rows the operation is purely the centroid double-permute. **Bit-identical.** ✓

---

## 3. Second-order risk audit

| Risk | Status | Evidence |
|------|--------|----------|
| `U_spinor` reader in BSE / PSP / Sternheimer code | **CLEAN** | `grep -rn "U_spinor" src/bse/ src/psp/` → zero matches. The only `sym.U_spinor` consumer in `src/` is `wfn_loader.py` (which was migrated). |
| Other `irk_to_k_map` / `irk_sym_map` readers | **CLEAN in production** | All remaining hits are in `tests/archive/` and `misc/archived_tests/` (dead code). Production `src/` is fully renamed. |
| `_q_irr_table_cache` / `find_irreducible_qpoints` consumers | **CLEAN** | No external readers; the method was deleted and replaced with eager `__init__` attrs. |
| `_I_SIGMA_Y` exposure to unmigrated code | **CLEAN** | Only `wfn_loader.py:458` (migrated) + `tests/test_unfold_psi_trs.py` (new test) read it. Leading-underscore convention preserves "private API" intent. |
| Test file rename hiding a regression | **CLEAN** | `git log --diff-filter=R 9e644e9..a45f039 -- tests/ src/` → no renames. Existing tests `test_q_ibz_and_centroid_perm.py` and `test_v_q_ibz_unfold.py` were updated in-place to use the new API; same physical scenarios exercised. |
| Cleanup commit `69ab42c` touching audited files | **NO** | `git show --stat 69ab42c` confirms it touches `AGENTS.md`, `README.md`, `docs/`, `pyproject.toml`, `src/gw/gw_init.py`, `src/psp/run_sternheimer.py`, `config/modulefiles/lorrax/0.1.0.lua` — neither audited file. |
| Stale-API fix `a45f039` touching audited files | **NO** | `git show --stat a45f039` confirms it touches `src/psp/dft_operators.py` only. |
| Stale references to `_unfold_v_q_ibz_to_full` name | **DOCSTRING/COMMENT ONLY** | `grep -rn "_unfold_v_q_ibz_to_full" src/` finds 7 hits — all in comments/docstrings (`zeta_reader.py:34`, `orbit_syms.py:274`, `v_q_tile.py` x5, `v_q_g_flat.py:28`). No code paths call the removed name; production calls `common.symmetry_maps.unfold_v_q`. Cosmetic-only; not a bug. |

---

## 4. Per-PR confidence ranking

| PR | Risk for non-TRS regression | Rationale |
|----|-----------------------------|-----------|
| **PR1** (`5da9ec7` — index-table consolidation, eager q-IBZ) | **LOW** | Pure rename + eager call into a refactored helper whose body byte-matches the deleted lazy method. The k-side path (`find_symmetry_ops_simple`) body is byte-unchanged. The q-side path produces the same `(q_irr_kgrid_int, irr_idx_q, sym_idx_q, q_irr_full_idx)` arrays as the old lazy method's tuple. No production consumers of the old field names remain. |
| **PR2** (`a00722d` + `796c043` — V_q unfold lift) | **VERY LOW** | Body lifted unchanged. Only narrowing: `n_sym_spatial` is required (production callers always pass it). Trivial-IBZ short-circuit, the inverse-permutation pad, the `take_along_axis(promise_in_bounds)` double-permute, and the `where(trs_mask, conj, V)` are all line-for-line identical. |
| **PR3** (`8504994` — `unfold_psi` + bispinor U_spinor TRS fix) | **LOW** for non-TRS rows / **MEDIUM** for TRS rows (intentional fix) | The eager-path migration through `unfold_psi` was traced operation-by-operation; for `sym_idx < ntran` every fp op (τ lookup, S·G rotate, phase exp, cnk × phase, U einsum) is bit-identical. The phdf5 static-table change uses `np.where(tr_mask=False, ...) = U_per_spatial` for spatial rows, again bit-identical. The phdf5 unfold kernel body is byte-unchanged. The U_spinor shape shrink only deletes rows ≥ ntran, which were unreachable for spatial sym_idx anyway. **The bispinor TRS-row fix is the real semantic change, by design, confined to `is_trs=True` and `tr_mask[k]=True` branches.** |

---

## 5. Specific code-line citations of concerns

None found. The audit raised no bugs.

A few cosmetic / minor stylistic notes (not regression risks, NOT requiring fix to ship):

1. **Stale comment references to `_unfold_v_q_ibz_to_full`**:
   - `src/file_io/zeta_reader.py:34`
   - `src/centroid/orbit_syms.py:274`
   - `src/gw/v_q_tile.py:1509, 1538, 1593, 1604, 1622, 1639, 1654`
   - `src/gw/v_q_g_flat.py:28`
   These are all in comments / docstrings referring to the now-moved function by its old name. Searching by name would mislead a reader. Low-priority docstring sweep candidate, not load-bearing.

2. **Hoisted `g_bar` slice in eager unfold** (`wfn_loader.py:860`): always computed, even when `tau == 0`. Pure host slice, no fp implications. Minor wasted memory bandwidth on symmorphic systems; not a correctness issue.

3. **`_I_SIGMA_Y` leading-underscore**: only intra-`common.symmetry_maps` plus the migrated wfn_loader; the leading underscore correctly signals "do not consume from elsewhere." No action.

4. **`unfold_v_q` slight contract narrowing**: `n_sym_spatial` is now required (was optional with `= None` default). All production callers pass it; the test suite tests pass it. The `None` default branch was a dead-on-arrival defensive default in the baseline; removal is fine. If a future caller forgets to pass it, they'll get a `TypeError`, which is the right failure mode.

---

## 6. Final verdict

**SHIP — high confidence no regression in non-TRS row math.**

If the sym-vs-nosym ground-truth check on the non-inversion bispinor system fails the <1 meV gate, the failure is NOT in this diff. Candidates to investigate next (outside this audit's scope):

- The iσ_y · conj(U) rule itself — does the sign convention match what BGW's `mtxel_cor.f90` assumes? (See `agent_4_reference_audit.md`.)
- The phase sign for TRS rows: the new code relies on `sym_mats_k[TRS row] = -S_spatial` giving `exp(+i (S·G_bar)·τ)`. If the input WFN file's `tnp` (BGW translations) has an opposite-sign convention to what `unfold_psi` assumes, the TRS phase will flip sign. Worth checking on a non-symmorphic non-inversion system specifically.
- The `s_spatial = s - n_tran` arithmetic: for TRS rows `s ∈ [ntran, 2·ntran)`, `s_spatial = s - n_tran ∈ [0, ntran)`. Verify the corresponding `S_spatial = sym_matrices[s_spatial]` matches BGW's `mtrx(sym_inv)` for TRS-augmented rows in the bispinor convention.
- Centroid `sym_perm` extension by `extend_trs=True`: PR3 design states rows `[ntran:]` duplicate `[:ntran]` (TRS keeps r fixed). Verify this on the same non-inversion bispinor case — for an op like C2(z) followed by TRS, does the centroid orbit really collapse onto the spatial-row permutation?

None of these candidates are in `wfn_loader.py` or `symmetry_maps.py`'s spatial-sym path.

**Audit complete. Read-only. No source modifications.**
