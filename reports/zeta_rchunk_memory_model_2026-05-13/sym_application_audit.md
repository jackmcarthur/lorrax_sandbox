# Sym-application audit: `compute_centroid_sym_perm` is correct; centroid set is unclosed

**Date:** 2026-05-14
**Triggered by:** CrI3 6×6 80 Ry warning
`compute_centroid_sym_perm: centroid orbit closure failed. sym 1 maps centroid μ=0 (at fft_idx [50, 22, 2]) to fft_idx [47, 25, 2], which is NOT in the centroid table. Total failures: 7444 / 9024.`
**Verdict:** **NO_BUG** in `orbit_syms.py`. The math is correct; the failure is that the centroid file was generated with non-orbit-aware kmeans.

---

## 1. BGW / QE convention verdict

**Ground truth — BGW source (`Common/symmetries.f90`):**

- spglib returns `mtrx_inv` — the real-space rotation in the convention "rotate, then translate" (`r' = mtrx_inv @ r + τ`). See line 159 of `wfn.h5.spec`: *"We use spglib, so convention is rotate, then translate."*
- BGW then stores `mtrx = invert_matrix_int(mtrx_inv)` (line 189). So **`mtrx` acts on G-vectors (column convention)**; the real-space rotation is `inv(mtrx)`.
- BGW stores `tnp = 2π · τ_frac` (line 190). So `τ_frac = tnp / (2π)`.

**Therefore the canonical real-space action is:**
```
r' = inv(mtrx) @ r + τ_frac    (column form)
   = r @ inv(mtrx).T + τ_frac  (row form)
```

**Code under audit (`orbit_syms.py:285`):**
```python
Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)   # S == BGW mtrx
images = np.einsum('rj,sij->sri', r_frac, Rinv) + tau_frac[:, None, :]
```

The einsum `'rj,sij->sri'` produces `images[s,r,i] = sum_j r_frac[r,j] * Rinv[s,i,j] = (Rinv[s] @ r_frac[r])_i`. That is exactly `r' = Rinv @ r + τ` in column form — the canonical BGW convention. **CORRECT.**

Cross-references in the codebase that use the same convention:
- `SymMaps.validate_atomic_symmetries` (`symmetry_maps.py:468`): `rot = inv(wfn.sym_matrices[s]); transformed = rot @ pos + tau` — same convention. CONSISTENT.
- `compute_rgrid_sym_perm` (`orbit_syms.py:415`): uses the identical einsum. CONSISTENT.
- `unfold_orbit_unique_with_id` (`orbit_syms.py:176, 184`): `np.einsum('ri,sji->srj', reps, Rinv)` — this is `r_frac[r,i] · Rinv[s,j,i] → result[s,r,j] = (Rinv[s].T @ r_frac[r])[j]`. That is `r @ Rinv.T = Rinv.T @ r` in col form, i.e. **`Rinv^T`, not `Rinv`**. ⚠️ This is a different convention! See §6 for impact.
- `SymMaps.find_irreducible_qpoints` (`symmetry_maps.py:409`): `np.einsum('sij,qj->sqi', Smk, full)` = `Smk @ q`. This acts on **k-points** with `sym_mats_k = sym_matrices.transpose(0,2,1)` (line 111), so it's `S^T @ k` = the correct k-vector action (since `mtrx` acts on G as col-vec, k transforms by `mtrx^T` from row → col convention). CONSISTENT.

**Misleading comment:** `symmetry_maps.py:110` says `sym_matrices "apply to real space coords as sym_matrices[i] @ [rx,ry,rz]"` — this is **wrong** (or at best ambiguous). The actual code throughout uses `inv(sym_matrices)` for real-space r. Comment should be fixed.

## 2. What `compute_centroid_sym_perm` actually computes (precisely)

For each sym `s`, each centroid `μ`:
```
r_μ_frac = idx[μ] / fft_grid
Rinv_s   = inv(sym_matrices[s])      (integer 3×3)
τ_s      = translations[s] / (2π)
r'_frac  = (Rinv_s @ r_μ_frac + τ_s) mod 1     [col-form math; einsum impl]
img_idx  = round(r'_frac * fft_grid) mod fft_grid
sym_perm[s, μ] = ν such that idx[ν] == img_idx,  or -1 if absent.
```
This is the **forward image** `r_{π_s(μ)} = (S_s, τ_s) · r_μ`. Identical recipe to `validate_atomic_symmetries`. ✓

## 3. Synthetic test results (`/tmp/sym_audit/test_compute_sym_perm.py`)

Five cases on 8³ or 9³ grids, with `expected` images hand-computed and centroid sets pre-closed under the test sym group:

| Case | S | τ | Result |
|------|---|----|----|
| Identity | I | 0 | **PASS** |
| Inversion | –I | 0 | **PASS** |
| C2 about z | diag(–1,–1, 1) | 0 | **PASS** |
| Pure translation | I | (½, 0, 0) | **PASS** |
| C3 about z (**non-orthogonal**) | [[0,–1,0],[1,–1,0],[0,0,1]] | 0 | **PASS** |

The C3 case is the discriminating test: orthogonal S has `S = S^T = S^{-T} = ±S^{-1}` so any wrong convention can accidentally pass. The non-orthogonal C3 only passes if the convention is exactly `Rinv = inv(S)` acting on column-vector r. **The implementation passes.**

## 4. Real CrI3 test (`/tmp/sym_audit/test_actual_failure.py`)

Reproducing the exact failure from `lorrax_B_round8_validation_2026-05-14/gw.out:186` —
sym 1 mapping `[50, 22, 2]` on `fft_grid = (75, 75, 200)`:

- `sym 1 mtrx = [[0,-1,0],[1,-1,0],[0,0,1]]` (det = +1, a C3 rotation in the hex plane)
- `τ_frac = (0, 0, 0)`
- `inv(S) = [[-1,1,0],[-1,0,0],[0,0,1]]`
- Hand-computed image: `Rinv @ (50/75, 22/75, 2/200) = (-28/75, -2/3, 1/100) → mod 1 → (47/75, 25/75, 2/200)` → grid `(47, 25, 2)`.

The function returns `(47, 25, 2)`. The log message reports `(47, 25, 2)`. **Math is correct.**

Comparison against all alternative conventions on the same centroid:

| Convention | Image |
|------|------|
| `Rinv @ r` (code) | **(47, 25, 2)** ← matches log |
| `S @ r` | (53, 28, 2) |
| `S^T @ r` | (22, 3, 2) |
| `S^{-T} @ r` | (3, 50, 2) |
| `r @ S` | (22, 3, 2) |
| `r @ inv(S)` | (3, 50, 2) |

Only the actual code path matches the log; nothing else would produce the reported failure index.

## 5. Why does closure fail? (`/tmp/sym_audit/check_centroids_closure.py`)

Loading the actual 1504-centroid file `centroids_frac_1504.txt` and counting closures per sym op:

| Sym op | Description | Images in centroid set |
|------|------|------|
| 0 | identity | 1504 / 1504 ✓ |
| 1 | C3+ | 10 / 1504 |
| 2 | C3– | 10 / 1504 |
| 3 | inversion (–I) | 24 / 1504 |
| 4 | S6+ | 16 / 1504 |
| 5 | S6– | 16 / 1504 |
| **Total** | | **1580 / 9024** — i.e. **7444 failures, exactly matching the log.** |

The centroids are essentially generic real-space points, closed only under identity. They were not produced by the orbit-aware kmeans path (`kmeans_cli.py:265–304`), or the orbit-awareness step silently degraded to identity.

I also tried the wrong conventions (`S @ r`, `S^T @ r`) as a sanity check: closures of 1580/9024 and 1552/9024 respectively — none of them give closure, so no wrong-convention "fixes" this. The centroid set just isn't generated to respect the sym group.

## 6. Side observation: `unfold_orbit_unique_with_id` uses `Rinv^T` not `Rinv`

In `orbit_syms.py:176` and `:184`:
```python
images = np.einsum('ri,sji->srj', reps_np, Rinv) + tau[:, None, :]
```
Expanding indices: `images[s,r,j] = sum_i reps[r,i] · Rinv[s,j,i] = (Rinv[s].T @ reps[r])[j]`.
That is `r' = Rinv^T @ r + τ` — **the transpose** of what `compute_centroid_sym_perm` uses (`Rinv @ r`).

For self-inverse symmetry groups (where every op is its own inverse, common for low-symmetry groups), `Rinv` and `Rinv^T` may both produce closed orbits and the inconsistency is invisible. But for groups containing e.g. C3 rotations, these give *different* orbits. This is an existing latent inconsistency between the two functions in the same file and **deserves a follow-up** even though it isn't the cause of the current 7444 failures.

Note that `_orbit_lex_winner` / `canonicalize_orbit` use `orbit_images` (line 78) which is `reps @ Rinv.T + τ` = `Rinv @ reps + τ` in col form — that **matches** `compute_centroid_sym_perm`. So `unfold_orbit_unique_with_id` is the odd one out; this is likely a typo (`'sij'` → `'sji'`) in either line 176 or line 184.

## 7. Verdict and recommendation

- **`compute_centroid_sym_perm` convention: CORRECT.** No bug there. The hypothesis that "real-space r needs S^{-T} or some non-Rinv variant" is **falsified** by both the synthetic non-orthogonal C3 test and the cross-check against BGW source.
- **The 7444/9024 failure is real, but its cause is upstream:** the kmeans pipeline emitted centroids that don't close under the WFN's sym group. Either:
  - the run was launched without the orbit-aware kmeans path (`--no-orbit`, or `wfn.ntran ≤ 1` detection misfired), or
  - orbit-aware kmeans converged but the post-kmeans snap-to-grid step (`snap_orbits_to_grid`, `orbit_syms.py:120`) didn't preserve orbit closure for this fft_grid, or
  - the centroids file is stale (generated against an older WFN with a smaller sym group).

**Concrete next steps for the user (in order):**
1. Check `kmeans_cli` invocation logs in the failing run dir — confirm `orbit_aware=True` was actually selected. If `ntran > 1` it should auto-enable; verify that ran.
2. Regenerate the centroid file from scratch with explicit `--orbit` and validate inline: every line in the output file's images-under-sym should be back in the file.
3. **Separately**, audit `unfold_orbit_unique_with_id` lines 176/184 — the `'ri,sji->srj'` einsum is inconsistent with the rest of the module (likely a typo for `'rj,sij->sri'`). This won't affect the current failure but is a latent correctness bug for non-self-inverse symmetry ops.
4. Fix the misleading docstring at `symmetry_maps.py:110` ("apply to real space coords as `sym_matrices[i] @ [rx,ry,rz]`") — should read "apply to G-vectors as `mtrx @ G`; real-space `r` transforms by `inv(mtrx) @ r + τ`".

---

**One-line summary:** `NO_BUG` — `compute_centroid_sym_perm` uses `r' = inv(S) @ r + τ` which matches BGW's `mtrx`-on-G convention; the 7444 failures are from a non-orbit-closed centroid set generated upstream by kmeans.
