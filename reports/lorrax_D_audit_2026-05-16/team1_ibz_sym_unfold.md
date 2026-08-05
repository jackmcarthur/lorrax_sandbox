# Team 1 — IBZ → full-BZ symmetry-routing audit on lorrax_D `agent/env-var-cleanup`

**Branch tip:** `488e870` (= `agent/env-var-cleanup`, lorrax_D)
**Audit date:** 2026-05-16
**Scope:** commits `f0af7c2 6075aa4 63c5eac f6f2372 991623a c4c8413 0880066 c13dbe0` plus their interaction with the pre-existing `sym_perm` consumers and the on-disk per-q sphere padding.

---

## 1 — Executive verdict

The Phase A/B/D landing on `agent/env-var-cleanup` is **not safe to merge to main as-is** on systems that fold k/q via time-reversal symmetry. The eight commits in scope are internally correct in algebraic and bit-level terms (per-element formulas check out for the rchunk ↔ G-flat pair, the IBZ-only triangular solve, the per-q sphere pad, and the BGW q-wrap), but they **leave intact the same TRS-blind index mismatch** that the prior incident (`reports/trs_sym_audit_2026-05-14/`) traced to a 6–14 eV Σ_X error on MoS2 3×3. The `agent/trs-aware-sym-fix` Phase 1 patch on lorrax_B was never ported across to lorrax_D `agent/env-var-cleanup`; `git log --all` confirms no `extend_trs` token, no length-`2·ntran` `sym_perm` construction, and no TRS-conjugation in `_unfold_v_q_ibz_to_full` exists in this tree. Independently, the unfold code path in `v_q_tile.py:1452-1735` is missing the **non-symmorphic L-phase factor** `exp(2πi q_irr · (L_μ − L_ν))` that the canonical formula (`SYMMETRY_CONVENTIONS.md:30-35`) requires — a separate bug class that fires on Si Fd-3m, P6_3/mmc, etc. and that was provisionally fixed on lorrax_B's `agent/trs-aware-sym-fix` (commit `0735c2a`) but again is not on this branch. The per-q sphere padding, phase-after-slice, and IBZ-only solve are clean.

---

## 2 — Per-element formulas

### 2.1 ζ unfold (full-BZ ζ from IBZ ζ)

Reference: BGW r-action `r' = inv(mtrx) · r + τ` (derivation `SYMMETRY_CONVENTIONS.md:238-279`). For spatial sym `s` with rotation `S_s = sym_matrices[s]` and translation `τ_s`, decompose

```
mtrx_s · (x_μ − τ_s) = x_{α_s(μ)} + L_{s, μ}     (column form; L ∈ ℤ³ is the lattice wrap)
```

Then `ζ_{S_s q, μ}(r) = exp(+2πi q · L_{s, μ}) · ζ_{q, α_s(μ)}(inv(S_s)·(r − τ_s))`. For TRS-augmented `s` composed with spatial `s'`: `ζ_{S_s q, μ}(r) = exp(−2πi q · L_{s', μ}) · conj(ζ_{q, α_{s'}(μ)}(inv(S_{s'})·(r − τ_{s'})))` (the conj from K-action on a single ζ leg).

### 2.2 V_q unfold (centroid double-permute + L-phase + TRS conj)

For full-BZ q1 with IBZ parent q_irr and spatial sym `s` such that `S_s · q_irr ≡ q1` (mod kgrid):

```
V_full[q1, μ', ν'] = exp(2πi q_irr · (L_{s, μ'} − L_{s, ν'}))
                   · V_ibz[parent, α_s(μ'), α_s(ν')]         (Eq. V — scalar/charge)
if s is TRS-augmented:   V_full = conj(V_full)
```

(Derivation: `SYMMETRY_CONVENTIONS.md:271-307`. The bilinear `R·V·Rᵀ` polarization mixing replaces the scalar identity in the transverse case.)

### 2.3 IBZ-only triangular solve

The CCT + Z_q system is **separable across q** (no inter-q coupling in either Cholesky or the back-solve). Given full-BZ `L_q` and full-BZ `Z_q`:

```
ζ_q = L_q^{-H} · L_q^{-1} · Z_q       (per q, independent)
```

So `ζ[q_irr_full_idx] = solve(L[q_irr_full_idx], Z[q_irr_full_idx])` is **algebraically identical** to running the full-BZ solve and indexing the result with `q_irr_full_idx`. No sym constraint enforced — it doesn't need one; the solve is q-local and the gather is just a row selection on a pure-q axis.

---

## 3 — Sub-gate grid

Notation: SYM = spatial-only sym op (`s ∈ [0, n_tran)`), TRS = TRS-augmented (`s ∈ [n_tran, 2·n_tran)`); SCALAR = charge `V_q^{0,0}`, TRANS = transverse `V_q^{ij}` (currently dead code, no call sites); SYMM-τ = symmorphic (τ_s = 0 for all s), NS-τ = at least one s has τ_s ≠ 0.

| Sub-gate | q location | sym row | channel | τ kind | Verdict | Citation |
|----------|-----------|---------|---------|--------|---------|----------|
| 1 | q interior | SYM `s<n_tran` | SCALAR | SYMM-τ | **PASS** | `v_q_tile.py:1452-1557` (correct double-permute; L=0 for SYMM-τ; no TRS conj needed) |
| 2 | q interior | SYM `s<n_tran` | SCALAR | NS-τ | **FAIL** (missing L-phase) | `v_q_tile.py:1549-1554`: gathers `V_at_irr` then `take_along_axis` on (μ,ν) only — no `exp(2πi q · (L_μ − L_ν))` factor anywhere. Formula at `SYMMETRY_CONVENTIONS.md:30-35` is mandatory for NS-τ. |
| 3 | q interior | TRS `s≥n_tran` | SCALAR | SYMM-τ | **FAIL** (silent OOB clip + missing conj) | `v_q_g_flat.py:172-177` builds `sym_perm` from `sym.sym_matrices[:n_tran]` → shape `(n_tran, n_rmu)`. `v_q_tile.py:1536` does `inv_perm_j[sym_j]` with `sym_j ∈ [0, 2·n_tran)` — JAX silently clips `s≥n_tran` to `n_tran−1`. Verified by direct `inv_perm_j[jnp.arange(8)]` experiment on a fabricated `(4,8)` table (this audit, in-session). Additionally, no `if TRS: V = V.conj()` in the helper. |
| 4 | q interior | TRS `s≥n_tran` | SCALAR | NS-τ | **FAIL** (combined sites 2+3) | Same code; both bugs compound. |
| 5 | q = 0 | (sym is identity at Γ for q-fold) | SCALAR | any | **PASS** (Γ is trivially self-symmetric; `full_to_irr_idx[Γ]=Γ`, `full_to_irr_sym[Γ]=0`) | `symmetry_maps.py:431-451` builds `q_full_to_irr_sym` by `np.where(irr_image_keys[:, irr_idx] == target)[0][0]` and identity is always among the matches. Confirmed by the `(idx == arange, sym == 0)` short-circuit at `v_q_tile.py:1494-1499`. |
| 6 | q on BZ boundary | SYM | SCALAR | SYMM-τ | **PASS** | Boundary q's resolve to the same IBZ rep under SYM only; centroid permutation is well-defined. |
| 7 | q on BZ boundary | TRS | SCALAR | SYMM-τ | **FAIL** | Same as Site 3. BZ boundary doesn't change the bug class — boundary q's that need TRS to fold (e.g. M, K under D3h on MoS2 3×3) hit the OOB clip. |
| 8 | q with star size < |G|-sphere | SYM | SCALAR | any | **PASS** in principle | The orbit-closure check at `_resolve_ibz_q_list:172-186` raises before reaching unfold; per-q sphere `ngk[q]` is decoupled from q-orbit size. The G-flat reader masks sentinel pads (`isdf_fitting.py:2203-2211`). |
| 9 | bispinor channel | TRS or SYM | TRANS | any | **CANNOT-DETERMINE** | `_unfold_v_q_ij_ibz_to_full` at `v_q_tile.py:1560-1666` is **dead code** — no caller in the tree (`grep -rn _unfold_v_q_ij_ibz_to_full src/` shows only the def + a self-referencing error string). Helper would still inherit Site-3's silent-OOB if called. |
| 10 | ζ on-disk pad slots | TRS | SCALAR | any | **PASS** (zeroed before write) | `isdf_fitting.py:2203-2211` `_mask = (_g_axis < _ngk_dev)` zeroes sentinel pad coefficients; reader keeps the sphere indexing per-q so the zeros never enter physical sums. |
| 11 | phase-after-slice (`to_rchunk`) | n/a | n/a | n/a | **PASS** | `wfn_transforms.py:408-417` slices first, then `apply_bloch_phase_on_slice` applies the separable factor on the r_len slab; `apply_bloch_phase_on_slice` at `wfn_transforms.py:736-794` decodes flat_r → (rx,ry,rz) on the slab indices. Algebraically `exp(2πi k · r) × ψ` then slice = slice then `exp(2πi k · r_slab)` exactly. |
| 12 | phase-on-slice (`accumulate_rchunk_to_gflat`) | n/a | n/a | n/a | **PASS** | `wfn_transforms.py:617-661`: per-q `exp(-2πi q · r)` table built once per trace; per-row gather by `q_row = i0+arange // n_mu_local`; pad rows clipped to `q=n_q-1` carry zero data (the rchunk pad was zero by construction). FFT box is zero-padded; phase is **NOT** applied to the zero-pad region (only to the data slab), which is the bug-free direction. |
| 13 | per-q sphere padding round-trip | n/a | n/a | n/a | **PASS** | Writer at `isdf_fitting.py:2203-2211` zeroes sentinel slots before write; reader (G-flat path) reads `zeta_q_G` directly without re-applying phase or FFT (`zeta_reader.py:1-42` doc). The sentinel `(nx//2, ny//2, nz//2)` is in-bounds for fancy-indexing under `mode='promise_in_bounds'`. |
| 14 | BGW q-wrap consistency writer↔consumer | n/a | n/a | n/a | **PASS** | Writer: `isdf_fitting.py:1778-1781` `_bgw_wrap_q: q > kg/2 → q − kg`. Consumer: `v_q_tile.py:1202-1204` `_qvec_wrap`: same direction. Confirmed identical wrap by inspection. |

Summary across 14 cells: **4 PASS in physically interesting NS-τ / TRS configurations**, **5 FAIL** (Sites 2, 3, 4, 7, 9 — all variants of the same two underlying bugs), **rest PASS or n/a**.

---

## 4 — Bug-class search results

### 4.1 `sym_perm` builders + sizes

| File:line | What's built | Returned shape | Index source used downstream |
|-----------|--------------|----------------|------------------------------|
| `centroid/orbit_syms.py:209-338` `compute_centroid_sym_perm` | per-rmu sym perm | `(n_sym, n_rmu)` where `n_sym = S.shape[0]` (caller chooses) | per-q `full_to_irr_sym` |
| `gw/v_q_g_flat.py:168-177` | calls `compute_centroid_sym_perm(sym.sym_matrices[:n_tran], ...)` | `(n_tran, n_rmu)` | `full_to_irr_sym ∈ [0, 2·n_tran)` |
| `gw/compute_vcoul.py:891-900` | identical pattern | `(n_tran, n_rmu)` | `full_to_irr_sym ∈ [0, 2·n_tran)` |
| `common/isdf_fitting.py:1755-1763` (closure pre-check) | identical pattern | `(n_tran, n_rmu)` | only used to raise — not downstream |

### 4.2 `sym_mats_k` (the TRS-augmented k-action table)

`common/symmetry_maps.py:111-124`: `sym_mats_k = concatenate([sym_matrices[:n_tran].T, -sym_matrices[:n_tran].T])` → length `2·n_tran`. **No spatial-only variant exists** anywhere in the tree (e.g. no `sym_mats_k_spatial` slice).

### 4.3 `find_irreducible_qpoints` and `q_full_to_irr_sym` range

`common/symmetry_maps.py:346-459`: uses `self.sym_mats_k` (length `2·n_tran`) in both `images = einsum('sij,qj->sqi', Smk, full)` (line 409) and the per-q resolution loop (line 440-451). Returned `q_full_to_irr_sym` therefore has values in **`[0, 2·n_tran)`**. Comment at line 377-381 documents this explicitly: "sym_mats_k already includes time-reversal (k → -k on every spatial op), so the resulting IBZ wedge is the smallest possible under (spatial + TRS). Callers that need to keep TRS off the q-fold should reduce sym_mats_k to its spatial-only slice before calling." — **but no caller in the changed files does this reduction**.

### 4.4 Index-vs-table-row collisions

In `_unfold_v_q_ibz_to_full` (`v_q_tile.py:1452-1557`), `_unfold_g0_ibz_to_full` (`v_q_tile.py:1669-1734`), and the dead `_unfold_v_q_ij_ibz_to_full` (`v_q_tile.py:1560-1666`): the line `perm_q = inv_perm_j[sym_j]` (resp. line 1536, 1725, 1649) indexes a `(n_tran, n_rmu_padded)` table with `sym_j` values in `[0, 2·n_tran)`. JAX `Array.__getitem__` silently clips OOB indices → all TRS rows resolve to `inv_perm[n_tran-1]` (the **wrong** permutation, regardless of which TRS op actually fires).

Verified in-session: with a fabricated `inv_perm = np.tile(arange(8)[None,:], (4, 1))` (n_tran=4, n_rmu=8), `inv_perm_j[jnp.arange(8)]` returns 8 copies of `arange(8)` (the identity-tiled rows happen to mask the symptom, but in production `inv_perm[s]` differs per `s`).

### 4.5 R_cart row count for the dead transverse helper

`common/symmetry_maps.py:163` `R_cart = self.syms_crystal_to_cartesian(wfn)`, which at line 525-546 applies `einsum('ij,njk,kl->nil', B_T_inv, self.sym_mats_k, B_T)` → length `2·n_tran`. So `R_cart[sym_j]` at `_unfold_v_q_ij_ibz_to_full:1660` would index correctly **if** the helper were called. TRS rows are `R_cart[n_tran + s] = −R_cart[s]` (linear in `sym_mats_k`); the bilinear `R V Rᵀ` is even in R, so polarization mixing is right by accident. The centroid-perm side at `inv_perm_j[sym_j]:1649` still has the Site-3 OOB-clip bug. So if/when this helper is wired, it inherits Site-3.

### 4.6 Missing L-table

`compute_centroid_sym_perm` (`orbit_syms.py:209-338`) computes `img_idx % fft_grid` (line 296) — it **discards** the integer lattice wrap `L_{s, μ}` that the canonical V_q unfold formula needs. The lorrax_B `agent/trs-aware-sym-fix` commit `0735c2a` added an `L_table` output (per `SYMMETRY_CONVENTIONS.md:32-39` and the test at `reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py`); lorrax_D has none of this. For symmorphic systems with all `τ = 0`, `L_{s, μ} = 0` for every (s, μ) and the missing phase is identically 1 → no observable effect; for NS-τ systems (Si Fd-3m, P6_3/mmc, …) it is a real bug.

---

## 5 — Specific issues found

### Issue I1 — TRS-augmented `full_to_irr_sym` indexes spatial-only `sym_perm` (silent OOB clip)

- **Severity:** Critical (10+ eV Σ_X errors on MoS2 3×3 at 30 Ry per prior audit)
- **Sites:** `gw/v_q_tile.py:1536`, `gw/v_q_tile.py:1649` (dead), `gw/v_q_tile.py:1725`
- **Code excerpt** (`v_q_tile.py:1527-1536`):
  ```python
  inv_perm_j = jnp.asarray(inv_perm)                  # (n_tran, n_rmu_padded)
  idx_j = jnp.asarray(np.asarray(full_to_irr_idx, dtype=np.int32))
  sym_j = jnp.asarray(np.asarray(full_to_irr_sym, dtype=np.int32))  # values in [0, 2·n_tran)
  ...
  perm_q = inv_perm_j[sym_j]                          # OOB-clipped silently for sym_j ≥ n_tran
  ```
- **What's wrong:** the table's first axis is sized `n_tran` (spatial-only), but `sym_j` carries TRS-augmented indices. JAX gather clamps to `n_tran−1` without raising.
- **Recommended fix:** port the `agent/trs-aware-sym-fix` Phase 1 from lorrax_B. Specifically: add `extend_trs: bool = False` to `compute_centroid_sym_perm`; when True, return `(2·n_sym, n_rmu)` with rows `[n_sym:]` duplicating rows `[:n_sym]` (TRS keeps r fixed). Have `_resolve_ibz_q_list` and `compute_vcoul`'s IBZ gate pass `extend_trs=True`. Add a hard-fail guard: if `extend_trs=False` but `full_to_irr_sym` has any value `≥ n_tran`, raise.
- **Recommended test:** synthetic test with `{I, σ_x}` and `noinv=False` (or any setup that forces TRS in `find_irreducible_qpoints`), run `_unfold_v_q_ibz_to_full` and compare against a hand-built reference. Also: add the "MoS2 3×3 same-basis IBZ-vs-full-BZ" e2e check (`reports/trs_sym_audit_2026-05-14/STATUS.md:14-18`) as a regression gate.

### Issue I2 — Missing TRS complex-conjugation in V_q unfold

- **Severity:** Critical
- **Sites:** `gw/v_q_tile.py:1452-1557` (`_unfold_v_q_ibz_to_full`), `gw/v_q_tile.py:1669-1734` (`_unfold_g0_ibz_to_full`)
- **Code excerpt:** `_do_unfold` at `v_q_tile.py:1533-1555` ends with `V_full = jnp.take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2, mode='promise_in_bounds')` and returns directly. No branch on `sym_j ≥ n_tran` to apply `V.conj()`.
- **What's wrong:** the formula at `SYMMETRY_CONVENTIONS.md:300-307` is `V_full = conj(V_S)` for TRS rows. For Hermitian V_q (charge channel) this is provably a no-op **only if the (μ, ν) double-permute also swapped to (ν, μ)** — see the discussion at `project_trs_blind_sym_bug.md:34-41`. The current code does neither the swap nor the conj, so the result is wrong for TRS rows even before the OOB-clip in I1 is fixed.
- **Recommended fix:** after the two `take_along_axis` calls, do
  ```python
  is_trs = sym_j >= n_tran   # static or device array
  V_full = jnp.where(is_trs[:, None, None], jnp.conj(V_full), V_full)
  ```
  with `n_tran` baked in at trace time. The g0 helper (single-leg ζ) needs the same conj.
- **Recommended test:** unit test `test_unfold_v_q_trs_row_conj` — synthetic 2-row sym table `{I, TRS·I}`, hand-built complex V_q_ibz, verify `V_full[q_trs] = conj(V_full[q_id])`.

### Issue I3 — Missing non-symmorphic L-phase in V_q unfold

- **Severity:** High (manifests as ~50–800 meV on NS-τ; was 160 eV in Si before related Phase-2 changes per `trs_sym_audit_2026-05-14`)
- **Sites:** `gw/v_q_tile.py:1452-1557`; root cause at `centroid/orbit_syms.py:286-296` (computes `img_idx % fft_grid`, discards `L = (img_idx - α_idx) // fft_grid`).
- **Code excerpt:** `orbit_syms.py:295-296`:
  ```python
  img_idx = np.rint(images * fft_grid_np[None, None, :]).astype(np.int64)
  img_idx = img_idx % fft_grid_np[None, None, :]   # ← L_{s, μ} thrown away here
  ```
- **What's wrong:** the canonical formula at `SYMMETRY_CONVENTIONS.md:30-35` requires multiplying `V_ibz[parent, α_μ, α_ν]` by `exp(2πi q_irr · (L_μ − L_ν))` where `L_{s, μ}` is the integer lattice wrap from `mtrx_s · (x_μ − τ_s) = x_{α_s(μ)} + L_{s, μ}`. lorrax_D never captures `L`; the unfold therefore omits this phase factor. For SYMM-τ (CrI3 P-3, MoS2 D3h) `L = 0` and the bug is silent; for NS-τ it is observable.
- **Recommended fix:** mirror lorrax_B's commit `0735c2a` on `agent/trs-aware-sym-fix`: extend `compute_centroid_sym_perm` to also return `L_table` of shape `(n_sym, n_rmu, 3)` int32; thread it through `_resolve_ibz_q_list` and the unfold helpers; apply `phase = exp(2πi · (L[sym_j, μ] − L[sym_j, ν]) · q_irr_frac[parent])` at the appropriate axes.
- **Recommended test:** `tests/test_v_q_ibz_unfold.py` currently has all-identity perms and zero τ; add `test_unfold_v_q_nonsymmorphic_L_phase` with a hand-constructed `(α, L, τ)` triple and a synthetic V_q_ibz.

### Issue I4 — `test_v_q_ibz_unfold.py` and `test_v_q_transverse_unfold.py` do NOT exercise the bug class

- **Severity:** Medium (test-suite blind spot — prior incident was "shipped silently broken" because of this)
- **Sites:** `tests/test_v_q_ibz_unfold.py:1-176`, `tests/test_v_q_transverse_unfold.py:1-168`
- **What's wrong:** every test uses `sym_perm` with first axis ≤ 2 and `full_to_irr_sym = np.array([0, 1, ...])` — none of which exceed the perm's first axis. The OOB-clip path is never hit; the TRS-conjugation path is never required; the L-phase is never non-zero. All 24 tests pass while the production code on a real WFN with `n_tr > 0` (e.g. MoS2 NSCF emits "SymMaps: 4/9 full-BZ k-points require time-reversal symmetry for unfolding" — `symmetry_maps.py:329-336`) silently produces wrong V_q.
- **Recommended fix:** add a `noinv=False` style sub-case to each test: deliberately pass `full_to_irr_sym` values that exceed `sym_perm.shape[0]` and assert that the helper either raises (with the recommended hard-fail guard from I1) or — once the fix lands — produces the correct conjugated/perm'd result.

### Issue I5 — `_unfold_v_q_ij_ibz_to_full` is dead code

- **Severity:** Low (latent — fires only if/when transverse IBZ-only is wired)
- **Sites:** `gw/v_q_tile.py:1560-1666`; no callers anywhere in `src/`
- **What's wrong:** introduced by commit `991623a` (Phase D) with a `# NOT YET WIRED into the bispinor V_q writer` note in the commit message. The helper has the Site-3 OOB-clip bug latent; if a future bispinor IBZ orchestrator calls it without first fixing I1, the bug ships silently in the transverse channel too. The transverse-specific polarization mixing `R V Rᵀ` is correct by R-evenness even for TRS rows (`R_cart[trs_row] = −R_cart[spatial_row]` and the bilinear is even in R), so only the centroid perm is wrong — same symptom as the scalar case.
- **Recommended fix:** either delete (preferred — re-add when wiring is ready) or guard at function entry: `if np.asarray(full_to_irr_sym).max() >= np.asarray(sym_perm).shape[0]: raise ValueError("TRS-augmented full_to_irr_sym requires extend_trs=True sym_perm").`

### Issue I6 — `accumulate_rchunk_to_gflat` pad-row phase + phase-after-slice math

- **Severity:** Informational (both PASS)
- I6a (`wfn_transforms.py:642-643`): pad rows from `pad_N` clamp to `q_row = n_q − 1`, but their slab data is zero (line 622's `jnp.pad`) so `contrib = 0` regardless of which `q_row` is used. Worth a one-line comment.
- I6b (`wfn_transforms.py:408-417` + `:778-787`): `apply_bloch_phase_on_slice` decodes `flat = r0 + arange(r_len)` then `rx = flat // (ny·nz); ry = (flat // nz) % ny; rz = flat % nz` — matches the C-order convention at line 351, so per-slab `px[k,rx]·py[k,ry]·pz[k,rz]` reproduces the global-box `apply_bloch_phase` at the same flat-r position. `test_rchunk_gflat_pair.py` enforces this 4/4.

---

## 6 — Open questions / unresolved by reading alone

### Q1 — Does any production run on lorrax_D `agent/env-var-cleanup` activate the TRS-augmented unfold today?

Reading alone: `gw_init.py:644` sets `_write_ibz_only_charge = not bool(cfg.bispinor)`; `_resolve_ibz_q_list` then gates `use_ibz=True` on centroid orbit closure succeeding. For a non-bispinor charge run on MoS2 3×3 with orbit-closed centroids and TRS-folded q's, the bug fires.

**What would resolve it:** run the MoS2 3×3 same-basis IBZ-vs-full-BZ check (`reports/trs_sym_audit_2026-05-14/STATUS.md:14-18` recipe) on this tree's HEAD; if max |ΔΣ_X| ≫ ULP, the bug is hot. (Cannot run heavy GPU jobs in this audit per scope.)

### Q2 — Is the L-phase actually zero for every (s, μ) on CrI3 6×6 30 Ry in the current centroid set?

The CrI3 P-3 group is symmorphic (`τ = 0`), so `L_{s, μ} = mtrx_s · x_μ − α_{s, μ}` is always zero **if** `mtrx_s · x_μ` lands inside the unit cell. For x_μ near a cell boundary it may wrap. `compute_centroid_sym_perm:286-296` discards L without recording its distribution.

**What would resolve it:** instrument `compute_centroid_sym_perm` to also compute and return `L_table = (img_idx - alpha_idx) // fft_grid`, then dump max|L| across (s, μ) for the production CrI3 centroid file. If max|L|=0, Issue I3 is silently a no-op on CrI3; if max|L|>0, the L-phase is missing in CrI3 too.

### Q3 — Will the `_n_q_disk` shape gate at `v_q_g_flat.py:325-330` ever raise on a mismatched zeta_q.h5 / sym pair?

The reader's `gvec_components.shape[0]` (from disk) must equal `n_q_ibz` (resolved at runtime from `sym.find_irreducible_qpoints`). If the writer's WFN had `extend_trs=True` semantics (more TRS-folding → smaller IBZ) but the reader's sym setup is different, the shapes diverge and the guard fires. Today both use the same `find_irreducible_qpoints` via `sym_mats_k` length `2·n_tran`, so it's consistent — but the day someone tries to mix sym-vs-nosym ζ files this is a tripwire.

**What would resolve it:** add a structural metadata write to `isdf_header` capturing the `extend_trs` flag at write time, validated at read time. (Out of scope for I1 but a natural extension.)

### Q4 — Is `_resolve_ibz_q_list:169` `n_tran` capture intentional?

`sym.sym_matrices` is `wfn.sym_matrices[:wfn.ntran]` (length `n_tran`); the slice is redundant but reads correctly. No `extend_trs` path is offered. Reading alone cannot distinguish "always meant to be spatial; TRS handling lives elsewhere" from "this is the bug." Cross-reference `agent/trs-aware-sym-fix:9e644e9` on lorrax_B would resolve.

---

## 7 — Recommended action

1. **Block** any plan to merge `agent/env-var-cleanup` to main until Issues I1 + I2 are resolved on this tree (port `agent/trs-aware-sym-fix` Phase 1 across, or recreate equivalently).
2. **Extend** the unfold test suite (Issue I4) before the fix lands — the prior incident shipped because tests were blind to TRS-augmented inputs. Sub-gate enumeration per Section 3 above.
3. **Triage** Issue I3 separately. The L-phase fix is independent of the TRS fix and benefits NS-τ systems specifically (Si, P6_3/mmc TMD polytypes, …); SYMM-τ users are not affected today.
4. **Keep** the Phase A (rchunk ↔ G-flat), Phase B (IBZ-only solve), per-q sphere, and BGW q-wrap work — those are sound and the audit didn't find issues with them.

---

## Appendix — File map + test coverage

Key sites: `sym_mats_k` TRS-augment `symmetry_maps.py:110-124`; `find_irreducible_qpoints` (sym ∈ `[0, 2·n_tran)`) `symmetry_maps.py:346-459`; `compute_centroid_sym_perm` (returns `(caller's n_sym, n_rmu)`) `centroid/orbit_syms.py:209-338`; `_resolve_ibz_q_list` (builds `sym_perm` with `[:n_tran]` slice) `gw/v_q_g_flat.py:153-204`; `_unfold_v_q_ibz_to_full` `gw/v_q_tile.py:1452-1557`; `_unfold_v_q_ij_ibz_to_full` (dead) `1560-1666`; `_unfold_g0_ibz_to_full` `1669-1734`; IBZ-only solve `isdf_fitting.py:1306-1330`; `accumulate_rchunk_to_gflat` `wfn_transforms.py:468-673`; `to_rchunk` phase-after-slice `wfn_transforms.py:338-429`; `apply_bloch_phase_on_slice` `wfn_transforms.py:736-794`; per-q sphere `coulomb_sphere.py:128-247`; sentinel zero-fill `isdf_fitting.py:2197-2212`; BGW q-wrap (writer/consumer) `isdf_fitting.py:1778-1781` / `v_q_tile.py:1202-1204`; gate `gw_init.py:644`; closure pre-check `isdf_fitting.py:1748-1771`.

Tests at HEAD `488e870` (24/24 pass): `test_v_q_ibz_unfold.py` (3 cases — identity sym + small perms only), `test_v_q_transverse_unfold.py` (4 cases — identity + −I + π/2 z-rot + μ-pad), `test_per_q_sphere.py` (6), `test_rchunk_gflat_pair.py` (4), `test_symmetry_maps_kpoint_map.py` (spatial-only path), `test_zeta_loader.py` (4 G-flat round-trip). **None** of the unfold tests pass a `full_to_irr_sym` value exceeding `sym_perm.shape[0]`; the production bug class is therefore unexercised — the same blind spot that let the prior incident ship.
