# Scope: TRS-blind sym sites

**Audit date**: 2026-05-14
**Tree**: `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap` HEAD `c7964207`
**Method**: read-only grep + per-element index-math verification. Every "should/should-not" claim is backed by explicit summation indices and (where possible) a concrete numerical example on the MoS2 3×3 sym table (`ntran=2`, in-plane mirror only; TRS adds 2 more ops → `sym_mats_k` length 4).

## Table-length contract (anchor)

Established once in `src/common/symmetry_maps.py`:

| Array | Length | Built at | Notes |
|---|---|---|---|
| `sym.sym_matrices`  | `ntran`     | line 116 | BGW `mtrx`; acts on G column-vectors |
| `sym.translations`  | `ntran`     | line 124 | BGW `tnp`; sliced to `[:ntran]` |
| `sym.sym_mats_k`    | `2·ntran`   | lines 117 + 130 | spatial + TR (`-sym_mats_k`); acts on k row-vectors |
| `sym.R_grid`        | `ntran`     | line 165 | = `sym_matrices` rint'd |
| `sym.Rinv_grid`     | `ntran`     | line 166 | = `inv(sym_matrices)` rint'd |
| `sym.R_cart`        | `2·ntran`   | line 169 + 549 | derived from `sym_mats_k` |
| `sym.U_spinor`      | `2·ntran`   | line 170 + 576-637 | derived from `R_cart`; the SU(2) for the TR-image rows is **not** the physical TRS spinor (see Site #6) |
| `sym.kfull_symmap`  | `(nk_full, 2·ntran)` | line 173, 720 | dead (no consumer in src) |
| `sym.irk_sym_map`   | length `nk_full`, **values ∈ [0, 2·ntran)** | line 144, 332 | per-full-k sym index |
| `q_full_to_irr_sym` | length `n_q_full`, **values ∈ [0, 2·ntran)** | line 403, 420, 445 | per-full-q sym index, from `find_irreducible_qpoints` |

Two distinct *consumer* mismatches show up:

1. Consumer indexes a length-`ntran` table with an index from `irk_sym_map` / `q_full_to_irr_sym` ⇒ JAX/numpy clipping → wrong physics, no exception. **Bug class A.**
2. Consumer indexes a length-`2·ntran` table (`U_spinor`, `R_cart`) with a TRS-augmented index — table lookup succeeds but the value at `[ntran:]` was computed by feeding `−S_spatial` through `get_spinor_rotations`, which contains an unrelated `if det<0: R = -R` step (line 580-582 of `symmetry_maps.py`). The resulting matrix is the spinor of `+S_spatial`, **not** the physical TRS spinor `iσ_y · U_spinor(S)*`. **Bug class B.**

## Summary table

| # | File:line | Function | Current sym table used | Correct sym table | Failure mode | Exercised? |
|---|---|---|---|---|---|---|
| 1 | `gw/v_q_tile.py:1513-1554` | `_unfold_v_q_ibz_to_full` | `sym_perm` axis-0 length **ntran** (built from `sym.sym_matrices[:n_tran]` upstream) | needs full TRS-aware perm: `2·ntran` rows where the TR half encodes `-S` + ζ→ζ\* leg-conjugation | A — JAX `inv_perm[sym_j]` clips OOB silently to last spatial row; for MoS2 3×3 the 4 TRS-q's get the σ_h centroid perm instead of TR×identity ⇒ wrong V_q. ΔΣ_X observed up to 6.89 eV. | **YES** (scalar IBZ cascade, via `compute_vcoul.py:1025` and `v_q_g_flat.py:459`) |
| 2 | `gw/v_q_tile.py:1706-1731` | `_unfold_g0_ibz_to_full` | same `sym_perm` (length ntran) | same fix as Site #1 | A — same OOB clip. g0 is only consumed at Γ where `sym_idx[Γ]=0`, so the wrong values at TRS q's are never read. Latent. | YES (called), but consumed cells safe (Γ-only). |
| 3 | `gw/compute_vcoul.py:895-918` and `gw/v_q_g_flat.py:180-194` | IBZ-mode gate / sym_perm builder | builds `sym_perm` from `sym.sym_matrices[:n_tran]` (length ntran), but reads `q_full_to_irr_sym` from `find_irreducible_qpoints` which uses `sym_mats_k` (length 2·ntran) | either pass full `sym_mats_k` to `compute_centroid_sym_perm` and extend that helper to handle the TR ζ-leg conjugation, or rebuild a spatial-only `find_irreducible_qpoints` view here | A — produces the mismatched (ntran, 2·ntran) pair that Site #1 consumes. The root coupling. | **YES** |
| 4 | `gw/v_q_tile.py:1618-1665` | `_unfold_v_q_ij_ibz_to_full` (current-channel/transverse) | same `sym_perm` length ntran + uses `R_cart[s]` (length 2·ntran) | needs TRS sym perm + the right TR rotation; `R_cart[s]` for s≥ntran is `−R_cart_spatial[s−ntran]` (rotation × parity), which is wrong for the j i tensor under TR | A on `sym_perm` axis; B on `R_cart` | NO (no caller — `_unfold_v_q_ij_ibz_to_full` is defined but unused; bispinor V_q is full-BZ on disk per `gw_init.py:650, 843`). **Dead but identical structure to Site #1.** |
| 5 | `file_io/wfn_loader.py:825-851` (eager) and `wfn_loader.py:446-494, 967-985` (phdf5) | ψ unfold (IBZ k → full-BZ k) | `irk_sym_map` (values 0..2·ntran), used to index `sym_mats_k` (length 2·ntran — OK), `translations` (length ntran — guarded by `if sym_idx >= ntran: continue`), `U_spinor` (length 2·ntran — looks OK; see Site #6 caveat) | for TR rows, ψ_{Sk,σ}(G) = Σ_{σ′} (iσ_y)_{σσ′} ψ\*_{kbar,σ′}(S^{-1}G); current code does `U_spinor[sym_idx] · conj(ψ)` which is missing the `iσ_y` factor and applies the wrong rotation row | B — silent: spinor multiplication uses a wrong matrix at TRS rows. The umklapp `kg0` for TR rows is also wrong (see Site #7). | **YES** (every full-BZ ψ load) |
| 6 | `common/symmetry_maps.py:170, 554-637` | `U_spinor` construction | iterates over `R_cart` (length 2·ntran); for TR rows feeds `−R_spatial` into `get_spinor_rotations`, which `if det<0: R=−R` flips back ⇒ `U_spinor[ntran+s] == U_spinor[s]` (modulo sign/branch) | TRS spinor should be `iσ_y · conj(U_spinor[s])`. The TR half of `U_spinor` as built is physically wrong. | B — root of Site #5's spinor error | **YES** (Site #5 reads from this) |
| 7 | `common/symmetry_maps.py:743-766` | `_get_umklapp_vector` (TRS branch) | `if sym_idx >= len(self.sym_matrices)` branch returns `wrap(S·kbar) − S·kbar` | for TR the full k is `−S·kbar`, so kg0 should solve `−S·kbar + kg0 = k_full` (numerically reduces to identity wrap, but the documentation/derivation is mis-stated — see also the explicit warning at line 336-342) | B — non-symmorphic τ-phase is then skipped (the file's own warning admits this). Bispinor ψ on TR rows is therefore wrong by a τ-phase plus the wrong spinor of Site #6. | YES (called from Site #5 — `wfn_loader.py:313`) |
| 8 | `common/symmetry_maps.py:531-552` | `syms_crystal_to_cartesian` | `np.einsum('ij,njk,kl->nil', B_T_inv, self.sym_mats_k, B_T)` — explicit `TODO` comment "NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS" (line 548) | depends on consumer; the only consumer is `U_spinor` construction (Site #6), which itself is currently wrong on the TR half | B — author-flagged uncertainty fed into Site #6 | YES |
| 9 | `gw/vcoul.py:160-172` | `compute_vcoul_comps_for_q` | reads `irk_sym_map` → indexes `sym_mats_k` (length 2·ntran — OK as far as table lookup), then uses `wfn.kpoints[iqbar]` and `wfn.get_gvec_nk(iqbar)` for G remap | TR branch needs ψ→ψ\* on the G-list and a sign flip on the q-vector reconstruction | A/B — but this function has **no callers** in `src/` (dead code) | **NO** |
| 10 | `gw/gw_driver_helpers.py:217-225` + `file_io/read_bgw_vcoul.py:33-81` | BGW-vcoul overlay sym search | passes full `sym.sym_mats_k` (length 2·ntran) | this one is **safe**: v(q+G) is rotation-invariant AND even in (q+G), so matching against `-S` images is mathematically equivalent to matching against `+S` images; the umklapp/G-remap on line 176 of `read_bgw_vcoul.py` (`G_input = S_k @ G − kg0`) preserves the integer FFT-grid invariance for both signs | none — correct by symmetry of the kernel | YES (when BGW vcoul overlay is on); behavior correct |
| 11 | `file_io/zeta_loader.py:411-462` | `_full_bz_unfold_tables` (q='full_bz' path in the eager loader) | explicit guard: if any `full_to_irr_sym ≥ ntran` raises `NotImplementedError` (line 432-441) | this is the CORRECT pattern — fail loudly | none (loudly raises) | NO (testing-only; production V_q does not call `ZetaLoader.load(q='full_bz')`) |

## Concrete numerical example (MoS2 3×3)

`runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5` has `ntran=2`, identity + σ_h. Reproduced with the on-disk `mtrx` and the integer kgrid sym fold (per `find_irreducible_qpoints`):

| full-BZ q (kgrid int) | best sym idx into `sym_mats_k` (length 4) | TR row? | canonical IBZ q |
|---|---|---|---|
| (0,0,0) | 0 | no | (0,0,0) |
| (0,1,0) | 0 | no | (0,1,0) |
| **(0,2,0)** | **2** | **YES** | (0,1,0) |
| (1,0,0) | 0 | no | (1,0,0) |
| (1,1,0) | 0 | no | (1,1,0) |
| (1,2,0) | 0 | no | (1,2,0) |
| **(2,0,0)** | **2** | **YES** | (1,0,0) |
| **(2,1,0)** | **2** | **YES** | (1,2,0) |
| **(2,2,0)** | **2** | **YES** | (1,1,0) |

⇒ `q_full_to_irr_sym` = [0,0,**2**,0,0,0,**2**,**2**,**2**]; IBZ has 6 q's.

`_unfold_v_q_ibz_to_full` builds `inv_perm = argsort(sym_perm, axis=-1)` with axis-0 length 2. Then `perm_q = inv_perm_j[sym_j]` with `sym_j` containing the value `2`.

JAX OOB behaviour (verified at runtime in a one-line script): out-of-range indices into a fancy-indexed array are silently clamped to the last valid row, **not** to zero. So `inv_perm[2]` returns `inv_perm[1]` — the σ_h centroid permutation. Concretely, for full-BZ q=(0,2,0), instead of

```
V_full[q=(0,2,0), μ, ν] = V_ibz[i=(0,1,0), π_TR^{-1}(μ), π_TR^{-1}(ν)] · (TR-leg conj on ζ contributions cancels in V)
```

we get

```
V_full[q=(0,2,0), μ, ν] = V_ibz[i=(0,1,0), π_{σ_h}^{-1}(μ), π_{σ_h}^{-1}(ν)]
```

— off by a σ_h-vs-TR centroid swap. For MoS2 (z-flat layer) σ_h sends every centroid to its z-mirror partner; for in-plane centroids those are identical, but for the small-z centroids the rows/cols of V_q get scrambled. The 6.89 eV ΔΣ_X mismatch quoted in STATUS.md is the integrated effect across the 4 TRS q's.

## Per-site detail

### Site #1: `_unfold_v_q_ibz_to_full` (`gw/v_q_tile.py:1452-1557`)

**Math (claimed in docstring)**: `V_full[q, μ', ν'] = V_ibz[i(q), π_{s(q)}^{-1}(μ'), π_{s(q)}^{-1}(ν')]`.

**Code (lines 1513, 1528-1554)**:
```python
inv_perm = np.argsort(sym_perm, axis=-1).astype(np.int32)   # (n_sym, n_rmu)
sym_j = jnp.asarray(np.asarray(full_to_irr_sym, dtype=np.int32))
...
perm_q = inv_perm_j[sym_j]                                  # (n_q_full, n_rmu_padded)
V_at_irr = V_ibz[idx_j]                                     # (n_q_full, μ, μ)
V_perm_mu = jnp.take_along_axis(V_at_irr, perm_q[:, :, None], axis=1, mode='promise_in_bounds')
V_full = jnp.take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2, mode='promise_in_bounds')
```

**Per-element**: `V_full[q, μ', ν'] = V_at_irr[q, inv_perm[sym_j[q], μ'], inv_perm[sym_j[q], ν']]`. The lookup `inv_perm[sym_j[q], …]` evaluates `inv_perm[sym_j[q]]` first → first-axis index in [0, n_sym_perm). If `sym_j[q] ≥ n_sym_perm`, JAX clips silently (verified above).

**Where `n_sym_perm = ntran` comes from**: both callers (Site #3) call `compute_centroid_sym_perm(sym_matrices=sym.sym_matrices[:n_tran], ...)`, so `sym_perm.shape[0] = ntran`. But `full_to_irr_sym` values are in `[0, 2·ntran)` because `find_irreducible_qpoints` uses `sym_mats_k`.

**TR-correct unfold**: needs π_TR (permutation generated by `-S` on the centroids; for a centroid orbit-closed under spatial sym it's not automatic that adding TR keeps closure — for inversion-containing groups it does; for σ_h-only it generally does not). The right structural fix is either (a) build `sym_perm` with axis-0 length 2·ntran by passing `sym.sym_mats_k` (and the negated `translations` for the TR half) into a TR-aware variant of `compute_centroid_sym_perm`, or (b) reduce `find_irreducible_qpoints` to the spatial-only subgroup when the centroid set isn't TR-closed.

**Exercised**: yes — every scalar (non-bispinor) IBZ cascade run since the cascade landed.

### Site #2: `_unfold_g0_ibz_to_full` (`gw/v_q_tile.py:1669-1734`)

Same `sym_perm` length-`ntran` issue as Site #1. The function only matters for the q=Γ slot of `g0` (head correction); for Γ, `sym_j[Γ]=0` (identity) so the OOB clip is never triggered at the read site. The TRS q rows in `g0_full` carry wrong values but are not consumed downstream (`head_correction.py` only reads `g0_acc[0]`). **Latent** — would activate if any future caller reads `g0_full[q≠0]`.

### Site #3: IBZ-mode gate (`gw/compute_vcoul.py:895-918`, `gw/v_q_g_flat.py:180-194`)

The two production V_q drivers each build `sym_perm` from `sym.sym_matrices[:n_tran]` (length ntran) and then independently call `sym.find_irreducible_qpoints()` (which uses `sym_mats_k`, returning length-2·ntran sym indices). The mismatch is wired in here.

Per-element check on `compute_vcoul.py:892-911`:
```python
n_tran = int(np.asarray(sym.sym_matrices).shape[0])              # = ntran (2 for MoS2)
sym_perm = compute_centroid_sym_perm(centroid_idx,
    sym_matrices=np.asarray(sym.sym_matrices[:n_tran]),          # shape (n_tran, 3, 3)
    translations=np.asarray(sym.translations[:n_tran]), ...)     # → sym_perm shape (n_tran, n_rmu)
...
(q_irr_kgrid_int, q_full_to_irr_idx,
 q_full_to_irr_sym, ...) = sym.find_irreducible_qpoints()        # values ∈ [0, 2·n_tran)
```

These two are passed in tandem to `_unfold_v_q_ibz_to_full(sym_perm=sym_perm, full_to_irr_sym=q_full_to_irr_sym)` (lines 1025-1030). Site #1 is then OOB.

### Site #4: `_unfold_v_q_ij_ibz_to_full` (`gw/v_q_tile.py:1560-1666`)

Transverse-current V_q^{ij}. Has the same structural OOB on `sym_perm` axis-0 PLUS an additional dependency on `R_cart[s]` (line 1635, 1660-1663) — `R_cart` does have length 2·ntran, but the TR rows store `−R_cart_spatial[s]` which is a rotation×inversion. For the current-current tensor `v^{ij}(K) ∝ (δ^{ij} − K^i K^j/|K|²)/|K|²`, TR maps `v^{ij}(K) = v^{ij}(−K)` (the kernel is even and rank-2 even), so the polarization mix `R^{ia}(s)·R^{jb}(s)·V^{ab}` with `R(s) = −R_spatial` produces `(−)(−)V = V` — accidentally correct on the polarization legs. But the centroid axis is still wrong (Site #1's bug). **Dead code** — bispinor V_q on disk is full-BZ (no IBZ cascade) by `gw_init.py:650, 843`. Worth flagging for the future bispinor IBZ port (memory note `project_lorrax_zeta_session_2026-05-13_14`).

### Site #5: ψ k-unfold (`file_io/wfn_loader.py:825-851` eager; `446-494` + `967-985` phdf5)

**Eager path (lines 825-850)**:
```python
ntran = int(sym.sym_matrices.shape[0])                  # ntran
U_per = np.asarray(sym.U_spinor)                        # length 2·ntran
for j, nk in enumerate(k_idxs):
    sym_idx = int(sym.irk_sym_map[nk_int])              # ∈ [0, 2·ntran)
    sym_krep = np.asarray(sym.sym_mats_k[sym_idx], ...) # OK (len 2·ntran)
    ...
    if sym_idx >= ntran:
        cnk = np.conj(cnk)                              # TR: ψ → ψ*
    else:
        # apply τ-phase (uses sym.translations[sym_idx], len ntran — guarded)
        ...
    cnk = np.einsum("jk,nkl->njl", U_per[sym_idx], cnk) # SU(2) rotation
```

For TR rows the math should be: `ψ_{Sk_full, σ}(G) = Σ_{σ′} (iσ_y)_{σσ′} ψ\*_{kbar, σ′}(S_spatial^{-1} G − kg0_TR)`. The code applies `U_spinor[ntran+s] · ψ\*` where `U_spinor[ntran+s]` was built from `−R_cart_spatial[s]` via Markley's quaternion algorithm with a `det<0 → −R` pre-flip (line 580-582). That algorithm produces the spinor of `+R_cart_spatial[s]` (i.e. the same spinor as the spatial row). So the effective rotation on TR rows is `U_spinor[s] · conj(ψ)` — missing the `iσ_y` factor. For non-spinor calculations (`nspinor=1`) this is a no-op; for spinor calcs (MoS2 in cohsex bispinor mode, but those use `kpoints_path` that doesn't load via this code branch, so it isn't currently triggered) it's wrong.

**phdf5 path (lines 446-494)**: same logic with the same outcome.

### Site #6: `U_spinor` construction (`common/symmetry_maps.py:170, 554-637`)

Constructed from `R_cart` (length 2·ntran). For each row, the helper applies `if det(R) < 0: R = -R` (line 580-582), then computes the SU(2) for the resulting proper rotation. For TR rows where `R_cart[ntran+s] = -R_cart[s]`, this means `det(R_TR) = -det(R_spatial)`, so the det-flip fires when the spatial op has det=+1 (pure rotation; for TR row it flips back to `+R_spatial`, giving the same spinor). For improper spatial ops (det=-1) it doesn't flip, and the TR row computes the spinor of `+R_spatial` directly (same outcome). So `U_spinor[ntran+s] ≡ U_spinor[s]` in all cases.

But the physical TRS spinor (Wigner): `K · S` in SU(2) is `iσ_y · K · U_spatial(S) · K = iσ_y · conj(U_spatial(S))` acting on a 2-spinor (where K is complex conjugation). The current `U_spinor[ntran+s]` is just `U_spatial(S)` — missing both the `iσ_y` and the conjugation-of-matrix. Site #5 applies the conjugation to ψ already, but it's the wrong half of the operation: matrix-conjugate of the rotation, not state-conjugate.

### Site #7: `_get_umklapp_vector` TR branch (`common/symmetry_maps.py:743-766`)

```python
if sym_idx >= len(self.sym_matrices):
    q_full = np.asarray(sym_krep @ wfn.kpoints[kbar_idx], dtype=np.float64)
    q_inzone = q_full % 1.0
    q_inzone[q_inzone > 0.9999] = 0.0
    return (q_inzone - q_full).astype(np.int32)
```

For TR, `sym_krep = -S_spatial`, so `q_full = -S·kbar`. The kg0 is computed as `wrap(q_full) − q_full` — the umklapp to wrap `q_full` into the 1BZ. The caller (`wfn_loader.py:313-315`) then does `g_rot = sym_krep @ k_gvecs − Gkk` and uses that G-list as the gvec for the full-BZ k. The TR-correct G transform is `G_full = −S·G_bar − kg0_TR + G_offset` where `G_offset` makes `−S·k_bar + G_offset = k_full`. The current code computes `kg0_TR = wrap(−S·k_bar) − (−S·k_bar)`, which equals the umklapp from `−S·k_bar` to its 1BZ image — that's the right answer if `k_full` IS the 1BZ image of `−S·k_bar`, which it is by construction in `find_symmetry_ops_simple` (the `_wrap_to_bz` step). So the kg0 computation here is **arithmetically correct**, but the warning attached at the kpoint-map construction step (line 336-342: "Non-symmorphic phases are NOT applied for these k-points") notes that `wfn_loader.py:842-847` skips the τ-phase entirely on TR rows. That τ-phase skip is a separate Bug class B issue: for non-symmorphic groups (e.g. CrI3 ⊃ S6 with non-zero `tnp`), the TR row should still pick up a phase `exp(-i (−S·G̅)·τ)` = `exp(+i (S·G̅)·τ)` = conj of the spatial phase. Currently it's set to 1.

### Site #8: `syms_crystal_to_cartesian` (`common/symmetry_maps.py:531-552`)

Explicit author-flagged TODO at line 548: "NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS". The current code uses `sym_mats_k` (length 2·ntran) so the output `R_cart` is length 2·ntran. Consumer is `get_spinor_rotations` (Site #6). If only the spatial half were needed, `R_cart` length would be ntran. Choice is consistent with `U_spinor` length but propagates the TR-half spinor problem.

### Site #9: `compute_vcoul_comps_for_q` (`gw/vcoul.py:160-172`)

Dead — no callers in `src/`. Documented for completeness; if reintroduced, the TR branch would need state-conjugation on ψ and a sign flip on the G-list reconstruction. Don't fix now; delete or guard with `if sym_idx >= ntran: raise`.

### Site #10: BGW-vcoul overlay (`gw/gw_driver_helpers.py:217-225`, `file_io/read_bgw_vcoul.py:33-81, 133-176`)

**Safe** despite passing `sym_mats_k` (length 2·ntran). Per-element check on `read_bgw_vcoul.py:74-81 + 166-176`:

```python
for S_k in np.asarray(sym_mats_k):
    for i, qf in enumerate(self.q_fracs):
        kg0 = _umklapp(q, S_k @ qf)               # q_lorrax = S_k · q_bgw + kg0
        if kg0 is not None:
            return i, np.rint(S_k).astype(np.int32), kg0
...
G_input = np.einsum('ij,gj->gi', S_k.astype(np.int32), G_miller) - kg0[None, :]
```

For a TR match `q_lorrax = -S·q_bgw + kg0` and `G_input = -S·G_miller − kg0`. Then `q_lorrax + G_input = -S·(q_bgw + G_miller)`. v(q+G) depends on `|q+G|` only (3D) or on `|q+G|` and the z-component (2D slab) — in both cases v is invariant under K→-K. So v(q_lorrax + G_input) = v(-S·(q_bgw+G_miller)) = v(S·(q_bgw+G_miller)) = v(q_bgw+G_miller). ✓ Correct.

### Site #11: `ZetaLoader._full_bz_unfold_tables` (`file_io/zeta_loader.py:411-462`)

**Reference for the good pattern**: explicit `NotImplementedError` raise at line 432-441 when `max(full_to_irr_sym) >= ntran`. Then the downstream call to `compute_centroid_sym_perm(sym.sym_matrices, ...)` uses spatial-only sym, consistent with the (now-asserted) all-spatial `full_to_irr_sym`. The asymmetry with Site #1 is striking: Site #1 silently miscomputes; Site #11 loudly refuses. This is the right shape for the fix.

## Notes for Agent 2

The patch isn't just "slice `q_full_to_irr_sym` to spatial-only" — that changes which q's are in the IBZ (would re-add the 4 TRS q's to the IBZ), defeating the IBZ savings. The structural fix has two components:

1. Generalise `compute_centroid_sym_perm` (or add a sibling) to handle TR rows. The forward map for a TR sym `−S` on real-space centroids is `r_{π_TR(μ)} ≡ −S r_μ − τ` (mod 1), same algebra as the spatial case with sign-flipped inputs — closure under TR is automatic for orbit-closed centroid sets that are symmetric under inversion+S (most physical configurations).

2. Add the ζ-leg conjugation. ζ_q(r, μ) transforms as a single Bloch leg (not bilinear) under TR, but **V_q is bilinear in ζ** so the conjugation appears twice and cancels — V_q itself transforms as `V_{Sq, π_S μ, π_S ν} = V_{q, μ, ν}` even under TR. So for the scalar (μ_L=0) channel, only Site #1's `sym_perm` axis needs to grow from ntran to 2·ntran rows; no leg conjugation is needed in the V_q-unfold itself.

The harder TR fix is on the ψ side (Sites #5, #6, #7): there ψ DOES need state-conjugation + `iσ_y` for bispinor, and τ-phase reconstruction. Those don't affect the scalar V_q IBZ→full unfold, but they're the production ψ-loader for every full-BZ k and feed every Σ_X/Σ_C consumer.

## Cross-references

- The good pattern: `file_io/zeta_loader.py:432-441` (explicit refuse).
- The "no caller, but identical structure" extension: `gw/v_q_tile.py:_unfold_v_q_ij_ibz_to_full` (Site #4).
- BGW WFN handling that hides this bug: `find_symmetry_ops_simple` (`common/symmetry_maps.py:315-344`) emits the warning at line 336-342 but the V_q-side `find_irreducible_qpoints` (line 352+) does NOT — it silently produces TR-tagged sym indices and trusts the consumer.
- Centroid-orbit closure assertion (`centroid/orbit_syms.py:312-336`) operates on spatial-only sym; passes for the orbit-aware kmeans output. A TR-aware closure check would either succeed (most groups including σ_h+TR for MoS2) or refuse — that refusal is the next layer of the fix.
