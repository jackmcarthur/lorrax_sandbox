# Agent 2 — Final design and patch

**Branch**: `agent/trs-aware-sym-fix` off `agent/zeta-bc-scan-shardmap` HEAD `c7964207`.
**Status**: V_q-side fix landed in a single commit.  ψ-side (Sites #5,
#6, #7 in Agent 1's scope) deferred to Phase 3.  Pytest green
(207 passed of the modules touched; 10 pre-existing failures in unrelated
modules — gw regression OOM, kmeans label, missing positional arg in
v_q_per_q_g_chunked, fixture errors in bispinor orchestrator).

---

## 1. Choice: option C (narrow patch) — and why not option A

The prior pre-Agent-1 sketch recommended option A (typed `SymTable`
abstraction with `is_trs` + `spatial_idx` accessors).  Agent 1's scope
report narrowed the actual exercised V_q sites to just three (the
gates in `compute_vcoul.py` + `v_q_g_flat.py`, the `_unfold_v_q_ibz_to_full`
helper itself, and its g0 sibling), making the option-A typed-object
refactor disproportionate to the surface area of the bug.

After Agent 1's findings and a per-element re-derivation of the V_q
TRS transformation rule, option C wins on:

1. **Surface area**: 4 source files touched (`orbit_syms.py`,
   `v_q_tile.py`, `compute_vcoul.py`, `v_q_g_flat.py`).  Option A
   would have required a `SymTable` dataclass + migration of every
   reader of `sym.sym_mats_k` / `sym.sym_matrices` — Agent 1
   enumerated 11 such sites, only 3 of which are exercised on the
   V_q side.  The ψ-side migration would still be needed in Phase 3
   regardless of which V_q option we picked.
2. **Bispinor extensibility is unchanged**: the bispinor V_q
   (when it gains IBZ support) reuses the same scalar V_q logic —
   the spinor index sits inside ζ and the bilinear-over-spinor sum
   in V_q makes the σ_y factor cancel just like the spatial
   conjugation does for the charge channel.  See §2.3.
3. **No abstraction layer for an empty contract**: the V_q TRS rule
   reduces to "conj at TRS-tagged rows" — a single mask operation.
   Wrapping that in a typed `SymTable.apply_to_vq` method buys
   nothing today and is undisputedly easy to add later if Phase 3 +
   future channels grow the consumer count.

The hard-fail guard the prior sketch left for Site #11 (the explicit
`NotImplementedError` in `ZetaLoader._full_bz_unfold_tables`) is
mirrored in `_unfold_v_q_ibz_to_full` and `_unfold_g0_ibz_to_full`
under option C, so the "loud refusal" pattern is preserved across all
consumers.

---

## 2. Math: the TRS rule actually applied

### 2.1 ζ-leg TRS

From `reports/zeta_ibz_2026-05-11/report.md`, Bloch + TRS yields:

```
ζ_{-q, μ}(G) = conj(ζ_{q, μ}(-G))                                  (eq. 1-TRS)
```

The centroid index `μ` is unchanged (TRS keeps r fixed).

Combined with the spatial transformation (eq. 1 of the same report)
the full TRS-augmented sym `K · {S | τ}` produces:

```
ζ_{S'q, π_s(μ)}(G_new) = e^{-i(S'q+G_new)·τ'} · conj(ζ_{q, μ}((S')^{-1} G_new))     (eq. 1-TS)
```

where `S' = -S`, `τ' = -τ`.

### 2.2 V_q-leg TRS — the actually-correct rule

Substituting eq. 1-TS into both ζ legs of `V_{q,μν} = Σ_G conj(ζ_{q,μ}(G)) · v(q+G) · ζ_{q,ν}(G)`:

```
V_{S'q, π_s(μ), π_s(ν)}
  = Σ_{G'} ζ_{q, μ}(G') · v(q+G') · conj(ζ_{q, ν}(G'))
  = conj( V_{q, μ, ν} )
  = V_{q, ν, μ}                          (last equality by V_q Hermiticity)
```

The per-element derivation:

```
V_{q, μ, ν}        = Σ_G' conj(ζ_{q,μ}(G')) · v · ζ_{q,ν}(G')
conj(V_{q, μ, ν})  = Σ_G' ζ_{q,μ}(G') · conj(v) · conj(ζ_{q,ν}(G'))
                   = Σ_G' ζ_{q,μ}(G') · v · conj(ζ_{q,ν}(G'))      (v real)
                   = V_{S'q, π_s(μ), π_s(ν)}
```

The τ-phases cancel between the bra-leg ζ* and the ket-leg ζ; the
double-conj-flip leaves one residual conjugation that does NOT cancel.

### 2.3 Why the prompt's "no-conj" reconciliation note was wrong

The reconciliation note claimed: "V_{-q,μν} = conj(V_{q,νμ}) (each leg
picks up conj + the μν indices swap).  For Hermitian V_q, conj(V_{q,νμ})
= V_{q,μν}, so TRS is a no-op on V_q."

Per-element derivation shows V_{-q,μν} = V_{q,νμ}, not conj(V_{q,νμ}).
The error in the note: the per-leg conj rule applied to a bra-ket pair
where ONE leg is already conjugated in the bilinear ⟹ the second conj
flips the bra back to non-conj and adds a new conj on the ket.  Net
result is μν-swap (or equivalently a single conj, by Hermiticity), NOT
"each leg picks up conj".

By Hermiticity `V_{q, ν, μ} = conj(V_{q, μ, ν})`, so V_{-q, μ, ν} =
conj(V_{q, μ, ν}) — equivalent statement.  This non-trivial transform
MUST be applied at the V_q-unfold level.

Implementation choice: apply `conj` (not transpose) — one mask · conj
op vs. one extra μν-transpose gather.  Output is identical by
Hermiticity.

### 2.4 g0 TRS

g0 is a single ζ leg, not bilinear:

```
g0_{-q, π_s(μ)} = conj(g0_{q, μ})            (s ≥ ntran)
```

The unfold applies `jnp.where(trs_mask, conj(g0), g0)` after the
centroid permutation.

### 2.5 Bispinor V_q extension (Phase 3 concern, not in this patch)

Bispinor V_q has an extra spinor index `a` inside ζ:
`V_q = Σ_{a, G} conj(ζ_{q, μ, a}(G)) · v(q+G) · ζ_{q, ν, a}(G)`.
TRS for bispinor is `T = i σ_y K`, so ζ picks up `(i σ_y)^{ab} ζ*_{q, μ, b}`
on TR rows.  The bilinear gives:

```
ζ*_{Tq, μ, a} · ζ_{Tq, ν, a} = (iσ_y)^{ab} (iσ_y)^{ac} · ζ_{q, μ, b} · ζ*_{q, ν, c}
                              = δ^{bc} · ζ_{q, μ, b} · ζ*_{q, ν, b}      (σ_y² = 1)
                              = ζ_{q, μ, b} · ζ*_{q, ν, b}
```

i.e. the spinor `σ_y` factors cancel and bispinor V_q transforms
exactly like the charge V_q under TRS.  **The current `conj` rule
extends cleanly to bispinor V_q without spinor plumbing** — when the
bispinor IBZ cascade lands.

---

## 3. Per-site change list (as implemented)

Keyed against Agent 1's scope table.

| Site | File | Change |
|---|---|---|
| #1 | `gw/v_q_tile.py:_unfold_v_q_ibz_to_full` | Add `n_sym_spatial` kwarg.  Add hard-fail guard if `max(full_to_irr_sym) >= sym_perm.shape[0]`.  Add TRS branch: `V_full = jnp.where(trs_mask, conj(V_full), V_full)` where `trs_mask = full_to_irr_sym >= n_sym_spatial`. |
| #2 | `gw/v_q_tile.py:_unfold_g0_ibz_to_full` | Same kwargs + guard.  TRS branch: `g0_full = jnp.where(trs_mask, conj(g0_full), g0_full)`. |
| #3a | `gw/v_q_g_flat.py:_resolve_ibz_q_list` | Pass `extend_trs=True` to `compute_centroid_sym_perm` (so the returned `sym_perm` has shape `(2·ntran, n_rmu)`). |
| #3b | `gw/v_q_g_flat.py:_compute_V_q_g_flat_one_tile` | Compute `n_sym_spatial = sym_perm.shape[0] // 2` and pass it to both unfold helpers. |
| #3c | `gw/compute_vcoul.py` | Same as #3a + #3b at the V_q tile orchestrator's IBZ-mode gate. |
| (new) | `centroid/orbit_syms.py:compute_centroid_sym_perm` | Add `extend_trs: bool = False` kwarg.  When True, return `(2·n_sym, n_rmu)` table with rows `[n_sym:]` duplicating `[:n_sym]`.  Default behaviour unchanged. |
| Test | `tests/test_trs_unfold_centroid_perm.py` (new) | Unit-test for the extend_trs shape + content invariants, the TRS unfold against a hand-rolled per-element reference, AND the hard-fail guard. |
| Validation | `tests/test_v_q_trs_roundtrip.py` (Agent 3 owns) | Synthetic V_q IBZ→full-BZ round-trip; passes with the fix on the codebase path. |

**Sites NOT touched in this commit** (per the prompt):

| Site | Reason |
|---|---|
| #4 (`_unfold_v_q_ij_ibz_to_full`) | Dead code (no callers in `src/`; bispinor V_q is full-BZ on disk per `gw_init.py:650, 843`).  Flagged for Phase 3 bispinor IBZ port. |
| #5 (ψ k-unfold in `wfn_loader.py`) | Phase 3.  Needs `(iσ_y)·conj` on bispinor ψ at TRS rows, separate audit. |
| #6 (`U_spinor` construction) | Phase 3.  Root of #5. |
| #7 (`_get_umklapp_vector` TR branch + τ-phase skip) | Phase 3.  Non-symmorphic τ-phase still missing on TR rows. |
| #8 (`syms_crystal_to_cartesian` TODO) | Phase 3.  Author-flagged uncertainty feeds Site #6. |
| #9 (`vcoul.compute_vcoul_comps_for_q`) | Dead code.  Flagged for deletion. |
| #10 (BGW vcoul overlay) | Mathematically correct by symmetry of v(q+G); no change needed. |
| #11 (`ZetaLoader._full_bz_unfold_tables`) | Already raises `NotImplementedError` on TRS — the reference good pattern; mirrored in #1 and #2. |

---

## 4. Backward compatibility and API contract

The new `n_sym_spatial` kwarg in `_unfold_v_q_ibz_to_full` /
`_unfold_g0_ibz_to_full` is **optional** with default `None`.  Legacy
callers (e.g. `tests/test_v_q_ibz_unfold.py`) that pass a small
`sym_perm` and a non-TRS `full_to_irr_sym` continue to work unchanged
(no TRS branch is taken; the hard-fail guard does NOT fire because
their sym indices are in-range for the length-`ntran` `sym_perm`).

Callers that DO need TRS handling (the production V_q drivers in
`compute_vcoul.py` + `v_q_g_flat.py`) must opt in by:

1. Calling `compute_centroid_sym_perm(..., extend_trs=True)` to get a
   `(2·ntran, n_rmu)` permutation table.
2. Passing `n_sym_spatial=ntran` (or
   `sym_perm.shape[0] // 2`) to the unfold helper.

If a caller passes only one of these two (mismatch), the unfold
helper raises a clear `ValueError` rather than silently corrupting
V_q.  This is the same "loud refusal" pattern as
`ZetaLoader._full_bz_unfold_tables` (Site #11).

---

## 5. Deferred Phase 3 work

### 5.1 ψ-side TRS (Sites #5, #6, #7)

The wavefunction loader (`file_io/wfn_loader.py`) applies a wrong
spinor rotation at TRS-augmented k-points.  Concretely:

- The TR row of `sym.U_spinor` is the SU(2) of `+S_spatial`, not the
  physical TRS spinor `iσ_y · conj(U_spatial)`.  This is a Bug class B
  (table lookup succeeds, value is silently wrong).
- The umklapp `_get_umklapp_vector` TR branch is arithmetically right
  but the non-symmorphic τ-phase on TR rows is set to 1 (the
  documented `WARNING: Non-symmorphic phases are NOT applied for these
  k-points` admits this).

For non-spinor calculations (`nspinor=1`) the spinor bug is moot
(2×2 identity) but the τ-phase miss still bites non-symmorphic groups.
For bispinor calculations the spinor bug is active but the cohsex
bispinor mode currently uses `kpoints_path` that bypasses this code
branch.

**Phase 3 plan**:

1. Fix `U_spinor` construction (`symmetry_maps.py:570-637`) to apply
   the physical TR spinor `iσ_y · conj(U_spatial)` for `s ≥ ntran`.
2. Resolve the `syms_crystal_to_cartesian` TODO at line 548 (Site #8)
   — likely use `sym_mats_k[:ntran]` (spatial only) since R_cart
   should be of length ntran.
3. Add the τ-phase to the TR rows in `_get_umklapp_vector` /
   `wfn_loader.py:842-847`.
4. Update `wfn_loader.py:825-851` (eager) and `wfn_loader.py:446-494,
   967-985` (phdf5) to apply `(iσ_y) · conj(ψ)` not `U_spinor · conj(ψ)`.
5. Independent unit test: build a 4×4×1 bispinor toy with an explicit
   TR-mapped k-pair, verify ψ at the TR-image satisfies the TR rule
   element-wise.

This is its own commit with its own audit because the surface area
(every full-BZ ψ load) is much larger than V_q's three sites and the
spinor index needs careful handling.

### 5.2 `_unfold_v_q_ij_ibz_to_full` (Site #4)

Dead today, but identical OOB structure to Site #1.  When the bispinor
IBZ cascade lands, this helper will be called.  The current-current
tensor `v^{ij}(K) ∝ (δ^{ij} − K_i K_j / |K|²) / |K|²` is real and
even-in-K; the polarization-leg `R · V · R^T` mix is correct under
TRS-augmented rows because `R(s ≥ ntran) = -R_spatial(s mod ntran)`
gives `(-)(-) V = V`.  Only the centroid axis needs the same fix as
Site #1.

Plan: same shape as Site #1 — accept `n_sym_spatial`, hard-fail guard,
TRS conj at tagged q's.  Bundle with the bispinor IBZ-cascade landing.

---

## 6. Verification

**Unit tests** (`tests/test_trs_unfold_centroid_perm.py`):
- `test_compute_centroid_sym_perm_extend_trs_shape_and_content` —
  4×4×1 toy with `ntran=2`, asserts shape `(4, 6)` and per-row
  duplication on the TRS half.
- `test_compute_centroid_sym_perm_extend_trs_each_row_is_permutation`
  — each of the 4 rows is a permutation of `[0, n_rmu)`.
- `test_unfold_v_q_ibz_to_full_handles_trs_rows` — synthetic
  Hermitian V_ibz with a hand-built full→IBZ table containing both
  spatial-only and TRS-augmented sym indices, compares the codebase
  unfold to a per-element reference; max |Δ| < 1e-12.
- `test_unfold_v_q_ibz_to_full_hard_fails_without_extend_trs` —
  passing a length-`ntran` sym_perm with TRS sym values raises
  `ValueError`.

**Synthetic integration test** (`tests/test_v_q_trs_roundtrip.py`,
Agent 3's): builds a 3×3×1 V_q from synthetic ζ at IBZ q's via
hand-rolled "unfold ζ properly then contract" reference, compares to
the codebase IBZ→full unfold helper output.  Passes at all 9 q's
(both spatial-only and TRS-only-reachable q's) at rel < 1e-12.

**Full pytest**: 207 passed, 18 skipped of touched-module tests.  10
pre-existing failures in unrelated modules (verified by git stash:
the same 10 failures appear without my patch applied).

**Anchor for Agent 4 R1-R3**: the MoS2 3×3 same-basis IBZ-vs-fullbz
ΔΣ_X (max 6.89 eV at k=0 band 44, max ≤ 13.85 eV for sex+coh on the
full per-k table) should collapse to ≤ basis-noise floor on the fix
branch.  Agent 4 owns the rerun.

---

## 7. Files in the commit

```
src/centroid/orbit_syms.py             — add extend_trs kwarg, docstring
src/gw/v_q_tile.py                     — add n_sym_spatial kwarg + TRS conj + hard-fail guard on the two unfold helpers
src/gw/v_q_g_flat.py                   — call extend_trs=True; pass n_sym_spatial to unfold (just the relevant hunks; pre-existing debug-env-var changes unaffected)
src/gw/compute_vcoul.py                — same as v_q_g_flat.py
tests/test_trs_unfold_centroid_perm.py — new
```

---

End of Agent 2 design report.
