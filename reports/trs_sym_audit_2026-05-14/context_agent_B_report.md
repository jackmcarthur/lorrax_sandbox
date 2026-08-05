# Context Agent B — ζ/V_q-side history cross-check

**Scope**: complement parallel math agent + agent A. Focus on V_q + ζ-side
historical context (vs ψ-side). Read-only.

## 1. Compare-skill conventions (`skills/compare/SKILL.md`)

Σ_X comparisons cross-code use `x_bare` (LORRAX `sigma_freq_debug.dat`
col 7) ↔ BGW `X` (`sigma_hp.log` col 3). Σ_X is **spinor-trace
invariant**, so it doesn't move when only the per-band spinor gauge is
wrong (consistent with the 5dc8813 ψ-side R_cart fix not collapsing the
6 eV CrI3 residual). LORRAX bands are 0-indexed; BGW 1-indexed; physical
= n+1. **Use the provided parsers — `parse_sigma_freq_debug` and
`parse_sigma_hp` — for all extraction.** SX-X+CH is not directly
comparable to Σ⁺+Σ⁻ piece-by-piece, only the sum `Cor`. None of these
conventions touch V_q-unfold directly; they only set how a successful
fix would be quantified.

## 2. Historical V_q-unfold tests — coverage gap that hid the bug

**Two synthetic tests previously "passed" the V_q TRS unfold round-trip
at 1e-14**, both with the **same coverage gap** that masks the non-
involutive C3/S6 case currently failing on CrI3:

- **Agent 3** (`test_v_q_trs_synthetic.py`, 2026-05-14 19:09): sym group
  `{I, σ_y}` → TRS-augmented `{I, σ_y, −I, −σ_y}`. **σ_y² = I, (−I)² = I,
  (−σ_y)² = I** — every op is involutive.
- **Agent 2 audit pr2** (`audit_pr2_perelement.py`, line 87-114): ntran=4
  with `{I, σ_y, σ_x, C2z}`. **All four involutive** (each squares to I).

For involutive S, `S = S^{-1}`, so the direction of
`compute_centroid_sym_perm` (currently `r ↦ S^{-1} r + τ`) is
indistinguishable from the forward direction `r ↦ S r + τ`. This is
**exactly the coverage gap** that lets MoS₂ pass and CrI3 fail. Agent A
likely independently rederives this.

The first time a non-involutive op was exercised in any pipeline test
was the algebraic_unfold_cri3 (2026-05-14) and the resulting
sym-vs-nosym Σ_X run — both immediately failed at O(eV) on C3-folded q's.

## 3. V_q TRS conjugation has been derived twice — both times correctly

The `V_{Tq}[π(μ), π(ν)] = conj(V_q[μ,ν]) = V_q[ν,μ]` (Hermiticity)
relation was originally argued away in agent_2_design_sketch.md as
"cancels in the bilinear", corrected by Agent 3 (2026-05-14 19:15-19:19)
via direct numeric verification, and landed in commit 9e644e9. The conj
branch (`jnp.where(trs_mask, jnp.conj(V_full), V_full)` at
symmetry_maps.py:243-244) is **present and correct** for the TRS rows.
**This is NOT where today's bug lives** — CrI3 P-3 contains spatial
inversion, so its IBZ→full map uses **zero TRS-augmented rows**
(verified in Agent 5's R2 entry: 0/36 TRS-tagged q's). The CrI3 6 eV
failure is on **purely spatial** C3/S6 unfold.

## 4. What the V_q consumer actually does — confirming no extra G-rotation or τ-phase

Traced `src/gw/v_q_g_flat.py:457-478` and `src/gw/compute_vcoul.py:1030-1050`:
both consumers (a) compute V_q at **IBZ q's only**, via
`_compute_V_q_g_flat_one_tile` looping `q ∈ [0, n_q_ibz)`, then (b) call
`unfold_v_q(V_acc, irr_idx, sym_idx, sym_perm, mesh_xy, n_sym_spatial)`
which does **only** the centroid double-permute + conj-on-TRS. **No
τ-phase is applied during V_q unfold** (it cancels in the bilinear per
the math agent's §3). **No G-rotation is applied to V_q matrix elements**
(G is contracted out per-IBZ-q before unfold). The Σ_X kernel downstream
treats `V[q_full, μ, ν]` as a black-box matrix indexed by **full-BZ q**;
it does not re-apply any sym to V_q.

**The only sym-direction-sensitive object that touches V_q is `sym_perm`**
— a single integer permutation table built by `compute_centroid_sym_perm`
at `src/centroid/orbit_syms.py:308`. That single direction choice
controls the whole V_q-side fate.

## 5. FLAG FOR MATH AGENT — three points to verify in your derivation

**(a) The "S" on the IBZ→full map is `sym_mats_k`, NOT `sym_matrices`.**
`find_irreducible_bz_points` line 52, 72: `images = einsum('sij,qj→sqi',
sym_mats_k, full_q)`. And `sym_mats_k = sym_matrices.transpose(0,2,1)`
(SymMaps `__init__` line 478) — i.e., `mtrx.T`. So the relation is
**`q_full = mtrx[s].T · q_irr`**, NOT `mtrx · q_irr`. The math agent
freely uses `q_full = S q_irr` with `S = mtrx` — for the **bug
diagnosis** this is fine (the question is just which permutation
direction matters), but if there's any cross-check against
`sym_idx_q[q_full]` lookups, remember to apply the transpose.

**(b) Bug surface analysis of the prior "passing" V_q TRS tests.** Both
the codebase regression (`test_v_q_trs_roundtrip.py`) and the PR2 audit
synthetic used **only involutive sym groups** (Section 2 above). Their
"pass at 1e-14" is **NOT evidence that the V_q centroid-permutation
direction is correct** — only that the TRS-conj branch is correct (and
that the legitimate involutive sub-case is bit-equal). The math agent's
proposed direction-flip diagnostic should hold these synthetic tests
constant (they will pass either direction, since for involutive S the
two directions coincide). The real validation gate is a non-involutive
op in synthetic form (e.g. C3 on a triangular grid) or the CrI3
end-to-end sym-vs-nosym Σ_X.

**(c) No τ-phase is ever applied at the V_q unfold level.** Confirmed by
direct read of `unfold_v_q` (symmetry_maps.py:110-247). The math agent's
§3 claim "τ cancels in the bilinear" is consistent with the production
implementation. CrI3 P-3 is symmorphic (τ=0) regardless, so this is moot
for the current failing test, but it is **relevant for future
non-symmorphic systems** — if the math agent's derivation predicts a
residual `exp(-i kg0·τ)` umklapp phase at the V_q level (it doesn't,
per §3.1 of the derivation), that would need to be added. Si's 160 eV
Σ_X failure is **NOT** at the V_q level (Si has zero TRS-augmented q's
because of inversion; the failure is the ψ-side `R_cart` bug per the
Si algebraic agent 00:25 entry).

## 6. Cross-check against parallel agents

**Math agent zeta_unfold_derivation.md** (read in full): §3 V_q rule and
§5.1 file:line localization both match what the production code does
modulo the direction choice. Their proposed fix at `orbit_syms.py:308`
(change `Rinv = inv(S)` → `R_fwd = S`) is consistent with the code I
traced. Their §5.2 note about **coupling to `compute_rgrid_sym_perm`
line 450** (which uses the same Rinv=inv(S) construction for the r-grid
ζ unfold) is correct — flip both atomically.

**Agent A**: report not yet posted at the time of this writing
(`context_agent_A_report.md` does not exist). Will append a cross-check
note if/when it lands.
