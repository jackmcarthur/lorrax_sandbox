# LORRAX Symmetry Conventions — empirical reference

**Status:** as of 2026-05-15, after the umklapp/orbit-closure investigation. This
document supersedes the conflicting prior reports in this directory; treat the
facts here as load-bearing and the older notes as historical.

## TL;DR

- `wfn.sym_matrices[s]` stores **U_s, the forward direct-space rotation**, in
  fractional crystal coords. Not K (reciprocal rotation), not K⁻¹, not U⁻¹.
- For a sym op {U | τ}, real-space points transform forward as
  `x → U·x + τ (mod 1)`. Inverse direction is `x → U⁻¹·(x − τ) (mod 1)`.
- For an ISDF basis ζ on a sym-closed centroid set {x_μ}, V_q transforms
  under (sym + umklapp) as

      V_{q1, μν} = exp(2π i q · (L_μ − L_ν)) · V_{q, α(μ), α(ν)}

  where q1 = K·q + G_R is the full-BZ q (K = U⁻ᵀ acts on q-fractional), q is
  the IBZ parent q (fractional reciprocal), and (α, L) come from the
  inverse-direction wrap:

      y_μ = U⁻¹ (x_μ − τ)     in fractional direct
      y_μ = x_{α(μ)} + L_μ    with x_{α(μ)} ∈ [0,1)³, L_μ ∈ ℤ³

- **Empirically verified** at ISDF noise floor (rel ~10⁻⁵, ≈ 21 eV out of
  |V|=2.5×10⁶) on the CrI3 6×6 30 Ry V_q dumps for every q where the centroid
  set is sym-closed.

## What the prior investigation got wrong

A 2026-05-15 agent investigating `sources/BerkeleyGW/Common/symmetries.f90:189`
concluded `mtrx = inv(spglib_R)` and therefore `wfn.sym_matrices = K` (= U⁻ᵀ).
**That conclusion is wrong by empirical test.** For CrI3 (P-3, hex coords) the
forward C3 direct-space sym is

    U_C3 = [[0, -1, 0],
            [1, -1, 0],
            [0,  0, 1]]

(derivation: maps a1 → a2, a2 → −a1−a2 in hex coords).

Empirically:

    wfn.sym_matrices[1] = [[0, -1, 0],
                           [1, -1, 0],
                           [0,  0, 1]]

These match. If `wfn.sym_matrices` were K = U⁻ᵀ, the matrix would be

    U⁻ᵀ_C3 = [[-1, -1, 0],
              [ 1,  0, 0],
              [ 0,  0, 1]]

which it is NOT. So `wfn.sym_matrices = U`, full stop.

Either the prior agent misread BGW's source, or BGW's "k-action" routine
`matmul(syms%mtrx, k)` is doing something inverse-flavored that effectively
makes mtrx coincide with U. Either way the empirical reading is decisive and
the user's longstanding test that "DFT degenerate subspaces transform correctly
under sym" relies on this convention.

## The convention, written without algebra games

In fractional crystal coords:

- **Direct-space forward sym**: `x' = sym_matrices[s] @ x + tau[s]`.
- **Direct-space inverse sym**: `x' = inv(sym_matrices[s]) @ (x − tau[s])`.
- **Reciprocal forward sym** (transformation of the q-label): `q' = K_s @ q`
  with `K_s = sym_matrices[s]⁻ᵀ`.
- **Reciprocal inverse**: `q' = sym_matrices[s]ᵀ @ q`.

In numpy (treating r as a row vector, the LORRAX convention):

- Forward: `r' = r @ U.T + τ` (where U = `sym_matrices[s]`). Equivalent to
  `U @ r + τ` if you treat r as a column vector. The actual code in
  `centroid/orbit_syms.py:313` does column-form `np.einsum('rj,sij->sri',
  r_frac, sym_matrices) + τ`, which is `sym_matrices @ r + τ`, the forward
  direct-space sym. **This is correct** — the post-flip change in commit
  `80edbe8` was just renaming from inverse-direction (pre-flip) to
  forward-direction; for abelian groups both produce the same orbit set.

For C3 in hex, U and U⁻¹ are not equal as matrices:

    U_C3   = [[0,-1,0],[1,-1,0],[0,0,1]]     (order 3)
    U⁻¹_C3 = U_C3²
           = [[-1,1,0],[-1,0,0],[0,0,1]]

For abelian groups (CrI3 P-3 = {E, C3, C3², −I, S6, S6²} is abelian) both
`{U_s}` and `{U_s⁻¹}` generate the same orbit on any point. The choice of
direction only affects the SIGN of L_μ and therefore the SIGN of the phase
factor in V_q; the magnitude residual is identical.

## ISDF-basis transformation under sym (derivation)

Given the ISDF identity at IBZ (k, q):

    ρ_{k,q}(r) = Σ_μ ζ_{q,μ}(r) · ρ_{k,q}(r_μ)        (1)

and forward sym {U|τ} acting on real space:

    ψ_{Sk}(r) = U_spinor · ψ_k(U⁻¹(r − τ))            (2)

The pair density at the rotated point:

    ρ_{Sk,Sq}(r) = ρ_{k,q}(U⁻¹(r − τ))                 (3)

(spinor U_spinor† U_spinor cancels). Plug into ISDF at (Sk, Sq):

    ρ_{Sk,Sq}(r) = Σ_μ ζ_{Sq,μ}(r) · ρ_{Sk,Sq}(r_μ)    (4)

Use (3) on the LHS and on each sample:

    ρ_{k,q}(U⁻¹(r − τ)) = Σ_μ ζ_{Sq,μ}(r) · ρ_{k,q}(U⁻¹(r_μ − τ))

The key step: `U⁻¹(r_μ − τ)` is the same as `x_{α(μ)} + L_μ` (the
inverse-direction wrap that defines α and L). The pair density at a shifted
lattice point satisfies the Bloch identity:

    ρ_{k,q}(r' + L) = exp(2π i q · L) · ρ_{k,q}(r')    (5)

So `ρ_{k,q}(U⁻¹(r_μ − τ)) = ρ_{k,q}(x_{α(μ)} + L_μ)
   = exp(2π i q · L_μ) · ρ_{k,q}(x_{α(μ)})`.

Reindex the sum and match coefficients (using `ρ_{k,q}(x_{α(μ)})` as basis):

    ζ_{Sq,μ}(r) = exp(2π i q · L_μ) · ζ_{q,α(μ)}(U⁻¹(r − τ))    (6)

Then V_{Sq}[μ, ν] = ∫∫ ζ*_{Sq,μ}(r) v(r-r') ζ_{Sq,ν}(r') dr dr':

    V_{Sq}[μ, ν] = exp(−2π i q · L_μ) exp(2π i q · L_ν)
                   · ∫∫ ζ*_{q,α(μ)}(r̃) v(r̃-r̃') ζ_{q,α(ν)}(r̃') dr̃ dr̃'
                 = exp(2π i q · (L_ν − L_μ)) · V_q[α(μ), α(ν)]

(change of integration variable r̃ = U⁻¹(r-τ); v is rotation-invariant; |det U|
= 1 for integer crystal-coord rotations of finite groups).

**Sign sanity check**: in the formula `V_{q1,μν} = exp(2π i q · (L_μ − L_ν)) ·
V_{q, α(μ), α(ν)}`, the user wrote `L_μ − L_ν` with positive sign on L_μ. My
derivation gives `L_ν − L_μ` (opposite). Verified empirically: both give the
same residual on the V_q dumps (`phase[:, None] * V * phase.conj()[None, :]`
vs `phase.conj()[:, None] * V * phase[None, :]`) — the formula is sign-symmetric
because V is Hermitian and the indices μ,ν are bookkeeping. Implementation
should follow the user's explicit spec.

## TRS-augmented case

For composed (TRS + spatial-S):

- L_table is the SAME as the spatial half (TRS acts on momenta, not r).
- α(μ) is the SAME as the spatial half.
- The final V_TRS[μ,ν] equals **complex conjugate** of the spatial-unfold
  V_S[μ,ν], NOT a sign-flipped phase. The L-sign flip from q → −q and the
  index-swap from V Hermitian both fall out into a single conj operation.

Implementation in `unfold_v_q`:

    sym_idx_spatial = sym_idx if sym_idx < n_tran else sym_idx − n_tran
    V_perm = V_ibz[parent][np.ix_(α[sym_idx_spatial], α[sym_idx_spatial])]
    qL = L[sym_idx_spatial] @ q_irr_frac
    phase = exp(2π i qL)
    V_full = phase[:,None] * V_perm * phase.conj()[None,:]
    if sym_idx ≥ n_tran:    # TRS row
        V_full = V_full.conj()

## The orbit-closure bug

**Symptom**: on CrI3 6×6 30 Ry, the centroid set
`runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt`
has 48/300 centroids whose C3-image is NOT in the set, even though the
manifest claims "orbit-closed 300 reps".

**Diagnostic data**:

- The 48 missing centroids are the SAME under both U and U⁻¹ direction (so
  it's a set-closure issue, not a direction issue).
- Same 48 fail under all 4 non-involutive ops (C3, C3², S6, S6²) and their
  4 TRS-augmented partners. Total 384/3600 (s,μ) pairs fail; 48 ought-to-be-
  in-set centroids missing.
- τ=0 for all sym ops (CrI3 P-3 is symmorphic), so this is NOT a
  non-symmorphic τ × FFT_grid commensurability issue.
- |L_μ| = 1 max (small integer wraps, as expected for centroids in [0,1)³
  rotated and wrapped).
- Example: μ=17 at (2/45, 34/45, 13/120). C3⁻¹(r) = (32/45, 43/45, 13/120).
  The latter is absent from the centroid set.
- The 48 misses cluster at specific z-coordinates that DO have hits at other
  μ — so it's not "vacuum centroids" being excluded.
- Coords like (2/45, 34/45, 13/120) suggest mixed FFT grids: x,y on a 45-grid
  (= 9·5, where 9 = 3·3 has the C3 factor), z on a 120-grid. 45 is divisible
  by 3 (C3 commensurate). So C3-mapping SHOULD land on the same 45-grid.

**Hypotheses** for what the orbit-closure pipeline is doing wrong:

1. The kmeans seed produced ~300 reps; `snap_orbits_to_grid` canonicalized
   them but didn't extend to cover full orbits — output is "canonical
   representatives" only, missing other orbit members.
2. `unfold_orbit_unique_with_id` is called somewhere to extend, but uses
   `Rinv` (U⁻¹) for the unfold matrix; for some specific centroids the
   unfold produces a point that gets deduped away due to a too-loose
   tolerance (`tol=1e-6` in `unfold_orbit_unique_with_id`).
3. The centroid-count was capped at 300 by the user, and the orbit-closure
   step had to drop 48 orbit members to fit the budget.
4. Something in the kmeans_cli's orbit-aware mode silently doesn't enforce
   closure post-snap.

## Files / functions in the orbit-closure pipeline

- `sources/lorrax_B/src/centroid/orbit_syms.py`:
  - `orbit_images:73-78` — applies sym to reps in row-form; physically does
    `U⁻¹·r + τ` in column form (= same orbit as forward U for abelian).
  - `snap_orbits_to_grid:120-153` — snaps to FFT grid + canonicalizes; the
    `n_dups` reporting is "snap+canon collisions", not "missing orbit members".
  - `canonicalize_orbit:107-117` — picks lex-smallest orbit member per rep.
  - `unfold_orbit_unique_with_id:156-196` — claims to unfold reps to all
    distinct orbit images; uses int64 lex keys with `tol=1e-6` rounding.

- `sources/lorrax_B/src/centroid/kmeans_cli.py` (and friends) — the entry
  point for centroid generation; controls whether orbit-aware mode is on.

- `sources/lorrax_B/src/centroid/` — other helpers; search for "orbit" or
  "close" to find the full pipeline.

## What needs to happen

1. Trace the centroid generation pipeline that produced the offending
   `centroids_frac_300.txt`. Identify the exact step that drops the 48
   missing orbit members.
2. Either FIX the orbit-closure to be complete (every (s, μ) maps into the
   set), or DOCUMENT clearly why 48 are missing and what should be done
   instead.
3. Once orbit closure is bulletproof, regenerate CrI3 30 Ry centroids and
   verify with a strict closure check (every (s, μ) → centroid index, no
   misses, no false matches).
4. Then add `L_μ` capture in `compute_centroid_sym_perm` + phase in
   `unfold_v_q` per the formula above.
5. Re-run V_q dump test → all 36 q's at ISDF floor.
6. Re-run CrI3 30Ry sym-vs-nosym Σ_X → <1 meV gate.

## Reference: the verification script

`reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py` is the
empirical gold standard for the V_q formula. It loads the V_q dumps and
applies the user-math formula directly; passes at ISDF floor (rel 8.67e-6)
on the 14 q's with closed orbits.

The V_q dumps are at
`reports/trs_sym_audit_2026-05-14/v_q_dumps/{Vq_ibz_sym.h5, Vqmunu_nosym.h5}`.

## Si Fd-3m: τ-table doesn't compose mod-lattice (2026-05-15 finding)

Si 4×4×4 sym-vs-nosym Σ_X residual at 791 meV (down from 160 eV, but
above the 1 meV gate) traces to a deeper issue than the V_q L-phase.
For Si's stored sym table:

- **Matrix products close**: every `S_a · S_b` matches some stored `S_c`
  (48 ops form a true matrix group). ✓
- **τ composition fails modulo lattice**: only 576/2304 = 25% of (a, b)
  pairs have `S_a · τ_b + τ_a ≡ τ_{ab} (mod 1)`. The remaining 75%
  differ by half-lattice translations (e.g. (1/2, 1/2, 0)) which do NOT
  vanish mod 1.

Consequence: forward action `r → S r + τ` is **not a true group action
modulo 1** for Si. Applying a sym op to a point in the "forward orbit"
of x can produce points OUTSIDE that orbit. Concretely:

```
Forward orbit of (4,8,12)/24:  44 distinct points
Forward orbit of (12,12,8)/24: 24 distinct points
(12,12,8) ∈ orbit-of-(4,8,12); (4,8,12) ∈ orbit-of-(12,12,8)
⇒ should be equal sets (orbits partition for true group actions)
⇒ but |intersection| = 20, |symdiff| = 28 — they're NOT equal
```

Atomic positions in Si DO map correctly under the stored sym ops
(`validate_atomic_symmetries` returns 0 failures), because Si atoms sit
on high-sym Wyckoff sites where the τ-mismatch lands on lattice
positions. For generic positions (centroids), the mismatch produces
half-lattice offsets that break orbit partition.

**Likely root cause**: BGW WFN.h5 stores τ values in a convention that
encodes the *coset representatives* of a non-symmorphic factor group,
not a true group of mod-1 actions. The convention works for atomic
sym checks (and for the BGW computation it was designed for) but breaks
LORRAX's orbit-aware kmeans + V_q IBZ-cascade closure assumption.

**Status**: Si bypasses the V_q IBZ cascade (use_ibz=False) because
`compute_centroid_sym_perm` fails closure validation, falling back to
full-BZ ζ-fit + V_q. The L-phase fix in `unfold_v_q` doesn't fire on
Si. The 791 meV residual is from somewhere else — likely
`unfold_psi`'s τ-phase application on the same non-closed τ table.

**Not fixed in this session.** Si needs a separate root-cause
investigation:
1. Verify the BGW τ-convention by cross-checking against the QE input
   atoms and their fractional translations.
2. Consider deriving an equivalent τ table that IS closed mod-1.
3. Or accept that for non-symmorphic systems with this BGW quirk, the
   IBZ cascade can't fire and full-BZ fallback is the right path —
   then fix any residual sym handling in the full-BZ path.
