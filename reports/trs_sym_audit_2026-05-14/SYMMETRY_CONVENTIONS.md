# LORRAX Symmetry Conventions — empirical reference

**Status:** as of 2026-05-15. This document records the BGW convention
LORRAX should use, the history of how we converged on it, what works,
and what's still broken. Treat the **Current state** section as
load-bearing; the **History** section is for context only.

## Current state (load-bearing)

### BGW convention (verified against pw2bgw writer + BGW reader + Si atom test)

- `wfn.sym_matrices[s] = mtrx[s]` is the **BGW k-action matrix**: `q' = mtrx·q`
  in column form. It is **NOT** the forward real-space rotation U; an earlier
  iteration of this doc claimed that, incorrectly.
- `wfn.translations[s] = tnp[s] = 2π · τ_frac`, per BGW spec
  (`docs/docs_bgw/wfn.h5.spec:180-185`). LORRAX divides by `2π` everywhere
  to get τ in fractional crystal coords. **This scaling is correct.**
- BGW's real-space sym action: **`r' = mtrx⁻¹ · r + τ`** in column form.
  Verified by `validate_atomic_symmetries` on Si Fd-3m (96/96 atom mappings
  pass under `inv(mtrx)`; 48/96 under `mtrx` direction).
- Group composition under this convention: `S_c = S_a S_b`,
  `τ_c = inv(S_a)·τ_a + τ_b` (mod lattice). Closes 0/2304 fails on Si.
- Reciprocal forward action: `q' = mtrx · q` (column form). Composition:
  `q'' = mtrx_a · mtrx_b · q`. So `sym_mats_k = mtrx` directly, NOT `mtrx.T`.
  **(POSSIBLE BUG: LORRAX currently uses `sym_mats_k = mtrx.T`. See
  `Outstanding hypotheses` below.)**

### The ISDF V_q unfold formula (validated on CrI3)

For full-BZ q1 with parent q_irr and sym index s:

    y_μ = mtrx_s · (x_μ − τ_s) = x_{α_s(μ)} + L_{s,μ},   L ∈ ℤ³
    V_full[q1, μ', ν'] = exp(2π i q_irr · (L_{s,μ'} − L_{s,ν'}))
                       · V_ibz[parent, α_s(μ'), α_s(ν')]
    if TRS row: V_full = conj(V_full)

`compute_centroid_sym_perm` produces `sym_perm = α` and `L_table = L`
directly per user-spec; `unfold_v_q` consumes them.

### What's verified

| System | Test | Result | Status |
|--------|------|--------|--------|
| CrI3 6×6 30 Ry | V_q dump unit test (synthetic q_irr, all 36 q's) | rel 8.7e-6 | ✓ ISDF floor |
| CrI3 6×6 30 Ry | sym-vs-nosym Σ_X e2e (3024 k×n pairs) | max 0.076 meV | ✓ <1 meV gate |
| MoS2 3×3 SOC | sym-vs-nosym Σ_X e2e | max 0.090 meV | ✓ <1 meV gate |
| Si 4×4×4 SOC | centroid orbit closure under BGW conv | 432 centroids, all permutations | ✓ closes |
| Si 4×4×4 SOC | sym-vs-nosym Σ_X e2e | max 48 eV | ✗ FAILS |

## History — what we've explored this session

### Code changes on lorrax_B `agent/trs-aware-sym-fix`

| Commit | Effect |
|--------|--------|
| `0735c2a` | unfold_v_q: per-centroid umklapp L-phase + forward-direction perm |
| `c657785` | (later reverted) orbit_syms: forward-S direction — wrong, was based on incorrect "mtrx = U" claim |
| `0b0fc37` | orbit_syms: apply BGW r-action `r' = mtrx⁻¹·r + τ` everywhere; matches subagent diagnosis |

### Diagnostic milestones

1. **Pre-session**: Si 4×4×4 sym-vs-nosym Σ_X had 160 eV gap (full-BZ fallback,
   centroid orbit closure failed under the old code). CrI3 6×6 30 Ry had 6 eV gap.
2. **The L-phase commit (0735c2a)** drove CrI3 from 6 eV → 0.076 meV. Si stayed
   in full-BZ fallback (no IBZ cascade), and incidentally moved from 160 eV →
   791 meV due to OTHER Phase 2 changes (unfold_psi refactor, R_cart fix) that
   happened to clean up Si's full-BZ ψ-side path. The L-phase fix itself
   doesn't fire on Si in full-BZ fallback.
3. **Subagent diagnosis** confirmed BGW r-action is `r' = mtrx⁻¹·r + τ`, NOT
   `r' = mtrx·r + τ`. Earlier code in `compute_centroid_sym_perm` and the
   centroid orbit-closure helpers used the wrong direction; harmless on
   symmorphic CrI3/MoS2 (τ=0 collapses both directions), fatal on Si Fd-3m.
4. **BGW-convention fix (0b0fc37)** fixed `compute_centroid_sym_perm` to use
   the user-spec `y_μ = mtrx·(x_μ−τ) = x_α + L`. Si centroid set now closes
   (432 centroids form valid permutations under all 96 sym ops including TRS).
   IBZ cascade fires for Si for the first time.
5. **But Si Σ_X is now 48 eV — WORSE than the 791 meV fallback state.** The
   IBZ cascade engagement exposes a downstream ψ-side bug that the fallback
   path was hiding. CrI3 is unaffected by the BGW-conv fix (τ=0).

## Outstanding hypotheses (Si's 48 eV residual)

Candidates for the downstream bug, in roughly decreasing likelihood:

### H1: `sym_mats_k = mtrx.T` is the wrong direction

`common/symmetry_maps.py:478` defines `self.sym_mats_k = self.sym_matrices.transpose(0,2,1)`.
Per the BGW reciprocal action `q' = mtrx · q` established above,
`sym_mats_k` should be `mtrx` directly — NOT the transpose. This matters
for:
- `find_irreducible_bz_points` (full→IBZ k-mapping uses sym_mats_k).
- `unfold_psi` (rotates G-vectors via `sym_mats_k[sym_idx] @ G`).
- Any q-related sym arithmetic.

If wrong: every Si k-mapping is off, every ψ unfold uses wrong G's.
Symmorphic systems (τ=0) might cancel out by accident. Non-symmorphic
Si: catastrophic.

**Evidence to produce**: open WFN.h5 for Si, take an IBZ k and one of
its sym-related full-BZ k from the k-list, compute `mtrx @ k_irr` and
`mtrx.T @ k_irr`, see which matches `k_full` mod kgrid.

### H2: `unfold_psi`'s τ-phase application uses wrong matrix on G

In `common/symmetry_maps.py:330-340`, unfold_psi computes:
```
rotated = (S_full @ g_bar.T).T          # S_full = sym_mats_k[sym_idx]
phase = np.exp(-1j * (rotated @ tau))
```
The τ-phase formula assumes a specific r-action convention. If
`sym_mats_k` is wrong (H1), the rotated G is wrong; the τ-phase compounds
the error.

**Evidence**: take a Si IBZ k with non-trivial sym op, compute
`ψ_{full}(G) = U_spinor · ψ_{irr}(?_G) · phase`, compare to a direct
load of ψ at the full-BZ k. Where does the mismatch land?

### H3: The L-phase sign or index convention in `unfold_v_q`

Currently `phase = exp(2π i q_irr · L[s, μ'])`, sign convention "+L_μ − L_ν".
The user's spec is mathematically clean but the orientation between
`q_irr` and `q_full` depends on the k-action direction. If `sym_mats_k`
is wrong (H1), `q_irr_frac[parent]` may not correspond to the right
physical IBZ q-point for the phase.

**Evidence**: take Si IBZ q and full-BZ q with non-trivial sym, compute
the user-spec V_full from V_ibz with various phase-sign choices and
both possible q-mappings, see which closes against a nosym reference V_q
sample.

### H4: `compute_centroid_sym_perm` τ_frac sign

Internal: `tau_frac = translations / (2π)`. For Si sym 1 we get
`τ[1] = (-1/2, 0, 0)`. Apply `inv(mtrx) · r + τ`: this gives the BGW
forward sym in real space. Verified by atom test 96/96.

But subagent noted the validate_atomic_symmetries impl uses
`transformed = rot @ pos + tau` with `rot = inv(mtrx)` — i.e., the same
direction. If LORRAX's `compute_centroid_sym_perm` now uses `mtrx · (r-τ)`
(which equals `inv(mtrx) · r + τ` for the SOURCE direction), the
**source** map is correct. But maybe the SIGN of τ in this formula is
flipped (should it be `mtrx · (r + τ)`?) Worth verifying.

**Evidence**: identity-check `mtrx · (r - τ) =? inv(rot_atom) · r - inv(rot_atom)·τ`
to confirm the source-map algebra is right.

### H5: Spin-side / U_spinor convention for non-symmorphic

Si is bispinor=false, so U_spinor is trivial. But there's still a
spinor branch in `unfold_psi` controlled by `ns=2`. If Si's WFN.h5 is
loaded with `ns=1` but unfold_psi takes a spinor path anyway, that's
a separate issue.

**Evidence**: print `wfn.spinor`, `ns`, and the U_spinor matrix used
during Si's run. Confirm Si uses the scalar path.

## References

- WFN.h5 spec: `docs/docs_bgw/wfn.h5.spec`
- BGW pw2bgw writer: `sources/BerkeleyGW/MeanField/ESPRESSO/version-7.2/pw2bgw_qe7.2_with_spinor_mag.f90:826-850`
- BGW Σ reader using `/(2π)`: `sources/BerkeleyGW/Sigma/sympert_utils.f90:754`
- BGW HDF5 writer: `sources/BerkeleyGW/Common/wfn_io_hdf5.p.f:557`
- LORRAX reader: `sources/lorrax_B/src/file_io/mf_header.py:119`
- LORRAX SymMaps + sym_mats_k: `sources/lorrax_B/src/common/symmetry_maps.py:471-490, 786-792`
- LORRAX centroid orbit: `sources/lorrax_B/src/centroid/orbit_syms.py` (after `0b0fc37`)
- LORRAX V_q unfold: `sources/lorrax_B/src/common/symmetry_maps.py:110-260` (`unfold_v_q`)
- LORRAX ψ unfold: `sources/lorrax_B/src/common/symmetry_maps.py:254-395` (`unfold_psi`)
- LORRAX IBZ k-mapping: `sources/lorrax_B/src/common/symmetry_maps.py:60-110` (`find_irreducible_bz_points`)
- CrI3 V_q dump unit test (passing): `reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py`
- Si run dir (currently failing 48 eV): `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym_bgw_conv_2026-05-15/`
- Si comparison script: `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/compare_sigma_x_bgw_conv.py`

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

## BGW r-action convention: `r' = mtrx⁻¹ · r + τ` (REVISED 2026-05-15)

**Earlier in this doc** I claimed `wfn.sym_matrices = U` (forward
direct rotation) based on CrI3 empirics. That was right in the
limited sense that for CrI3 P-3 (symmorphic τ=0), the matrix
direction doesn't matter — orbits coincide.

**The actual BGW convention** (verified by reading `pw2bgw_qe7.2_with_spinor_mag.f90:826-850`,
`BerkeleyGW/Sigma/sympert_utils.f90:754`, and the `validate_atomic_symmetries`
test on Si Fd-3m which passes 96/96 only under this convention) is:

    r' = mtrx⁻¹ · r + τ        (forward real-space sym)
    τ = wfn.translations / (2π) (per BGW spec)
    Reciprocal: q' = mtrx · q   (BGW's "k-action" via `matmul(syms%mtrx, k)`)

So `wfn.sym_matrices = mtrx = U⁻¹` where U is the user's "forward
direct rotation." Equivalently, `inv(wfn.sym_matrices) = U`. This
reconciles the earlier-confusing reading: BGW stores K (k-action,
not U), and the r-action uses `Rinv = inv(mtrx) = inv(K)`.

For the user's umklapp-aware unfold formula:

    y_μ = U⁻¹(x_μ − τ) = mtrx · (x_μ − τ) = x_{α(μ)} + L_μ

`compute_centroid_sym_perm` computes this directly (see
`orbit_syms.py:309`). For symmorphic systems both old and new
conventions give the same numerical result; for non-symmorphic
Si the BGW convention is required for orbit closure.

## Si Fd-3m: progress, but still failing the 1 meV gate (2026-05-15)

**Earlier I claimed Si's τ-table didn't compose mod-1.** That was a
**wrong** diagnosis — I was using the wrong closure rule. The correct
BGW closure law under `g·r = mtrx⁻¹·r + τ` is `τ_c = mtrx_a⁻¹·τ_a + τ_b`
(when `S_c = S_a S_b`). With this rule, Si's stored τ values compose
correctly: **0/2304 fails**. Closure under the right convention holds.

**After applying the BGW-convention fix to compute_centroid_sym_perm**
(orbit_syms.py commit `0b0fc37`):

- Si centroid regen produces 432 orbit-closed centroids.
- All 96 sym ops × 432 centroids form valid permutations under
  `compute_centroid_sym_perm`, `|L|max = 3` (real lattice wraps).
- The V_q IBZ cascade fires for the first time on Si (`n_q_ibz=8`,
  `unfold=IBZ→full`, vs the prior `use_ibz=False` fallback).

**Si Σ_X trajectory:**

| State | max |ΔΣ_X| |
|---|---|
| Pre-this-session (Phase 2 only) | 160 eV |
| L-phase commit only, full-BZ fallback | 791 meV |
| **BGW-conv fix + IBZ cascade firing** | **48 eV** |

The 48 eV is from a separate bug that the cascade engagement now
exposes. Likely candidates:

- `unfold_psi` has its own τ-handling that may use a different
  convention than `compute_centroid_sym_perm` now uses.
- The L-phase formula in `unfold_v_q` may interact incorrectly with
  the new α-direction sym_perm semantics for non-symmorphic ops.
- The k-mapping (`sym.irr_idx_q`, `sym.sym_idx_q`) is built using
  `sym_mats_k = mtrx.T` (forward k-action under BGW convention is
  `q' = mtrx · q`, so `sym_mats_k` should be `mtrx`, not `mtrx.T`).
  Worth auditing.

Symmorphic systems (CrI3, MoS2) are not affected by any of these
because τ=0 collapses all conventions. The CrI3 gate at 0.076 meV
still passes after the BGW-convention fix.
