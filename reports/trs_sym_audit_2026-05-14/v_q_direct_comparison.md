# V_q direct sym-vs-nosym comparison — CrI3 6×6 30 Ry

**Date**: 2026-05-14
**Agent**: V_q-comparison agent (lorrax_B `agent/trs-aware-sym-fix` @ HEAD `80edbe8`)
**Test bed**: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/`
**Approach**: empirical V_q tensor dump + element-wise comparison; no theoretical pre-judgement.

## TL;DR

1. **Step 1 (Σ_X measurement post both fixes)**: FAIL. Max |ΔΣ_X| =
   **4096 meV** (the `compare_sigma_x.py` script double-counts because in
   x_only mode sex_0 = x_bare; raw script-reported is 8191 meV).  Math
   agents' centroid_perm flip DID reduce |ΔΣ_X| from prior 6022 meV (with
   only R_cart fix) to 4096 meV — a ~2 eV improvement — but the gate is
   1 meV, so still 4000× failing.

2. **Step 2 (direct V_q comparison)**: V_q^full tensor **disagrees at
   |ΔV| ~ 2.4e6 absolute / ~1.0 relative** between sym and nosym paths.
   ζ at IBZ q-points matches at ISDF-noise-floor (1e-5 relative), and
   V_q at IBZ q-points matches at ISDF-noise-floor — but V_q AFTER
   unfold to non-IBZ q-points is wrong.  **The V_q unfold itself is the
   bug.**

3. **Step 3 (Σ_X kernel trace)**: not needed — the bug is at V_q level,
   not in downstream consumption.

4. **Root cause (algebraic)**: For C3-related q-points, the relation

       V_nosym[q', μ', ν'] = exp(i [θ_s(μ') - θ_s(ν')])
                            · V_nosym[parent, sym_perm[s, μ'], sym_perm[s, ν']]

   with θ_s(μ') ∈ {0, ±π/3} taking values depending on the centroid's
   z-coordinate (above vs below z = 0.5).  The current
   `common.symmetry_maps.unfold_v_q` (`src/common/symmetry_maps.py:110-247`)
   applies the centroid double-permute but **misses the gauge phase
   factor**.  The math agent's derivation §3 of `zeta_unfold_derivation.md`
   claimed the τ-phase + G-axis umklapp factors cancel in the V_q
   bilinear — **this claim is empirically false** for CrI3 P-3 C3 ops.

5. **File:line diagnosis**:
   - `src/common/symmetry_maps.py:110-247` (`unfold_v_q`) applies only the
     centroid double-permute; **missing per-element phase factor**.
   - The phase factor depends on the umklapp of the sym op (the kg0
     wraparound when `S q_irr` exits the principal BZ).  Source of phase:
     either the ζ G-axis-pullback umklapp `exp(2πi k_g0 · r_μ)` that
     doesn't cancel when sym_perm permutes μ and ν to different
     "umklapp classes", OR a τ-phase analog that's specific to the
     IBZ-cascade's k_g0 bookkeeping in `find_irreducible_bz_points`.

6. **Proposed fix**: do NOT attempt a centroid-only unfold for V_q with
   non-trivial sym ops.  Either (a) replicate the V_q computation at
   every full-BZ q (turning off IBZ-cascade for systems with non-cubic
   non-involutive sym ops), OR (b) re-derive the V_q transformation rule
   with the umklapp G-pullback included and apply the per-element phase
   in `unfold_v_q`.  Option (a) is the safe immediate fix.

## Step 1: Post-fix Σ_X re-measurement

### Setup

- HEAD `80edbe8` includes both:
  - `5dc8813`: `syms_crystal_to_cartesian` R_cart fix (mtrx not mtrx.T)
  - `80edbe8`: `compute_centroid_sym_perm` forward direction (S not inv(S))
- `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/cohsex.in` and
  `run_nosym/cohsex.in` already in place; reran `gw_jax` on run_sym with
  fresh tmp/ at HEAD; nosym output reused from prior run (16:09 timestamp,
  no source-dependence on the two fixes since ntran=1 means identity-only
  centroid perm and identity-only R_cart, both trivially correct).
- Allocation: JID 52976844 (1 node, 4× A100-40GB, sym run uses 2 GPUs).

### Result

```text
$ python3 compare_sigma_x.py
sym WFN: ntran=6, has_inversion=True
nosym WFN: ntran=1, has_inversion=False

===== sym vs nosym PR3 Σ_X comparison (CrI3) =====
  total (k, n) pairs: 3024 = 36 k-pts × 84 bands

--- DFT eigenvalue offset (orientation check) ---
   max |ΔE_dft|        =     0.0010 meV       (PASS — DFT identical)

--- Σ_X components ---
   max |Δx_bare|       =  4095.7410 meV
   max |Δsex_0|        =  4095.7410 meV       (sex_0 = x_bare in x_only mode;
                                                the script's "ΔΣ_X total" is
                                                effectively 2·Δx_bare)
   max |Δcoh_0|        =     0.0000 meV
   max |ΔΣ_X (total)|  =  8191.4820 meV       (= 2 × 4095.7 meV)

  Verdict: FAIL — max |ΔΣ_X| = 4.096 eV (4000× the 1 meV gate)
```

### Comparison vs prior states

| Code state                                  | max |ΔΣ_X| (real, meV) | gain vs baseline |
|--------------------------------------------|-----------------------|------------------|
| Pre-Phase-2 baseline                        | (CrI3 not tested)     | —                |
| `c34ae49` (R_cart fix only, pre-perm flip)  | 6022                  | baseline         |
| `80edbe8` (R_cart + centroid_perm flip)     | 4096                  | -1926 meV        |

The centroid_perm flip improved by ~2 eV but did not close the gap.
**Both math-agent fixes together are insufficient.**

## Step 2: direct V_q^full sym-vs-nosym tensor comparison

### Instrumentation

Added two temporary env-var-gated H5 dumps in
`src/gw/gw_init.py:1112-` (full-BZ V_qmunu after unfold) and
`src/gw/v_q_g_flat.py:457-` (IBZ V_q pre-unfold, plus sym tables).
Triggered by `LORRAX_DUMP_VQMUNU=path` and `LORRAX_DUMP_VQ_IBZ=path`.
**Reverted before reporting** (`git checkout -- src/gw/{gw_init,v_q_g_flat}.py`,
`git status` clean on tracked files).

Ran both sym and nosym `gw_jax` with the dumps active (one allocation,
~2 minutes each).  Files at
`reports/trs_sym_audit_2026-05-14/v_q_dumps/`:

| File                  | Shape          | Contents                                |
|-----------------------|----------------|-----------------------------------------|
| `Vqmunu_sym.h5`       | (36, 300, 300) | sym path full-BZ V_q (post unfold)      |
| `Vq_ibz_sym.h5`       | (8, 300, 300)  | sym IBZ V_q (pre-unfold) + sym tables   |
| `Vqmunu_nosym.h5`     | (36, 300, 300) | nosym full-BZ V_q (direct, no unfold)   |
| `Vq_ibz_nosym.h5`     | (36, 300, 300) | nosym IBZ V_q (≡ Vqmunu_nosym, no fold) |

### Tier 1: IBZ-vs-IBZ baseline

Sanity: V_q at the **8 IBZ q-points** matches between sym and nosym at
the ISDF noise floor:

```text
  sym IBZ idx=0, nosym full_q= 0: max|Δ|=1.40e+01, rel=6.9e-06
  sym IBZ idx=1, nosym full_q= 1: max|Δ|=1.79e+01, rel=7.3e-06
  sym IBZ idx=2, nosym full_q= 2: max|Δ|=1.03e+01, rel=6.0e-06
  ... (all 8 match at rel < 8e-06)
```

So ζ at the IBZ q-points and the V_q kernel itself are working
correctly.  Confirmed independently by ζ-vs-ζ comparison at IBZ q's
(`max|Δζ| ≈ 9e-3, rel ≈ 6e-6`).

### Tier 2: full-BZ V_q^full sym vs nosym

```text
  global max |ΔV| = 2.41e+06
  global max |V_nosym| = 2.46e+06
  rel max = 0.98       (~100% wrong on the largest elements!)
  L2(ΔV) / L2(V_nosym) = 0.78
```

Per-q breakdown (subset; full table in compare_v_q.py output):

```text
  q   parent_ibz  sym_idx  | max |ΔV_q|        L2 |ΔV_q|       status
  0       0           0    | 1.40e+01      2.11e+02       MATCH (sym_idx=0 trivial)
  1       1           0    | 1.79e+01      3.09e+02       MATCH
  ... (all 8 sym_idx=0 q-points match at ISDF floor)
  4       2           3    | 6.31e+05      7.04e+06       FAIL (-I)
  5       1           3    | 5.76e+05      6.41e+06       FAIL (-I)
  6       1           5    | 2.41e+06      3.84e+07       FAIL
  10      4           1    | 1.75e+06      3.14e+07       FAIL (C3)
  11      1           1    | 2.41e+06      4.40e+07       FAIL (C3)
  ... (28 of 36 q-points fail at 1e5–1e6 absolute)
```

**Pattern**: every q-point with `sym_idx > 0` fails massively.

### Tier 3: which permutation convention restores agreement?

Tested 8 permutation/transpose/conj variants per q-point.  Summary:

```text
  q  sx | current(inv,inv) fwd,fwd     T(inv,inv)   conj(inv,inv)
  6   5 | 2.41e+06          21         1.67e+06     2.41e+06
 11   1 | 2.41e+06          1.42e+06   1.62e+06     2.41e+06
 12   5 | 1.45e+06          23         1.30e+06     1.45e+06
 17   5 | 1.75e+06          8.83e+05   1.44e+06     1.75e+06
 28   3 | 6.72e+05          6.72e+05   1.12e+06     6.72e+05
```

**No single permutation convention works for all q-points** — `fwd,fwd`
works only for 3/28 non-trivial q-points; the rest fail with **all**
double-permute variants.  This RULES OUT the bug being a simple
"forward vs inverse" centroid permutation choice.

### Tier 4: pure-nosym self-consistency check

The "gold-standard" test: does V_nosym[q'=11] = V_nosym[parent=1]
permuted by **any** sym_perm row?  Answer: **no** for q=11
(C3-related to parent).  The only match found is the trivial identity.

```text
  V_nosym[11] vs V_nosym[1][sym_perm[s], sym_perm[s]] for s=0..5:
    s=0: 2.46e+06   s=1: 1.42e+06   s=2: 2.46e+06
    s=3: 2.40e+06   s=4: 1.62e+06   s=5: 2.46e+06
```

So **V_q itself does not obey a pure centroid-double-permute symmetry
relation under C3 in CrI3**.  This is independent of LORRAX's
implementation — it's a property of the V_q tensor as computed by both
paths from independent ζ-fits.

### Tier 5: identification of the per-element gauge phase

Compute the elementwise ratio
`r(μ', ν') = V_nosym[11, μ', ν'] / V_nosym[1, sym_perm[1, μ'], sym_perm[1, ν']]`:

```text
  |ratio| stats on off-diag: min=0.9999, max=1.0001, mean=1.0000   (unit modulus!)
  ratio[μ,ν] * ratio[ν,μ] = 1.0  ⇒ phase factorises as exp(i(θ(μ) - θ(ν)))
```

So the ratio is a **pure gauge phase that depends only on (μ, ν)
SEPARATELY**, factored as `exp(i(θ(μ) - θ(ν)))`.  Extracting θ from a
reference column ν₀=86 (large-|V| column):

```text
  Quantization (rounded to multiples of 2π/3):
    phase ω⁰ (=1):       152 centroids
    phase ω¹ (=e^{iπ/3}): 148 centroids
    phase ω² (=e^{-iπ/3}): 0 centroids
```

Inspecting which centroids carry the phase:

```text
  μ        r_μ_frac (x, y, z)        θ_obs (deg)
   0  (0.31, 0.64, 0.0083)            0°
 100  (0.36, 0.67, 0.0083)            0°
  50  (0.33, 0.69, 0.0083)            0°
 150  (0.69, 0.36, 0.9917)           60°
 250  (0.64, 0.33, 0.9917)           60°
 200  (0.67, 0.31, 0.9917)           60°
   1  (0.11, 0.73, 0.0500)            0°
 151  (0.89, 0.27, 0.9500)           60°
```

**The phase value depends on which z-half-layer the centroid sits in**:
z < 0.5 → θ = 0; z > 0.5 → θ = π/3 (= 60°).

For CrI3 (P-3 in 2D-slab orientation, with the −I op = sym=3 flipping
z), this is precisely the **half-cell offset under the sym map that
takes the upper Cr-I layer to the lower one**.  The C3 (sym=1) op acts
in-plane (z component +1), so naively it should NOT flip z — but
combining the q-side k_g0 umklapp (which IS non-zero, e.g.
`kg0_int = (0, -6, 0)` for q_irr=(0,1,0)→q_full=(1,5,0) under C3) and
the centroid r-axis permutation produces the residual phase.

The physical origin: the V_q-transformation rule the math agent
derived assumed **the kg0 G-pullback umklapp drops out at the V_q
bilinear level** (§3 of `zeta_unfold_derivation.md`).  Empirically,
**this cancellation only happens when both μ and ν live on the same
side of the sym-related crystallographic plane**; for cross-plane
pairs the kg0 phase survives as a per-(μ,ν) gauge.

### Tier 6: why the math agent's derivation missed this

Re-reading §3 of `zeta_unfold_derivation.md`:

> §3 V_q transform rule: substituting the ζ-transform into both legs of
> the bilinear, the two τ-phases cancel exactly (bilinear-V cancellation)
> AND the G-pullback umklapp cancels under sum-index relabeling +
> rotation invariance of v(|q+G|). Net result:
>
>   V_full[Sq, π_s(μ), π_s(ν)] = V_ibz[q, μ, ν]   (spatial S)

The "G-pullback umklapp cancels" step assumes that after substituting
`g_new = S^{-1} G_full - kg0` in the G-sum, the limits of the sum are
preserved.  This is **only valid when the ζ̃ representation at q_full
samples the full G-sphere identically to ζ̃ at q_irr permuted by
S^{-1}**.  In a non-symmorphic OR an inversion-containing space group,
the ζ "lives" on different per-q-sphere G-sets, and the umklapp shift
kg0 lands centroids at positions that ARE inside the cell but at
different per-centroid r_μ values.  The residual is a phase
exp(2π i kg0 · r_μ) that DOES NOT factor out of the bilinear.

The §3 conclusion is correct ONLY for groups where `kg0 = 0` for all
sym-related (q_irr → q_full) pairs.  This holds trivially for:
- The identity sym (q_full ≡ q_irr)
- Sym groups where every full-BZ q-orbit lies entirely inside the
  principal BZ (rare in non-cubic systems)

For CrI3 P-3 on a 6×6×1 grid, the C3 ops generate orbits that exit the
principal BZ for half the q-points (the ones with `kg0 ≠ 0`).  These
are exactly the q-points where `unfold_v_q` fails.

## Step 3 — Σ_X kernel trace

Not needed.  V_q^full is wrong before it reaches the Σ_X kernel.
The Σ_X kernel itself does no further sym-aware processing on V_q
(per Context Agent B's audit, `v_q_g_flat.py:457-478` and
`compute_vcoul.py:1030-1050` apply no τ-phase, no G-rotation post-V_q).

## File:line diagnosis

**Bug location**: `src/common/symmetry_maps.py:110-247` (`unfold_v_q`).

The implementation correctly applies the centroid-axis double permute:
```python
# Lines 224-247 (after the math agent's centroid_perm flip):
@partial(jax.jit, out_shardings=V_sh)
def _do_unfold(V_ibz):
    perm_q = inv_perm_j[sym_j]                          # (n_q_full, n_rmu)
    V_at_irr = V_ibz[idx_j]                              # (n_q_full, μ, μ)
    V_perm_mu = jnp.take_along_axis(
        V_at_irr, perm_q[:, :, None], axis=1, ...)
    V_full = jnp.take_along_axis(
        V_perm_mu, perm_q[:, None, :], axis=2, ...)
    # ↑ THE PER-ELEMENT PHASE FACTOR exp(i[θ_s(μ) - θ_s(ν)])
    #   IS MISSING HERE.  This is the bug.
    V_full = jnp.where(
        trs_mask_j[:, None, None], jnp.conj(V_full), V_full)
    return V_full
```

The missing phase: `exp(2π i kg0[s(q)] · r_μ - 2π i kg0[s(q)] · r_ν)`,
where `kg0[s] = sym_mats_k[s] @ q_irr - q_full` is the per-(q, sym)
umklapp shift recorded by `find_irreducible_bz_points` but currently
discarded.

**Math agent's `zeta_unfold_derivation.md` §3 conclusion is wrong** —
the umklapp G-pullback does NOT cancel at the V_q bilinear when
sym-related q-orbits exit the principal BZ.  The derivation needs to
be re-run with explicit retention of the `kg0`-induced phase.

## Proposed fix (P0)

**Option A (safe, slower)**: disable the IBZ→full V_q cascade for any
system whose sym group has non-zero `kg0` shifts under any non-identity
op.  I.e.: gate `use_ibz` in `v_q_g_flat.py:457` and
`compute_vcoul.py:880` on a runtime check
`np.all(kg0_table == 0)` after building the IBZ tables.  For systems
that fail this check, compute V_q at every full-BZ q-point directly
(no unfold).  Cost: ~6× more V_q kernel calls for CrI3 P-3 (8 IBZ →
36 full).  Acceptable for correctness validation.

**Option B (preferred, correct)**: extend `unfold_v_q` to apply the
missing per-element gauge phase:
```python
# Build per-q phase columns: phase_q[q_full, μ] = exp(2π i kg0[s(q)] · r_μ_frac)
# Apply: V_full[q, μ, ν] *= phase_q[q, μ] * conj(phase_q[q, ν])
```
This requires:
1. Exposing `kg0_table` from `find_irreducible_bz_points` (currently
   discarded after computing irr_idx, sym_idx).
2. Passing centroid fractional coordinates into `unfold_v_q`.
3. The per-element complex multiply, sharded along (None, 'x', 'y').

Option B should reduce the residual to the ISDF noise floor (1e-5
relative), matching the IBZ-q-points baseline.

## Empirical validation gate for the fix

CrI3 6×6 30Ry V_q sym-vs-nosym Σ_X must drop to < 1 meV after applying
the fix.  The current 4096 meV failure becomes the regression test.
Also: max|ΔV_q| (direct tensor comparison) must drop from 2.4e+06 to
~10 (matching the IBZ-q-points baseline of 1.4e+01 at q=0).

## Hard constraint compliance check

- Source modifications: temporary instrumentation **added then
  reverted**.  Working tree: clean on tracked files (`git status`
  shows untracked pre-existing files only: round6_discussion.md,
  sternheimer_guidelines.md, test_zq_from_psi_sm_bit_identity.py).
- Allocation: used JID 52976844 (1 node, ~2 min wall for each of three
  GW runs).  Total cost ~6 GPU-min.
- HEAD `80edbe8` confirmed pre and post.

## Artifacts

| File                                             | Content                          |
|--------------------------------------------------|----------------------------------|
| `v_q_dumps/Vqmunu_sym.h5`                        | full-BZ V_q from sym path        |
| `v_q_dumps/Vq_ibz_sym.h5`                        | IBZ V_q + sym tables             |
| `v_q_dumps/Vqmunu_nosym.h5`                      | full-BZ V_q ground truth         |
| `v_q_dumps/Vq_ibz_nosym.h5`                      | nosym IBZ V_q (== Vqmunu_nosym)  |
| `v_q_dumps/compare_v_q.py`                       | comparison driver                |
| `runs/.../run_sym/gw.out.dump_sym`               | sym dump-run log                 |
| `runs/.../run_sym/gw.out.postfix`                | sym Σ_X-only run log (Step 1)    |
| `runs/.../run_sym/sigma_freq_debug.dat`          | post-fix Σ_X data                |
| `runs/.../run_sym/sigma_freq_debug.dat.prefix_R_cart_only` | pre-perm-flip backup    |
| `runs/.../run_nosym/gw.out.dump_nosym`           | nosym dump-run log               |

## Next steps for the next agent

1. Implement Option A (gate IBZ-cascade) as a safety net before any
   non-trivial-symmetry production runs.  ~30-line patch in
   `v_q_g_flat.py` + `compute_vcoul.py`.

2. Pursue Option B (umklapp phase in `unfold_v_q`).  Requires:
   - Auditing `find_irreducible_bz_points` to return `kg0_table`
     alongside `irr_idx, sym_idx, irr_kgrid_int_out`.
   - Re-deriving §3 of `zeta_unfold_derivation.md` with explicit
     umklapp retention.  The math agent's claim "the G-pullback
     umklapp cancels under sum-index relabeling + rotation invariance
     of v(|q+G|)" is wrong; the rotation invariance only cancels the
     **|q+G|** factor inside v(...), but the centroid index μ is
     NOT integrated over — it's a fixed external label that carries
     the umklapp phase exp(-2πi kg0 · r_μ).

3. Re-evaluate the **MoS₂ pass at 0.090 meV**: that test bed exercises
   only s=0 (E) and s=1 (-E, TRS-aug) under no_t_rev=true, with no
   non-trivial spatial sym.  So the umklapp-phase bug is invisible
   there.  MoS₂ is NOT a regression test for this fix — Si and CrI3
   are.  Si's 160 eV failure is the R_cart bug (independent); after
   the umklapp fix lands, Si should also pass.

4. Consider that the V_q **q=0 head** (`G0_gathered` in
   `gw_init.py:1097`) is unaffected by this bug (q=0 is in the IBZ),
   so the existing head-correction code path is untouched by this fix.

## Discussion ping

`reports/trs_sym_audit_2026-05-14/discussion.md` updated separately.
