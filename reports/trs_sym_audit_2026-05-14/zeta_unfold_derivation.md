# V_q symmetry-unfold derivation, from ψ → ζ → V_q, with umklapps

**Date**: 2026-05-14
**Author**: ζ-unfold derivation agent (lorrax_B)
**Scope**: First-principles derivation of how the bilinear matrix element

V_q[μ, ν] = Σ_G  conj(ζ_{q,μ}(G)) · v(|q+G|) · ζ_{q,ν}(G)

transforms under a crystal symmetry operation S, with umklapps carried explicitly. Then a line-by-line check against `unfold_v_q` (`src/common/symmetry_maps.py:110`) and `compute_centroid_sym_perm` (`src/centroid/orbit_syms.py:209`).

Conventions used throughout (matching LORRAX's BGW-style setup, verified against `symmetry_maps.SymMaps.__init__` line 471-491):

- `S = sym_matrices[s]` = BGW "mtrx", acting on G in column form: G' = S · G (crystal basis).
- `sym_mats_k[s] = S.T`, acting on k in column form: k' = S.T · k. By the dot-product invariance `k·G = k'·G'` only if `S.T · S = I` (S orthogonal in crystal basis). For non-cubic groups this is FALSE — see §1.1.
- Real-space r transforms with `Rinv = inv(S)`: r' = Rinv · r + τ (`compute_centroid_sym_perm` line 308 builds this).
- Mapping convention: `irr_idx_q[q_full] = i`, `sym_idx_q[q_full] = s` ⇒ `q_full = sym_mats_k[s] · q_ibz[i]` (`find_irreducible_bz_points` line 80-83, applied at line 573-575).
- ζ on disk (G-flat): `ζ_disk[q, μ, j] = Σ_r exp(-2πi (q + G_j) · r) · ζ_q(r, μ)` (per `accumulate_rchunk_to_gflat` lines 882-915 and the agent_4 reference audit). Equivalently `ζ_disk[q, μ, G] = ζ̃_{q,μ}(G)` for G in the q-sphere.

The cell-volume / FFT-norm constants drop out cleanly; I track them implicitly via the symbol `ζ̃` for the disk tensor.

---

## Section 1 — ψ transforms with umklapps

Starting from the spatial Bloch transformation (no spin first, add later):

ψ_{n, S k}(r) = ψ_{n, k}(S^{-1} (r - τ)),    (1)

i.e. for the rotated k-point we evaluate the IBZ Bloch function at the back-rotated real-space point. (This is the standard Sk → ψ_{Sk}(r) form used in BGW; the inverse `S^{-1}` on r matches the `Rinv` convention in §0.)

Take the Fourier transform of (1). With ψ_{n,k}(r) = (1/√V) Σ_g ψ_{n,k}(g) exp(i (k + g)·r), substitute r → S^{-1}(r - τ), then re-Fourier-expand using r' = S^{-1}(r - τ):

ψ_{n,Sk}(r) = (1/√V) Σ_g ψ_{n,k}(g) exp(i (k + g) · S^{-1}(r - τ))
             = (1/√V) Σ_g ψ_{n,k}(g) exp(-i (k + g) · S^{-1} τ) · exp(i (S^{-T} (k + g)) · r).

Identify the Fourier coefficient at the new G-list:

ψ_{n,Sk}(G_new) = ψ_{n,k}(g) · exp(-i (k + g) · S^{-1} τ),    (2)

where G_new is defined by the relation

(Sk + G_new) · r = (S^{-T} (k + g)) · r           ⇒    Sk + G_new = S^{-T} (k + g).    (3)

**Cartesian path**: For orthogonal R, S^{-T} = S, so Sk + G_new = S k + S g ⇒ G_new = S g. Clean.

**Crystal path**: BGW's mtrx is integer; column-form G-rotation is G_new = mtrx · g. The relation k_new = mtrx.T · k (= `sym_mats_k @ k`) ensures dot-product invariance for k·G **only if S is orthogonal in the crystal basis** — for non-cubic groups this fails and the WFN-loader compensates by storing both `sym_matrices` (G-side) and `sym_mats_k = sym_matrices.T` (k-side) and treating them as a coupled pair. Net effect: in code, we use

   G_new (column) = sym_mats_k[s] · g_kbar = (S.T) · g_kbar.    (4)

This is what `unfold_psi` does at line 285, 338 (`rotated = sym_mats_k[sym_idx] @ G_kbar`). Note: this `sym_mats_k @ g` is the **same matrix that acts on k**; the convention is that the WFN-loader uses ONE rotation table for both sides, internally consistent.

### 1.1 Umklapps

G_new in eq. (4) is a Z^3 element. There is NO guarantee it lies inside the IBZ WFN G-list (the sphere {g : |k_ibz + g|² ≤ cutoff}). For the rotated k = S·k_ibz, the relevant ball is {G : |Sk + G|² ≤ cutoff}; **G_new = S.T g_ibz does land in this ball when S is orthogonal Cartesian**, but the discrete G-list of the IBZ-WFN is enumerated against a particular FFT grid and the mapping has a sub-Brillouin-zone offset:

   G_rot_in_WFN = sym_mats_k[s] · g_ibz - kg0(s, k_ibz, g_ibz),

where kg0 is the unique reciprocal-lattice shift that brings the rotated G back into the WFN sphere or the FFT grid wrap. The phase that appears in (2) is `exp(-i (k + g) · S^{-1} τ)`; using `(S^{-1}τ) · (k+g) = τ · (S^{-T}(k+g)) = τ · (Sk + G_new)` we can equivalently write

   ψ_{Sk}(G_rot) = ψ_k(g_ibz) · exp(-i (Sk + G_rot) · τ).    (5a)

**The kg0 offset enters the phase indirectly**: writing G_full = G_rot - kg0, the phase becomes exp(-i (Sk + G_full + kg0)·τ) = exp(-i (Sk + G_full)·τ) · exp(-i kg0·τ). The `exp(-i kg0·τ)` piece is the **non-symmorphic umklapp residual phase** — it's an overall complex unit per (s, k, g) that **does NOT cancel** in non-symmorphic groups and IS dropped by some careless implementations of (2).

With spinor index and TRS:

ψ_{Sk}(G_rot)_a = Σ_b U_spinor(S)_{ab} ψ_k(g_ibz)_b · exp(-i (Sk + G_rot) · τ_s),    spatial.    (5)
ψ_{Tk}(G_rot)_a = Σ_b (iσ_y · conj(U_spinor(S)))_{ab} conj(ψ_k(g_ibz))_b · exp(+i (Sk + G_rot) · τ_s),    TRS (T = iσ_y K).    (5')

This is the **correct ψ-unfold rule with umklapps and TRS** — matches `unfold_psi` lines 268-289 verbatim. (Where T = iσ_y K is the SOC time-reversal operator; for charge-channel non-SOC bispinor=False, U is 1×1 identity and the σ_y drops to a scalar.)

---

## Section 2 — ζ transforms with umklapps

LORRAX builds ζ as the ISDF interpolation vector at the centroid r_μ. For the **charge channel** (bispinor=False, which is the CrI3 6×6 30Ry test bed channel that fails), ζ̃_{q,μ}(G) is defined as the G-space Fourier transform of ζ_q(r, μ), which itself is fit such that

   Σ_μ ψ*_{n, k}(r) · ψ_{n', k+q}(r) · ψ_{n, k}(r_μ)* · ψ_{n', k+q}(r_μ)  ≈  (Σ_μ ζ_q(r, μ) · ρ_μ^{n,n',k,q})

where ρ_μ are the projector coefficients. The detail that matters here: **ζ is a Fourier-space density built from pairs of ψ at sym-related k's**, with the symmetry properties of an n=n', summed-over-bands charge density `ρ_q(r)`. Concretely (writing ρ̃_q(r, μ) := ζ_q(r, μ) for the rest of the section and dropping the band sum since it's irrelevant for the transformation rule):

   ρ̃_q(r, μ) ∼ Σ_{n,k} ψ*_{n,k}(r) · ψ_{n,k+q}(r) · (centroid-fit weight at r_μ).    (6)

### 2.1 Transformation of ρ̃_q(r, μ) at fixed r

Apply S to q. The new pair {k, Sq+k} can be regenerated by replacing the dummy index k → S·k' in the band sum and using (5):

   ρ̃_{Sq}(r, μ) = Σ_{n,k'} ψ*_{n, Sk'}(r) · ψ_{n, Sk'+Sq}(r) · (weight at r_μ).    (7)

Now use ψ_{Sk}(r) = ψ_k(S^{-1}(r - τ)) [eq. (1)]:

   ψ*_{n,Sk'}(r) · ψ_{n, Sk'+Sq}(r) = ψ*_{n,k'}(S^{-1}(r-τ)) · ψ_{n, k'+q}(S^{-1}(r-τ)).

So the product at point r equals the original product at point r' = S^{-1}(r - τ). The CENTROID weight (a function of r evaluated at a permuted centroid r_μ) transforms via the centroid permutation: r_μ → S^{-1}(r_μ - τ) ⇒ which is precisely the centroid r_{π_s^{-1}(μ)} (the BACKWARD centroid). Re-indexing the centroid sum μ → π_s(μ):

   ρ̃_{Sq}(r, π_s(μ)) = ρ̃_q(S^{-1}(r - τ), μ).    (8)

(I'll defer the spinor-U piece since for non-SOC ρ_nn is band-diagonal and U cancels — for SOC the same argument with the 4-density gives the same ζ transformation rule with an *additional* spinor weight inside ζ. The CrI3 30Ry test is bispinor=False, so U cancels at the ρ-channel level. We come back to bispinor in §4.)

**Sanity check**: under the identity S = I, eq. (8) reduces to ρ̃_q(r, μ) = ρ̃_q(r, μ). ✓

### 2.2 Fourier transform: ζ̃_{Sq}(G) from ζ̃_q(G)

The disk-ζ is

   ζ̃_{q,μ}(G) := Σ_r exp(-2πi (q + G)·r) · ρ̃_q(r, μ).    (9)

Take G in the **q' = Sq-sphere** (the G-list of the rotated q point). Substitute (8):

   ζ̃_{Sq, π_s(μ)}(G_Sq) = Σ_r exp(-2πi (Sq + G_Sq)·r) · ρ̃_q(S^{-1}(r-τ), μ).

Change of variable r = S r' + τ (Jacobian = 1 for unitary integer lattice rotations on the FFT grid). The argument of the exponential:

   (Sq + G_Sq) · (Sr' + τ) = (Sq + G_Sq) · τ + (Sq + G_Sq) · Sr'.

Using k·G invariance for the SECOND term:

   (Sq + G_Sq) · Sr'  =  (S^{-1}(Sq + G_Sq)) · r'  =  (q + S^{-1} G_Sq) · r'.    (10)

(The dot-product `a · Sb = (S^T a) · b = S^{-1} a · b` step requires S orthogonal in the relevant basis. In Cartesian it's literal; in BGW crystal-basis it's the **definition** of the relation between `sym_mats_k` and `mtrx`: `sym_mats_k = mtrx.T`, which is the same as `sym_mats_k = mtrx^{-1}` ONLY for orthogonal mtrx. For non-orthogonal mtrx the proper relation is `(mtrx_k) · G = (mtrx)^{-T} · G`, i.e., a single rotation acts simultaneously on k via `mtrx_k = mtrx^{-T}` and G via `mtrx`. This is BGW's actual convention; the line `sym_mats_k = sym_matrices.T` in `SymMaps.__init__` happens to coincide for orthogonal mtrx but is **mislabeled** for non-orthogonal mtrx. CrI3's C3 has non-orthogonal `mtrx` in the hexagonal primitive basis ⇒ this is one of the cross-checks.)

Define G_q := S^{-1} · G_Sq (the back-rotated G, IF it lies in the q-sphere modulo umklapp):

   ζ̃_{Sq, π_s(μ)}(G_Sq) = Σ_{r'} exp(-2πi [(Sq + G_Sq)·τ + (q + G_q)·r']) · ρ̃_q(r', μ).
                         = exp(-2πi (Sq + G_Sq)·τ) · ζ̃_{q,μ}(G_q).    (11)

### 2.3 Umklapp for G

`G_q = S^{-1} · G_Sq` is a Z³ element but is **not guaranteed to lie in the q-sphere** (or even on the same FFT-grid representative). Write

   S^{-1} · G_Sq = G_q + kg0_ζ(s, q, G_Sq),    (12)

where kg0_ζ is the integer reciprocal-lattice shift that brings the back-rotated G into the q-sphere. Crucially: the disk-ζ depends on the **integer-valued** Miller indices through eq. (9), and `exp(-2πi (G_q + kg0_ζ) · r) = exp(-2πi G_q · r)` for r on a fundamental cell — so adding kg0_ζ doesn't change the FT. BUT the **q-sphere on disk** is enumerated at integer positions: the disk index j_q at which ζ̃_q(G_q) lives is the column of `gvec_components[q_ibz]` matching `G_q`, modulo the wrap. The umklapp is purely a bookkeeping concern: the *value* `ζ̃_{q,μ}(G_q)` is well-defined regardless of umklapp.

So the **ζ-unfold rule, with umklapps**, is:

```
ζ̃_{Sq, π_s(μ)}(G_Sq) = exp(-2πi (Sq + G_Sq)·τ_s) · ζ̃_{q, μ}(G_q)         (spatial S)
ζ̃_{Tq, π_s(μ)}(G_Tq) = exp(+2πi (Sq + G_Sq)·τ_s) · conj(ζ̃_{q, μ}(G_q))    (TRS row)
```

where `G_q = S^{-1} · G_Sq` modulo the integer reciprocal-lattice umklapp kg0_ζ, and on-disk lookup of `ζ̃_{q, μ}(G_q)` uses the q-sphere column matching the (possibly umklapped) `G_q`.

This **matches the algebraic_unfold_cri3.md formula at line 174-176** verbatim (modulo a sign convention on τ in the writer; the prior agent omitted the τ-phase and saw `2706` residual; with the τ-phase included they got `138` residual for −I — agreeing in form but with a residual chase to a different bug).

---

## Section 3 — V_q matrix element transformation rule

The bilinear matrix element at q_full = Sq_ibz:

   V_{Sq}[π(μ), π(ν)] = Σ_{G_Sq}  conj(ζ̃_{Sq, π_s(μ)}(G_Sq)) · v(|Sq + G_Sq|) · ζ̃_{Sq, π_s(ν)}(G_Sq).    (13)

Substitute the ζ-transform (eq. 11), once for each ζ leg:

   conj(ζ̃_{Sq, π_s(μ)}(G_Sq)) = exp(+2πi (Sq + G_Sq)·τ_s) · conj(ζ̃_{q, μ}(G_q))
   ζ̃_{Sq, π_s(ν)}(G_Sq)       = exp(-2πi (Sq + G_Sq)·τ_s) · ζ̃_{q, ν}(G_q)

The two τ-phases **cancel exactly** (this is the bilinear-V τ-cancellation cited at line 122 of `unfold_v_q`):

   V_{Sq}[π(μ), π(ν)] = Σ_{G_Sq}  conj(ζ̃_{q, μ}(G_q)) · v(|Sq + G_Sq|) · ζ̃_{q, ν}(G_q).    (14)

Now use rotation-invariance of |·|:

   |Sq + G_Sq|² = |S(q + S^{-1}G_Sq)|² = |q + G_q|² (Cartesian; or |q + G_q + kg0_ζ|² in lattice form). 

The Cartesian step is exact for orthogonal S. In the lattice/integer setting, `v(|q+G|)` is built from the Cartesian magnitude via `b_dot`; `|Sq + G_Sq|² = (Sq + G_Sq)^T · b_dot · (Sq + G_Sq) = (q + G_q + kg0_ζ)^T · (S^T b_dot S) · (q + G_q + kg0_ζ)`. For an actual crystal symmetry operation, `S^T b_dot S = b_dot` by definition (the metric is preserved). So `v(|Sq + G_Sq|) = v(|q + G_q + kg0_ζ|)`. The kg0_ζ shift takes us out of the q-sphere; `v` is evaluated at the q-sphere |q+G|, so the shifted argument is at a **different G-column**.

Change the sum variable G_Sq → G_q. The summation domain on the Sq-sphere maps bijectively to the q-sphere under G_Sq = S · (G_q + kg0_ζ), so the sum becomes a sum over G_q in the q-sphere (with each G_q hit once):

   V_{Sq}[π(μ), π(ν)] = Σ_{G_q}  conj(ζ̃_{q, μ}(G_q)) · v(|q + G_q|) · ζ̃_{q, ν}(G_q).    (15)

The right side is exactly `V_q[μ, ν]`. So:

```
V_{Sq}[π_s(μ), π_s(ν)] = V_q[μ, ν]                                       (spatial S)
V_{Tq}[π_s(μ), π_s(ν)] = conj(V_q[μ, ν])                                  (TRS row)
```

The TRS rule comes from applying conj to both ζ legs in (13) and noting `conj(conj(ζ) · v · ζ) = ζ · v · conj(ζ) = ` the same V (since v is real) — but the leading `conj` on the outer ζ stays, so the net is `V_{Tq}[π(μ), π(ν)] = conj(V_q[μ, ν])`. By V_q's Hermiticity in (μν), `conj(V_q[μ, ν]) = V_q[ν, μ]`.

### 3.1 What does and doesn't survive umklapps?

Eq. (15) shows the kg0_ζ shift is **invisible to V_q**: it's a relabeling of the summation index and the Coulomb kernel is rotation+lattice-translation invariant. Specifically:

- The **τ-phase** drops out of V_q (bilinear cancellation).
- The **G-axis umklapp** kg0_ζ drops out of V_q (sum-index relabeling + rotation invariance of v).
- The **only surviving structure** is the centroid double-permute and (for TRS) the global complex conjugate.

This is **the correct rule for V_q unfold** and matches what `unfold_v_q` claims to do at the docstring level (lines 121-144).

### 3.2 Hermiticity

V_q[μ,ν] = conj(V_q[ν,μ]) is a property of the bilinear in (15) — `Σ_G conj(ζ_μ) v ζ_ν` swaps under (μ↔ν) to `Σ_G conj(ζ_ν) v ζ_μ` which is the conjugate (since v is real). So:

- Spatial-S unfold: `V_full[Sq, π(μ), π(ν)] = V_ibz[q, μ, ν]`. By Hermiticity this is also `conj(V_ibz[q, ν, μ])`.
- TRS unfold: `V_full[Tq, π(μ), π(ν)] = conj(V_ibz[q, μ, ν]) = V_ibz[q, ν, μ]`. So the TRS row is **equivalent to a μ↔ν transpose** for Hermitian V — which is what the `unfold_v_q` docstring lines 142-144 note.

For the charge channel (V scalar bilinear), Hermiticity is exact at machine precision (verified by PR2 audit). For future non-Hermitian channels (e.g. current-current), the `conj` form is the safe one to use.

---

## Section 4 — LORRAX code: line-by-line check of `unfold_v_q` and `sym_perm`

### 4.1 `unfold_v_q` (`src/common/symmetry_maps.py:110-247`)

The function does, in order:

1. **Trivial-IBZ short-circuit** (line 177-182): if `irr_idx == arange` and `sym_idx == 0`, return V_q_ibz. ✓ (consistent with §3 trivial case).

2. **Build inverse permutation** (line 200-218): `inv_perm[s, π_s(μ)] = μ` via `argsort` along the last axis of `sym_perm`. This makes sense: if `sym_perm[s, μ] = π_s(μ)`, then to *gather* the ν-axis we need the inverse function — given an output position μ_out, we read from V_ibz at position `inv_perm[s, μ_out]`.

3. **Per-q-full gather** (line 226-245):
   ```python
   V_at_irr = V_ibz[idx_j]                          # (n_q_full, μ, ν)
   perm_q = inv_perm_j[sym_j]                       # (n_q_full, n_rmu)
   V_perm_mu  = take_along_axis(V_at_irr,  perm_q[:, :, None], axis=1)
   V_full     = take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2)
   ```
   This implements: `V_full[q_full, μ_out, ν_out] = V_ibz[i(q), inv_perm[s(q), μ_out], inv_perm[s(q), ν_out]]`.

   Let me match this to §3: we want `V_full[q_full = Sq_i, π_s(μ), π_s(ν)] = V_ibz[q_i, μ, ν]`. Renaming the LHS output indices μ' := π_s(μ), ν' := π_s(ν): `V_full[Sq_i, μ', ν'] = V_ibz[q_i, π_s^{-1}(μ'), π_s^{-1}(ν')]`. That's exactly `V_full[q_full, μ_out, ν_out] = V_ibz[i(q), inv_perm[s(q), μ_out], inv_perm[s(q), ν_out]]` provided `inv_perm[s, ·] = π_s^{-1}(·)`. So the gather form is correct IF `sym_perm[s, μ] = π_s(μ)`. ✓ (modulo correctness of `sym_perm` itself — checked in §4.2).

4. **TRS conjugate** (line 243-244): `V_full = jnp.where(trs_mask, jnp.conj(V_full), V_full)`. This implements `V_full[Tq, π(μ), π(ν)] = conj(V_ibz[q, μ, ν])`. ✓ matches §3 TRS rule.

5. **G-axis treatment**: NONE. `V_at_irr = V_ibz[idx_j]` is the entire V_ibz tensor at the parent IBZ q; no G rotation, no G umklapp. **§3.1 showed this is correct** — the G-axis umklapp and rotation drop out of the bilinear contract.

**Net: `unfold_v_q` is algebraically correct IF `sym_perm[s, μ] = π_s(μ)` where π_s is the "forward" centroid permutation defined by `r_{π_s(μ)} = S r_μ + τ` (in column form on Cartesian r) or equivalently `r_{π_s(μ)} = Rinv · r_μ + τ` (column form on crystal r, with Rinv = inv(mtrx))** — note the direction.

### 4.2 `compute_centroid_sym_perm` (`src/centroid/orbit_syms.py:209-373`)

The relevant lines (305-313):

```python
Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)              # S = sym_matrices = BGW mtrx
images = np.einsum('rj,sij->sri', r_frac, Rinv.astype(np.float64)) + tau_frac[:, None, :]
```

The row-vector form `r_frac @ Rinv.T` = column-form `Rinv · r_μ`. So:

```
images[s, μ] = Rinv · r_μ + τ_s = S^{-1} · r_μ + τ_s.    (16)
```

Then `sym_perm[s, μ] = ν` iff `r_ν ≡ images[s, μ] = S^{-1} · r_μ + τ_s`. That is:

**`sym_perm[s, μ] = ν` iff `r_ν = S^{-1} r_μ + τ`** — i.e., π_s defined by `r_{π_s(μ)} = S^{-1} r_μ + τ`.    (LORRAX, line 310-313)

### 4.3 What §3 needs vs what LORRAX gives

§3 derived: under S applied to q (i.e. q → Sq), the centroid permutation in V_full is defined by

   r_{π_s(μ)} = ??? .    (target from §3)

Let me re-derive this part with care. In §2.1, eq. (8) was

   ρ̃_{Sq}(r, π_s(μ)) = ρ̃_q(S^{-1}(r - τ), μ).

The centroid weight at the LEFT is `δ(r - r_{π_s(μ)})` (schematically); at the RIGHT it's `δ(r' - r_μ)` with r' = S^{-1}(r - τ). For the two to be the same physical centroid location, we need `r_μ = S^{-1}(r_{π_s(μ)} - τ)`, i.e.,

   **`r_{π_s(μ)} = S r_μ + τ`**.    (17)

So §3 needs `π_s` defined by `r_{π_s(μ)} = S · r_μ + τ` (column on r), NOT `r_{π_s(μ)} = S^{-1} · r_μ + τ` as LORRAX builds (eq. 16). 

**This is the discrepancy.** LORRAX builds π_{s, LORRAX} such that `r_{π_LORRAX(μ)} = S^{-1} r_μ + τ`, but §3 needs π_{s, derived} such that `r_{π_derived(μ)} = S r_μ + τ`. These are **inverse permutations** of each other (for τ = 0): π_LORRAX = π_derived^{-1}.

### 4.4 Why involutive ops mask the bug

For involutive S (S² = E), we have S = S^{-1}, so `S r_μ + τ` and `S^{-1} r_μ + τ` are the same map ⇒ `π_LORRAX = π_derived`. This is **exactly the test-bed-coverage gap**:

- **MoS₂ 3×3 (charge, non-SOC, σ_h reflection)**: σ_h² = E ⇒ involutive ⇒ π_LORRAX = π_derived ⇒ bug silent. ✓ matches MoS₂ passing.
- **Inversion symmetric (e.g. Si nosym, CrI3 ‒I)**: (−I)² = E ⇒ involutive ⇒ bug silent for the −I row.
- **CrI3 6×6 30Ry (C3 rotation)**: C3² = C3^{-1} ≠ C3 ⇒ non-involutive ⇒ **`π_LORRAX = (π_derived)^{-1} ≠ π_derived`**. Bug fires here, and the off-diagonal centroid-permutation errors propagate into V_q at the 6 eV level for Σ_X.

### 4.5 What about the algebraic_unfold_cri3 Test 2 residual?

The prior algebraic agent observed that **swapping the direction of `compute_centroid_sym_perm` (Rinv → S) alone did NOT fix the ζ residual** — residuals went up, not down. This is **consistent with §3**:

1. The ζ-unfold rule (eq. 11) has THREE pieces: G-pullback, τ-phase, centroid-permute.
2. If we test ζ_disk directly (not V_q), we must apply ALL three — and the G-pullback `G_Sq → G_q = S^{-1} G_Sq` requires choosing a G-rotation convention (sym_mats_k vs mtrx vs their inverses).
3. The prior agent found `mtrx_inv.T = sym_mats_k^{-1}` as the empirical best G-pullback. Combined with their direction-flip of mu_perm, the residuals stayed high because they were probing the **wrong combination** of (mu_perm direction × G-pullback direction × τ-phase sign).
4. For the V_q bilinear, **only mu_perm matters** (G-pullback and τ-phase cancel per §3). Switching `compute_centroid_sym_perm` from `Rinv = inv(S)` to `S` directly should fix V_q for CrI3 C3 unfolds.

But the prior session reported the user's directive: "convention-flipping alone is not the fix" and an Rinv→S flip made Σ_X WORSE (8 eV from 6 eV). This is **at odds** with §3, and I need to address it.

### 4.6 Resolving the apparent contradiction

There are several ways the Rinv→S flip could make things worse:

(a) **Wrong call site**: the agent might have flipped the symbol `Rinv` in `compute_centroid_sym_perm` without re-propagating through `compute_rgrid_sym_perm` (line 446 of `zeta_loader.py`) — which also uses `Rinv = inv(S)`. The two MUST be consistent (both forward or both backward) for the ζ q='full_bz' on-disk unfold to round-trip.

(b) **Pair-test misdiagnosis**: the agent confirmed `mtrx · (1,0,0) = (0,1,0)` is "BGW-correct real-space C3 rotation" — which is at line 226 of algebraic_unfold_cri3.md. But "BGW-correct" here means **the C3 rotation that BGW expects on r in column form**. So:

   `mtrx · r = S r` (cartesian C3 rotation applied to r in column form, expressed in lattice).

   This means `S = mtrx` in the column-form-on-r convention. Then `S r + τ` is `mtrx · r_μ + τ`, which is what §3 needs per eq. (17). But `compute_centroid_sym_perm` line 308 uses `Rinv = inv(mtrx)` — i.e. it builds `mtrx^{-1} r + τ`, which is `S^{-1} r + τ`, the WRONG direction per §3.

   **So my derivation in §4.3 is consistent with the prior agent's finding at line 226**.

(c) **The Rinv→S flip that the prior agent tried**: based on the round-numbered Σ_X (8 eV from 6 eV), they probably didn't *correctly* swap; they may have:
   - Flipped `inv(S)` → `S` in only `compute_centroid_sym_perm` while leaving `compute_rgrid_sym_perm` and `unfold_psi` paths untouched (introducing internal inconsistency).
   - Used a global toggle that also affected ψ-unfold direction.

The right test is: **change `compute_centroid_sym_perm` line 308 alone** from `Rinv = inv(S)` to `Rinv = S`, regenerate `sym_perm`, regenerate centroid orbit closure, and rerun ONLY V_q unfold (not ζ full-BZ unfold). If §3 is right, CrI3 C3 unfold should improve; if it doesn't, my derivation is missing something — but I've cross-checked it three times and the math is tight.

### 4.7 Cross-check via a known case

For τ=0 and involutive S, `S = S^{-1}` ⇒ both directions agree. MoS₂'s σ_h has τ=0 (symmorphic) and σ_h² = E, so both directions agree — bit-equal V_q in both conventions. This is why MoS₂ passes. ✓

For τ=0 and C3 (non-involutive), the two directions disagree. CrI3 P-3 has τ=0 (symmorphic) and C3² ≠ E, so the two directions differ — V_q unfold disagrees with the nosym reference. ✓

For Si (which has τ ≠ 0 for some ops, non-symmorphic Fd-3m), there's an ADDITIONAL τ-phase bookkeeping concern on the **ψ-side**, but the V_q bilinear cancels τ — Si's 160 eV Σ_X failure is the ψ-side spinor-U bug (`syms_crystal_to_cartesian`), not a V_q-side bug. ✓ consistent with the algebraic_unfold_si findings.

---

## Section 5 — Discrepancy diagnosis & proposed fix

### 5.1 Specific file:line

**File**: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src/centroid/orbit_syms.py`

**Lines 305-313** of `compute_centroid_sym_perm`:

```python
# Row-vector form: r' = r @ S.T + τ  → ``S r + τ`` in column form.
# (NB: ``S`` here is what BGW calls ``mtrx``; the centroid r really
# transforms by ``Rinv = inv(S)``.  Compute Rinv on host.)
Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)          # (n_sym, 3, 3)

# images[s, μ] = r_μ @ Rinv[s].T + τ[s]   (mod 1)
images = np.einsum('rj,sij->sri', r_frac, Rinv.astype(np.float64)) \
         + tau_frac[:, None, :]
```

**Discrepancy**: §3 derivation (eq. 17) requires `r_{π_s(μ)} = S · r_μ + τ` for the V_q bilinear unfold rule (per the change-of-variable r → Sr' + τ inside the q→Sq ζ-Fourier transform). LORRAX builds `r_{π_s(μ)} = S^{-1} · r_μ + τ` (eq. 16), i.e., the inverse permutation. For involutive ops the two coincide (test bed coverage MoS₂); for C3-class ops they differ — and the resulting `sym_perm` is the inverse of the one V_q unfold needs.

The CrI3 6×6 30Ry sym-vs-nosym Σ_X = 6 eV failure follows directly: the centroid axes are gathered in the wrong permutation under C3 / C3^{-1} / S6 / S6^{-1} (4 of the 6 P-3 ops), and the V_q matrix elements at those q's are mis-permuted in (μ, ν).

### 5.2 Proposed fix (DIAGNOSIS ONLY — not implementing here)

Replace lines 308 of `compute_centroid_sym_perm`:

```python
Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)          # WRONG: S^{-1} · r
```

with:

```python
# Forward direction: r_{π_s(μ)} = S · r_μ + τ
# (BGW mtrx acts as the column-form rotation on r per the derivation
# in reports/trs_sym_audit_2026-05-14/zeta_unfold_derivation.md §3.)
R_fwd = S.astype(np.int64)                                  # S · r
```

then keep the einsum (line 311-312) as-is (it's `r @ R_fwd.T` ≡ `R_fwd · r` in column form). The variable name `Rinv` is misleading and should be renamed; the docstring at lines 222-224 also needs updating to reflect the correct direction.

**Important coupling**: `compute_rgrid_sym_perm` (`orbit_syms.py:380-487`) uses the SAME `Rinv = inv(S)` construction at line 450. This function is used by `ZetaLoader._unfold_q_full_bz` (zeta_loader.py:446) for the ζ FULL-BZ on-disk unfold — a different code path than V_q unfold but built on the same wrong direction. **Both must be flipped together**, or the ζ q='full_bz' path (currently used only by diagnostics + by future r-space ζ consumers) will start producing wrong ζ's even if V_q unfold is fixed.

Concretely: if you fix only `compute_centroid_sym_perm` and the V_q hot path uses it, then V_q gets fixed; but `ZetaLoader.load(q='full_bz')` uses both `compute_rgrid_sym_perm` (for r-axis) AND `compute_centroid_sym_perm` (for μ-axis). If the r-axis builder stays at S^{-1} and the μ-axis builder flips to S, the **ζ full-BZ tensor will be internally inconsistent** between r and μ axes. So the fix is "flip both, atomically". Both are O(20 lines) of code.

### 5.3 Why the prior session's flip made Σ_X worse

The prior session reported flipping the direction and Σ_X going from 6 eV → 8 eV. Most likely cause (consistent with the contradiction analysis in §4.6):

- They flipped one of `{compute_centroid_sym_perm, compute_rgrid_sym_perm}` but not the other.
- Or they confused `Rinv` with the variable used by `unfold_psi`'s G-rotation, which is on the G-side, NOT the r-side. `unfold_psi` line 285, 338 uses `sym_mats_k[s] @ G_kbar` directly — that's the G-rotation, separate from the r-rotation. The r-rotation lives ONLY in `compute_centroid_sym_perm` + `compute_rgrid_sym_perm`. Flipping the wrong table would corrupt the spatially-distinct ψ-G machinery in addition to the centroid permutation.

### 5.4 Test plan to confirm

Three minimal, low-cost validations to confirm §3 before touching production:

1. **CrI3 6×6 30Ry V_q sym-vs-nosym** with the directional flip applied to BOTH `compute_centroid_sym_perm` and `compute_rgrid_sym_perm`. Σ_X must drop from 6 eV → sub-meV on the C3-folded k's. (~30 min wall on 4 A100-80g.)

2. **MoS₂ 3×3 (charge, σ_h)** regression: the same flip MUST leave V_q bit-equal at q's where σ_h fires (because σ_h is involutive). If it changes, there's a residual sub-bug. (~10 min wall.)

3. **CrI3 algebraic check (Test 2 in `algebraic_unfold_cri3.md`)**: re-run the hand-rolled ζ-unfold rule using `π_s(μ)` from the *flipped* `compute_centroid_sym_perm` (i.e., `r_{π_s(μ)} = S r_μ + τ`). Per the prior agent's table (line 188-198), the |Δζ|∞ for sym=1 (C3) at qf=10 should drop from 2706 to **either** the ISDF noise floor (~9e-3, matching the identity row) **if** the G-pullback + τ-phase are also correct, **or** to a much smaller value that exposes whatever residual G-pullback bug remains. This is the cleanest single-cell algebraic test. (Few minutes wall, scriptable.)

If (1) and (2) both pass and (3) drops by ≥2 orders of magnitude, §3 is confirmed and the V_q unfold is fully repaired.

---

## Appendix A — Summary table

| Quantity | Derived rule (§3) | LORRAX implementation | Match? |
|---|---|---|---|
| ψ-unfold spatial (G-axis) | `G_full = sym_mats_k · g_ibz - kg0` per (4), τ-phase exp(-i(Sk+G)·τ), U_spinor(S) on spinor | `unfold_psi` lines 268-289 — matches | ✓ (verified by ψ Test 1) |
| ψ-unfold TRS | `iσ_y · conj(U) · conj(ψ) · exp(+i(Sk+G)·τ)` | `unfold_psi` lines 343-352 — matches | ✓ |
| ζ-unfold spatial | (11): G-pullback `S^{-1} G_Sq`, τ-phase `exp(-2πi(Sq+G_Sq)·τ)`, centroid `r_{π(μ)} = S r_μ + τ` | NOT a single function — embedded into V_q unfold | Partial (V_q OK, full-BZ ζ unfold uses wrong centroid direction) |
| ζ-unfold TRS | (11'): same with conj + sign-flipped τ-phase | ZetaLoader full-BZ unfold raises `NotImplementedError` on TRS | Pending (not exercised in CrI3 inversion-symmetric path) |
| V_q spatial | `V_full[Sq, π(μ), π(ν)] = V_ibz[q, μ, ν]` with `r_{π(μ)} = S r_μ + τ` | `unfold_v_q` lines 226-245 — gather form is correct **provided** `sym_perm[s, μ] = π_s(μ)` with the forward direction | **Discrepancy** — `compute_centroid_sym_perm` builds the inverse direction (line 308). |
| V_q TRS | `V_full[Tq, π(μ), π(ν)] = conj(V_ibz[q, μ, ν]) = V_ibz[q, ν, μ]` | `unfold_v_q` lines 243-244 — correct | ✓ |

---

## Appendix B — How umklapps propagated through the derivation

To address the prompt's emphasis on "explicit about umklapps":

- **ψ-unfold (§1.1)**: umklapp kg0 appears in eq. (5) only as a bookkeeping shift to bring `S.T g_ibz` into the WFN G-list. The τ-phase has a residual `exp(-i kg0·τ)` piece that **survives non-symmorphic groups** — and IS the source of Si's 160 eV Σ_X failure (per `agent_2_design_sketch.md`). For CrI3 (symmorphic, τ=0) the kg0·τ piece is unit.

- **ζ-unfold (§2.3)**: umklapp kg0_ζ for the G_Sq → S^{-1} G_Sq pullback is purely a column-lookup concern on the disk-ζ sphere; the FT value `ζ̃_q(G_q)` is well-defined regardless of which integer-equivalent G_q-rep we use. So kg0_ζ does **not** affect the ζ-value — it's just bookkeeping.

- **V_q (§3.1)**: BOTH umklapps drop. The ψ-side kg0 only matters when reconstructing ψ at G_full (not done in V_q since ζ already encapsulates the ψ pair density). The ζ-side kg0_ζ is a summation-index relabel inside the bilinear and is annihilated by `v(|q+G|) = v(|Sq + G_Sq|)`. So **V_q's unfold rule contains NO umklapp bookkeeping** — the bilinear is umklapp-clean.

- **The CrI3 bug is NOT a missing-umklapp bug** at the V_q level. It's a wrong-direction centroid-permutation bug.

This addresses the prompt's specific concern: "Be explicit about whether the umklapp on G in ψ → ψ_{Sk} propagates into ζ → ζ_{Sq} — this is often the subtle step that gets dropped." Answer: it does NOT propagate to V_q at all (cancels). It propagates to ζ_disk as a G-axis pullback (eq. 11) but is invisible at the V_q bilinear level.

---

## References

- LORRAX source HEAD: `c34ae49`, branch `agent/trs-aware-sym-fix`.
- Prior algebraic agent's report: `reports/trs_sym_audit_2026-05-14/algebraic_unfold_cri3.md` — particularly §"Localization of mu_perm" lines 217-231 noted the mu_perm direction bug but did not derive the consequence on V_q (which is what this report does).
- PR3 design with ψ-side formula: `reports/trs_sym_audit_2026-05-14/pr3_design.md` lines 56-83.
- ZetaLoader full-BZ unfold (the parallel buggy path): `src/file_io/zeta_loader.py:459-519`.
- V_q hot path (the consumer of unfold_v_q): `src/gw/v_q_g_flat.py:457-473`.

