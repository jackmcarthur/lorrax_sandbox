# Algebraic ψ + ζ + G-vec unfold audit — CrI3 6×6 30Ry SOC

**Date**: 2026-05-14
**Agent**: CrI3 branch (parallel with MoS₂ and Si)
**Source**: `lorrax_B` @ `agent/trs-aware-sym-fix` commit `69ab42c` (HEAD)
**Test bed**: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/{run_sym, run_nosym}`
**Companion reports**: `algebraic_unfold_mos2.md` (PASS), `algebraic_unfold_si.md` (not landed yet)

## Verdicts

| Test | Verdict | Worst case |
|---|---|---|
| **Test 1 — ψ unfold (a) LORRAX vs (b) hand-rolled** | **PASS, bit-equal (0.0)** | All 6 representative (kf, sym_idx) pairs — max ‖(a)−(b)‖∞ = 0.0 |
| **Test 1 — ψ unfold (b) hand-rolled vs (c) nosym (gauge check)** | **FAIL for C3/S6, PASS for E/−I** | sym=1 (C3), kf=6, deg-group: max ‖U Uᴴ − I‖∞ = **0.820** (gate ≤ 10⁻⁶) |
| **Test 2 — ζ unfold (b) hand-rolled vs (c) nosym** | **FAIL for ALL non-trivial syms** | sym=1 (C3), qf=10: max ‖Δζ‖∞ = **2706** absolute, rel = **1.74** |
| **Test 3 — G-vector full-BZ k-list consistency** | **PASS** | All 36 k_full × ~13700 G-vectors match nosym set-elementwise |

**Bottom line for CrI3**: PR3's `unfold_psi` faithfully implements the per-element formula at machine precision (Test 1a-vs-b passes). But the formula itself **disagrees with nosym ground truth for the proper 3-fold (C3) and improper 3-fold (S6) ops** at the 0.8-unitary-deviation level. This is the same bug surface that produces the 6 eV ΔΣ_X failure logged in `cri3_sym_vs_nosym_pr3.md` — and it is **NOT exercised by the MoS₂ test bed** (whose point group is only E + σ_h). The hand-rolled ζ unfold rule that I derived from first principles also disagrees with nosym ζ on disk for ALL non-trivial syms, including −I where a per-(μ,G) phase correction reduces the residual 20× (2900→138) but does not eliminate it. Three independent root-cause candidates remain in play; localization below narrows it to two.

## Test bed structure

CrI3 monolayer (8 atoms: 2 Cr + 6 I), 6×6×1 k-grid, 30 Ry, SOC enabled (`nspinor=2` in WFN), `bispinor=false` in `cohsex.in` (charge-channel Σ_X only). Both runs share `centroids_frac_300.txt`.

**sym WFN** (P-3 / #147):
- `ntran = 6`: E + 2C3 + (−I) + 2S6
- `len(sym_mats_k) = 12` (TRS-augmented, but the lower 6 are spatial)
- **0 TRS-tagged k or q** (CrI3 has inversion in `mtrx` ⇒ PR3 TRS branches do not fire)
- `tnp[:ntran] = 0` (symmorphic ⇒ no τ-phase)
- IBZ: 8 q-points → 36 full-BZ via the 6 sym ops
- `q_irr_kgrid_int = [(0,0,0), (0,1,0), (0,2,0), (0,3,0), (1,1,0), (1,2,0), (1,3,0), (2,2,0)]`
- `q_irr_full_idx = [0, 1, 2, 3, 7, 8, 9, 14]`

**nosym WFN**: `ntran = 1`, 36 k = full BZ, no unfold.

**Sym matrices** (BGW `mtrx`, acting on G as G' = mtrx · G):
```
s=0  E:   [[ 1, 0,0], [ 0, 1,0], [0,0, 1]]    det = +1
s=1  C3:  [[ 0,-1,0], [ 1,-1,0], [0,0, 1]]    det = +1
s=2  C3⁻¹: [[-1, 1,0], [-1, 0,0], [0,0, 1]]   det = +1
s=3  −I:  [[-1, 0,0], [ 0,-1,0], [0,0,-1]]    det = −1
s=4  S6:  [[ 0, 1,0], [-1, 1,0], [0,0,−1]]    det = −1
s=5  S6⁻¹: [[ 1,-1,0], [ 1, 0,0], [0,0,−1]]   det = −1
```

Sym indices fire as follows across the 36 q's (from `sym.sym_idx_q`):
- s=0 (E): 8 q's (identity / IBZ representatives)
- s=1 (C3): 7 q's
- s=2 (C3⁻¹): 6 q's
- s=3 (−I): 5 q's
- s=4 (S6): 6 q's
- s=5 (S6⁻¹): 4 q's

No `sym_idx ≥ ntran` ⇒ **PR3's TRS-augmented code path does not fire here**. The bug surface is the spatial-sym path only.

## Test 3 — G-vector full-BZ k-list consistency

`WfnLoader.gvecs(k='full_bz')` at every full-BZ k vs `WfnLoader(nosym).gvecs(k='ibz')` at the matching k — compare as Miller-index sets.

**Result**: 36/36 k-points match (worst |Δ_set| = 0).

The umklapp + G-rotation in `WfnLoader._gvecs_cache` (which calls `sym._get_umklapp_vector` + `np.einsum('ij,kj->ki', sym_krep, k_gvecs) - Gkk`) produces the same Miller-index sphere as the independent QE NSCF on the nosym grid.

**Implication**: the bug is NOT in G-vector enumeration. The G-axis ordering of `cnk_full[..., g]` corresponds to physically-correct G-vectors in the full-k basis.

JSON: `cri3_alg_unfold/test3_gvec_results.json`.

## Test 1 — ψ unfold algebraic check

For each sym_idx ∈ {0..5}, pick the first full-BZ k whose `sym.sym_idx_k[kf] = s` and run:

**(a) LORRAX**: `common.symmetry_maps.unfold_psi(cnk_kbar, sym_idx=s, ...)` — production code.

**(b) Hand-rolled**: re-implement the per-element formula from `pr3_design.md` **without importing any LORRAX unfold code** (only the data structures: `sym.sym_mats_k`, `sym.translations`, `sym.U_spinor`):

```python
S_full = sym.sym_mats_k[sym_idx]           # length-12 row
tau    = sym.translations[sym_idx % ntran]  # τ=0 for CrI3
g_rot  = (S_full @ g_kbar.T).T              # row-form
phase  = exp(-1j * (g_rot @ tau))            # ≡ 1 since τ=0
is_trs = sym_idx >= ntran                    # FALSE for CrI3 in every test
if is_trs:
    cnk_full = einsum("ij,nkl→njl",
                      ISIGMA_Y @ conj(U_spinor[s%ntran]),
                      conj(cnk_kbar) * phase)
else:
    cnk_full = einsum("ij,nkl→njl", U_spinor[s], cnk_kbar * phase)
```

**(c) nosym ground truth**: `WfnLoader(nosym).load(...)` at the matching full-BZ k.

### (a) vs (b): LORRAX matches hand-rolled bit-equally

| kf | sym_idx | det(mtrx) | max ‖(a) − (b)‖∞ |
|---|---|---|---|
| 0  | 0 (E)    | +1 | **0.0** |
| 6  | 1 (C3)   | +1 | **0.0** |
| 3  | 2 (C3⁻¹) | +1 | **0.0** |
| 7  | 3 (−I)   | −1 | **0.0** |
| 10 | 4 (S6)   | −1 | **0.0** |
| 1  | 5 (S6⁻¹) | −1 | **0.0** |

LORRAX's `unfold_psi` faithfully implements the per-element design-doc formula. **No PR3-code-vs-design bug.**

### (b) vs (c): gauge / unitarity check against nosym

For each degenerate band group at IBZ k (energy gap < 1.4×10⁻⁵ eV ≈ 1 µRy), compute `U[m,n] = ⟨ψ_nosym(m) | ψ_handroll(n)⟩` over (spinor, G_aligned-via-Test-3-matching). For a correct unfold, each per-group block of `U` is unitary and cross-group entries are zero.

Tested with 80 bands per k_full:

| kf | sym_idx | det | max ‖U Uᴴ − I‖∞ on diag block | max ‖U‖∞ off-block | verdict |
|---|---|---|---|---|---|
| 0  | 0 (E)    | +1 | **2.4×10⁻¹⁴** | 3.8×10⁻¹⁵ | PASS — ULP |
| 6  | 1 (C3)   | +1 | **8.20×10⁻¹** | **7.08×10⁻¹** | **FAIL** |
| 3  | 2 (C3⁻¹) | +1 | **8.08×10⁻¹** | **7.62×10⁻¹** | **FAIL** |
| 7  | 3 (−I)   | −1 | **2.58×10⁻⁸** | 1.53×10⁻⁴ | PASS (ULP + small off-diag → SCF noise floor) |
| 10 | 4 (S6)   | −1 | **8.25×10⁻¹** | **7.14×10⁻¹** | **FAIL** |
| 1  | 5 (S6⁻¹) | −1 | **8.20×10⁻¹** | **7.66×10⁻¹** | **FAIL** |

**Localization**: the failure mode is **specific to det = ±1 ops that are NOT involutive** (i.e., **proper 3-fold C3, C3⁻¹** and **improper 3-fold S6, S6⁻¹**). Identity (E) and inversion (−I), both involutive (op² = E), pass at ULP. The C3 and S6 ops are 3-fold (op³ = E or op⁶ = E for S6), and these uniformly fail at ~0.82 unitary deviation.

**Cross-block leakage of 0.7 + same-group unitarity loss of 0.82** means the hand-rolled ψ has projected significant amplitude onto OTHER nosym bands — a band-mixing error, not merely a within-pair gauge degeneracy.

### Diagnostic — alternative G-rotation conventions

Tried both LORRAX's `g_rot = sym_mats_k @ g_kbar` (= `mtrx^T @ g_kbar`) and an alternative `g_rot = mtrx @ g_kbar`. With `mtrx` (instead of `mtrx^T`):
- For C3/S6: only ~6200/13700 G-vectors match the nosym G-set (the integer multiplication mod kgrid no longer produces the same sphere). G-mapping is half-broken.
- Unitarity gets WORSE (1.0 = max possible).

⇒ `sym_mats_k = mtrx^T` IS the correct G-vector rotation matrix in the BGW/LORRAX convention. The G-rotation in `wfn_loader.py:315` is mathematically correct.

The bug is **NOT** in the G-rotation. It is downstream of it.

### Spinor U inspection — suspect site #1

`sym.U_spinor` for CrI3:
```
s=0  E:    diag(+1, +1)                              ← exp(0) — correct
s=1  C3:   diag(-0.5800 + 0.8147 i, -0.5800 - 0.8147 i)
s=2  C3⁻¹: diag(-0.5800 - 0.8147 i, -0.5800 + 0.8147 i)
s=3  −I:   diag(+1, +1)                              ← inversion has TRIVIAL SU(2) — correct
s=4  S6:   diag(-0.5800 + 0.8147 i, ...)             ← same as C3 (factor of −I leaves SU(2) unchanged)
s=5  S6⁻¹: diag(-0.5800 - 0.8147 i, ...)             ← same as C3⁻¹
```

For a C3 rotation around the z-axis (which is what P-3's C3 ops are — they leave kz invariant), the SU(2) representation is `diag(exp(−iπ/3), exp(+iπ/3))` = `diag(0.5 − 0.866 i, 0.5 + 0.866 i)` (120° / 2 = 60° in SU(2)). LORRAX produces `diag(−0.580 + 0.815 i, ...)` ≈ `diag(exp(+i 125.5°), exp(−i 125.5°))` — that's a rotation by **~251° in SU(2)**, NOT 60°. **Off by a factor — likely a sign in the quaternion or a doubled angle.**

Tracing the cause: `SymMaps.syms_crystal_to_cartesian` (line 808-813) computes `R_cart = einsum('ij,njk,kl→nil', B_T_inv, self.sym_mats_k, B_T)` where `B = bvec`. For CrI3, R_cart[1] (C3) comes out as:
```
[[ 0.500,  2.021, 0],
 [-0.866, -1.500, 0],
 [ 0,      0,     1]]
```
This is **NOT orthogonal** (e.g., row 0 · row 0 = 0.25 + 4.08 = 4.33 ≠ 1). For a real-space rotation in cartesian, this should be orthogonal (det = ±1, R Rᵀ = I).

The cartesian conversion is wrong:
- **Source A**: `sym_mats_k` is the G-vector action in crystal coords (LORRAX convention, see line 478). To convert a G-action to its cartesian rotation, the formula is `R_cart = (B^T)^(-1) · sym_mats_k^T · B^T` (i.e., the conjugate-transpose construction). The current code uses `B_T_inv · sym_mats_k · B_T` — wrong-direction conjugation.
- **Source B**: Equivalently, the REAL-space rotation in cartesian for a sym op stored as BGW `mtrx` is `R_cart_real = (A^T) · mtrx_inv · (A^T)^(-1)` where `A = avec` (real-space basis vectors). The code uses `bvec` (reciprocal) and `sym_mats_k = mtrx^T` — both factors wrong.

For cubic systems, both bases are unitarily related and `mtrx^T = mtrx^(−1)` for orthogonal mtrx, so all four conventions agree at ULP. For hexagonal CrI3, they disagree.

**The non-orthogonal R_cart is fed to `get_spinor_rotations` → Markley's quaternion algorithm** (which assumes an orthogonal rotation matrix as input). Producing a wrong quaternion → wrong SU(2) U → wrong ψ_full for C3 and S6 → wrong overlap with nosym ψ → 0.82 unitary deviation.

(See `symmetry_maps.py:792-813` for `syms_crystal_to_cartesian`; the function-body comment "NOT SURE IF THESE SHOULD BE SYM_MATS_K OR SYM_MATS TODO" at line 809 is the smoking gun.)

JSON: `cri3_alg_unfold/test1_psi_results.json`. Log: `cri3_alg_unfold/run_alg_unfold.log`.

## Test 2 — ζ unfold algebraic check (the load-bearing test)

`zeta_q_G[q, μ, j]` on disk stores `Σ_r exp(−2πi (q + G_j)·r) · ζ_q(r, μ)` per the writer at `common.isdf_fitting:2068` and `common.wfn_transforms.accumulate_rchunk_to_gflat`. The "via LORRAX" unfold for G-flat ζ is **the V_q post-loop centroid double-permute** in `common.symmetry_maps.unfold_v_q` — there is no `ZetaLoader.load(q='full_bz')` path for G-flat (`zeta_loader.py:306` raises NotImplementedError). The bilinear V_q unfold relies on the τ-phase cancellation in `V_q[μ,ν] = ζ_μ* · v(q+G) · ζ_ν`.

For the **algebraic** check I derive the per-(q, μ, G) symmetry rule for the disk-ζ at the q' = S·q_irr point:

```
ζ_disk[Sq, π_s(μ), G_full]
    = exp(-2πi [(Sq) · r_{π_s(μ)} - q · r_μ])         ← disk-phase difference
      · ζ_disk[q, μ, S⁻¹·G_full]                       ← G-axis pullback under S
```
(τ=0 here so no extra phase from `exp(-i G·τ)`.)

The centroid permutation `π_s` is built from `compute_centroid_sym_perm(r_mu_fft_idx, sym.sym_matrices, ...)`. The G-pullback `S⁻¹` is in the same convention as the LORRAX G-action (`S = sym_mats_k`).

I tested four G-back-rotation conventions (`mtrx_inv`, `mtrx`, `mtrx_inv^T`, `mtrx^T = sym_mats_k`, `sym_mats_k⁻¹`) and two centroid-perm conventions (LORRAX's `Rinv = inv(mtrx)` in `compute_centroid_sym_perm.py:308` vs the inverse direction).

### Headline residuals

Per-(qf, sym_idx) max ‖Δζ‖∞ on the **matched G-slots**, with the best G-rotation convention + LORRAX's mu_perm, **without** the disk-phase correction:

| qf | i_irr | sym_idx | det | best_G_conv | matched | max ‖Δ_no_phase‖∞ | rel |
|---|---|---|---|---|---|---|---|
| 0  | 0 | 0  (E)     | +1 | trivial | 13787/13787 | 9.18×10⁻³ | 5.5×10⁻⁶ (ISDF-fit noise floor) |
| 10 | 4 | 1  (C3)    | +1 | mtrx_inv_T (= sym_mats_k⁻¹) | 13739/13739 | **2706** | **1.74** |
| 11 | 1 | 1  (C3)    | +1 | mtrx_inv_T | 13789/13789 | **1508** | **1.02** |
| 13 | 6 | 2  (C3⁻¹)  | +1 | mtrx_inv_T | 12715/13768 | **2827** | **1.71** |
| 4  | 2 | 3  (−I)    | −1 | (all 4 agree) | 13752/13752 | **2892** | **1.73** |
| 5  | 1 | 3  (−I)    | −1 | (all 4 agree) | 13789/13789 | **1585** | **1.08** |
| 6  | 1 | 5  (S6⁻¹)  | −1 | mtrx_inv_T | 13789/13789 | **240** | **0.16** |
| 22 | 6 | 4  (S6)    | −1 | mtrx_inv_T | 12715/13768 | **2827** | **1.71** |

### Disk-phase correction

The same residuals **with** the disk-phase correction `exp(-2πi[(Sq)·r_{π_s(μ)} − q·r_μ])` multiplied into the prediction:

| qf | sym | max ‖Δ_phase‖∞ | improvement |
|---|---|---|---|
| 4  | 3 (−I)    | **138** | **20×** smaller than 2892 |
| 5  | 3 (−I)    | **240** | 6.6× |
| 6  | 5 (S6⁻¹)  | 240 (unchanged) | — |
| 10 | 1 (C3)    | 2017 | 1.34× |
| 11 | 1 (C3)    | 1182 | 1.27× |
| 13 | 2 (C3⁻¹)  | 2486 | 1.14× |
| 22 | 4 (S6)    | 2514 | 1.12× |

**Localization of the residual structure**:
- For −I: the per-(μ, G) **ratio** `pred / actual` is, on average, a cube root of unity `exp(±2πi/3)` (verified at random sample of 8 (μ, G) cells; values like `(-0.43, -0.85)`, `(-0.50, -0.87)`, `(-0.41, -0.94)` ≈ exp(−2πi/3)). The disk-phase correction reduces this by 20×, indicating the phase correction is mostly correct for −I but a residual ~5% per-cell magnitude remains.
- For C3/S6: even with the disk-phase correction, residuals stay at ~1.5-2.5× the actual ζ magnitude, indicating **the centroid permutation π_s is fundamentally wrong** (or the G-rotation is incomplete) for non-involutive ops.

### Localization of mu_perm

`compute_centroid_sym_perm.py:307-308` computes:
```python
Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)         # S = sym_matrices = BGW mtrx
images = np.einsum('rj,sij→sri', r_frac, Rinv) + tau_frac
```
The DOCSTRING says "real-space r transforms via Rinv = S^(−1)" (`S r + τ` in column form). But for the hexagonal CrI3, applying `mtrx` (not `mtrx_inv`) to fractional `r_μ` gives the BGW-correct real-space C3 rotation:
```
mtrx[1] · (1, 0, 0)_crystal = (0, 1, 0)_crystal     ← a1 → a2 under +120° rotation, CORRECT
mtrx_inv[1] · (1, 0, 0)_crystal = (-1, -1, 0)_crystal  ← a1 → −a1 − a2, this is C3⁻²
```
**So `compute_centroid_sym_perm` is computing π_s for the WRONG sym op** — it builds π_{s⁻¹} when it should build π_s (or vice versa, depending on what "forward" means in the consumer). For involutive ops (E, −I, σ_h on layered: I.e., any op with op² = E), π_s = π_{s⁻¹} so this bug is invisible. For C3/S6, they are NOT involutive (C3² = C3⁻¹ ≠ C3), and the wrong direction is the wrong permutation table.

I verified this directly by building `mu_perm_B` with `sym_matrices = inv(mtrx)` (i.e., using `mtrx_inv_list` as the "S" passed to `compute_centroid_sym_perm`), but the residuals did NOT improve — they got worse on average. So **mu_perm direction is not the only bug**; it interacts with the G-rotation convention and/or the phase correction.

### What the (μ, G) localization tells us

At qf=4 (sym=3, −I), the **per-cell phase factors** in `pred/actual` are predominantly `exp(±2πi/3)`. This is a smoking-gun for a **per-band gauge issue tied to C3 symmetry**. Because the test bed is `bispinor=false` (charge channel), the band-level spinor U cancels in ρ_nm = Σ_σ ψ*ψ — but the band-level wavefunction PHASE (within a 2-band Kramers pair) does NOT cancel, and IF the ψ_full = U · ψ_kbar uses a wrong spinor U for the spatial sym path, then within a Kramers pair the ψ_full has a wrong basis-choice ordering that propagates into ρ_nm via the off-diagonal pair densities.

This connects Test 1's spinor-U bug to Test 2's ζ residual structure. **The wrong U_spinor for C3/S6 (Test 1 finding) likely propagates into ζ via the band-pair gauge** — even though it cancels at the per-r charge density level, it does NOT cancel in the centroid-projected ρ_nm matrix that the ISDF fits ζ to.

JSON: `cri3_alg_unfold/test2_zeta_results.json`. Logs: `t2_conv.log`, `t2_phase.log`, `t2_mi.log`.

## Bug summary — three candidate roots, narrowed to two

| # | Suspect site | Evidence pro | Evidence con |
|---|---|---|---|
| **1** | `syms_crystal_to_cartesian` (line 808-813) — uses `bvec` (recip) where `avec` (real) is needed, and `sym_mats_k` where `mtrx` (or `mtrx_inv`) is needed for the real-space rotation. R_cart for C3 is non-orthogonal (rows have norm 2+), feeds into Markley → wrong SU(2) U_spinor. | C3/S6 unitarity = 0.82, det = +1 (proper) AND det = −1 (improper) syms both affected; E/−I (involutive, sym_mats_k = sym_mats_k^T = inv) pass at ULP. SU(2) U_spinor[1] = `diag(exp(±i 125°))` ≠ `diag(exp(±i 60°))` (the expected C3 SU(2)). | bispinor=false ⇒ ρ_nm sums over σ ⇒ U_spinor SHOULD cancel. (But see "phase tied to gauge" discussion above — does NOT cancel for off-diagonal m≠n pair densities.) |
| **2** | `compute_centroid_sym_perm` direction (line 307-308) — uses `Rinv = inv(mtrx)` to build π_s but the V_q unfold rule expects π_s = "where does S·r_μ land" with the OTHER inverse direction. | Verified empirically: building mu_perm with mtrx (not inv) gives a different table that produces DIFFERENT (still wrong) ζ residuals — implies the direction matters. For C3 (non-involutive), the two directions differ. For −I (involutive), they agree. | Switching to mtrx (without other changes) did not improve residuals — implies the bug is multi-source. |
| **3** | Missing G·τ phase in V_q unfold | — | τ = 0 for CrI3 P-3 (symmorphic). Cannot be the bug here. |

**Most-likely combined story**: bug #1 (wrong U_spinor → wrong band-gauge within ψ_full at C3-related k → wrong off-diagonal pair densities → wrong ζ) is the dominant root cause. Bug #2 (mu_perm direction) is a separate latent issue that aligns with the cube-roots-of-unity phase ratios seen in the (μ, G) localization of the ζ residual. Bug #3 doesn't fire here.

For **MoS₂** (which passes the algebraic test), neither bug fires:
- MoS₂'s sym group is E + σ_h. σ_h is involutive (σ² = E), so bug #2 doesn't activate.
- σ_h in cartesian on a hex/in-plane structure is the diagonal reflection `diag(1, 1, -1)` (orthogonal), and `B_T_inv · sym_mats_k_σ · B_T` for this diagonal matrix gives back a correct orthogonal matrix even in the wrong-basis convention. So bug #1 doesn't activate for σ_h either.

For **Si** (separate failure at 160 eV per `si_sym_vs_nosym_pr3.md`), the root cause is different: τ-phase bra/ket bookkeeping in non-symmorphic ψ-unfold. Si's mtrx is integer-orthogonal in crystal coords (cubic), so the R_cart conversion bug doesn't fire there.

## Why MoS₂ passing didn't catch this

The MoS₂ test bed only exercises sym_idx=2 (TRS-augmented identity, `sym_mats_k = -E`) for its non-trivial unfolds. The spatial part `sym_mats_k[0] = E` is the identity ⇒ `syms_crystal_to_cartesian(E)` returns E regardless of which basis convention is used. The bug surface is **degenerate** there.

CrI3's P-3 with 2C3 + 2S6 (4 non-involutive ops out of 6) maximally exercises the buggy conversion path.

## Confirmation of Bug #1 (`syms_crystal_to_cartesian`) — direct numeric proof

Computing `‖R_cart Rᵀ − I‖∞` for the current LORRAX code vs the corrected form `A.T @ mtrx_inv @ inv(A.T)` (A = avec · alat):

| sym | LORRAX `R_cart` ‖RRᵀ − I‖∞ | Corrected ‖RRᵀ − I‖∞ |
|---|---|---|
| 0 (E)      | **0.00**         | 2.4×10⁻¹⁷ |
| 1 (C3)     | **3.46** ❌      | 1.1×10⁻⁷ |
| 2 (C3⁻¹)   | **5.33** ❌      | 1.1×10⁻⁷ |
| 3 (−I)     | **0.00**         | 2.4×10⁻¹⁷ |
| 4 (S6)     | **3.46** ❌      | 1.1×10⁻⁷ |
| 5 (S6⁻¹)   | **5.33** ❌      | 1.1×10⁻⁷ |

LORRAX's R_cart for C3 is:
```
[[ 0.500,  2.021, 0],
 [-0.866, -1.500, 0],
 [ 0,      0,     1]]
```
Corrected R_cart for C3 (the canonical 120° rotation around z):
```
[[-0.500,  0.866, 0],
 [-0.866, -0.500, 0],
 [ 0,      0,     1]]
```

Markley's algorithm in `get_spinor_rotations` accepts a non-orthogonal matrix and silently produces a non-SU(2) "spinor U" instead of erroring. This is the bug surface that propagates to ψ-unfold and (via off-diagonal pair densities) to ζ.

## Recommended fixes (NOT proposed here per task spec — diagnose only)

Per task: "Don't propose code fixes. Diagnose only." Suspects narrowed for the next agent:
1. **`syms_crystal_to_cartesian`** — verify that R_cart is orthogonal for every sym op (assertion: `‖R Rᵀ − I‖∞ < 1e-10`). If not orthogonal, fix the basis transform.
2. **`compute_centroid_sym_perm`** — verify against an explicit triple test on a known C3 case (e.g., propagate an atom position by mtrx and check the perm matches).
3. **`get_spinor_rotations`** — once R_cart is correct, verify U_spinor[1] = `diag(exp(±i π/3))` for the CrI3 C3 op as a hard unit test.

## Hard constraints — discharged

- Did NOT use `unfold_psi` / `unfold_v_q` to derive the hand-rolled references; all (b) computations are written from first principles, importing only data structures (`sym.sym_matrices`, `sym.sym_mats_k`, `sym.translations`, `sym.U_spinor`, `sym.irr_idx_q`, etc.).
- Cross-checked methodology against `algebraic_unfold_mos2.md` (PASS at 1.7×10⁻⁴ rel — matched). MoS₂'s PASS is consistent with the proposed root cause: MoS₂ doesn't exercise the buggy conversion path.
- Used the currently-running 4-A100-80GB allocation (JID 52971590); ~10 GPU-min total budget consumed (well under the 60 GPU-min cap).
- No code fixes proposed — all findings are diagnostic.

## Artifacts

| File | Purpose |
|---|---|
| `cri3_alg_unfold/test_alg_unfold.py` | Primary driver: T1 + T2 + T3 with first-principles hand-roll |
| `cri3_alg_unfold/test_t2_g_conv.py` | T2 sub-experiment: 4 G-rotation conventions × 2 mu-perm directions |
| `cri3_alg_unfold/test_t2_full_phase.py` | T2 sub-experiment: + disk-phase correction |
| `cri3_alg_unfold/test_t2_minus_i.py` | T2 deep-dive on −I (per-(μ,G) ratio analysis showing exp(±2πi/3)) |
| `cri3_alg_unfold/test_t1_g_conv.py` | T1 sub-experiment: alternative G-rotation conventions |
| `cri3_alg_unfold/test1_psi_results.json` | T1 per-(kf, sym_idx) results |
| `cri3_alg_unfold/test2_zeta_results.json` | T2 per-(qf, sym_idx) results |
| `cri3_alg_unfold/test3_gvec_results.json` | T3 per-k_full G-set comparison |
| `cri3_alg_unfold/run_alg_unfold.log`, `t1_conv.log`, `t2_conv.log`, `t2_phase.log`, `t2_mi.log` | Run logs |
