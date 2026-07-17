# Design: BSE at arbitrary exciton momentum Q via htransform (direct-kernel-only)

Reconnaissance + design, 2026-07-16. Read-only on `sources/lorrax_A`
@ `agent/bse-phase2` (f19136e). Numerical checks: 1 GPU, module-free
srun+shifter, `runs/MoS2/A_bse_w0_resolvent_2026-07-16/arbitrary_q_recon/`.
Companion designs: `coulomb_sr_lr.md` (SR/LR split), `kernel_dataflow_trace.md`
(BSE kernel spine). Ry units throughout.

Owner framing: htransform interpolates the wavefunction r_μ coefficients →
enables BSE at arbitrary Q; the hard problem is the long-range/short-range
Coulomb separation needed IF V_Q is to be interpolated; **direct-kernel-only
scope is acceptable**. This design confirms that scope is the right cut, and
shows (with numbers) that the "interpolate V_Q from one master ζ" shortcut does
NOT hold — the exchange is correctly deferred.

---

## 0. Verdict summary (for main)

1. **htransform is ready as the off-grid ψ(r_μ) + ε source.** `bse_setup.compute_wfns_fi`
   already produces exactly `u_{c,k+Q}(r_μ)` + recovered `ε_c(k+Q)` for a
   shifted uniform grid, via one coarse-grid Fourier interpolation of the
   f-transformed DFT Hamiltonian — **no H(k) rebuild, no Sternheimer**. It needs
   generalization from "fine uniform grid" to "coarse grid + arbitrary shift Q",
   ~1 function.
2. **Direct-kernel-only arbitrary-Q BSE is cheap and clean.** Per-element check
   confirms: `k, k'` both stay on-grid ⇒ `k−k'` on-grid ⇒ **the existing coarse
   `W_{μν}(k−k')` tiles and the 3-D-FFT-over-coarse-k convolution serve
   UNCHANGED**. Q enters ONLY through which conduction ψ the loader supplies
   (`k+Q` not `k`), evaluated at the *existing* charge centroids `r_μ` — **no ζ
   refit for the direct term**. The change is a loader swap + a D-diagonal swap.
3. **The ζ-structure shortcut FAILS (measured, MoS2 3×3, 30 Ry).** LORRAX's
   `ζ_q(μ,G)` is **NOT** the sphere-windowed FT of a q-independent real-space
   `ζ_μ(r)`. The real-space interpolation vectors are genuinely different objects
   per q: the *magnitude* field `|ζ_q(μ,r)|` changes by **~40% (median) between
   adjacent q** even at fixed band-limit; predicting `V_q` at an on-grid q from a
   Γ "master ζ" + analytic phase gives **90–340% Frobenius error**. So
   `V_Q[μν] = Σ_G ζ̃*_μ(Q+G) v(Q+G) ζ̃_ν(Q+G)` is **not** directly evaluable from
   one stored object. The exchange V_Q, when eventually added, needs a per-Q ζ
   refit (compute-don't-interpolate) or the Gaussian SR/LR interpolation of
   `coulomb_sr_lr.md`. **Deferring exchange in the direct-only scope is
   vindicated, not a corner cut.**

**Recommendation:** build direct-kernel-only arbitrary-Q TDA BSE now (~1 loader
generalization + finite-Q roll of the conduction leg; W untouched). It is
unblocked by the ζ finding. Exchange is a separate, larger workstream.

---

## 1. The htransform interface contract

### 1.1 What it is

`src/bandstructure/htransform.py` + `src/bandstructure/bse_setup.py`. htransform
is **Hamiltonian interpolation in a rank-α centroid basis**, NOT a per-k solve.
It builds a small `(rank, rank)` effective Hamiltonian `fH_R` in the lattice-R
representation ONCE on the coarse grid, then Fourier-sums it to any off-grid q
and diagonalizes the tiny matrix. Cost per off-grid point is one `(rank,rank)`
eigendecomposition — there is **no `H(k)` plane-wave rebuild and no Sternheimer**.

The "f-transform" (`f_transform_eigs`, htransform.py:324) is a smooth
bandwidth-bounded map `f(ε) ≤ 0 for ε<shift, =0 above` applied to the DFT
eigenvalues so that `fH_k = Σ_n f(ε_{n,k}) c_{n,k} c_{n,k}^H` is a well-behaved
low-rank operator whose eigenpairs recover `(f(ε_{n,q}), c_{n,q})` at any q;
`newton_inv` (htransform.py:405) inverts `f` to return physical energies.

### 1.2 Setup (one-time, coarse grid)

`initialize_wfns` (htransform.py:576) → `streaming_galerkin_solve`
(htransform.py:43) produces the three durable objects:

| object | shape | sharding | meaning |
|---|---|---|---|
| `ctilde` | `(nk_co, nb, rank)` | replicated `P()` | Galerkin coeffs of ψ in the rank-α basis |
| `B_at_mu` | `(rank, ns, n_μ)` | replicated `P()` | α-basis evaluated at the charge centroids `r_μ` (= `L⁻¹Vᴴ` from the SVD, htransform.py:202) |
| `enk_sigma` | `(nb, nk_co)` | replicated | coarse DFT band energies (Ry); optional EQP override |

`rank` = SVD rank of `A = ψ_at_centroids.reshape(nk_co·nb, ns·n_μ)` truncated at
`rtol=1e-8` (htransform.py:98-113); bounded by `min(nk_co·nb, ns·n_μ)`, in
practice `≈ nk_co·nb` (MoS2 3×3, nb≈8 ⇒ rank ≤ 72). Setup reads ψ at centroids
(`load_centroids_band_chunked`) and streams ψ at full r once
(`iter_psi_rchunk_bandwise`) to build the Gram `G` → Cholesky → `ctilde`. This is
the same ψ machinery the ISDF/GW side already loads; residency is host-streamed,
device peak is one r-chunk.

### 1.3 The BSE loader entry point — the exact contract to call

`bandstructure.bse_setup.compute_wfns_fi(*, ctilde, B_at_mu, enk_sigma,
kgrid_co, kgrid_fi, band_window_fi, mesh_xy, a_band_index=None, batch_size=32,
log_fn) -> SimpleNamespace` (bse_setup.py:58). Internals (bse_setup.py:117-155):

```
fH_k, fH_R, (a,n,shift), _ = build_fH_R(ctilde, enk_sigma, kgrid_co, mesh_xy)   # once
for q_batch in fine_grid:                                                        # per-q, batched
    fH_q   = Σ_R e^{-2πi q·R} fH_R                        # (bs, rank, rank)  Fourier sum
    lam,U  = eigh(fH_q)                                   # ascending; lam = f(ε_{n,q})
    c      = U[:, :, b_min:b_max]                         # (bs, rank, nb_fi)
    psi    = einsum('qan,asm->qnsm', c, B_at_mu)          # (bs, nb_fi, ns, n_μ)  ψ(r_μ)
energies_fi = newton_inv(a,n,shift, lam)                  # DFT-equivalent ε(q)
```

Returns the canonical wfn bundle:

| field | shape | sharding | contents |
|---|---|---|---|
| `psi_rmu_Y` | `(nk_fi, nb_fi, ns, n_μ)` | `P(None,None,None,'y')` | `u_{n,q}(r_μ)` at coarse centroids |
| `psi_rmuT_X` | `(nk_fi, n_μ, nb_fi, ns)` | `P(None,'x',None,None)` | same, μ on x |
| `enk_full` | `(nk_fi, nb_fi)` | replicated | `ε_{n,q}` via `newton_inv` |
| `lam_fi` | `(nk_fi, nb_fi)` | replicated | raw `f(ε)` eigenvalues (diagnostic) |

The bundle **exactly matches `load_centroids_band_chunked`'s output layout** —
it is designed to drop straight into any ISDF/BSE consumer that reads ψ at
centroids. This is the seam the arbitrary-Q BSE loader calls.

### 1.4 Cost model (per off-grid point)

- Setup (once): SVD `(nk_co·nb, ns·n_μ)` + streamed Gram + Cholesky. One-time,
  coarse-grid, same order as one ISDF centroid load.
- `build_fH_R`: one IFFT over coarse k of `(nk_co, rank, rank)`. Once.
- **Per off-grid q: `O(rank³)` eigh + `O(nk_co·rank²)` Fourier sum +
  `O(rank·ns·n_μ)` ψ reconstruction.** `rank³` dominates. Batched `batch_size`
  q's per compile; `_kpath_batch`/`_q_batch` shard the batch axis over devices
  (htransform.py:701, bse_setup.py:138).
- Memory: `fH_R` replicated `(nk_co, rank, rank)` (~240 MB at "our scale" per the
  htransform.py:698 note); B_at_mu replicated. All small vs the GW tensors.

### 1.5 Accuracy / validation surface

Built-in diagnostics (htransform.py `h_transform`, 655-733), no external gate
recorded (bandstructure is flagged *experimental*, codebase.md:121):
- `fH(k=0)` eigenvalues vs `f(ε)` — exactness of the low-rank rebuild (target ~0).
- Γ FFT round-trip `‖fH_k − Σ_R e^{−2πik·R}fH_R‖` (target ~1e-12).
- Γ `Δε` (mRy) — interpolated vs exact DFT energies at Γ.
Accuracy is governed by centroid count `n_μ`, band window, and the f-transform
`a_band_index` (set `a` from the highest band you need accurate,
`_f_params_from_energies`, htransform.py:261). **Gate to add for BSE use:**
htransform ψ(r_μ), ε at an *on-grid* q vs the directly-loaded ψ(r_μ), ε — must
match to interpolation tolerance (this closes the "is the interpolation good
enough for the kernel" question the experimental flag leaves open).

### 1.6 Config plumbing (already wired)

`BSEConfig` (gw_config.py:815): `get_centroids_fi` (gate), `wfn_fi_min/max`
(band sub-window), `kgrid_fi` ("nx ny nz"). Driven from `htransform.main`
(htransform.py:897). For arbitrary Q, add `qshift_fi = "qx qy qz"` (fractional)
— see §2.

---

## 2. Arbitrary-Q pair-basis — per-element dataflow (direct-kernel-only)

Exciton at momentum Q: transitions `|v k → c, k+Q⟩`, `k` on the coarse grid,
`Q` arbitrary. TDA `H^BSE = D − W` (exchange V deferred; `kernel_dataflow_trace.md`
§Purpose). Per-ingredient:

### (a) ψ_c at off-grid k+Q — htransform

`u_{c,k+Q}(r_μ)` for all coarse `k` = htransform evaluated on the shifted grid
`{k + Q : k ∈ coarse}`. `compute_wfns_fi` today builds a *Γ-centred uniform*
`kgrid_fi` (`_uniform_kgrid_frac`, bse_setup.py:48). **Generalization needed:**
accept an explicit q-list `{k+Q}` (coarse grid + constant shift Q) instead of a
uniform fine grid. One helper: `compute_wfns_at_qlist(ctilde, B_at_mu,
enk_sigma, kgrid_co, q_list=coarse_k + Q, band_window, mesh_xy)` — identical body,
`q_all = coarse_k_frac + Q` in place of `_uniform_kgrid_frac`. The valence leg
`u_{v,k}(r_μ)` is the ordinary on-grid ψ (already loaded; or htransform at Q=0
for a single consistent source).

### (b) D_Q = ε_c(k+Q) − ε_v(k) — off-grid energies

`ε_c(k+Q)` = `enk_full` from the SAME `compute_wfns_fi` call (`newton_inv` of the
`fH_{k+Q}` eigenvalues, bse_setup.py:159). `ε_v(k)` = on-grid coarse energies
(or EQP via `--eqp`). So the D diagonal is `enk_full_c[k,·] − eps_v[k,·]` — a
one-line change from today's `eps_c[k]−eps_v[k]` (kernel_dataflow_trace.md
§"Diagonal D", bse_io.py:436). **No band interpolation needed beyond htransform**
— the eigenvalues come from the same interpolation as the wavefunctions, so
ε(k+Q) and ψ(k+Q) are mutually consistent (both are the eigenpairs of one
`fH_{k+Q}`). This is cleaner than BGW's separate WFN_fi + energy interpolation.

### (c) Direct kernel W(k−k') — UNCHANGED (per-element verified)

Claim: both `k, k'` stay on-grid ⇒ `k−k'` on-grid ⇒ the existing `W_{μν}(q)`
tiles + the coarse-k FFT convolution serve unchanged. Per-element, from
`kernel_dataflow_trace.md:37-43` (the ISDF direct term):

```
(W X)[b,c,v,k] = (1/Nk) Σ_{k'} Σ_{c'v'}
     [ Σ_{tμ} ψ*_{c,k+Q}(μ) ψ_{c',k'+Q}(μ) ]_μ   ·   W_{μν}(k−k')   ·
     [ Σ_{sν} ψ_{v,k}(ν)   ψ*_{v',k'}(ν)   ]_ν      X[b,c',v',k']
```

vs the Q=0 code, which has `ψ*_c(k) ψ_{c'}(k')` on the conduction leg. Element by
element, the ONLY difference is the conduction pair density uses `ψ_{·,k+Q}`
instead of `ψ_{·,k}`:
- `W_{μν}(k−k')`: argument `k−k'` is independent of Q (both shift by +Q would
  cancel; here only c-leg shifts) — index is `q = k−k'`, on the coarse grid
  exactly as today (`kernel_dataflow_trace.md:48-49`). **W tile untouched.**
- The convolution `U[k] = (1/√Nk) Σ_q W[q] T[k−q]` is a 3-D FFT over the coarse
  `(nkx,nky,nkz)` (bse_serial.py:71-75, bse_simple.py:147-157). **FFT grid
  untouched.**
- The conduction pair density `Σ_μ ψ*_{c,k+Q}(μ) ψ_{c',k'+Q}(μ)` is evaluated at
  the SAME charge centroids `r_μ` — it needs `ψ_{·,k+Q}(r_μ)` (from (a)) but **no
  ζ refit** (the direct term contracts ψ-at-centroids with the precomputed W, it
  never re-fits interpolation vectors).
- Valence leg `Σ_ν ψ_{v,k}(ν) ψ*_{v',k'}(ν)` is fully on-grid — identical to today.

**Conclusion (c): the direct term needs only a finite-Q roll of the conduction ψ
in the T-encode; `W` and the k-convolution are literally the same arrays.** This
is the owner's "direct kernel only" scope with an *exact* (not approximated)
direct term. The finite-q `W_q` roll machinery already landed on this branch
(6ca714b, c74a189: `kgrid_shift_map`, finite-q `W_q` resolvent) is the same
plumbing pattern; here Q shifts the ψ leg, not W.

### (d) Exchange V_Q — the SR/LR problem, and finite-q ζ

Exchange `⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν} M*_{cvk}(μ) V_Q(μν) M_{c'v'k'}(ν)`
with pair density `M_{cvk}(μ) = Σ_s ψ*_{c,k+Q}(μ) ψ_{v,k}(μ)` and the **bare**
Coulomb tile at exciton momentum Q, `V_Q(μν) = Σ_G ζ̃*_{Q,μ}(G) v(Q+G) ζ̃_{Q,ν}(G)`
(kernel_dataflow_trace.md:30-34). Two off-grid needs:
- `v(Q+G)` at arbitrary Q — carries the `1/|Q+G|²` divergence at the head; this
  is the SR/LR problem `coulomb_sr_lr.md` owns.
- `ζ̃_Q(μ,G)` — the ISDF interpolation vectors AT momentum Q. The manual
  (`manual/05_isdf/5.3`) is explicit that the fit is **per momentum transfer**:
  `C_q ζ_q = Z_q`, with `C_q = Σ_k Σ_{ss'} P*_{k−q} P_k` and `Z_q` likewise
  (k-convolution of quasi-density matrices). So `ζ_Q` at an off-grid Q requires
  the pair densities `u*_{v,k}(r) u_{c,k+Q}(r)` — i.e. htransform'd `ψ_{c,k+Q}`
  on the FULL r-grid (not just centroids) — fed through the existing
  `fit_zeta_to_h5` machinery at the single momentum Q. **The `zeta_q` machinery
  DOES admit arbitrary Q**: it is already a per-q normal-equation solve; point it
  at `q=Q` with htransform'd conduction ψ and it produces `ζ_Q`. Cost = one
  r-chunk-loop ζ-fit for one momentum (C_Q build + Cholesky + per-r-chunk solve),
  ~1/nq of a full GW ζ-fit, plus the htransform ψ(full-r) at `{k+Q}`.

This is why **exchange is correctly deferred** in the direct-only scope: it costs
a per-Q ζ refit *or* the interpolation machinery — see §3/§4.

---

## 3. The ζ-structure question — VERDICT WITH NUMBERS

**Owner-invited hypothesis (tested, not assumed):** is `ζ_q(μ,G)` the
|q+G|-sphere-windowed FT of a q-INDEPENDENT real-space `ζ_μ(r)`, so that
`V_Q[μν] = Σ_G ζ̃*_μ(Q+G) v(Q+G) ζ̃_ν(Q+G)` is directly evaluable at any Q from
one stored object + analytic v, with no SR/LR split and no interpolation
(only the G=0 head stays analytic-separate)?

**Answer: NO.** The fit is genuinely q-dependent.

### 3.1 Method

Fixture: `runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5`
(charge channel, `vertex_mu_L=0`, full-BZ 3×3×1 = 9 q, n_μ=640, ζ-cutoff 30 Ry,
FFT 24×24×80). Script + log:
`runs/MoS2/A_bse_w0_resolvent_2026-07-16/arbitrary_q_recon/zeta_structure_test.py`.

Verified G-flat convention from source (`accumulate_rchunk_to_gflat`
wfn_transforms.py:1018, `_do_disk_to_G` zeta_loader.py:668): `ζ̃_q(μ,G) = Σ_r
ζ_q(μ,r) e^{-2πi(q+G)·r}` on the sphere `{G : |q+G|²≤cutoff}` — coeffs are
sampled at physical reciprocal vector `k_phys=q+G`. Reconstruct real-space
`ζ_q(μ,r)` per q (scatter→ifftn→undo Bloch phase), compare across neighboring q.
Index mapping validated: `g0_mu == ζ̃[q,:,G=0 slot]` to machine zero.

### 3.2 Results

**Part A — neighbour-q shape/phase (64-centroid subset, 18 pairs).**
Metric `p = min_φ ‖ζ_a(μ) − e^{iφ}ζ_a'(μ)‖ / ‖ζ_a(μ)‖` (residual after removing
the best per-μ global phase — a gauge, per the tile-gauge MEMORY note):

| quantity | median | max |
|---|---|---|
| `p_ζ` (real-space ζ, H1/H3 test) | **1.35** | 2.12 |
| `p_R` (cell-periodic `e^{-iq·r}ζ`, H2 test) | **1.50** | 2.46 |
| `m_mag` = `‖|ζ_a|−|ζ_b|‖/‖|ζ_a|‖` (gauge-free) | **0.39** | 1.07 |

`p ≈ √2 ≈ 1.41` is the *orthogonal-vectors* value — neighbouring-q `ζ` for the
SAME centroid are nearly uncorrelated after optimal phase alignment, in BOTH the
lab-frame (`p_ζ`) and cell-periodic (`p_R`) representations. Neither H1 (pure
r-object) nor H2 (cell-periodic object) holds.

**Sanity check that reconstruction is correct:** the ±q pairs (q and −q, related
by TRS) give `m_mag = 0.000` *exactly* (identical magnitude field, as
`|ζ_q|=|ζ_{-q}|` demands) yet `p_ζ ≈ 1.5` — consistent with `ζ_{-q}=ζ_q*`
(conjugation, magnitude-preserving, non-removable by a global phase). The machinery
is right; the q-dependence is physical.

**Part B — band-limit-CONTROLLED magnitude (common 1507-G set, identical
truncation for every q).** With the sphere fixed so truncation cannot cause the
difference: `m_mag` median **0.396**, max 1.07. **The magnitude field
`|ζ_q(μ,r)|` genuinely changes by ~40% between adjacent q.** Not a window
artifact.

**Part C — master-ζ prediction (the design payoff, all 640 μ).** Build the
"master" object from Γ (`ζ_Γ`), predict `ζ̃_q` at every other on-grid q via
`fftn(e^{-2πiq·r} ζ_Γ)|_sphere(q)`, form `V_q^pred` with `v=8π/|q+G|²`, compare
to `V_q^actual` (from disk). Applied the analytic centroid phase
`e^{2πiq·(r_ν−r_μ)}` (the umklapp/L-phase already in `unfold_v_q`) as the best
possible correction:

| target q (frac×3) | ζ̃ phase-aligned resid | V_q raw rel-Frob | V_q + centroid-phase | diag rel |
|---|---|---|---|---|
| (0,1) nearest | 0.85 | 1.35 | 1.49 | 0.74 |
| (1,0) nearest | 0.80 | **0.91** | 1.15 | **0.40** |
| (1,1) | 0.95 | 1.06 | 0.99 | 0.39 |
| (0,2) | 1.46 | 3.36 | 3.41 | 2.27 |
| (2,2) far | 1.31 | 3.44 | 3.52 | 2.01 |

Even the BEST (nearest-neighbour) prediction is **40% wrong on the diagonal and
~90% in Frobenius**; it degrades to 200–350% at farther q. The analytic
centroid-phase correction does not help (often hurts), ruling out the
centroid-phase law (H3). **`V_Q` is not reconstructible from one master ζ.**

### 3.3 Why (physics)

`ζ_q` interpolates the SPAN of momentum-q pair densities `{u*_{v,k} u_{c,k+q}}`
(manual 5.3). That span rotates substantially with q — different band pairings,
different k-coupling through the convolution `Σ_k P*_{k−q}P_k` — so the
least-squares interpolation vectors are genuinely q-specific. This is the ISDF
analogue of "the screened-exchange basis is q-dependent"; it is not removable by
bookkeeping.

### 3.4 Consequence

The direct-evaluation route for bare exchange is dead. `V_Q` needs one of:
(i) a per-Q ζ refit, or (ii) interpolation of a *smoothed* `V^SR_q` (the SR/LR
design). Note even (ii) inherits the ~40% coarse-grid q-variation of ζ as
interpolation error unless the fine grid is dense — the SR/LR split cures the
*divergence*, not the basis rotation. **For the direct-kernel-only scope this is
moot: exchange is not evaluated at all.** The finding converts "exchange
deferred" from a corner-cut into a justified scope boundary.

---

## 4. Exchange-term options, ranked

For when exchange is added on top of direct-only (all need `V_Q`/`ζ_Q`):

1. **Compute-don't-interpolate — per-Q ζ refit (RECOMMENDED, honest).** Run
   `fit_zeta_to_h5` at the single momentum Q with htransform'd `ψ_{c,k+Q}` on the
   full r-grid. Exact (no interpolation error), reuses the whole validated ζ-fit
   path. Cost = one r-chunk ζ-fit for one q + htransform ψ(full-r) at `{k+Q}`
   (§2d). Head `v(Q+G=0)` stays analytic-separate (mini-BZ average), the ONLY
   piece that must be handled outside the tile. Best when few Q's are needed.
   **The ζ finding makes this the default — it is the only route with no
   uncontrolled error.**
2. **Gaussian SR/LR split + interpolate `V^SR_Q`** (`coulomb_sr_lr.md`). Split
   `v = v_SR + v_LR`; `V^SR` is smooth (divergence removed), interpolate it across
   the coarse grid, re-add `v_LR` analytically at Q. Basis-agnostic (per-G scalar
   multiply through the existing centroid contract), no ζ-fit surgery. **Caveat
   from §3:** the ~40% coarse-q variation of ζ still lives inside `V^SR`, so the
   interpolation error is bounded by coarse-grid ζ-smoothness, not by the split.
   Viable for dense Q-sampling / many Q; needs the interpolation design + epsdiag
   `ε⁻¹₀₀(q)` to exist. Larger, multi-design effort.
3. **Compute-don't-interpolate for W_Q (for screened exchange / full BSE, not
   TDA-direct).** htransform ψ → pair basis at Q → the validated `ω=0` resolvent
   generates `W_Q` directly — no W interpolation. Cost = one resolvent solve per
   Q (the finite-q `W_q` resolvent already on this branch, c74a189/6ca714b). This
   is the W analogue of option 1 and the natural partner if screening is wanted
   at finite Q. Not needed for bare-exchange direct-only.

**Direct-eval-from-master-ζ (the tested hypothesis): REJECTED** by §3 — do not
build it.

---

## 5. Cost model per Q (direct-kernel-only)

| stage | cost | one-time? |
|---|---|---|
| htransform setup (SVD+Gram+Chol) | ~1 ISDF centroid load | once |
| `build_fH_R` | 1 IFFT over coarse k of `(nk_co,rank,rank)` | once |
| **per Q:** htransform ψ(r_μ),ε at `{k+Q}` | `nk_co ×` [`O(rank³)` eigh + `O(rank·ns·n_μ)` recon], batched | per Q |
| **per Q:** D-diagonal rebuild | `O(nk·nb)` | per Q |
| **per Q:** TDA `H^BSE=D−W` solve | identical to Q=0 (Lanczos/FEAST); `W` tiles + FFT reused | per Q |

W tiles, W_R FFT, valence ψ, mesh/sharding: all reused unchanged. The marginal
cost of a new Q is one htransform pass (dominated by `nk_co` small eigh's) + one
BSE solve. No new large tensors.

## 6. Gates (1-GPU, MoS2/Si fixtures — no 16-GPU gating)

1. **htransform on-grid consistency (unit).** `compute_wfns_at_qlist` with Q=0 on
   the coarse grid must reproduce `load_centroids_band_chunked` ψ(r_μ) and coarse
   ε to interpolation tolerance. Closes the "experimental" accuracy gap.
2. **Direct-term Q=0 non-regression (integration, load-bearing).** Arbitrary-Q
   path at Q=0 must reproduce the existing TDA `D−W` eigenvalues bit-for-bit
   (the Q shift is identity) — proves the loader/roll refactor is exact. Anchor:
   the validated Si-SOC ledger (`STATUS.md` ~3 meV vs BGW).
3. **`k−k'` on-grid invariance (unit).** Assert the `W_q` index set and FFT grid
   are byte-identical between Q=0 and Q≠0 runs (they must be — §2c).
4. **Finite-Q dispersion smoke (diagnostic).** Lowest exciton eigenvalue vs Q
   along a small path; expect a smooth, physical `E(Q)` with the correct Q→0
   limit. No BGW anchor for direct-only finite-Q, so this is qualitative until a
   reference exists.
5. **(exchange, later)** per-Q ζ refit round-trip: `ζ_Q` at an on-grid Q via
   htransform-fed refit must match the stored `ζ_q` to fit tolerance.

## 7. LOC estimate

- `compute_wfns_at_qlist` (generalize `compute_wfns_fi` to coarse+shift): **~40
  LOC** (mostly the q-list builder; body reused).
- BSE loader: conduction ψ source = htransform bundle instead of on-grid read;
  D-diagonal `enk_full_c − eps_v`; finite-Q conduction roll in the T-encode:
  **~120 LOC** across `bse_io` + one matvec variant (start with `bse_simple`).
- `qshift_fi` config key + CLI `--q-exciton`: **~15 LOC** (`gw_config` +
  `bse_jax`).
- Gates 1–4: **~120 LOC** test code.
- **Total direct-kernel-only: ~180 LOC prod + ~120 test.** Exchange (option 1
  per-Q refit) is a separate ~200–300 LOC workstream; SR/LR (option 2) is the
  ~430 LOC of `coulomb_sr_lr.md` plus the interpolation/epsdiag designs.

## 8. Open questions for Jack

1. **Direct-only sufficiency.** Is bare-direct `E(Q)` (no exchange, no finite-Q
   screening) the intended first deliverable, or is screened `W_Q` (option 3
   resolvent-per-Q) wanted in the same pass? The former is ~180 LOC and unblocked
   today; the latter adds the finite-q resolvent already on-branch.
2. **htransform accuracy budget.** What Γ `Δε` / on-grid ψ(r_μ) tolerance
   qualifies htransform for kernel use? (Sets the centroid count / band window /
   `a_band_index` for the BSE fine bundle; today's diagnostics report it but no
   pass/fail threshold exists.)
3. **Exchange route when it lands.** Per-Q ζ refit (exact, option 1) vs SR/LR
   interpolation (option 2, amortized over many Q but carries the measured ~40%
   coarse-q ζ-variation as interpolation error). Depends on how many Q's and
   whether a dense exciton-dispersion is the goal.
4. **Valence ψ source.** Use on-grid ψ_v directly, or route BOTH legs through
   htransform (Q=0 for valence) so the pair density is built from one consistent
   interpolation? The latter is cleaner but pays htransform on the valence leg too.
```
