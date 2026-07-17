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

---

## 9. SR/LR interpolation in the literature — survey + mapping

Literature survey (2026-07-17, read-only, web only). Purpose: adjudicate the
owner's suspicion that the BGW-style **multiplicative** trick — interpolate
`|Q|²·(divergent quantity)` and divide by `|Q|²` after — does **not** transfer to
LORRAX's *contracted* tile `V_Q[μν]=Σ_G ζ̃*_μ(Q+G) v(Q+G) ζ̃_ν(Q+G)`; and to test the
parent's hypothesis that the correct transfer is **rank-1-head-channel
factorization** (interpolate the smooth body + the smooth `g0(Q)` vector,
reassemble with analytic `1/|Q|²`). Three literatures do exactly this problem for
Coulomb-mediated quantities; all three converge on the **subtractive** (not
multiplicative) convention, and the closest analogue (exciton-Wannier) is
*literally* the rank-1-head factorization.

### 9.0 Verdict (for main)

- **`|Q|²`-multiply-and-divide: REJECTED for the contracted tile.** The trick is
  only well-posed on an *isolated single divergent channel* with a common `1/|Q|²`
  prefactor (a scalar `ε⁻¹₀₀(q)`, a single-G Coulomb `v(Q+G)`, or BGW's head entry
  stored as `1.0`). `V_Q[μν]` is a **sum** of a divergent `G=0` head + a smooth
  `G≠0` body; there is no common `1/|Q|²` to factor out of a sum. Multiplying the
  whole tile by `|Q|²` sends the body amplitude → 0 as `Q→0`, and dividing back is
  `0/0` on the body — it destroys the smooth information it was meant to preserve.
  The owner's suspicion is correct. **BGW never does this to a summed object**: it
  keeps head/wing/body as *separate G-indexed channels* (`mtxel_kernel.f90`,
  `w_head_wings_interp.md §Reference`) and applies multiply/divide (or the stored-
  `1.0` strip) only to the already-isolated scalar head/wing. LORRAX has already
  contracted over G, so it has no separate channel to multiply — unless it first
  *reconstructs* one, which is precisely the rank-1 factorization.
- **subtract-analytic-LR: CORRECT, and it is the universal convention.** Verdi–
  Giustino, Sjakste, Brunin (e-ph) and Haber–Qiu–da Jornada–Neaton (excitons) all
  form `X_SR = X − X_LR`, interpolate the finite/smooth `X_SR`, and add the
  closed-form `X_LR` back at the target point. Strictly better-conditioned than any
  whole-object multiply *because subtraction respects the additive head+body
  structure*: it removes only the singular part and leaves the body's amplitude
  untouched (no `0/0`, no amplitude collapse). This is the e-ph "SUBTRACTIVE"
  convention and it is not a stylistic choice — `g = g_S + g_L` is a sum and `g_L`
  is the only divergent term, so a multiplicative `q·g` would kill the smooth `g_S`
  the same way it kills LORRAX's body.
- **rank-1-head factorization (parent's hypothesis): VINDICATED — it *is* the
  subtractive scheme specialized to LORRAX's known rank-1 head.** The exciton-
  Wannier long-range exchange kernel is *literally* rank-1 in a dipole vector and is
  subtracted/re-added analytically (Haber et al. Eqs 27–34). LORRAX's head is
  already stored in exactly this form — `V_qmunu` is persisted with `G=0` zeroed,
  `g0_μ = ζ(q,μ,G=0)` is kept separately, and `apply_q0_head_rank1` injects
  `v(q)·conj(g0)⊗g0` (`head_correction.py:743-816`, `coulomb_sr_lr.md §Current
  state`). **The SR/LR split LORRAX already has at Q=0 *is* the rank-1
  factorization; interpolation just applies it per-fine-Q instead of only at the
  single coarse Q=0.** Recommended for both `V_Q` and the exchange kernel.
- **subtractive vs multiplicative is a false dichotomy *once the head is
  isolated*.** After you name the rank-1 channel `g0(Q)⊗g0*(Q)·v(Q)`, adding it
  back (`V_SR + v·g0⊗g0`) and "dividing a stripped factor back in" coincide — both
  operate on one clean channel. The real dichotomy is **isolate-the-channel
  (rank-1 factorization / subtraction) vs operate-on-the-summed-tile (naive
  `|Q|²`-multiply, which fails)**. The e-ph subtractive convention is "better
  conditioned" precisely because subtraction is *how you isolate the channel
  additively*; the rank-1 factorization isolates it even more cleanly by naming its
  exact form.

### 9.1 Exciton-band-structure interpolation — the direct analogue

**Haber, Qiu, da Jornada, Neaton, "Maximally Localized Exciton Wannier
Functions," PRB 108, 205109 (2023) / arXiv:2308.03012.** This is the exciton
BSE-kernel transcription of the whole problem and it maps 1:1 onto LORRAX.

They split the singlet exciton Hamiltonian (their Eq. 32) and interpolate only the
short-range part:

```
H^Xct(Q) = H^SR(Q) + 2 δ_S K^LR(Q)                                    (Eq 32)
K^LR(Q)  = K^{X,Dip}(Q) − K̄^{X,Dip}(0)                               (Eq 33)  ← SUBTRACT the Q=0 dipole
K^{X,Dip}_{MN}(Q) = (4πe²/V_uc) Σ_G [P*_M·(Q+G)][P_N·(Q+G)] / |Q+G|²  (Eq 27)  ← rank-1 in dipole P
K^{NA}_{MN}(Q) ≡ lim_{G=0, Q→0} K^{X,Dip} = (4πe²/V_uc)(P*_M·Q̂)(P_N·Q̂) (Eq 28-29) ← direction-dep., finite
```

- **What is rank-1:** `K^{X,Dip}` is a dyadic in the *exciton transition-dipole
  vector* `P_M`. The `G=0` head is `(P*_M·Q̂)(P_N·Q̂)` — direction-dependent,
  bounded in 3D (numerator `|Q|²` cancels `v`'s `|Q|²`), and this **is g0⊗g0**: `P_M`
  is Haber's `g0`.
- **What is interpolated:** the short-range `H^SR(Q)` is Fourier-transformed to the
  exciton-Wannier lattice `R̄` (Eq 20); `H^SR_{MN}(R̄)` decays rapidly ⇒
  interpolation-safe. The head-removed `Q=0` anchor is `H^SR(0)=T(0)−K^D(0)+2δ_S
  K̄^X(0)` (Eq 43).
- **What is added back analytically:** `K^LR(Q)` from the *smooth/Q-independent
  dipoles* `P_M` and the closed-form `1/|Q+G|²` at each target Q.
- **Direction-dependent Q→0:** handled by subtracting `K̄^{X,Dip}(0)` before the FT
  (regularizes the sum) and a Löwdin down-folding (their §V.2) for the residual
  head; the `G=0` cusp lives entirely in the analytic `K^LR`, never in the
  interpolant.

This is *exactly* the parent's rank-1-head factorization: interpolate the smooth
body (`H^SR`), carry the smooth `g0` (dipoles `P_M`), reassemble with analytic
`1/|Q|²`. Earlier precedent for interpolating BSE across the grid via Wannier +
analytic-singular separation: **Kammerlander, Botti, Marques, Marini, Attaccalite,
arXiv:1209.1509** (double-grid BSE; interpolate transition dipoles, treat the
Coulomb head/wing singularity analytically).

### 9.2 Polar electron-phonon matrix elements — the subtractive lineage

The `g = g_SR + g_LR` split is the *additive* analogue of the owner's multiplicative
idea, and the community adopted it precisely because the multiplicative form fails on
a sum.

**Verdi & Giustino, PRL 115, 176401 (2015) / arXiv:1510.06373** — Fröhlich vertex:

```
g_{mnν}(k,q) = g^S_{mnν}(k,q) + g^L_{mnν}(k,q)                         (Eq 2)
g^L_{mnν}(k,q) = i(4π/Ω)(e²/4πε₀) Σ_κ (ℏ/2NM_κω_qν)^½ ·
   Σ_{G≠−q} [(q+G)·Z*_κ·e_κν(q)] / [(q+G)·ε^∞·(q+G)] ·
            ⟨ψ_{m,k+q}|e^{i(q+G)·r}|ψ_{n,k}⟩                          (Eq 4)  ← 1/q dipole divergence
```

Their explicit recipe (verbatim): "(ii) subtract `g^L` so as to obtain the
short-ranged part `g^S`; (iii) apply Wannier-Fourier interpolation to `g^S`; (iv)
add up the short-range and long-range parts at arbitrary k and q **after**
interpolation." **SUBTRACT → interpolate remainder → ADD analytic LR back.**
Parallel construction: **Sjakste, Vast, Calandra, Mauri, PRB 92, 054307 (2015)**
(GaAs polar-optical Wannier interpolation, same split).

**Brunin, Miranda, Royo, Stengel, Verstraete, et al., PRL 125, 136601 (2020) /
arXiv:2002.00628** — the *cautionary* result: dipole-only subtraction is **not
enough**. The next order in q (the **dynamical quadrupole**) is finite at `q→0` but
angular-discontinuous; if it is left inside `g^S`, Fourier interpolation produces
unphysical oscillations near Γ (their Fig. 2, "FI" vs "FI+Q"). Fix = extend `g^L` to
the quadrupole term (their Eq 3, adds `(q_β+G_β)(q_γ+G_γ)(Z* v^Hxc + ½Q^βγ)` in the
numerator) so the interpolated remainder is truly smooth. **Lesson for LORRAX: the
subtractive split is only as good as the analytic LR model; if the removed channel
does not capture *all* the nonanalytic structure, the "smooth body" is still
non-interpolable.** This is the e-ph mirror of the ζ-rotation caveat in §3.

**2D modifications — Sohier, Calandra, Mauri, Nano Lett. 17, 3758 (2017) /
arXiv:1612.07191** (and Sohier–Gibertini–Marzari, mobility framework): in 2D the
bare Coulomb changes power, `v(q) = 2π/(|q| ε_2D(q))` with `ε_2D(q)=ε_ext+r_eff|q|`,
so the LO/head is **linear in `|q|`** with a *finite but direction-discontinuous
slope* at `q→0` (nonanalytic first derivative, not a `1/q²` pole). The analytic
re-add must use the 2D-truncated Coulomb, not the 3D `1/q²`.

### 9.3 ISDF/THC across momentum — two conventions (target 3)

Direct precedent for (not-)interpolating the density-fitting vectors themselves:

- **q-independent auxiliary basis (interpolation trivial by construction).** Lee &
  Reichman, "Even Faster Exact Exchange for Solids via THC," JCTC (2023) /
  arXiv:2304.05505; and the k-point RPA-THC with a *momentum-dependent auxiliary
  basis* (JCTC 2023, doi:10.1021/acs.jctc.3c00615). In these, the interpolation
  vectors `ζ_μ(r)` are **cell-periodic and k-independent**; the entire `q=k′−k`
  dependence is folded into Bloch **phase factors** `e^{iq·r}` and the `M`-matrices.
  Interpolating/reusing ζ across q is a non-issue there *because ζ was fit to the
  union span of all orbital pairs at once* (larger rank, not tuned per q).
- **per-q least-squares ζ_q (LORRAX's convention).** LORRAX fits `C_q ζ_q = Z_q` per
  momentum to the *specific* pair-density span at that q (manual 5.3). That span
  rotates with q — §3 measured ~40% median magnitude variation between adjacent q
  and 90–340% Frobenius error predicting `V_q` from a Γ "master ζ". So LORRAX sits
  in the convention where ζ is **genuinely q-dependent and not directly
  interpolable**; the H1/H2 "cell-periodic master ζ" hypothesis that the k-THC
  schemes rely on is exactly what §3 rejected for LORRAX's per-q fit.

Implication: a precedent for reusing ISDF vectors across q **exists**, but only in
the global-auxiliary-basis convention. Two honest routes for LORRAX: (a) recompute
`g0(Q)`/`ζ_Q` per Q (compute-don't-interpolate, §4 option 1 — cheap for the `g0`
G=0 slice), or (b) migrate to a union-span/global ζ if dense-Q dispersion makes
per-Q refits dominate (a larger-rank, separate design). AFQMC-ISDF (Malone, Lee,
Morales, arXiv:1810.00284) and complex-k-means ISDF (arXiv:2208.07731) are further
ISDF-with-k references but do not interpolate ζ across q either.

### 9.4 Mapping table — literature object ↔ LORRAX object

| literature object | source | LORRAX object |
|---|---|---|
| exciton dipole `P_M` (Haber Eq 27) | 2308.03012 | `g0_μ = ζ(q,μ,G=0)` — the head channel vector (`head_correction`, `tagged_arrays.py:94`) |
| rank-1 dipolar head `K^{X,Dip}=Σ_G P*·(Q+G) P·(Q+G)/|Q+G|²` (Eq 27) | 2308.03012 | rank-1 head `V_Q^LR[μν] = Σ_G ζ̃*_μ v(Q+G) ζ̃_ν` restricted to the g0 channel = `v(Q)·conj(g0)⊗g0` (`apply_q0_head_rank1`) |
| short-range `H^SR(R̄)` (Wannier-interpolated, Eq 32/20) | 2308.03012 | `V_Q^SR[μν] = V_qmunu` (already G=0-zeroed) — the smooth body to interpolate |
| subtract `K̄^{X,Dip}(0)` before FT (Eq 33) | 2308.03012 | subtract `V_Q^LR` before storing/interpolating (= `coulomb_sr_lr.md` split, α→∞ limit) |
| e-ph `g = g_S + g_L`, `g_L ∝ (q·Z*·e)/(q·ε·q)` (Verdi Eq 2,4) | 1510.06373 | `v(Q+G)=v_SR+v_LR` per-G split; `V_Q=V_SR+V_LR` (`coulomb_sr_lr.md` Gaussian split) |
| SUBTRACT-interpolate-ADD recipe (Verdi step ii–iv) | 1510.06373 | interpolate `V_SR/W_SR`, re-add `v_lr_at_qG` analytically (`range_sep.readd_lr_direct`) |
| quadrupole term needed for smooth `g_S` (Brunin Eq 3) | 2002.00628 | the ζ-rotation residual (§3): analytic head removal alone leaves ~40% coarse-q body variation ⇒ body still not perfectly interpolable |
| 2D `v=2π/(|q|ε_2D)`, linear-`|q|` head (Sohier) | 1612.07191 | slab-truncated Coulomb `f_2D` envelope (`slab_2d.py:29-37`); 2D head `~2π/|Q|` |
| 2D exchange head `A|Q| + A|Q|e^{−i2θ}` winding-2 (Qiu Eq 9,10) | 1507.03336 | directional `g0(Q̂)`-carried head; `S_cart` anisotropic generator thrown away today (`w_head_wings_interp.md`) |
| k-THC q-independent ζ_μ(r), q in phases | 2304.05505 | the H1/H2 "master ζ" hypothesis §3 REJECTED for LORRAX's per-q ζ_q |

### 9.5 2D nonanalytic q̂ — implication for the MoS2 fixtures

Qiu, Cao, Louie, PRL 115, 176801 (2015) / arXiv:1507.03336 is the load-bearing 2D
reference. Their exchange kernel (Eq 2) `⟨vckQ|K^x|v'c'k'Q⟩=Σ_G M_cv v(Q+G)
M*_{c'v'}` with 2D `v(Q+G)=2πe²/|Q+G|` (Eq 4) gives, on expansion (Eqs 9–10):

```
intravalley:  ⟨S^K_Q|K^x|S^K_Q⟩   = C + A|Q| + βQ²
intervalley:  ⟨S^K_Q|K^x|S^{K'}_Q⟩ = A|Q| e^{−i2θ} + β'Q²        (θ = polar angle of Q)
```

Two consequences for MoS2 (a slab fixture, `sys_dim=2`):

1. **The head is nonanalytic — a `|Q|` cusp, not a `1/|Q|²` pole — and the analytic
   re-add must use the 2D-truncated Coulomb `2π/|Q|·f_2D`, not the 3D `1/|Q|²`.**
   LORRAX already has `f_2D` (`slab_2d.py`), and `coulomb_sr_lr.md`'s Gaussian split
   keeps the dimensional envelope as an outer factor, so `v_SR+v_LR=v` holds for the
   slab. The rank-1 factorization inherits this for free **iff** the head's `v(Q)` is
   evaluated through `get_kernel(sys_dim).v_qG`, not a hardwired 3D form.
2. **The head is direction-dependent with winding number 2** (`e^{−i2θ}`). A single
   isotropic scalar (today's `wcoul0`/`vhead` in `apply_q0_head_rank1`) **averages
   this away** — correct only at the single coarse `Q=0` point where Baldereschi–
   Tosatti makes the direction average out (which is *why* the isotropic head passed
   the coarse Si/MoS2 gates, `w_head_wings_interp.md`). Once Q is refined toward 0
   — exactly the finite-Q / fine-grid regime this design targets — the isotropic
   scalar is wrong. The rank-1 head `g0(Q)⊗g0*(Q)·v(Q)` carries the winding-2
   angular structure **naturally, provided `g0(Q)` is the Q̂-dependent G=0 projection
   (the transition-dipole orientation), not a frozen vector.** This is the same point
   `w_head_wings_interp.md` makes about the discarded `S_cart` anisotropic generator:
   for MoS2 finite-Q, `g0` must rotate with Q̂.

### 9.6 Concrete recommended scheme

**For `V_Q` (bare exchange tile), arbitrary Q — rank-1-head factorization
(= subtract-analytic-LR with LORRAX's known rank-1 head):**

1. Persist the smooth body `V_Q^SR[μν] = V_Q[μν] − v(Q)·conj(g0(Q))⊗g0(Q)` — **this
   is the already-stored `V_qmunu` (G=0 zeroed)**; no new production of the body is
   needed, only the recognition that the stored G=0-zeroed tile *is* `V^SR`.
2. Persist/carry the head vector `g0(Q)=ζ̃(Q,μ,G=0)`.
3. Interpolate `V_Q^SR` across the fine grid (uniform-refinement FFT of the body, or
   the dcc/dvv interpolation of the wfn design — its choice, not this note's).
4. Reassemble at target `Q_fi`:
   `V_{Q_fi} = interp(V^SR)(Q_fi) + v(Q_fi)·conj(g0(Q_fi))⊗g0(Q_fi)`,
   with `v(Q_fi)` the **analytic** `get_kernel(sys_dim).v_qG` (3D `8π/|Q|²`; 2D
   `2π/|Q|·f_2D`), and `g0(Q_fi)` the Q̂-dependent G=0 projection.
5. **Do NOT** multiply the whole tile by `|Q|²` and divide back (§9.0). Do the
   removal/re-add on the **isolated rank-1 channel only**.

Caveat carried from §3 and mirrored by Brunin §9.2: `g0(Q)` and the body inherit the
ζ-rotation. Two sub-cases, gate them:
- If the **G=0 slice `g0(Q)` is smooth** across the coarse grid (an open, cheap-to-
  test question — the G=0 projection may be far smoother than full `ζ_q`, whose §3
  rotation is dominated by high-G components), interpolate it too.
- If `g0(Q)` is **not** smooth, **compute-don't-interpolate the head vector**: one
  G=0 projection of an htransform-fed per-Q ζ (or dipole) refit per Q — cheap, exact,
  no interpolation error (§4 option 1 specialized to the single G=0 row).

**For the exchange kernel at arbitrary Q (finite-Q exciton):** same rank-1
factorization on `V_Q`, plus the **2D directional head** of §9.5 (Q̂-dependent
`g0(Q̂)`, 2D Coulomb `2π/|Q|·f_2D`, winding-2 preserved). For MoS2 the honest first
cut is **per-Q ζ refit of the body too** (§4 option 1) until the in-flight
ingredient-interpolation *falloff study* (which quantifies how fast the body/`g0`
coarse-q variation decays — the §3 40% number is one datum of it; **do not
duplicate that study here**) shows the body is smooth enough to interpolate at the
target fine-grid density. The rank-1 factorization is the *correct container* either
way: it cleanly separates the divergence (analytic, exact) from the basis rotation
(the residual interpolation error), so the falloff study's verdict decides only
*how* the body is produced (interpolate vs refit), never whether the head is right.

**Relationship to the existing designs:** the rank-1 factorization **is**
`coulomb_sr_lr.md`'s Gaussian SR/LR split in the `α→∞` / G=0-only limit (all singular
weight on the head channel), ISDF-compressed. `coulomb_sr_lr.md`'s finite-α Gaussian
split is the *smooth generalization* that also de-weights near-head G≠0 terms; it is
the better choice if the body FFT-interpolation needs extra smoothness. Either way
the seam is the same `v_qG_split`/`v_lr_at_qG` + the existing `apply_q0_head_rank1`
made per-fine-Q. The anisotropic-head machinery for the winding-2 term is
`w_head_wings_interp.md`'s promoted `head_wing` module with the Q̂-directional
`W_head`.

### 9.7 Load-bearing citations (returned to main)

1. Haber, Qiu, da Jornada, Neaton, PRB 108, 205109 (2023), arXiv:2308.03012 — the
   exciton-Wannier SR/LR split; `K^{X,Dip}` rank-1 in the dipole, subtracted and
   re-added analytically (Eqs 27–34, 43). **The direct proof the rank-1-head
   factorization is the right transfer.**
2. Verdi & Giustino, PRL 115, 176401 (2015), arXiv:1510.06373 — polar e-ph
   `g=g_S+g_L`, explicit SUBTRACT→interpolate→ADD recipe (Eqs 2, 4). The
   subtractive convention.
3. Qiu, Cao, Louie, PRL 115, 176801 (2015), arXiv:1507.03336 — 2D exchange head
   `A|Q|` + winding-2 `e^{−i2θ}` nonanalyticity (Eqs 2, 4, 9, 10). Governs the MoS2
   fixtures' directional head.
4. Brunin et al., PRL 125, 136601 (2020), arXiv:2002.00628 — dipole-only subtraction
   leaves a non-interpolable remainder; quadrupole (next order in q) needed (Eq 3).
   The e-ph mirror of the ζ-rotation caveat: subtraction is only as good as the
   analytic LR model.
5. Lee & Reichman, JCTC (2023), arXiv:2304.05505 (+ k-point RPA-THC,
   doi:10.1021/acs.jctc.3c00615) — ISDF/THC across q with a q-**independent**
   auxiliary basis; the precedent-and-contrast for LORRAX's q-dependent per-q ζ.

### 9.8 OWNER RULING (2026-07-17) — g0 winding kills direct head-vector interpolation; finite-α split promoted

§9.6's hedge ("if g0(Q) is smooth … interpolate it too") is resolved in the
negative, by the owner's argument: `g0(Q) = ζ̃(Q, G=0)` **winds across the BZ**
— the "G=0" label is not periodic (at the zone boundary the G=0 channel at Q
maps to a different G-channel at the equivalent Q+G point), so componentwise
interpolation of `g0(Q)` chases a multivalued object, on top of the 2D
winding-2 `e^{−i2θ}` of §9.5. Direct `g0` interpolation is REJECTED.

Consequence: the analytic LR channel must be **finite-range**, spanning a
decent shell of small `|Q+G|` — the finite-α Gaussian split of
`coulomb_sr_lr.md`:

    v_LR(Q+G) = v(Q+G) · exp(−|Q+G|²/(4α²))     (summed over ALL G — periodic in Q)
    v_SR(Q+G) = v(Q+G) − v_LR(Q+G)              (bounded, smooth)

`Σ_G ζ̃* v_LR ζ̃` is evaluated analytically/exactly at each target Q (the
divergence and the small-G winding both live here and are handled in closed
form); the SR tile `Σ_G ζ̃* v_SR ζ̃` is the interpolable object (subject to the
ζ-rotation falloff study). The α→∞ / G=0-only rank-1 form of §9.6 is demoted
to what it actually is: the single-coarse-point Q=0 special case in production
today. α selection policy remains the open question flagged in
`coulomb_sr_lr.md` (c·Δk default vs exposed knob).
