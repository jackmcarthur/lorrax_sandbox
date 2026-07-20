<!-- Superseded for HANDOFF purposes by ARBITRARY_Q_PRIMER.md (self-contained, fresh-agent-ready). This doc keeps living: the "Ingredient-interpolation falloff study" results section still lands HERE first, then folds into PRIMER PART III §III.4. -->

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

### 3.5 Ingredient-interpolation falloff study — VERDICT WITH NUMBERS

The reserved landing (see doc header). §3.2 killed the *master-ζ* shortcut
(interpolate `V_Q` from one Γ object). This asks the follow-up the owner scheduled:
does interpolating the **ingredients** across the coarse grid rescue arbitrary-q
`V_Q`? The Gram matrix `C_q[μν] = Σ_k P*_{k−q,μ} P_{k,ν}` (fixed centroid basis, so
consistently gauged across q) and `Z_q = C_q ζ_q` both carry the C_R falloff, so
they should interpolate where a single master ζ cannot. Fixtures: MoS2 3×3 / 4×4
(`00_mos2_3x3_cohsex/05...`, `01_mos2_4x4_cohsex_gnppm/00...`), Si 4×4×4. Scripts +
logs under `runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/`.

**(1) C_R falls off — owner's premise is TRUE.** Per-shell `‖C_R‖_F` normalised to
R=0, Green's-function-like decay to a `~1e-3–1e-4` floor:

| |R| (Bohr) | MoS2 3×3 | MoS2 4×4 | Si 4×4×4 |
|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 |
| ~6–7 | 2.3e-2 | 2.0e-2 | 2.9e-1 |
| ~10 | 6.7e-4 | 3.7e-4 | 1.7e-2 |
| ~12–15 | — | 1.7e-4 | 5.3e-2 → 5.2e-4 |

Reach is material-dependent: MoS2 (2D) dead by ~10 Bohr; Si (3D) only by ~14 Bohr.

**(2) The ingredients DO interpolate** — far better than master-ζ (§3.2C, 90–340%).
Leave-one-out Fourier interp of `C_q` (drop one q, rebuild from the rest via the
R-stencil), median rel-Frobenius:

| | on-grid loo (nR≥7) | off-grid midpoint (coarse→fine truth) |
|---|---|---|
| MoS2 4×4 | **1.3e-3** | **4.0e-2** (2×2 → 4×4 midpoints) |
| Si 4×4×4 | 3.3e-1 | 7.2e-1 (2×2×2 → 56 midpoints) |

MoS2 sub-percent on-grid, ~4% off-grid. Si is poor — the 4×4×4 coarse grid does not
resolve the slower 3D falloff (needs a denser grid). So far this *supports* the
ingredient route for 2D.

**(3) But V_Q reconstruction is defeated by the C⁻¹ solve.** `ζ_q = C_q⁻¹ Z_q` and
`cond(C_q) ~ 1e7 (Γ) – 1e9`. The sub-percent ingredient-interp residual is amplified
past 100%. Leave-one-out at target q, MoS2 3×3, nR=7, sweeping the solve
regularisation (‖ΔV‖_F/‖V‖ tile error, and the physical scalar `d*V_q d` with
`d ∈ range(C_q)`):

| solve | tile med | tile max | phys med | phys max |
|---|---|---|---|---|
| raw | 3.7e6 | 6.7e7 | 7.4e4 | 4.9e5 |
| rankcut 1e-8 | 3.7e6 | 8.4e6 | 7.3e4 | 2.0e5 |
| rankcut 1e-6 | 1.2e4 | 1.8e4 | 1.5e3 | 4.5e3 |
| rankcut 1e-4 | 1.1e1 | 2.5e1 | 2.1e1 | 7.3e1 |
| rankcut 1e-2 | **1.00** | 1.00 | **0.89** | 1.70 |

There is **no regularisation window**: light regularisation lets the conditioning
blow the answer up (10³–10⁶×); the only λ aggressive enough to tame it (rankcut 1e-2,
keeping ~top modes) has already thrown away the signal → ~100% error. The **physical
observable is not protected** — `d*V_q d` tracks the tile, refuting a gauge-artifact
escape. **Density does not rescue it:** the 6×6 grid at rankcut 1e-6 gives V med
8.2e2 / 1.8e3 / 1.2e3 at nR=4/7/13 (error grows, not shrinks, with more R-vectors).

**(4) Mechanism — the falloff does not transfer to ζ.** DFT the fit itself:
`ζ_R = (1/nq) Σ_q e^{2πi q·R} ζ_q(μ,r)`. Unlike C_R, **ζ_R does not fall off**
(MoS2 3×3, `|ζ_R|/max`): 1.00 → 0.82 → 0.65 out to the largest |R| — nearly flat.
The `C⁻¹` in `ζ = C⁻¹Z` **de-localizes** ζ in R (inverse of a short-ranged operator
is long-ranged). Interpolating ζ directly (skip the solve) therefore also fails, and
*worsens* with more R-vectors as it tries to resolve a non-existent decay: nR=4/7/8
→ phys 0.17 / 1.34 / 4.87. This is the single root cause of both failures — the
master-ζ shortcut (§3.2C) and ingredient-interp+solve here.

**(4b) Final 6×6 split (landed post-write-up, confirming):** separating the two
ingredients at nR=19 shows **Z-interpolation is the dominant error source**
(V med 2.4e5 via interp-Z vs 878 via interp-C-only) — consistent with the
mechanism: `Z_q = C_q ζ_q` inherits ζ's non-compactness, so BOTH paths into the
solve carry the delocalized object. The aggressive-rank-cut sweep at 6×6 was
reaped at end-of-allocation; it was purely confirmatory (the 3×3 sweep already
established no-window, and the mechanism is density-independent).

**Verdict.** The ingredient-interpolation middle path is **not viable at accessible
grid densities.** `C_R`'s falloff is real but `C_q` is the wrong object to
interpolate — the object you must produce, `ζ_q` (hence `V_Q`), does not inherit the
falloff, and the `C⁻¹` that produces it is both ill-conditioned and R-delocalizing,
with no regularisation window and no gauge escape. This **confirms §3.4 / §4**: the
per-Q ζ refit (§4 option 1, "compute-don't-interpolate") remains the only route with
no uncontrolled error. It does **not** kill §4 option 2 (SR/LR interpolation of the
*smoothed* `V^SR_Q`, a divergence-removed potential object, not ζ) — but it removes
any hope of a cheaper ζ-side or `C⁻¹Z`-side shortcut: route arbitrary-q exchange
through a per-Q refit or the SR/LR-smoothed potential, never through interpolated ζ.

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
Functions," PRB 108, 125118 (2023) / arXiv:2308.03012.** This is the exciton
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

1. Haber, Qiu, da Jornada, Neaton, PRB 108, 125118 (2023), arXiv:2308.03012 — the
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

---

## 10. External-response prototype campaign (2026-07-17) — APPENDED: §3.5 re-based under the physical metric; frame-transport counterproposal killed

Three parallel prototypes tested the ARBITRARY_Q_PRIMER_RESPONSE counterproposal
(zeta as ill-conditioned dual basis; locality in whitened/half-inverse objects;
smooth BZ-periodic frame + parallel transport + G-channel sewing) on the MoS2
3×3 (+6×6) fixtures, adjudicated under the OWNER-GOVERNING metric (2026-07-17
pushback): physical pair-amplitude contractions — gap-window exchange block
`B = M^H V_Q M` (81 gap-window rows) and TDA exciton swap shifts — NOT tile
Frobenius. Full synthesis:
`runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/CAMPAIGN_REPORT.md`.

| construction | null test | phys on-grid (B relF med) | verdict |
|---|---|---|---|
| C1 target-frame transported V^SR interp | gates only (chain aborted at a fixture-trap gate) | not landed | INCOMPLETE — conventions/gates delivered (production disk-match 1.9e-15) |
| C2 global periodic frame + four-tails + transported-Phi interp | PASS 4.4e-14 | 0.96 (exciton 36 meV) | **NEGATIVE — clean kill** by the response's own sec-10C criterion |
| C3 rank-r solve on interpolated C/Z (§3.5 re-base) | PASS 6.6e-13 | 1.14–1.19 in §3.5's own q-labeling (exciton 18 meV) | NEGATIVE as scheme; re-based bar delivered; INTERP rows superseded (wrap trap) |
| derived: same §3.5 ladder, production BGW-wrapped labeling | inherits C2 PASS | **rankcut 1e-4: 4.7e-3 (max 3.2e-2)** | **OPTIMISTIC — surviving candidate, on-grid; off-grid pending** |

**Supersession notice for this document (measured, logs on disk):**

1. **§3.5(3) "no regularisation window" and §3.5(4b) "Z-interp dominates" FALL**
   under the physical metric in the production q-labeling. Two compounded
   artifacts produced the old verdict: (i) tile-Frobenius/random-`d` metrics are
   junk-weighted — truncating TRUE ingredients at κ1e6 destroys 90% of tile
   Frobenius yet moves B by 7.6e-4 and excitons by 0.01 meV (the tail is
   physically inert; the full-rank tile is ~100% junk); (ii) the §3.5 harness
   used the unwrapped `mf_header/rk` while the stored zeta spheres are
   BGW-wrapped, scrambling 5/9 training fields with a spurious `e^{iG0·r}` —
   worth **155×** on the physical ladder (A/B same q0/solve/truth: 4.5e-3
   wrapped vs 0.70 unwrapped; KNOWN_SANDBOX_ERRORS 2026-07-17). Corrected,
   rankcut ~1e-4 ingredient interpolation delivers **0.47% median / 3.2% max**
   on-grid LOO (exciton ≤5.4 meV at the 1e-2 rung) while the tile stays "100%
   wrong" — the owner's few-percent scenario is REAL on-grid. Still standing
   from §3.5: the conditioning-dominated rows (raw/rankcut ≤1e-6 fail in every
   convention), the C_R-falloff premise, §3.2's master-zeta kill, and the
   zeta-direct rejection (7% corrected — better, still 15× off the ladder).
2. **The counterproposal's mechanism is dead by its own falsification
   criterion:** transported whitened tails are ROUGHER than raw zeta (Phi~_R
   1.89/1.60 at 3×3 shells vs raw 0.39/0.16 vs C_R 2.3e-2/6.7e-4; 6×6
   replicates to 26 Bohr), adjacent-q whitened subspaces sit at the
   random-subspace floor (0.098/0.054 at 3×3/6×6), holonomy at the random
   ceiling, exact sewing/gauge/densification change nothing. The smoothness
   lives in the frame-free quadratic ingredients (C_q, Z_q), not in any frame,
   section, or half-inverse object; the winning scheme never interpolates a
   frame (rankcut solve in the target's own frame).
3. **Ranking update to §3.5's verdict / PRIMER §III.4:** per-Q ζ refit remains
   the production default, but "never route arbitrary-q exchange through
   `C⁻¹Z` on interpolated inputs" is RELAXED to: rankcut-regularized ingredient
   interpolation (wrapped labeling, physical-metric-validated) is a measured
   few-percent ON-GRID fallback; the decisive 3×3-subgrid → 6×6-complement
   off-grid-with-truth test is pending, plus a Si 4×4×4 negative control
   (never ran). §3.5's off-grid factor (~30× for C_q) may make off-grid
   marginal — measure, don't assume.

## 11. Off-grid follow-up (2026-07-17) — APPENDED: owner redesign mid-execution; 6×6 LOO anchor + Γ→x̂ path smoothness PASS; midpoint ζ-refit truth pending; Si control fails off-grid as predicted

Follow-up to §10's "decisive missing measurement". All numbers grep-verified
from `runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/`
`offgrid_{mos2,si,path,path_htr}.log` (+ `offgrid_*_results.npz` /
`offgrid_prep.py`, `offgrid_mos2.py`, `offgrid_si.py`, `offgrid_path.py`,
`offgrid_path_htr.py`). Scheme under test throughout: the §10 surviving
candidate — plain rank-cut interpolation of C_q/Z_q in production BGW-wrapped
labeling, one truncated solve in the target's own frame, physical metrics
(gap-window `B = M^H V_Q M`, TDA exciton swap).

### 11.0 OWNER REDESIGN (supersedes §5a-item-1's test design; do not re-attempt)

The originally-specified 3×3-subgrid → 6×6-complement off-grid-with-truth
test is **withdrawn as a scheme verdict**: exciton/physical values shift
strongly with k-grid convergence between the two grid classes, so
interpolation error cannot be separated from convergence shift, and 3×3 is
judged never-useful as a coarse base grid. (The leg had already completed
when the redesign landed — its output is retained as §11.5 ingredient-level
appendix ONLY.) Replacement design, owner's words: "start with 6×6 and see
if it's possible to get smoother interpolation just between the lowest
eigenvalues at like k=0 and k=1/6 x̂ or something, even if it's harder to
compare those to ground truth" — i.e. (1) 6×6 on-grid LOO anchor, (2) a
Γ → (1/6,0,0) path with SMOOTHNESS of physical observables as the judge +
(where affordable) midpoint ground truth via per-Q ζ refit from htransform'd
wavefunctions on the same 6×6 data; (3) Si control deprioritized.

### 11.1 Gates and nulls (all fixtures)

Campaign chain inherited and extended; every gate green before any result:

| gate | MoS2 3×3 | MoS2 6×6 | Si 4×4×4 (work_old) |
|---|---|---|---|
| sphere max\|q+G\|²−cutoff (post-wrapfix) | 0.0 | 0.0 | 0.0 |
| makeVq vs disk V_qmunu, all q (max) | 1.30e-9 | 2.81e-9 | **1.42e-15** |
| X^H X == C_q | 8.7e-11 | 4.2e-11 | 1.1e-15 |
| solve-chain null (true C/Z, raw) | 4.90e-13 | 4.6–6.0e-10 | 3.9–4.0e-13 |
| rankcut-1e-4 floor on TRUE data | 3.58e-3 | 3.2–3.7e-3 | 0.9–1.4e-3 |
| trig-interp exactness (to a training pt) | 4.9e-16 | 4.9e-16 | 7.3e-16 |
| solve/to_sphere commutation | 2.16e-14 | — | — |
| harness continuity vs §10 logs | **rc1e-4 LOO B 4.699e-3/3.235e-2 + exc 5.444 meV @1e-2 — exact reproduction** | | |

Two new fixture traps found and recorded (KNOWN_SANDBOX_ERRORS 2026-07-17):
the **half-boundary wrap trap** (at q-components exactly 1/2 the stored
sphere center is per-q irregular — writer FP fuzz; 2/36 q on the 6×6
mislabeled by `rk−round(rk)`; fix = sphere-derived center, implemented in
`offgrid_prep.fix_sphere_wrap`) and the **Si 3D mini-BZ head** (disk V_qmunu
carries the MC-averaged v(q,G=0) at q≠0 per `build_v_head_miniBZ_avg_3d`;
without it makeVq-vs-disk fails at 9.6e-3 med). Also: §5a-item-2's fixture
pointer (work_sym/792) is IBZ-only zeta — the control ran on work_old
(full-BZ, n_μ=960).

### 11.2 Redesign item 1 — 6×6 on-grid LOO (35-train) + the missing 3×3 exciton

| rung | 6×6 LOO nR7 B med (max) | exc meV med (max) | 6×6 nR13 B med | 3×3 LOO B med | 3×3 exc meV med (max) |
|---|---|---|---|---|---|
| rankcut 1e-3 | 7.55e-3 (4.46e-2) | 0.050 (0.263) | 7.39e-3 | 8.18e-3 | 0.160 (0.608) |
| **rankcut 1e-4** | **3.73e-3 (3.63e-2)** | **0.020 (0.185)** | 3.54e-3 | 4.70e-3 | **0.110 (0.242)** |
| rankcut 1e-5 | 3.12e-3 (4.59e-2) | 0.019 (0.234) | 2.34e-3 | 6.62e-3 | 0.131 (0.270) |
| raw | 2.31e-1 (3.24) | 1.77 (6.4) | 6.48e-2 | 2.64e-1 | 14.7 (20.9) |

- The 3×3 headline gets its 6×6 counterpart: **0.37% median / 3.6% max B,
  excitons 0.020 meV** at the rankcut-1e-4 optimum. Ingredients: dC med
  1.0e-3 (nR7) / 4.8e-4 (nR13), dZ 3.3e-2. The §10 caveat "0.5% is 3×3-only"
  is closed — densifying the grid does not degrade the window; excitons
  improve ~5× (0.110 → 0.020 meV).
- The §6 missing number: **3×3 LOO exciton at rankcut 1e-4 = 0.110 meV med /
  0.242 max** (rc1e-2 rung reproduces the logged 5.444 meV exactly).
- Subgrid-LOO bridging row (8-train at 1/3-spacing ON the 6×6 data): rc1e-4
  B 4.92e-3/4.40e-2 — statistically the 3×3-fixture number: q-spacing, not
  fixture, controls the error.

### 11.3 Redesign item 2a — Γ→(1/6,0,0)x̂ path smoothness (`offgrid_path.py`)

9-point t-grid, training = all 36 on-grid q, fixed G-superset (2012 G, union
of path spheres, **G=0 excluded** — the slab head diverges ~1/|q_par| toward
Γ and is the analytic rank-1 channel; its interpolant coefficient
ζ̃(t, G=0) is tracked separately), fixed Γ gap-window probe (324 rows).

- **Rank-cut trajectories are smooth**: B̃ entries evolve in gentle monotone
  arcs (d²/range med 5–7e-2 across stencils nR7/13/36); the top eigenvalue
  curves are smooth in the interior; the head-channel coefficient's
  successive-t overlap stays ≥ 0.9899 at every rankcut rung.
- **raw is chaotic everywhere off-grid** (eigenvalue excursions 10²–10⁴,
  head-channel overlap dropping to 0.11) — the regularization window is not
  an on-grid artifact; it is what makes off-grid evaluation possible at all.
- **Exact-stencil chain null**: nR36 + raw solve reproduces the stored truth
  at both on-grid endpoints through the full off-grid contraction machinery
  at 2.4e-9 / 6.5e-8.
- Honest caveat (metric hygiene): with this FIXED-Γ-probe, G0-excluded
  contraction the rankcut truncation costs 3.3–4.5e-2 at t=0 and ~0.19 at
  t=1 vs full-rank truth — junk-inertness is a property of the physical
  pair-row metric (0.3% floors above), NOT of arbitrary probes. Consistent
  with §10's "tile is junk-weighted" adjudication; quote no fixed-probe
  number as a physical error.
- A visible first-step at Γ on the 2nd/4th eigenvalue (t=0 → 1/8) is the
  Γ-adjacent nonanalyticity of the G0-excluded 2D exchange body (winding
  structure survives the rank-1 head removal) + the Γ truncation offset —
  present identically at every rung, not an interpolation kink.

### 11.4 Redesign item 2b (partial) — physical swap-H(t) via htransform (`offgrid_path_htr.py`)

Production entry points (`bandstructure.htransform.initialize_wfns` +
`bse_setup.compute_wfns_fi`, kgrid_fi=24×24×1 — contains every k−q(t) for
t = j/4) on the same 6×6 dataset. Galerkin rank 1280 (= ns·n_μ ceiling).

- **Content finding (resolves the direction of the band-span trap):**
  htransform ψ(r_μ) at on-grid k matches **psi_full_y** at window-subspace
  cos med 0.9987 while matching raw-WFN centroids only at med 0.716 — the
  LORRAX loader itself produces the psi_full_y content class; the stored
  ζ/W0/restart and htransform are ONE consistent class, raw WFN.h5 is the
  outlier. Cross-content contraction is therefore NOT an issue inside this
  pipeline. Norm convention ratio med 1.0073 (applied once).
- htransform fidelity: on-grid ε err med 220 meV / max 837 meV (bands
  20–32) at 640 centroids — adequate for smoothness/anchors (D and H_dir
  errors are t-consistent and cancel in swaps), NOT yet for absolute
  dispersion physics.
- **Swap-H(t) lowest-4 trajectories** (true M(t), D(t), H_dir(t) from
  htransform; B(t) from the interp scheme, G0-excluded): smooth finite-Q
  exciton dispersion curves (range ~240 meV, max|d²| 38–62 meV ≈ curvature
  scale of a 5-point parabola, no kinks); **rung-sensitivity of the whole
  trajectory ≤ ~0.3 meV** across rankcut 1e-3/1e-4/1e-5.
- **Endpoint swap anchors (interp vs stored-fit truth, same M): 0.024–0.057
  meV**; endpoint B relF (G0-excluded convention) 4–5% at Γ, 1.4–1.6% at
  q=(1/6,0,0).
- **Still pending — the true (b):** the per-Q ζ REFIT truth at the
  MIDPOINTS needs the fit RHS on the full r-grid, i.e. full-grid htransform
  ψ (centroid-basis reconstruction beyond `compute_wfns_fi`'s ψ(r_μ)
  contract). Marked as the follow-up; until it lands the midpoint accuracy
  is bounded only by smoothness + endpoint anchors, not measured directly.

### 11.5 APPENDIX (no scheme conclusions — §11.0): the withdrawn 3×3-subgrid → 6×6-complement leg

Completed before the redesign landed; ingredient-level diagnostics retained:
interpolating from the 9-point subgrid OF THE 6×6 DATASET to the 27
complement q gave dC med 7.9e-4, dZ med 4.1e-2, and through the rankcut-1e-4
solve B med 3.88e-3 / max 8.07e-3 with exciton swaps 0.026/0.080 meV against
the same-dataset stored fits (raw: 3.5e-2/0.30; window shape as in §11.2).
Read per §11.0 as: the q-interpolation operator itself is benign at
1/3-spacing on 6×6-converged ingredients — NOT as an off-grid capability
claim for a 3×3-based production run.

### 11.6 Si 4×4×4 negative control (deprioritized; ran at zero marginal cost)

Full-BZ fixture work_old (n_μ=960, 64 q, bare-3D + mini-BZ head Coulomb).

| test | dC med | dZ med | B med best rung | B med rc1e-4 | raw |
|---|---|---|---|---|---|
| off-grid 2×2×2 → 56 complement (nR8 exact) | 0.67 | 0.69 | 0.194 (rc1e-2) | 9.4 | 4.6e4 |
| on-grid LOO 63-train, nR13 (R0+full fcc shell) | 0.136 | 0.137 | **2.87e-3 (rc1e-3)** | 3.06e-3 | 3.4e-2 |
| on-grid LOO, nR7 (broken 6-of-12 shell) | 0.457 | 0.449 | 8.5e-2 | 1.8e-1 | 3.1e-1 |

**The control PASSES (= the scheme fails where theory says it must):**
off-grid from a 2×2×2 base the ingredient error is ~67–69% (§3.5's 72%
reproduced) and B fails at EVERY rung with the window inverted (more
truncation = less bad) — error tracks the unresolved 3D C_R falloff, not the
solve. Two informative surprises: (i) on-grid LOO in 3D still achieves
MoS2-class 0.29% B despite 13.6% ingredient error — junk-inertness under the
physical metric extends to 3D on-grid; (ii) 3D R-stencils must take
COMPLETE coordination shells (nR7 = an argsort-tiebreak subset of the
12-vector fcc shell is 30–60× worse than nR13).

### 11.7 Verdict and standing defaults

1. **6×6 on-grid anchor: PASS** — 0.37% med / 3.6% max B, 0.020 meV
   excitons at rankcut 1e-4 (window 1e-3..1e-5 flat).
2. **Path smoothness (a): PASS** — rank-cut trajectories smooth Γ→x̂ with
   machine-level exact-stencil nulls and 0.02–0.06 meV endpoint swap
   anchors; raw chaotic; window persists off-grid.
3. **Midpoint ground truth (b): PENDING** — htransform ζ-refit at off-grid
   q needs full-grid ψ reconstruction; the single remaining measurement
   before any production adoption.
4. **Si control: behaves as predicted** (off-grid fail tracks falloff
   resolution; on-grid 3D fine with complete shells).
5. **Production default unchanged: per-Q ζ refit.** The scheme is a
   measured few-tenths-of-a-percent on-grid + smooth-and-anchored
   near-grid interpolant; it is not yet certified at generic off-grid Q.

## 12. Owner-spec-compliant tile-level schemes (2026-07-17/18) — APPENDED: no-r_tot constraint MET at 0.6% B / 0.05 meV; multipole counterproposal adjudicated; "frames are dead" narrowed by operator theory

**Governing constraint (owner).** The §10/§11 surviving candidate is REJECTED
for production because it stores/interpolates `Z_q` (`n_μ × r_tot`, ~17 GB at
MoS2 6×6). Spec: the interpolation machinery and the per-target-`Q` cost may
touch only SR/LR-split V-tile-level objects (`n_μ²`) or moment-class vectors
(`n_μ × small`); coarse-grid production fits may touch `Z` once (the existing
GW pipeline). Everything below obeys this: no `Z_r` array is ever formed.

Scripts/logs (all numbers grep-verified from disk, per the phantom-table
rule): `primer_response_study/tile_{prep,t1t2_mos2,smooth_filter,path,`
`wannier_pair}.py`, logs `tile_t1t2_{3x3,6x6}.log`,
`tile_smooth_filter.log`, `tile_path.log`, `tile_wannier_pair{,_nw2}.log`,
npz alongside. Fixtures: the §11 MoS2 3×3/6×6 (wrapped labels via
`offgrid_prep.fix_sphere_wrap`); metric: gap-window `B = M^H V_Q M` (3v×3c,
stored-fit truth) + TDA exciton swap; stencil nR7; truth and harness
continuity anchored to the campaign (below).

### 12.0 The constructions

All coarse-side objects are built once from stored data; interpolation is a
truncated-R Fourier stencil on them; per-target work is `n_μ²` AXPYs + an
analytic LR rebuild. Per-element math (full derivations in `tile_prep.py`
docstring):

- **Cleaning without Z.** rank-cutting the stored fit is a projection:
  `ζ_rc = R_r Λ_r^{-1} R_r^H Z = P ζ_stored` with `P = R_r R_r^H` from
  `eigh(C_q)` (`C_q` is ψ-level, `n_μ²`). On the tile:
  `V_c = conj(P) V_ref conj(P)` (`= Π V Π`, `Π = conj(P)` Hermitian).
  Gate: `Π V Π == makeVq(P ζ̃)` at 2.8e-14; the B-metric clean-floor
  reproduces the campaign's rankcut-on-TRUE-data floor **exactly**
  (3×3: 3.572e-3 vs logged 3.58e-3; 6×6 med 3.2–3.7e-3) — bit-level
  continuity with §10/§11 without ever forming `Z`. Smooth-filter variant
  (owner amendment): `S_ε = R g_ε(Λ) R^H`, `g_ε(λ)=λ²/(λ²+ε²)` (the Z-free
  form of the Tikhonov solve `f_ε(C)Z`, `f_ε(λ)=λ/(λ²+ε²)`).
- **T1 split.** `v_LR = v·exp(−K²/4α²)` (full slab `f_2D` envelope, only the
  true divergence zeroed), `v_SR = v·(−expm1)` per-G on the stored sphere;
  `V_SR_c = V_c − Π V_LR Π`; sphere-tail bound `exp(−cutoff/4α²)` ≤ 8e-17
  for α ≤ 0.45. Re-add at target over a FIXED global Miller superset 𝒢(α)
  (`min_{q∈BZ}|q+G|² ≤ 4α²ln(1/ε_LR)`, ε_LR=1e-8; 125/337/1007 G at
  α=0.2/0.3/0.45). Out-of-sphere (q,G) superset channels are zero in the
  stored representation (26 channels at α=0.45, worst Gaussian weight
  5.8e-17 — bounded, harmless).
- **T2 moments (slab-adapted, frame-free).** Winding cure: factor the FULL
  centroid phase, `ζ̃_μ(K) = e^{−iK·s_μ} M_μ(K)`; since `q_z = 0` on the
  coarse grid, `K_z = G_z` is an exact discrete channel (per-`G_z` moments —
  the pasted response's own §10 slab refinement), and only `K_∥` is
  Taylored: `M_μ(K_∥,G_z) ≈ m0_μ(G_z) − iK_∥·d_μ(G_z) − ½K_∥·Θ_μ(G_z)·K_∥`
  with minimal-image in-plane displacements (Cartesian via the exact dual
  `A = 2π(B^T)^{-1}`, gated vs `adot` at 1.4e-17). Storage
  `n_μ × n_{Gz} × 6` (27 G_z channels). Model evaluated at any `Q+G` with
  the analytic phase — no G-slot label anywhere, BZ-periodic by
  construction.
- **T2' channels (no Taylor).** `F_μ(q;G) = e^{+2πi(q+G)·s_μ} ζ̃_c,μ(q+G)`
  on 𝒢 — the exact form factor `M_μ(K)` sampled at `K=q+G`; the T2 moments
  are its in-plane Taylor coefficients. `n_μ × n_G_LR` per coarse q
  (124 MB total at 6×6/α=0.3 vs 17 GB for `Z_r`). Componentwise stencil
  interpolation at fixed Miller G is winding-safe (phase factored).
- **Assembly variants** (target q0, LOO weights w): **A** raw-tile
  `Σw V_ref`; **B** cleaned tile `Σw V_c`; **C** `Σw V_SR_c + Π_0 V_LR(q0)
  Π_0` from the target's own stored ζ̃ — the DIAGNOSTIC ceiling (breaks LOO
  on the LR channel only; `Π_0` needs only `C_q0` = ψ-level); **D**
  `Σw V_SR_c + V_MP[interp moments](q0)` (mixed split — quantifies the
  response's §4 subtract/re-add inconsistency warning); **E**
  `Σw [V_c − V_MP_own] + V_MP[interp moments](q0)` (same model both sides:
  exact-at-coarse for ANY model — the pasted approach's structure,
  frame-free; exact-stencil null 2.7e-15); **F** `Σw V_SR_c +
  V[interp F-channels](q0)` (T2 without Taylor; own-rebuild gate 1.8e-9).

### 12.1 Head-to-head (LOO over all coarse q, B med (max); exc meV med/max)

| scheme | interp objects (per q) | 3×3 B med | 6×6 B med (max) | 6×6 exc |
|---|---|---|---|---|
| §11 ingredient (rc1e-4) — REJECTED (r_tot) | C (n_μ²) + **Z (n_μ×r_tot)** | 4.70e-3 | **3.73e-3 (3.63e-2)** | 0.020/0.185 |
| A raw tile | V (n_μ²) | 5.04e-1 | 1.61e-1 (3.03e-1) | 0.52/1.88 |
| B cleaned tile, no split (rc1e-4) | V_c | 4.59e-1 | 1.53e-1 (2.69e-1) | 0.16/0.94 |
| C clean-SR + exact-LR α=0.3 (ceiling) | V_SR_c | 1.89e-2 | 5.22e-3 (4.42e-2) | 0.037/0.181 |
| C at α=0.45 / 0.6 | V_SR_c | 7.9e-3 / 5.2e-3 | 3.69e-3 / **3.56e-3 (3.63e-2)** | — / 0.021/0.088 |
| **F clean-SR + channel-LR α=0.3 (honest, spec-compliant)** | V_SR_c + F (n_μ×337) | 1.87e-2 | **6.23e-3 (4.57e-2)** | 0.046/0.212 |
| F at α=0.45 | + F (n_μ×1007) | 1.36e-2 | 5.72e-3 (3.97e-2) | — |
| F, Tikhonov cleaning ε=1e-4 | same | — | **5.85e-3 (3.78e-2)** | 0.045/0.167 |
| D mixed moment-LR (α=0.3, o2) | + moments (n_μ×162) | 3.35e-2 | 4.32e-2 | 0.079/0.423 |
| E consistent moment model o0/o1/o2 | same | 2.8/5.1/6.0e-2 | 1.18/1.90/2.06e-2 | o2: 0.053/0.647 |
| W pair-level LR, projection gauge | + M̃ (npair×337) | — | 3.7e-1 (gauge-blocked, §12.4) | 1.6/8.0 |

Reading, in causal order:

1. **The parent's bounds sketch, scored.** (i) "raw tile fails at 5–10%":
   REFUTED downward — raw is worse (16% at 6×6, 50% at 3×3). (ii)
   "cleaned tile reaches ingredient level": TRUE for the full T1
   construction (clean + split): C sits at 3.6–5.2e-3 ≈ the ingredient
   3.73e-3; cleaning ALONE (B, 15%) does nothing for the physical metric —
   the tile's q-roughness is overwhelmingly the near-head Coulomb-weight
   variation, not the junk (junk is inert under B by §10, and neither hard
   nor Tikhonov cleaning rescues the un-split tile: Btik ≈ Bhard ≈ 0.15).
   The SPLIT is the load-bearing move; cleaning matters only in that the SR
   remainder then interpolates at the clean floor. (iii) "moments of
   cleaned ζ smooth where raw g0 is not": PARTIAL — see 12.2.
2. **The spec is MET at 0.59–0.62% B / 0.05 meV excitons** (F at α=0.3–0.45,
   hard or Tik cleaning), within ~1.6× of the r_tot-carrying ingredient
   scheme (0.37% / 0.020 meV) at ~1/100 the per-target cost and ~1/70 the
   storage, with NO solve, NO eigh, and NO r_tot object at or after
   interpolation. α-ladder: C/F improve 0.2→0.45 and plateau 0.45→0.6
   (α ≈ 1.5–2× Δq); rc-window flat 1e-3..1e-5 (E rows and B rows) — same
   inert-window behaviour as §10.
3. **Γ→(1/6,0,0) path (tile_path.py, fixed-Γ probe, G0-excluded rows
   comparable to §11.3):** F trajectories smooth — entries d²/range med
   3.6e-2 (nR36) / 4.3e-2 (nR7), slightly better than the §11.3
   ingredient-scheme 5–7e-2; top-eig arc smooth+monotone; the eig-1 first
   step at Γ is §11.3's known G0-excluded winding nonanalyticity (present
   at every rung, not an interpolation kink). Exact-stencil chain null at
   the endpoints 1.6e-14 / 7.0e-10 (raw), and the t=1 anchor 0.19–0.26 on
   the split variants is the FIXED-Γ-PROBE truncation cost (§11.3 measured
   ~0.19 for the same reason), not scheme error.

### 12.2 T2 verdict — the winding cure works; literal moments do not

- Adjacent-q roughness (6×6, +x̂ pairs, rel diff med): raw g0 slot vector
  0.678 (max 1.96 — the winding object) → phase-factored cleaned
  F-channels **0.313** → cleaned m0 0.357; but dipole/quadrupole moments
  0.72/0.78 — WORSE than the monopole. R-falloff mirrors it (m0
  1→0.40→0.11 vs C_R 1→4.2e-2→7.8e-4; d does not decay at shell 1).
- Moment-model fidelity vs the exact cleaned LR tile: 58–70% relF, and
  orders HURT (o1/o2 worse than o0) — the ISDF ζ envelopes are not
  compact, so literal polynomial-weighted moments are dominated by
  delocalized tails (the response's own §9 warning about literal real-space
  moments, realized). Consequently the Brunin hierarchy INVERTS in the
  assembled scheme: E o0 (1.18e-2) beats o1/o2 (1.9/2.1e-2).
- The Brunin smoothness criterion itself registers only weakly and in the
  remainder alone: interpolated-remainder trajectories along Γ→x̂ smooth
  out mildly with quadrupole subtraction (entries d²/range max
  0.143→0.098, eigs 0.114→0.081 at nR36) — the DIRECTION is right, but the
  re-add model error (58–70%) swamps the gain: total-E is
  model-accuracy-limited, not smoothness-limited.
- The pure-3D z-Taylor is DOA for slabs, quantified: at α=0.3 the |G_z|=1
  channels carry 19% of the LR weight with 73% 3D-Taylor error (|G_z|=2:
  2.8% weight, 460% error). Per-G_z moments (or channels) are MANDATORY —
  the pasted response's §10 caveat confirmed with numbers.
- **The working object is the phase-factored exact channel `F`, not its
  Taylor compression.** F ≈ C at α=0.3 (6.2e-3 vs 5.2e-3) — once the
  winding is carried analytically, the channels interpolate essentially at
  the exact-LR ceiling; every Taylor/integral-moment compression of them
  loses an order of magnitude. And the compression buys almost nothing:
  moments are n_μ×162 vs F's n_μ×337 at α=0.3.

### 12.3 Operator-theory checks (owner amendment 1) — "frames are dead" NARROWED

`tile_smooth_filter.log`, 6×6, all 36 adjacent +x̂ pairs:

- **Check A (matrix-function continuity):** ‖ΔC‖/‖C‖ = 3.18e-2;
  Tikhonov-cleaning-weight continuity ‖Δg_ε(C)‖/‖ΔC‖ ≈ 7e2–1e3 across
  ε_rel 1e-3..1e-6 — i.e. ε_abs·ratio = 0.006–0.04, a factor 16–100 BELOW
  the analytic Lipschitz bound (max|g′_ε| ≈ 0.65/ε): the filtered
  operators are far smoother than worst-case. Hard-cut projector distance
  ‖ΔP_r‖_F/√2r = 0.21–0.22 — NOT at the random floor (0.65–0.85) but a
  persistent ~22% edge-rotation, exactly the Davis–Kahan picture: modes
  within ‖ΔC‖ of the cut rotate freely, the bulk does not. Cleaned-TILE
  smoothness: ‖ΔV_c‖/‖ΔV_ref‖ = 0.015 (hard) / 0.011 (Tik) — the cleaned
  tile is ~70–90× smoother in q than the raw tile, Tik mildly smoother
  than hard, and downstream Ftik ≥ Fhard (5.85e-3/3.78e-2 vs
  6.23e-3/4.57e-2) — the amendment's prediction confirmed in sign,
  small in size.
- **Check B (re-audit of the C2 "random floor"):** plain top-m
  eigen-subspace principal cosines between adjacent q (no whitening, no
  transport, no ζ — C_q is ψ-level and label-free): cos med ≈ 1.0000 and
  affinity 0.975–0.9998 for EVERY m ∈ {10..480}, vs random floors
  0.125–0.87. Only the MINIMUM cosine degrades with m (0.9986 at m=10 →
  0.083 at m=480) — the cut-edge modes, DK-consistent (‖ΔC‖/gap grows 1.1
  → 5e6). **The C2 probe was not wrap-trapped and not buggy; it measured a
  different statistic** — whitened (S^{-1}-amplified) angles with band
  transport, whose median over the gapless tail is REQUIRED to sit at the
  floor by perturbation theory. The subspace geometry of the pair space is
  in fact q-smooth. §10's "frames are dead" is hereby narrowed to:
  **hard-cut/whitened EIGENFRAME objects are lawless on C_q's gapless
  spectrum, exactly as Davis–Kahan and the BBR gap-hypothesis require
  (Benzi–Boito–Razouk, SIAM Rev. 55, 3 (2013): decay/continuity bounds for
  f(H) need f analytic in a Bernstein ellipse clearing the spectrum — a
  step INSIDE a gapless spectrum has no theorem, an analytic filter of
  width ε does); smoothed spectral functions and whole-subspace/filtered
  objects are the licensed ones.** Do not re-attempt eigenvector-frame
  transport; smooth-filtered functionals of C_q are fine.

### 12.4 Wannier-theory spine (owner amendment 2) and the pair-frame test

**(a) Why the ingredients are analytic — the Wannier statement.** For an
isolated band group, an analytic periodic Bloch gauge exists (Panati,
Ann. Henri Poincaré 8, 995 (2007); Marzari–Vanderbilt) and yields
exponentially localized Wannier functions `w_nR`; BBR-class bounds give the
same decay for the gauge-invariant density matrix. The window density
matrices `P_k` entering §1's Grams are gauge-invariant, and
`C_R, Z_R ∝ |P_R|²` inherit exponential decay (the measured C_R falloff) ⇒
`C_q, Z_q` are analytic in q. The momentum-q PAIR SPACE is spanned by Bloch
sums of Wannier pair products `w*_n(r−R) w_m(r−R−ΔR)` — a canonical,
exponentially-localized, q-ANALYTIC frame for exactly the space ζ lives in.
**(b) What this licenses for ζ:** posed in Wannier-pair coordinates the fit
has analytic inputs AND an analytic frame; the q-roughness of ζ (and of
every ζ-linear centroid-frame object measured here: g0 0.68, m0 0.36,
F 0.31) is a property of the CENTROID POINT-VALUE DUAL coordinates — the
LSQ dual basis of an ill-conditioned frame — not of the physical content.
Any fixed ANALYTIC-filtered functional of C_q (12.3) and any frame-free
quadratic contraction (the tiles) is q-analytic; that is precisely what
the working schemes (§11 ingredient, T1 SR tiles, F under contraction)
exploit. **(c) The C2 diagnosis:** C2's "global smooth frame" was the
SPECTRAL frame of C_q — lawless on a gapless spectrum by 12.3 — where the
theory offers the WANNIER frame, analytic by theorem. That substitution is
exactly why C2 hit the floor with a correct implementation.
**(d) The constructive test, attempted and honestly blocked.**
`tile_wannier_pair.py` builds the cheapest standard smooth gauge
(Γ-anchored trial projection + Löwdin on the stored ψ-at-centroids; NO
htransform) on the gap-window legs and measures the pair-level LR channels
`M̃(q) = x̃_q ζ̃_c` in that gauge. Result: with the 3v/3c trio windows the
gauge fails its own nonsingularity diagnostic (smin/smax down to
0.006–0.016) — the trio windows SPLIT KRAMERS DOUBLETS (C3's flag; §I.2
lesson), so their subspace is k-discontinuous and no smooth gauge exists
even in principle. Kramers-clean 2v×2c windows FIX the valence gauge
completely (med 0.991, min 0.880) but the conduction window remains
entangled (med 0.452, min 0.056 — Γ trials do not span the K-point
conduction character; bottom-2 conduction is not an isolated group), and
the moving (conduction) leg dominates M̃, which stays rough (1.21 adjacent
diff) and the pair-LR variant W fails (0.37). **Verdict: not a refutation —
the fixture's centroid-sampled ψ + Γ-trial projection cannot construct the
conduction-side Wannier gauge; a proper test needs atom-centered trials on
full-grid ψ or genuine disentanglement.** Flagged per amendment item 5;
the theory's empirical support here is Check B (smooth subspaces) + the
tile/ingredient analyticity it predicts.

### 12.5 Adjudication of the pasted multipole approach (RESPONSE.md, final section)

The owner-pasted "effective covariant dipole/quadrupole tensors in the
parallel-transported half-whitened frame" approach, scored against the
campaign + this section's evidence:

1. **Its subtract-model / interpolate-remainder / re-add-model skeleton is
   CORRECT and now measured** — variant E implements it frame-free and is
   algebraically exact at coarse points (null 2.7e-15) for any model
   quality, exactly as the response §"scheme I would now favor" claims.
2. **Its frame is the killed one.** The tensors are defined in the
   half-whitened transported frame (`Φ_q = S R^H ζ`, `D → T_{Q←q}D`) — the
   C2-measured-dead machinery; 12.3 explains WHY it is dead (gapless
   spectral frame) and shows the content it wanted (localized, analytic
   frames) belongs to Wannier theory instead. As pasted: **UNWORKABLE.**
3. **Its literal multipole content fails frame-free too.** In the
   centroid basis with the winding phase factored — the most favorable
   frame-free reading — literal moments are a 58–70%-error model of the LR
   channel, orders invert Brunin (12.2), and the promised compression is
   marginal (n_μ×162 vs n_μ×337 for exact channels). Its own §9 warning
   (literal real-space moments vs form-factor Taylor coefficients) and §10
   caveat (slab needs G_z-resolved moments — confirmed: 73% 3D-Taylor
   error at 19% weight) are the operative failure modes. Its §4
   "fit-based" extraction (LSQ in K-space over the LR support) is the one
   untested variant that could close part of the moment↔channel gap; it
   can at best reach the F-channel numbers it would be fit to.
4. **What survives, and is now adopted:** the finite-α Gaussian window
   over a fixed reciprocal superset (its §6 — implemented, incl. the
   worst-case tolerance rule), the stable `expm1` SR evaluation and slab
   small-K series (§5 — implemented), consistency-of-LR-definitions (§4 —
   quantified: mixed D 4.3e-2 vs consistent E 2.1e-2 vs matched-channel F
   6.2e-3), α from a held-out plateau (§11 — observed), and "the LR term
   naturally has a different, extremely low-rank representation" (§3) —
   realized as the phase-factored exact channels F (and, once a real
   disentangled Wannier gauge exists, the pair-level `F_Q D F_Q^H` form).
   **Bottom line: dipole+quadrupole LR tensors — UNWORKABLE as pasted
   (frame dead, literal moments inaccurate, 3D form DOA for slabs);
   the surviving 20% of the idea is the subtractive container and the
   low-rank LR channel, which the F-scheme implements better with the
   Taylor summed to all orders.**

### 12.6 Standing verdict

1. **Owner spec (no r_tot, n_μ²-level everywhere): MET** at 6×6 on-grid
   LOO **0.59–0.62% median B / 3.8–4.6% max, excitons 0.045–0.046 meV med
   / 0.17–0.21 max** (F-composition, α 0.3–0.45, Tik or hard cleaning) —
   ~1.6× the ingredient scheme's error at ~1/100 the per-target cost, no
   solve at the target. The exact-LR ceiling (C, 0.36% at α=0.6) shows the
   SR-tile side is already AT ingredient level; the remaining gap is
   entirely the LR-channel interpolation.
2. Off-grid: path smoothness + endpoint anchors at §11.3 quality (entries
   d²/range 3.6–4.5e-2, better than the ingredient scheme's 5–7e-2).
   The §11 caveat is inherited unchanged: no off-grid capability number
   until an off-grid ground truth exists (per-Q ζ refit at midpoints —
   note the htransform route for it is owner-barred in this thread).
3. **Production default remains per-Q ζ refit.** The F-scheme replaces the
   §10/§11 ingredient scheme as the designated fallback/interpolation
   candidate (it dominates it on every axis the owner cares about:
   storage, per-target cost, spec compliance; accuracy within 1.6×).
4. Do not re-attempt: eigenvector-frame transport (12.3), literal moment
   compression of the LR channels (12.2), trio-window gauges that split
   Kramers doublets (12.4), pure-3D multipoles on slabs (12.2).

Citations for 12.3/12.4: M. Benzi, P. Boito, N. Razouk, "Decay properties
of spectral projectors with applications to electronic structure," SIAM
Review 55(1), 3-64 (2013) (archive/designs copy; Thm 8.1, Cor 8.6, §1157
gap-dependence remark); Davis-Kahan sin-theta (via BBR §presentation);
G. Panati, Ann. Henri Poincaré 8, 995 (2007); Marzari-Vanderbilt
RMP 84, 1419 (2012); Brunin et al., PRL 125, 136601 (2020); Haber et al.,
PRB 108, 125118 (2023).

## 13. Compact LR-channel representation (2026-07-17) — APPENDED: the K-ball fit program; F's n_μ×337-per-q block collapses to n_μ×26 GLOBAL coefficients at the exact-LR ceiling; M(K) is single-valued to ~1% in the Tikhonov gauge (the q-fiber was the hard-cut edge); literal-moment pinning refuted a second way

**Governing question (owner).** The F-scheme (12.6) carries the LR channel
as `n_μ × 337` explicit G-channels per coarse q. Wanted: a representation
that is "just simpler and more consistent system to system" — target
`≤ n_μ × (10–30)` coefficients, accuracy at the F level (6e-3 B; the
literal-moment 1.2e-2+ is the failure to beat), analytic at any Q.

Scripts/logs (all numbers grep-verified from disk):
`primer_response_study/lr_{prep,singlevalued,basis_ladder,fiber_source,`
`transfer,pin}.py`, logs `lr_singlevalued_{6x6,3x3}.log`,
`lr_basis_ladder_6x6{,_tik}.log`, `lr_fiber_source_6x6.log`,
`lr_transfer.log`, `lr_pin_6x6.log`, npz alongside. Fixtures, truth,
B/exciton metric, nR7 stencil, α=0.3, rc*=1e-4: unchanged from §12.

### 13.0 The reframing, and a structural fact about the samples

`F_μ(q;G) = e^{+iK·s_μ} ζ̃_c,μ(K)`, `K = q+G`, are scattered samples of a
would-be single function `M_μ(K)` on the LR ball
`|K| ≤ 2α√(ln 1/ε_LR)` (2.57 bohr⁻¹ at α=0.3). The §12.2 literal moments
are the K=0 Taylor of that function — they fail because Taylor departs
from the truth by |K| ~ α; the right object is a WEIGHTED LSQ FIT over
the whole ball (the pasted response's own untested §4 "fit-based
extraction"). Structural fact (gated: integrality 2e-9, duplicates 0):
with q on the N×N grid (q_z=0) and G in the fixed superset, the in-plane
sample points tile a REGULAR fine lattice of spacing |b|/N per exact
discrete `K_z = G_z` channel, with exactly ONE sample per K point — so
"is M single-valued?" cannot be a coincidence test; it decomposes into
seam parity, q-fiber, and plateau (13.1). Fit machinery (`lr_prep.py`):
per-G_z in-plane bases `Φ_b(K_∥)`, weight `w = v_LR(K)` (this makes the
LSQ objective exactly `‖ΔA‖²_F` of the tile factor `A = ζ̃√v_LR` — the
fit minimizes what the physical contraction sees); design matrix shared
across μ, so one normal solve per (G_z, LOO target) serves all rows;
per-q normal blocks make LOO refits honest and O(n_b²). The winding
phase is never fit through — it is re-applied analytically (T2'
convention). Per-G_z v_LR weight shares (6×6, α=0.3, summed ±):
0.421 / 0.466 / 0.090 / 0.020 / 0.0025 for |G_z| = 0..4.

### 13.1 Single-valuedness (experiment 1) — and the fiber's identity

`lr_singlevalued_{6x6,3x3}.log`, `lr_fiber_source_6x6.log`:

- **Seam parity.** Adjacent fine-lattice pairs (|ΔK| = |b|/N) that cross
  a BZ boundary (Miller label changes) vs pairs that do not: med rel diff
  ratio cross/same = **0.983** (6×6; 0.932 at 3×3) on the weight-relevant
  subset — statistically identical classes. NO residual seam beyond the
  analytically-carried phase: the winding is fully cured.
- **q-fiber (hard-cut gauge).** After a rich smooth fit (gto3×poly4 on
  |G_z|≤2, weighted rel resid 0.091), **65% of the residual power is
  coherent per-(q,G_z)** — 11.5× the white-residual dof floor (0.056);
  the per-q coherent component is 9.3–11.6% of |Y| at G_z=0. M is NOT a
  function of K alone at the 1% level in the hard gauge. At 3×3 the same
  fit reaches 0.018 with per-q coherence 0.1–0.4%.
- **The fiber is the hard cut.** Same rich fit on differently-cleaned
  channels (`lr_fiber_source`): hard rc=1e-4 wres 0.0911 / per-q
  coherent (G_z=0 med) 0.0928 → raw 0.0567 / 0.0640 → **Tikhonov ε=1e-4:
  0.0112 / 0.0097**. The q-fiber is overwhelmingly the Davis–Kahan
  cut-edge rotation of the rank-cut projector (12.3's ~22% edge
  rotation, seen here in the channels), plus dual-basis jitter that the
  smooth filter also tames. **In the Tikhonov gauge M(K) IS
  single-valued to ~1%** — the parent's hypothesis holds in exactly the
  gauge that 12.3's operator theory licenses. (The fiber is also
  physically inert: the hard-gauge ladder below reaches the same B floor
  through 32% tile infidelity.)

### 13.2 Basis ladder (experiment 2) — LOO over all 36 coarse q, 6×6

Budget specs (per-|G_z| in-plane poly degree, following the weight
shares): b16p = {2,1,0,0}, b26p = {3,2,0,0}, b45p = {4,3,1,0,0} for
|G_z| = 0,1,2,(3,4); |G_z| beyond spec dropped (exact-channel truncation
to |G_z|≤3 costs B med 3.4e-3 = the clean floor; ≤2 costs 4.6e-3).
Tikhonov gauge headline (hard-gauge in parens where instructive);
D = clean-SR interp + model re-add at target (the F-scheme's own
structure), E = consistent subtract/re-add; fid = own-fit
relF(V_model, Π V_LR Π) med:

| rung | coeffs/μ | fid | B med (max) | exc meV med/max |
|---|---|---|---|---|
| C exact-LR ceiling | — (target's own LR) | 0 | 5.40e-3 (3.87e-2) | 0.039/0.149 |
| F channel-interp (≡12.1 Tik row — continuity anchor) | 337 per q | — | 5.85e-3 (3.78e-2) | 0.045/0.167 |
| **D b26p global fit** | **26 global** | 7.4e-2 | **5.37e-3 (3.96e-2)** | **0.043/0.144** |
| D b16p | 16 global | 9.2e-2 | 5.26e-3 (4.17e-2) | — |
| D b45p | 45 global | 5.8e-2 | 5.47e-3 (3.97e-2) | — |
| D unif-poly0 (fitted per-G_z monopoles) | 19 global | 1.8e-1 | 5.69e-3 [hard: 4.98e-3] | — |
| D rich gto3×poly4 (fit ceiling) | 477 global | 1.25e-2 | 5.46e-3 (3.83e-2) | 0.043/0.148 |
| D svd r=4 / 16 / 24 ("learned multipoles") | 4/16/24 +shared | 1.7e-1 / 6.3e-2 / 4.9e-2 | 5.26e-3 / 6.33e-3 / 5.34e-3 | svd16: 0.035/0.143 |
| D hyb26 (b26p + per-(q,G_z) mean-residual corr) | 26 global + 7 per q | 1.8e-2 | 5.13e-3 (3.88e-2) | 0.046/0.145 |
| D ftop30 (exact top-weight channels + model tail) | 30 per q + 26 | 5.9e-3 | 5.62e-3 (3.78e-2) | 0.047/0.170 |
| D b28g (gto radial ladder) | 28 global | 1.1e-1 | 1.06e-2 (5.54e-2) | — |
| E b26p | 26 global | 7.4e-2 | 5.69e-3 (4.02e-2) | 0.042/0.168 |
| pinned-m0 b26p (13.3) | 26 global | 3.2e-1 | 1.21e-2 (6.62e-2) | — |

Reading:

1. **The spec is BEATEN.** b26p — per-G_z in-plane polynomials, degrees
   {3,2,0,0}, ONE global weighted LSQ over all coarse samples — carries
   the whole LR channel in `n_μ × 26` complex coefficients TOTAL (not
   per q): B med 5.37e-3 / max 3.96e-2 / excitons 0.043/0.144 meV — at
   the exact-LR ceiling (5.40e-3) and better than the F-anchor
   (5.85e-3, 0.045/0.167) on every metric, at 1/467 of F's per-μ LR
   storage (26 vs 337×36) and with NO per-q LR object at all. LOO
   coefficient stability 0.7% med / 1.7% max.
2. **B forgives model infidelity up to ~10–20%.** The SR-interp side
   dominates the floor (C = 5.4e-3), so every rung with fid ≲ 0.2 lands
   at 5.1–5.8e-3 — including the hard-gauge b26p (fid 32%, B 5.89e-3,
   exc 0.048/0.180: the inert fiber is invisible to B). The failure
   modes only surface beyond ~40%: literal-moment D o2 (fid 58–70%)
   4.3e-2, pinned-m0 (32%, but biased IN the head weight) 1.21e-2.
   The fit-based extraction is thereby vindicated exactly where the
   literal moments failed: fitted per-G_z constants ALONE (19 numbers)
   give 4.98e-3 (hard) where the literal o0 E-scheme gave 1.18e-2.
3. **What does NOT pay:** SVD compression — the weighted sample-matrix
   spectrum decays slowly (1.4e-1 → 7.9e-3 over 32 svals even in Tik
   gauge; the LR data has no small effective rank across μ), so
   "learned multipoles" never beat the direct 16–26-coefficient
   analytic fit. Per-q corrections (hyb26, ftop30) buy ≤4% on B med —
   not worth any per-q storage. The gto radial ladder (b28g) is the one
   conditioning casualty (three near-collinear widths, B 1.06e-2):
   plain polynomials × the v_LR weight are better AND simpler.
4. E ≈ D throughout (≤6% relative): once fid ≲ 10%, split consistency
   (response §4) is a non-issue.

### 13.3 System-consistency (experiment 3): transfer, and the physical-tensor question settled negative

- **Grid transfer** (`lr_transfer.log`; the 3×3 and 6×6 fixtures share
  ALL 640 centroids, so coefficients are μ-comparable): b26p fit on the
  3×3 data alone, deployed on 6×6 — coefficient distance 3–12% per G_z,
  fitted monopole correlation 0.9981, tile fidelity 10.5% (vs own
  7.4%), and 6×6 LOO **B med 5.382e-3 vs own-fit 5.368e-3 — zero
  downstream loss.** The representation transfers across datasets; fit
  on the cheap grid, deploy on the fine one.
- **Literal-moment pinning refuted a second way** (`lr_pin_6x6.log`,
  the literature-suggested Poisson-DF/MDF move: pin the monopole
  exactly, fit only the rest — in the poly basis this is just freezing
  the constant term to the literal `m0_μ(G_z)`): the literal m0 of the
  Tik-cleaned ζ is still 23% rough in q (P1: med 0.230 vs hard-gauge
  0.357) — NOT a physical-tensor-like object in any gauge tried — and
  pinning it drags B to 1.21e-2 (2.3× worse) with fid 32%. The
  delocalized ζ envelopes contaminate literal real-space moments
  everywhere; only the v_LR-weighted FITTED constants are well-defined,
  and it is those (not the literal moments) that transfer across grids
  at 0.998 correlation. **There is no Born-charge-like per-μ tensor
  file here; the system-consistent object is the weighted fit itself.**
- Scope caveat: per-G_z decomposition presumes a slab with q_z = 0
  (K_z = G_z·b₃ exact discrete channels — evaluation is analytic at any
  IN-PLANE Q, the physical case). A 3D-bulk variant needs a
  K_z-continuous basis; untested (Si fixture is 3D — future work).

### 13.4 Literature mapping (lit-agent survey, 2026-07-17)

- **Range-separated Gaussian density fitting** — Ye & Berkelbach, JCP
  154, 131104 (2021): LR channel of periodic DF integrals evaluated
  analytically in reciprocal space from Gaussian form factors on an
  e^{−K²/4ω²}-windowed ball; Sun et al., JCP 147, 164119 (2017) (MDF):
  multipole-matched compensation Gaussians carried analytically.
  Identical machinery class to our per-G_z polynomial(×Gaussian) fits;
  none of them FIT arbitrary numeric form factors — that step appears
  novel here.
- **Poisson DF / charge-constrained fitting** (Manby–Knowles Poisson
  DF; Dunlap charge-conserving fitting): the pin-the-multipoles move —
  tested (lr_pin) and REFUTED for ISDF ζ (13.3): the constraint that
  stabilizes GTO aux bases hurts when the "multipole" is a literal
  moment of a delocalized numeric envelope.
- **Data-driven momentum-resolved compression** — Hummel, Tsatsoulis &
  Grüneis, JCP 146, 124105 (2017) (SVD of the Coulomb vertex);
  Yeh & Morales, JCTC 19, 6197 (2023) (k-point THC): the SVD rung is
  their analogue; our measured spectrum says the LR channel is NOT
  low-rank across μ at the accuracy needed — analytic K-fits beat
  learned bases here.
- **Form-factor fit practice** (X-ray Cromer–Mann 4–5 Gaussians;
  nuclear sum-of-Gaussians, Sick NPA 218, 509 (1974), N ≈ q_max·R/π;
  2D Slepian/PSWF dof counts with our weight → ~10–50 real dof per
  (μ,G_z)): our measured 10/6/1/1 per-G_z allocation (b26p) sits
  exactly in this window — the coefficient count is
  information-theoretically expected, not lucky.
- Small-K exciton exchange forms: Qiu, Cao & Louie, PRL 115, 176801
  (2015); Haber et al., PRB 108, 125118 (2023) (citation corrected
  2026-07-17, commit de90147f).

### 13.5 Standing verdict

1. **Owner target (≤ n_μ×(10–30), F-level accuracy, analytic any-Q):
   EXCEEDED** — `n_μ × 26` complex coefficients GLOBAL (b26p, Tikhonov
   gauge), B 5.37e-3 med / 3.96e-2 max, excitons 0.043/0.144 meV,
   ≥ F-anchor on all metrics, 1/467 of F's LR storage, zero per-q LR
   objects, LOO-stable to 0.7%, grid-transferable at zero B-loss.
   Even 16 coefficients (b16p) holds the ceiling on B med.
2. **Production shape.** Offline: Tikhonov-clean (ε_rel=1e-4) + §12
   split; ONE weighted LSQ (w = v_LR, shared design, per-q blocks) of
   per-G_z in-plane polys {3,2,0,0} over all coarse samples. Per
   target: §12's SR stencil + closed-form model rebuild (poly eval ×
   phase × v_LR) — strictly cheaper than F (no n_μ×337 stencil AXPYs).
3. **Do not re-attempt:** fitting hard-cut channels (the q-fiber is the
   cut edge — Tik gauge is mandatory for the fit program, 13.1);
   literal-moment pinning (13.3, second refutation of literal moments);
   SVD/learned-multipole compression (no low rank to find, 13.2);
   multi-width GTO radial ladders without a conditioning treatment
   (b28g); pure-3D bases on slabs (12.2, unchanged).
4. Open: α=0.45 budget re-allocation (more G_z channels in the
   superset); 3D-bulk (K_z-continuous) variant; off-grid ground truth
   inherited §11/12 unchanged (fit evaluation at off-grid Q is
   analytic, but no truth exists to score it against yet).

### 13.6 Consolidation (2026-07-17): one reference implementation + e2e test — the formalism survives its own development mess

The winning pipeline (Tikhonov clean → Gaussian SR/LR split → SR-tile
stencil + global b26p LR fit → closed-form assembly at any Q) is now ONE
self-contained module:
`runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/`
`REFERENCE_arbitrary_q_vq.py` — three procedural stages
(`prepare_coarse` / `fit_lr_model` / `eval_vq`), per-element math in the
docstrings, copy-with-attribution from the campaign scripts (arithmetic
preserved), fixture wrap-fix + gate battery + machine-level nulls built
in. Companions: `test_reference_e2e.py` (MoS2 3×3 smoke: prepare → fit →
eval at every held-out q → thresholds; PASS in 15 s compute at B med
1.409e-2 / max 3.553e-2, exc 0.642/2.542 meV — the 3×3 q-spacing
baseline, ~3× the 6×6 B per the §11 bridging row; `test_reference_e2e.log`)
and `README.md` (the scratch ledger: every script family classified
superseded-by-reference vs evidence-only; nothing deleted).

Acceptance (`REFERENCE_acceptance_6x6.log`, 124 s): the module re-runs
the 6×6 Tik-gauge LOO from scratch (fresh gates — all identical to the
ladder log bit-level; fresh eigh; honest LOO refits) and reproduces the
§13.2 pins to every printed digit, per-target as well as in summary:

| metric | §13.2 pin | reference impl |
|---|---|---|
| b26p B med / max | 5.368e-3 / 3.960e-2 | 5.3676e-3 / 3.9597e-2 |
| b26p exc med / max (meV) | 0.043 / 0.144 | 0.0427 / 0.1444 |
| F-anchor B med / max | 5.848e-3 / 3.779e-2 | 5.8480e-3 / 3.7790e-2 |
| F-anchor exc med / max (meV) | 0.045 / 0.167 | 0.0449 / 0.1671 |
| coeff stability med / max | 6.899e-3 / 1.739e-2 | 6.8991e-3 / 1.7389e-2 |

Mode `transfer` re-runs the §13.3 grid transfer (3×3-fit → 6×6-deploy):
B med 5.3817e-3 / max 3.9081e-2 vs pins 5.382e-3 / 3.908e-2 — PASS
(`REFERENCE_transfer.log`, 52 s).

Doc sync landed with this consolidation: `F_SCHEME_NOTE.html` gained
§3.5 ("The long-range channel compresses") + the b26p results-table row
(repo copy — the published artifact needs republishing by the owner);
`ARBITRARY_Q_PRIMER.md` §III.5 gained item 8 (b26p + consolidation
status); `CAMPAIGN_REPORT.md` §7 carries a pointer note. The §14
stress/edge campaign (parallel session, `stress_*.py`) was in flight
during this consolidation and is untouched by it. Production default
remains the per-Q ζ refit (§12.6/§13.5 unchanged).

## 15. Two owner investigations (2026-07-20) — APPENDED: (1) the b26p LR basis is EMPIRICAL but that is irrelevant — the DOF is genuinely low, a principled disk basis ties on accuracy and LOSES on conditioning, symmetry-adaptation halves the coeff count at ~10% cost, the SVD failure is a µ-property; (2) full cutoff audit — RIDGE/pinv/EPS_LR are n_µ-inert, cleaning-ε is the only n_µ-coupled knob and stays relative-to-λmax; under FAITHFUL n_µ growth a small fixed ε_rel drifts B by 0.2% to cond 2.5e14 (junk-inert), crossover ε=cond⁻¹ᐟ² is the conservative floor

**Scope.** Two questions the owner raised about the §13 b26p long-range fit.
Both are read-only studies on the §12/§13 MoS2 3×3/6×6 slab fixtures, run in
the reference harness (`REFERENCE_arbitrary_q_vq.py` loaders/pipeline verbatim,
Tikhonov gauge, α=0.30, nR7, verdict = gap-window `B = MᴴV_QM` med/max over the
full-BZ LOO + TDA exciton swap). All numbers grep-verified from disk (phantom-
table rule): scripts `primer_response_study/study2_{basis_lib,run_study1,`
`run_study2,probe_env}.py`; on-disk provenance = the `.npz` result arrays
(gitignored; written directly by Python — stdout logs are flaky under
srun+shifter) reloaded by `study2_verify_npz.py` → `study2_provenance_6x6.txt`
(committed; regenerates every number below). Production
default is unchanged (per-Q ζ refit; §12.6/§13.5).

### 15.1 STUDY 1 — is the b26p basis fundamental, or empirical?

b26p = per-`G_z` in-plane Cartesian **monomials** `(K_x/2α)^a(K_y/2α)^b`,
`a+b ≤ d`, degrees `{|G_z|=0:3, 1:2, 2:0, 3:0}` → 26 complex coeffs/µ, allocated
by measured `v_LR` weight share (§13.0). Not derived from completeness or
symmetry. Tested against three **principled** disk bases, each fit with the
identical `v_LR`-weighted per-q-normal-block LSQ (only the design matrix
`Φ(K_par)` changes) at matched coefficient count:

- **zernike** — Zernike disk polynomials `R_n^m(r/R){cos,sin}(mθ)` to radial
  degree `d`: the ORTHOGONAL basis of the disk spanning the *same* polynomial
  function space as monomials of degree `d`.
- **bessel** — Fourier-Bessel (Neumann) disk harmonics
  `J_m(k_{m,l}r/R){cos,sin}(mθ)`, `k_{m,l}=` roots of `J_m'` (so `m=l=0` is the
  constant): the bandlimited-optimal orthogonal basis on the physical ball
  `R = 2α√(ln 1/ε_LR)` — the owner's "spherical harmonics of the disk."
- **bessel3 / zernike3** — the same, angular-restricted to `m ≡ 0 (mod 3)`
  (MoS2 C3v/D3h symmetry-adapted).

**Head-to-head (MoS2 6×6, LOO over all 36 coarse q;
`study2_study1_MoS2_6x6_results.npz`):**

| rung | coeff/µ | cond(block) max | LOO-stab med | B med | B max | exc med/max meV |
|---|---|---|---|---|---|---|
| **monomial (b26p, anchor)** | 26 | **3.1e1** | **6.9e-3** | **5.368e-3** | 3.960e-2 | 0.043/0.144 |
| zernike (same span) | 26 | 7.5e5 | 3.1e-2 | 5.368e-3 | 3.960e-2 | 0.043/0.144 |
| bessel (bandlimited) | 26 | 3.3e6 | 1.0e-1 | 5.319e-3 | 3.952e-2 | 0.044/0.147 |
| zernike3 (m≡0 mod 3) | **12** | 6.7e5 | 1.2e-1 | 5.903e-3 | 3.963e-2 | 0.052/0.147 |
| bessel3 (m≡0 mod 3) | 26 | 3.0e8 | 2.2e-1 | 6.052e-3 | 3.869e-2 | 0.053/0.152 |
| monomial b45p `{4,3,1,0}` | 43 | 9.4e1 | 8.0e-3 | 5.249e-3 | 3.967e-2 | 0.043/0.152 |
| bessel b45p | 43 | 2.6e7 | 9.9e-2 | 5.340e-3 | 3.953e-2 | 0.040/0.147 |

Reading (each point measured, not asserted):

1. **Zernike ≡ monomial to every digit** (B 5.368e-3, identical excitons,
   identical across-µ SVD spectrum). A matched-degree orthogonal disk basis and
   the ad-hoc monomials span the *same* polynomial function space, so the
   weighted-LSQ projection — hence B — is bit-identical. **The accuracy is set
   by the function SPACE (its DOF), not by the basis within it.** This is the
   direct answer to "does basis choice matter?": for accuracy, no.
2. **Bandlimited (bessel) ties, does not beat** (5.319e-3 vs 5.368e-3 — noise).
   The principled bandlimited-optimal basis buys nothing on B.
3. **The empirical monomial WINS on conditioning and stability.** Its `1/(2α)`
   scaling is matched to the physical `1/K²·e^{−K²/4α²}` weight, giving
   cond(block) **31** and LOO-coeff stability **0.7%**; the disk-orthogonal
   bases are orthogonal under the *uniform-disk* measure (the wrong one for this
   weight), so they are 4–7 orders worse conditioned (7e5–3e8) and 4–30× less
   LOO-stable. b26p is already near-optimally conditioned — a principled basis
   is a downgrade here.
4. **Diminishing returns confirm low DOF:** b45p (43 coeff) improves B by 2%
   over b26p (26). ~26 is already on the plateau.
5. **Symmetry-adaptation is the one principled lever with value, and it is
   modest.** `zernike3` reaches B 5.9e-3 with **12** coeffs (vs 26) — a >2×
   compression at ~10% B cost. But the payoff is small *because there is little
   angular structure to exploit*: the µ-averaged angular power of `M_µ(K)` on
   the richest `|G_z|=0` channel is **m0 0.959, m1 0.033, m2 0.005, m3 0.001**
   (6×6; 0.920/0.060/… at 3×3) — the form factor is ~96% isotropic **monopole**,
   its largest anisotropy is a `m=1` dipole (from centroids at generic, non-
   symmetry FFT sites), and the genuinely C3-specific `m=3,6` harmonics are
   ≲0.001. "m ≡ 0 (mod 3) dominant" is true only in the trivial sense that
   `m=0` dominates; there is no rich C3 tensor structure to compress.
6. **The SVD-multipole failure (§13.2) is reconciled and is basis-independent.**
   The across-µ SVD of the fitted coefficient matrix decays *identically slowly*
   in monomial, Zernike, and Bessel bases (norm 1.0, 0.14, 0.086, 0.056, …).
   The slow decay is a **µ-rank property** — the 640 centroids are 640 genuinely
   distinct form factors — orthogonal to the K-basis choice. A principled
   K-basis fixes the (already low, ~10) K-dimension; it cannot reduce the
   µ-dimension, which is where "learned multipoles" tried and failed.
7. **No transfer advantage over b26p.** The owner's hypothesised edge — a fixed
   physical ball radius `R` is system-independent — does not distinguish the
   bases, because b26p is *already* physical-scale (its `1/(2α)` normalization,
   α physical). Grid transfer 3×3-fit → 6×6-deploy: monomial 5.382e-3, zernike
   5.408e-3, bessel 5.522e-3 — all at zero downstream B-loss (§13.3).

**VERDICT (Study 1).** The b26p basis is **empirical** (a weight-informed
polynomial-degree ladder), and this is **immaterial**: the LR form factor lives
in a genuinely low-dimensional, near-isotropic function space (~10 DOF on the
rich channel), so (i) no principled basis is more accurate at matched count
(Zernike is bit-identical, Bessel ties); (ii) none is more transferable (all
transfer at zero loss; b26p is already physical-scale); (iii) the polynomial's
success *does* reveal that the DOF is just low — and, because the physical
`1/K²·Gaussian` weight is strongly non-uniform, the ad-hoc monomials with `1/2α`
scaling are in fact *better conditioned and more LOO-stable* than the
"principled" disk-orthogonal bases, which are orthogonal under the wrong
measure. The only principled idea that pays is **symmetry-adaptation**
(`m≡0 mod 3` → 12 vs 26 coeffs at ~10% B cost), and it pays little because the
form factor carries almost no angular structure. **Keep b26p.** If a smaller
footprint is ever wanted, drop to the symmetry-adapted 12-coeff `zernike3`, not
a bandlimited basis.

### 15.2 STUDY 2 — SVD/rank/Tikhonov cutoffs and ISDF-basis-size dependence

**Every cutoff in the V_Q interp + trainer pipeline, audited
(`study2_run_study2.py`; `REFERENCE_arbitrary_q_vq.py`/`lr_prep.py`):**

| cutoff | site | value | scaling | grows with n_µ? | measured verdict |
|---|---|---|---|---|---|
| **EPS_TIK** | tile cleaning `g_ε(λ)=λ²/(λ²+(ε·λ_max)²)` | 1e-4 | **relative to λ_max** (per q) | **couples to cond_C(n_µ)** | the only n_µ-sensitive knob; §15.2 policy below |
| RIDGE | b26p normal solve `A+=RIDGE·(trAᐟnb)·I` | 1e-11 | relative to block trace | **no** — `nb`≤15 is the BASIS size | **inert**: B *identical* for RIDGE ∈ {0,1e-14,…,1e-6} (5.368e-3) |
| pinv rcond | stencil `f0@pinv(F)` | 1e-15 | relative to σ_max(F) | **no** — `F` is `(n_q×nR)` q-geometry | safe; unrelated to ISDF size |
| EPS_LR | LR ball `K²_max=4α²ln(1/ε)` | 1e-8 | Gaussian tail bound | **no** — depends on α+lattice | safe; sets the 337-G superset only |
| hard rank-cut `Π=top-r` | (alternative to Tik) | — | fraction of n_µ | — | spectrum-*shape*-invariant but narrow sweet spot + §13.1 q-fiber |

So three of four cutoffs are **structurally n_µ-independent** (RIDGE acts on the
K-space Gram whose size `nb` and conditioning do not grow with n_µ; pinv and
EPS_LR are pure geometry/physics). RIDGE is additionally *load-free* — the fit
survives RIDGE=0. Only **cleaning-ε** touches the C_q spectrum, which deepens
(longer gapless tail, higher cond_C) as n_µ → 20k.

**Cleaning-ε sweep (6×6, completes the killed §14/stress-axisA;
`study2_study2_MoS2_6x6_results.npz`):** cond_C med **1.57e7**, crossover
ε*=cond⁻¹ᐟ²=**2.5e-4**.

| ε_rel | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | hard r=0.5n_µ |
|---|---|---|---|---|---|---|
| B med | 8.46e-3 | 5.37e-3 | **4.71e-3** | 6.17e-3 | 1.10e-2 | 4.88e-3 |

Broad, shallow optimum ε≈1e-4–1e-5 (best 1e-5); over-cleaning (1e-3) and the
§13.1 q-fiber (ε≤1e-6, exciton max 0.14→0.23 meV) bound it. Hard rank-cut peaks
at r≈0.5n_µ (4.88e-3) but degrades sharply at r=0.75n_µ (8.3e-3, the Davis-Kahan
cut-edge modes) — narrower and worse-behaved than Tikhonov, as §13.1 predicts.

**n_µ scaling — synthetic spectrum stretch (`λ'=λ_max(λ/λ_max)^σ`, cond_C →
cond_C^σ, eigenvectors fixed; the owner's tail-scaling probe).** Two stretch
models — *uniform* (deepens ALL sub-maximal modes, pessimistic) and *tail*
(keeps the resolved top-half fixed, deepens only the lower tail — faithful to
"larger n_µ adds inert junk, physics unchanged") — under the current fixed
ε_rel=1e-4 vs the crossover-tracking ε=cond_C^−1/2:

| σ (cond_eff) | 1.00 (1.6e7) | 1.25 (1e9) | 1.50 (6e10) | 1.75 (4e12) | 2.00 (2.5e14) |
|---|---|---|---|---|---|
| **tail**, fixed 1e-4 | 5.368e-3 | 5.379e-3 | 5.379e-3 | 5.379e-3 | **5.379e-3** |
| **tail**, track cond⁻¹ᐟ² | 6.63e-3 | 4.70e-3 | 4.88e-3 | 4.88e-3 | 4.88e-3 |
| uniform, fixed 1e-4 | 5.368e-3 | 8.27e-3 | 1.16e-2 | 1.81e-2 | **4.91e-2** |
| uniform, track cond⁻¹ᐟ² | 6.63e-3 | 6.74e-3 | 6.79e-3 | 6.81e-3 | 6.81e-3 |
| hard r=0.5n_µ | 4.88e-3 | 4.88e-3 | 4.88e-3 | 4.88e-3 | 4.88e-3 |

The decisive rows are the **tail** ones (the faithful n_µ-growth model): a small
**fixed relative** ε_rel drifts B by **0.2%** (5.368→5.379e-3) even as cond_C
runs to 2.5e14, because the deepened tail is **inert under `MᴴV_QM`** — damping
it more (ε fixed) or the same (ε tracked) never touches the physical block. This
directly confirms the junk-inertness thesis for the BSE tile. The *uniform*
rows are the stress case: when the deepening also reaches physically-resolved
modes, fixed ε_rel over-damps them (9× drift by σ=2, ndamp 374→559), and only
the crossover-tracking ε=cond⁻¹ᐟ² holds B flat (2.7% over 7 decades of cond,
ndamp constant 406). The hard top-r projector is B-invariant by construction
(spectrum-shape-independent) but inherits §13.1's q-fiber and narrow sweet spot.

**Connection to the ridge-ζ A/B verdict (`reports/zeta_ridge_ab_2026-07-17/`).**
There the GW Σ pathway is junk-**sensitive** (V0/W0 device-covariance and
Σ_c swing 4–200 meV as ε shrinks; ε MUST scale relative to λ_max; crossover
ε*≈cond(C)⁻¹ᐟ²). The BSE tile pathway measured here is the **forgiving** cousin:
identical relative-ε *logic*, but the junk is inert under the physical
contraction, so the fixed-ε policy that is *dangerous* for GW is *safe* for BSE
to cond 2.5e14. The scaling law transfers; the tolerance does not.

**RECOMMENDED ε/rank policy (safe 640 → ~20k centroids):**

1. **Cleaning-ε stays RELATIVE to λ_max (per q).** Never an absolute floor —
   an absolute floor drifts with the spectrum (measured). This is already the
   code (`EPS_TIK·λ.max()`); keep it.
2. **For the BSE tile, a small fixed ε_rel (1e-4, or 1e-5 for peak on-grid
   accuracy) is safe and near-optimal to 20k centroids** — the owner's small-ε
   preference is vindicated *for this pathway*, because the growing tail is
   junk-inert (tail-stretch B drift 0.2%). No n_µ-scaling of ε is required for
   BSE tile interpolation.
3. **The conservative floor, if the physical/junk gap ever closes** (window
   redesign that admits near-degenerate physical pairs, near-metallic system,
   or when sharing the cleaning with a junk-sensitive GW Σ path), is
   `ε_rel = max(ε_small, cond_C⁻¹ᐟ²)` evaluated per-q from the actual spectrum —
   the ridge-ζ crossover law, which held B flat under BOTH stretch models. Cost:
   ~24% worse on-grid at MoS2 scale (6.6e-3 vs 5.4e-3), bought as robustness.
4. **RIDGE, pinv-rcond, EPS_LR need no change from 640 to 20k** — they are
   n_µ-independent by construction (K-space Gram size, q-geometry, physical ball
   radius). RIDGE may even be set to 0; 1e-11 is a harmless guard.
5. Prefer Tikhonov over the hard rank-cut (§13.1 q-fiber; narrower sweet spot);
   if a spectrum-shape-invariant option is ever wanted, r≈0.5n_µ is the plateau.

**VERDICT (Study 2).** The pipeline is well-posed for the n_µ→20k regime: the
only spectrum-coupled cutoff (cleaning-ε) is already correctly relative-to-λmax,
and for the junk-inert BSE tile a small fixed ε_rel is measured-safe to
cond_C≈2.5e14 (0.2% B drift). The crossover ε=cond_C⁻¹ᐟ² is the recommended
conservative default whenever junk-inertness cannot be assumed (it is *mandatory*
on the GW Σ side); the fit-ridge and stencil/ball cutoffs are n_µ-inert and need
no revision.
