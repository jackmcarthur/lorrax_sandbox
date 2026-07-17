# Arbitrary-Q interpolation in LORRAX — self-contained primer

**Purpose.** Hand this file to a fresh physicist-agent with *no* prior LORRAX
context and they can attack the open problem — "find the most viable scheme to
produce the Coulomb tile `V_Q` (and the BSE exchange kernel) at an *arbitrary*
exciton momentum `Q`, off the coarse `q`-grid." Everything needed is here: the
ISDF machinery LORRAX runs, the per-`q` fit and its conditioning, the physical
argument for why `q`-interpolation is even plausible, what is already settled,
and the crisp open question with its evaluation criteria.

**Provenance / coordination.** This is a reorganized, self-contained superset of
`arbitrary_q_bse.md` (§0–§9.8), which remains the living working doc. A separate
agent is appending a **"Ingredient-interpolation falloff study" results section**
to `arbitrary_q_bse.md`; that section is the decisive empirical input to PART III
here. The falloff study has LANDED (arbitrary_q_bse.md §3.5) and its numbers
are folded into PART III §III.4 of this file. `arbitrary_q_bse.md`
stays as-is and is the place the falloff study is written; this primer is the
handoff artifact. Ry units throughout. Fractional `q`/`Q` are in reciprocal-
lattice coordinates unless a Cartesian `q+G` is written.

---

# PART I — ISDF and the LORRAX Coulomb pipeline

## I.0 Orientation

LORRAX is a real-space, real-axis GW/BSE package that runs alongside BerkeleyGW
(BGW) and Quantum ESPRESSO (QE): QE supplies DFT spinor wavefunctions `ψ_{nk}`,
LORRAX builds the GW self-energy and the BSE kernel. Its distinguishing move is
**ISDF compression** (Interpolative Separable Density Fitting): the `O(N²)` pair
densities per k-point are replaced by a shared low-rank real-space basis, so
every two-point Coulomb object becomes a small dense `n_μ × n_μ` matrix. All
wavefunctions are two-component spinors (SOC native). Large arrays are sharded
over GPUs with JAX; the physics below is independent of the sharding.

## I.1 The ISDF ansatz

Every expensive GW/BSE object is built from **pair densities**

$$
\rho_{mn\mathbf{k}}(\mathbf{q};\mathbf{r}) \;=\; \psi^*_{m,\mathbf{k}-\mathbf{q}}(\mathbf{r})\,\psi_{n,\mathbf{k}}(\mathbf{r}),
$$

with SOC the coupling object is the **spin-traced** pair density

$$
\rho_{mn\mathbf{k}}(\mathbf{q};\mathbf{r}) \;=\; \sum_{\alpha\beta}\psi^*_{m,\mathbf{k}-\mathbf{q},\alpha}(\mathbf{r})\,\sigma^0_{\alpha\beta}\,\psi_{n,\mathbf{k},\beta}(\mathbf{r}),
\qquad \sigma^0=\text{spin identity}.
$$

There are `O(n_k n_b²)` of them on `N_r` grid points, all built from the same
`O(n_b)` orbitals — massively redundant. ISDF replaces them with one ansatz:

$$
\boxed{\;\rho_{mn\mathbf{k}}(\mathbf{q};\mathbf{r}) \;\approx\; \sum_{\mu=1}^{n_\mu}\zeta_{q,\mu}(\mathbf{r})\;\rho_{mn\mathbf{k}}(\mathbf{q};\mathbf{r}_\mu)\;}
$$

**A pair density anywhere is a linear combination of its own values at a fixed
set of interpolation points `{r_μ}` (centroids)**, with vectors `ζ_{q,μ}(r)`
shared by all band pairs and k-points, depending only on `q`. The `(mnk)` label
separates from the `r` dependence — the "separable" in ISDF. Storage drops from
`O(n_k n_b² N_r)` to `O(n_q n_μ N_r)`; band sums over `m,n` still factorize at
the points (the "two-sums rule" survives compression). `ζ` is spin-independent
even though the metric couples all four spin channels (§I.2).

**Centroids.** The points are chosen once, before the run, by k-means / CVT
clustering of the FFT grid weighted by the band-window charge density
(`lorrax-centroids`); they concentrate where orbitals overlap. `n_μ` is the
basis-size dial (the analogue of a plane-wave cutoff): convergence sets in near
**8–12 centroids per band** in the fitting window; cost is `n_μ²` (matrices) to
`n_μ³` (factorizations). Symmetry post-processing (§I.2, orbit closure) enlarges
a requested set slightly.

## I.2 The per-q fit, as LORRAX does it

Minimizing the squared interpolation error over all band pairs and k-points in
the window gives, **per momentum transfer `q`**, the normal equations

$$
\boxed{\;C_q\,\zeta_q \;=\; Z_q\;}\qquad \sum_\mu C_{q,\nu\mu}\,\zeta_{q,\mu}(\mathbf{r}) = Z_{q,\nu}(\mathbf{r}).
$$

The coefficients collapse into products of **quasi-density matrices** evaluated
at the centroids / grid. Define the band-window density matrix

$$
P_{\mathbf{k},ss'}(\mathbf{r}_\nu,\mathbf{x}) \;=\; \sum_{n\in\text{window}}\psi^*_{n\mathbf{k},s}(\mathbf{r}_\nu)\,\psi_{n\mathbf{k},s'}(\mathbf{x}),
$$

then

$$
C_{q,\nu\mu} = \sum_{\mathbf{k}}\sum_{ss'} P^*_{\mathbf{k}-\mathbf{q},s's}(\mathbf{r}_\nu,\mathbf{r}_\mu)\,P_{\mathbf{k},ss'}(\mathbf{r}_\nu,\mathbf{r}_\mu),
\qquad
Z_{q,\nu}(\mathbf{r}) = \sum_{\mathbf{k}}\sum_{ss'} P^*_{\mathbf{k}-\mathbf{q},s's}(\mathbf{r}_\nu,\mathbf{r})\,P_{\mathbf{k},ss'}(\mathbf{r}_\nu,\mathbf{r}).
$$

Read the structure:

- `C_q` is the **centroid–centroid** Gram (both legs at centroids, `n_μ × n_μ`,
  Hermitian). `Z_q` is the **centroid–to–(r or G)** cross Gram (one leg at
  centroids, one leg on the full real-space / reciprocal grid).
- Both are **k-space convolutions** `Σ_k P*_{k-q} P_k`: the same
  `k`-convolution for every `q`, so LORRAX evaluates all `q` at once by lattice
  FFT (`C_R = Σ_ab |P_{R,ab}|²`, then `C_q = FFT_{R→q}[C_R]`; likewise `Z`).
- The spin trace couples **all four** `P_{ss'}` — even the off-diagonal
  `P_{↑↓}` — because the Galerkin error factorizes as
  `Σ_{ss'} P^*_{s's} P_{ss'} = ‖P_k‖²_F` (Frobenius spin norm), *not*
  `|P_{↑↑}+P_{↓↓}|²`. (Fitting only the trace still needs the full 2×2 `P`.)
- The **band windows** enter through the `n∈window` sum in `P`: the *screening*
  fit uses `[0, n_cond)` occupied×conduction; the exchange/charge fit uses the
  charge window. Different windows → different `P` → different `C_q, Z_q`.

`C_q` is factorized once per `q` — Cholesky in the charge channel (PSD up to
numerical rank deficiency; single-device adds a small trace-proportional ridge),
pivoted LU for the indefinite bispinor transverse channels. `ζ_q` is then a
triangular / LU solve. Fits run on the **IBZ `q` set only** when the cascade is
active (§I.4).

### The `|q+G|` sphere and the G-flat layout

The Coulomb-facing form of `ζ` lives in reciprocal space on a **per-`q` sphere**
`{G : |q+G|² ≤ cutoff}` (the `zeta_cutoff_ry` dial):

$$
\boxed{\;\tilde\zeta_{q,\mu}(K) \;=\; \frac{1}{N_r}\,\mathrm{FFT}_{r\to G}\!\big[e^{-2\pi i\,\mathbf{q}\cdot\mathbf{r}}\,\zeta_{q,\mu}(\mathbf{r})\big](K),\qquad K=\mathbf{q}+\mathbf{G}\;}
$$

sampled at the **physical** reciprocal vector `K = q+G` (a cell-periodic /
"Bloch-stripped" convention). On disk (`zeta_q_G.h5`, dataset
`zeta_q_G (n_q_disk, n_rmu, ngkmax)`, c128) the sphere is padded to
`ngkmax = max_q ngk[q]` with pad slots zeroed; **`G=(0,0,0)` is always slot 0**
of every `q`'s sphere (this is why the head channel is `g0_μ(q) = ζ̃(q,μ,G=0) =
ζ̃[q,μ,0]`, §I.4). Per-`q` fractional vectors follow the BGW wrap convention
`q > kgrid/2 → q − kgrid`. This is the "flat-`q` C-order" layout: a single flat
`q`-axis (IBZ subset when the cascade fires), each row one HDF5 chunk.

### Conditioning — the load-bearing numerical fact

`C_q` is **badly conditioned**: `cond(C) ~ 1e7–1e9` in practice (measured
`cond(CCT) = 3.6e9` on the Si screening window; `cond(H) ~ 1e8` for the
head-injected finite-`q` tiles). The solve `ζ_q = C_q^{-1} Z_q` therefore
**amplifies** any error/non-smoothness in the right-hand side by up to
`~cond(C)`. This is the crux of PART III: the *ingredients* `C_q, Z_q` may be
smooth in `q`, but the *solved* `ζ_q` need not inherit that smoothness, because
the near-singular inversion rotates and amplifies. Fit figure of merit is the
per-`q` relative residual `‖ρ − ρ_fit‖_F / ‖ρ‖_F`.

### The degeneracy-window covariance lesson (PHASE2_LOG, Round-2)

A concrete way the fit silently breaks symmetry — worth internalizing before
trusting any `ζ`-derived tile. On Si, BSE exciton multiplets that should be
exactly degenerate split by hundreds of μeV. Root cause traced by bisection
`ψ(1e-15) → CCT-input(0.4%) → ζ-head G0(8.6%) → V0(3.2%) → W0(3.0%)`
(`corr(V0,W0)=0.997`, W inherits V): **splitting a degenerate multiplet at the
top of the ISDF/screening band window makes the fitted `ζ̃` — and every `V_q/W_q`
tile — non-covariant under the crystal symmetry** at the high-symmetry `k` where
the multiplet lives, then `cond(CCT) ~ 1e10` amplifies the seed ~20×. Fix:
degeneracy-round the fit window down to a closed shell (`gw/degen_average.py`
`round_band_window_to_closed_shell`, BGW `TOL_Degeneracy = 1e-6` Ry); the
padded-up world-size divisibility must pad with **zero** bands, never real ones.
Lesson for arbitrary-`Q`: `C`/`ζ`/`V` covariance is only as good as the window's
degeneracy hygiene, and the conditioning turns a small window defect into a
large tile defect.

## I.3 Why `q`-interpolation is even plausible — the density-matrix ↔ Green's-function argument

*This is the theoretical spine of the whole approach; the owner wants it stated
properly.*

`C_q` and `Z_q` are built from products of **single-particle density matrices**
`P_{k,ss'}(r_ν, x)` — literally the band-window-projected one-body density matrix
`ρ(r,r')` sampled at centroid/grid pairs. In a **gapped** system the density
matrix obeys **Kohn nearsightedness** (Kohn 1996; Kohn–Nenciu–Prodan): the
real-space density matrix decays **exponentially** (insulator) or
**super-algebraically** (the general gapped statement) in the separation
`|r − r'|`,

$$
|\rho(\mathbf{r},\mathbf{r}')| \;\lesssim\; e^{-\gamma|\mathbf{r}-\mathbf{r}'|},\qquad \gamma \sim \sqrt{E_\text{gap}} .
$$

Take the lattice Fourier transform to the **R-representation**: the
Bloch-summed density matrix
`P_R(r_ν, r_μ) = Σ_k e^{ik·R} P_k(r_ν, r_μ)` connects home cell `0` to cell `R`,
and inherits the same exponential falloff in the cell index `R`. Now the ISDF
Grams are, by the convolution theorem applied to `Σ_k P*_{k-q} P_k`, elementwise
**products of two R-space density matrices** before the `R→q` FFT:

$$
C_R(\mu,\nu) = \sum_{ss'}\big|P_{R,ss'}(\mu,\nu)\big|^2,\qquad C_q = \mathrm{FFT}_{R\to q}[C_R],
$$

and identically for `Z`. A product of two exponentially-decaying matrices decays
**at least as fast** — so `C_R` and `Z_R` are **short-ranged in R**. A function
with support confined to `|R| ≲ R_c` in the lattice has a Fourier transform
whose variation scale in `q` is `~1/R_c`: **superalgebraic/exponential spatial
falloff of `C_R, Z_R` is exactly the smoothness-in-`q` of `C_q, Z_q`.** This is
the Wannier-interpolation premise (rapidly-decaying real-space matrix elements ⇒
interpolable band structure) transcribed to the ISDF Grams.

**What this does and does not license.** It licenses interpolating the
*ingredients* — `C_q`, `Z_q`, and any object built *linearly* from short-ranged
R-space data (the smooth part of the Coulomb tile). It does **not** automatically
license interpolating the *solved* `ζ_q = C_q^{-1} Z_q`, because the near-
singular inversion (`cond ~ 1e7–1e9`, §I.2) can convert smooth inputs into a
`q`-rough output. So the theory predicts: **interpolate on the smooth-ingredient
side of the inversion, not the fit-vector side.** The measured §II.3 result
(naive "master-ζ + analytic phase" fails at 90–340%) and the falloff study
(`C_R` decay length, leave-one-out on the ingredient tile) are the two
measurements that bracket the inversion — one on each side.

**Confirmations of the falloff prediction** (authoritative; study complete —
full tables in `arbitrary_q_bse.md` §3.5 and PART III §III.4, including the
decisive negative result: the falloff does NOT transfer through `C⁻¹Z`):

- `C_R` decays to a `1e-3–1e-4` floor by `~10 Bohr` (MoS₂) / `~14 Bohr` (Si) —
  a fixed *physical* length, independent of `k`-mesh, exactly as nearsightedness
  predicts.
- **Ingredient** leave-one-out interpolation error falls `0.41%` (3×3 mesh) →
  `0.13%` (4×4 mesh): denser `q`-sampling puts more `R`-shells inside the decay
  length, so the interpolant improves — the signature of a short-ranged kernel.

## I.4 From `ζ` to the Coulomb tiles; the head; 2D truncation; what is on disk

### The contraction

The Coulomb matrix every later stage consumes is

$$
\boxed{\;V_{q,\mu\nu} \;=\; \sum_{\mathbf{G}\in\text{sphere}(q)} \overline{\tilde\zeta_{q,\mu}(\mathbf{q}+\mathbf{G})}\; v_q(\mathbf{G})\; \tilde\zeta_{q,\nu}(\mathbf{q}+\mathbf{G})\;}
$$

a bilinear contraction of the two `ζ̃` legs weighted by the bare Coulomb scalar.
For bispinor transverse channels the weight carries an extra angular factor
`t^{μ_L ν_L}(K)` (`1` for charge–charge; `1 − K̂_i²`, `−K̂_iK̂_j` for the
transverse tiles). The **screened** tile is `W_q = (1 − V_q χ_q)^{-1} V_q` built
from the *same* bare `V_q` — note **`W` is nonlinear in `V`**, so a split of `V`
is not a split of `W` (§II.6).

### The bare kernel and 2D truncation

$$
v_q(\mathbf{G}) \;=\; \frac{8\pi}{|\mathbf{q}+\mathbf{G}|^2}\;T_\text{cell}(\mathbf{q}+\mathbf{G})
$$

(already divided by `V_cell` internally). `T_cell` is set by `sys_dim`:

- **3D (bulk):** `T = 1`. Head diverges as `1/|q+G|²`.
- **2D (slab):** `T = 1 − e^{-|q_∥+G_∥| L_z/2}\cos(G_z L_z/2)`. Cuts periodic
  images along `z`; **mandatory** for 2D — gaps/binding otherwise depend on
  vacuum thickness. Head is `~2π/|q_∥|` (a `|q|` behavior, not `1/q²`).
- **0D (box):** Wigner–Seitz truncation, **finite** `v(q=G=0)` from the WS FFT →
  no divergence, split-exempt. Implemented but not wired through the production
  `V_q` driver (driver accepts 2, 3).

### The `q→0` head and its rank-1 storage

At `q→0` the bare kernel diverges `~1/q²` while the polarizability head vanishes
`~q²` (f-sum rule / gauge invariance), so `W`'s head is **finite but not
computable from the discrete grid** — LORRAX builds the `q²` coefficient from
transition dipoles (`psp.get_dipole_mtxels`, s-tensor machinery) or takes
overrides. Operationally the head is isolated as a **rank-1 channel** in the
centroid basis. LORRAX persists the body with `G=0` zeroed and keeps the head
vector separately:

$$
g0_\mu(q) = \tilde\zeta(q,\mu,G{=}0),\qquad V_{q0} \mathrel{+}= \frac{v_\text{head}}{V_\text{cell}}\,\overline{g0}\otimes g0,\qquad W_{q0} \mathrel{+}= \frac{w_\text{head}}{V_\text{cell}}\,\overline{g0}\otimes g0,
$$

with `v_head = ⟨v⟩_mBZ` (mini-BZ average of the diverging kernel),
`w_head = ⟨v⟩_mBZ · ε⁻¹₀₀` (`apply_q0_head_rank1{,_sharded}`,
`head_correction.py:743`). **The `SR/LR` split LORRAX already runs at `q=0` *is*
this rank-1 factorization** — the singular weight collapsed onto the `G=0`
channel; arbitrary-`Q` generalizes it to per-fine-`Q` (§II.6).

### IBZ cascade (symmetry)

When the centroid set is closed under the space group + TRS (orbit closure), the
fit runs on the IBZ `q` set only and `V_q`/`W_q`/`Σ_x` are **unfolded** to the
full BZ (`unfold_v_q`): a centroid double-permute + per-centroid umklapp phase
`exp(2πi q_irr·(L_μ − L_ν))`; the `τ`-phase cancels leg-to-leg by bilinearity.
`~6–12×` savings on the dominant stages. Gate: closure holds or fall back to
full BZ.

### What is on disk — the restart map (one page)

Two artifacts carry the arbitrary-`Q`-relevant state:

**`zeta_q_G.h5`** (the ISDF fit output):

| dataset / header | shape | meaning |
|---|---|---|
| `zeta_q_G` | `(n_q_disk, n_rmu, ngkmax)` c128 | `ζ̃_{q,μ}(q+G)` on the per-`q` sphere, IBZ-only when cascade active |
| `gvec_components` | `(n_q_disk, 3, ngkmax)` | the `G`-list per `q` (slot 0 = `G=0`) |
| `ngk_per_q` | `(n_q_disk,)` | true sphere length per `q` (rest is zero pad) |
| `r_mu_fft_idx`, `fft_grid` | — | centroid grid indices; FFT box |
| `zeta_cutoff_ry`, `vertex_mu_L`, `zeta_layout='G_flat'` | — | fit metadata |
| `mf_header` | — | copied verbatim from `WFN.h5` |

**`tmp/isdf_tensors_*.h5`** (the static-W restart bundle BSE reads):

| dataset | shape | meaning |
|---|---|---|
| `V_qmunu` | `(nq, μ, ν)` flat-`q` | bare Coulomb tile, `G=0` head zeroed (`= V^SR` at `α→∞`) |
| `W0_qmunu` | `(nq, μ, ν)` | static screened tile; `W0_ready=True` gate |
| `G0_mu_nu` | `(μ,)` | head channel vector `g0_μ` |
| `psi_full_y` | `(nk, nb, ns, μ)` | ψ at centroids `u_{n,k}(r_μ)` |
| `enk_full` | `(nk, nb)` | DFT band energies (Ry) |
| `vhead`, `whead[ω]`, `omega_grid` | scalars / `(nω,)` | head scalars (`⟨v⟩`, `⟨v⟩·ε⁻¹₀₀`) |
| `kgrid` | int64[3] | coarse `k`-mesh |

Everything the direct-only path needs is `ψ(r_μ)`, `enk_full`, `W0_qmunu`,
`kgrid`; exchange additionally needs `V_qmunu`/`ζ̃`/`g0`/`vhead`.

---

# PART II — The arbitrary-Q problem, and what is already settled

## II.0 Problem statement

An exciton at finite momentum `Q` couples transitions `|v,k → c,k+Q⟩` with `k`
on the coarse grid and **`Q` arbitrary** (off-grid). TDA BSE Hamiltonian
`H^BSE = D − W + K^x` with diagonal `D_Q = ε_c(k+Q) − ε_v(k)`, direct kernel `W`,
exchange kernel `K^x` (uses the bare tile `V_Q`). Producing `H^BSE(Q)` at
arbitrary `Q` needs four off-grid ingredients: `ψ_c(k+Q)`, `ε_c(k+Q)`, the direct
`W`, and the exchange `V_Q`. The question is which of these are cheap/exact and
which require an *interpolation across `q`* — and for the latter, **what scheme**.

## II.1 htransform — the off-grid `ψ(r_μ) + ε` source

`src/bandstructure/htransform.py` + `bse_setup.py` do **Hamiltonian
interpolation in a rank-α centroid basis**, not a per-`k` solve. Build a small
`(rank, rank)` effective Hamiltonian `fH_R` in the lattice-`R` representation
**once** on the coarse grid, Fourier-sum it to any off-grid `q`, diagonalize the
tiny matrix. **No `H(k)` plane-wave rebuild, no Sternheimer.**

The "f-transform" applies a smooth bandwidth-bounded map `f(ε)≤0 for ε<shift, =0
above` to the DFT eigenvalues so `fH_k = Σ_n f(ε_{nk}) c_{nk} c_{nk}^H` is a
well-behaved low-rank operator whose eigenpairs recover `(f(ε_{nq}), c_{nq})` at
any `q`; `newton_inv` inverts `f` to physical energies.

Setup (once, coarse grid) produces three durable objects: `ctilde`
`(nk_co, nb, rank)` (Galerkin coeffs), `B_at_mu` `(rank, ns, n_μ)` (α-basis at
the centroids), `enk_sigma` `(nb, nk_co)` (coarse energies). `rank` = SVD rank of
`ψ_at_centroids.reshape(nk_co·nb, ns·n_μ)` at `rtol=1e-8`, `≈ nk_co·nb`
(MoS₂ 3×3, nb≈8 ⇒ rank ≤ 72).

The BSE loader entry `compute_wfns_fi(...)` returns, per `q` in a batched grid,
`ψ_rmu = u_{n,q}(r_μ)` and `enk_full = ε_{n,q}` (via `newton_inv`). **The
eigenvalues and wavefunctions come from the same `fH_q` diagonalization → `ε(q)`
and `ψ(q)` are mutually consistent** (cleaner than BGW's separate WFN_fi +
energy interpolation). Cost per off-grid `q`: `O(rank³)` eigh + `O(nk_co·rank²)`
Fourier sum + `O(rank·ns·n_μ)` ψ reconstruction; `rank³` dominates; batched over
devices. `fH_R` replicated `~240 MB` at MoS₂ scale — small vs the GW tensors.

**Generalization needed (small):** today `compute_wfns_fi` builds a Γ-centred
*uniform* fine grid; the arbitrary-`Q` path needs one helper
`compute_wfns_at_qlist(..., q_list = coarse_k + Q)` — identical body, explicit
`q`-list = coarse grid + constant shift `Q`. ~40 LOC.

**Accuracy caveat.** `bandstructure` is flagged *experimental*; built-in
diagnostics (Γ FFT round-trip `~1e-12`, Γ `Δε`) exist but **no pass/fail
threshold**. The load-bearing gate to add: htransform `ψ(r_μ), ε` at an
*on-grid* `q` must reproduce the directly-loaded `ψ(r_μ), ε` to interpolation
tolerance.

## II.2 Direct-kernel-only arbitrary-Q BSE is exact and cheap

Per-element the direct term is (ISDF form)

```
(W X)[c,v,k] = (1/Nk) Σ_{k'} Σ_{c'v'}
   [ Σ_μ ψ*_{c,k+Q}(μ) ψ_{c',k'+Q}(μ) ] · W_{μν}(k−k') · [ Σ_ν ψ_{v,k}(ν) ψ*_{v',k'}(ν) ] X[c',v',k']
```

vs the `Q=0` code (`ψ*_c(k) ψ_{c'}(k')` on the conduction leg). Element by
element the **only** difference is the conduction pair density uses `ψ_{·,k+Q}`
instead of `ψ_{·,k}`. Both `k, k'` stay on-grid ⇒ `k−k'` stays on-grid, so:

- `W_{μν}(k−k')` — argument independent of `Q` (only the c-leg shifts) ⇒ the
  existing coarse `W_q` tiles serve **unchanged**.
- The convolution `U[k] = (1/√Nk) Σ_q W[q] T[k−q]` is a 3-D FFT over the coarse
  `k`-grid — **unchanged**.
- The conduction pair density is evaluated at the **same** centroids `r_μ`, using
  `ψ_{·,k+Q}(r_μ)` from htransform — **no `ζ` refit** (the direct term contracts
  ψ-at-centroids with the precomputed `W`, it never re-fits interpolation
  vectors).
- Valence leg fully on-grid — identical to `Q=0`.

**Conclusion: the direct term needs only a finite-`Q` roll of the conduction ψ
in the T-encode; `W` and the `k`-convolution are literally the same arrays.**
The finite-`q` `W_q` roll machinery already landed on-branch (roll conduction by
`+q`, NO umklapp phase — validated to `1e-4–1e-5` vs disk tiles across all IBZ
`q`; convention derived from the periodic FFT-convolution `χ_q ∝ Σ_k G_c^k
G_v^{*,k+q}`). Direct-kernel-only arbitrary-`Q` TDA BSE is **~180 LOC prod /
~120 test**, unblocked today.

## II.3 The `ζ`-structure shortcut FAILS — measured

**Tested hypothesis (owner-invited, not assumed):** is `ζ_q(μ,G)` the
`|q+G|`-sphere-windowed FT of a **q-independent** real-space `ζ_μ(r)`, so that
`V_Q[μν] = Σ_G ζ̃*_μ(Q+G) v(Q+G) ζ̃_ν(Q+G)` is directly evaluable at any `Q` from
one stored object + analytic `v`, with only the `G=0` head separate?

**Answer: NO. The fit is genuinely `q`-dependent.**

Fixture: MoS₂ 3×3, 30 Ry, charge channel, `n_μ=640`, full-BZ 9 `q`. G-flat
convention verified from source; index mapping validated (`g0_μ == ζ̃[q,:,G=0]`
to machine zero); `±q` TRS pairs give `m_mag = 0.000` exactly, confirming the
reconstruction is correct.

**Part A — neighbour-`q` shape/phase** (64-centroid subset, 18 pairs). Metric
`p = min_φ ‖ζ_a − e^{iφ}ζ_a'‖ / ‖ζ_a‖` (residual after removing best per-μ global
phase — a gauge):

| quantity | median | max |
|---|---|---|
| `p_ζ` (real-space ζ, H1 test) | **1.35** | 2.12 |
| `p_R` (cell-periodic `e^{-iq·r}ζ`, H2 test) | **1.50** | 2.46 |
| `m_mag = ‖\|ζ_a\|−\|ζ_b\|‖ / ‖\|ζ_a\|‖` (gauge-free) | **0.39** | 1.07 |

`p ≈ √2 ≈ 1.41` is the *orthogonal-vectors* value — neighbouring-`q` `ζ` for the
same centroid are nearly uncorrelated after optimal phase alignment, in both the
lab-frame and cell-periodic representations. Neither H1 nor H2 holds.

**Part B — band-limit-controlled magnitude** (common 1507-G set, identical
truncation every `q`): `m_mag` median **0.396**, max 1.07. **The magnitude field
`|ζ_q(μ,r)|` genuinely changes ~40% between adjacent `q`** — not a window
artifact.

**Part C — master-ζ prediction** (all 640 μ). Build "master" `ζ_Γ`, predict
`ζ̃_q` via `fftn(e^{-2πiq·r} ζ_Γ)|_sphere(q)`, form `V_q^pred` with
`v = 8π/|q+G|²`, apply the analytic centroid phase `e^{2πiq·(r_ν−r_μ)}` as the
best correction:

| target `q` (frac×3) | ζ̃ resid | `V_q` raw rel-Frob | `V_q`+centroid-phase | diag rel |
|---|---|---|---|---|
| (0,1) nearest | 0.85 | 1.35 | 1.49 | 0.74 |
| (1,0) nearest | 0.80 | **0.91** | 1.15 | **0.40** |
| (1,1) | 0.95 | 1.06 | 0.99 | 0.39 |
| (0,2) | 1.46 | 3.36 | 3.41 | 2.27 |
| (2,2) far | 1.31 | 3.44 | 3.52 | 2.01 |

Even the best (nearest-neighbour) prediction is **40% wrong on the diagonal and
~90% in Frobenius**, degrading to 200–350% at farther `q`. The analytic
centroid-phase correction does not help (often hurts). **`V_Q` is NOT
reconstructible from one master `ζ` (H3 rejected).**

**Why (physics).** `ζ_q` interpolates the *span* of momentum-`q` pair densities
`{u*_{v,k} u_{c,k+q}}`. That span rotates substantially with `q` (different band
pairings, different `k`-coupling through `Σ_k P*_{k-q}P_k`), so the least-squares
vectors are genuinely `q`-specific. This is the ISDF analogue of "the screened-
exchange basis is `q`-dependent" — not removable by bookkeeping. It is precisely
the *inversion side* of the §I.3 argument: the ingredients are smooth, but
`ζ = C^{-1}Z` rotates. **Deferring exchange in the direct-only scope is
vindicated, not a corner cut.**

## II.4 Exchange-term options, ranked

For when exchange is added on top of direct-only (all need `V_Q`/`ζ_Q`):

1. **Compute-don't-interpolate — per-`Q` `ζ` refit (default; honest).** Run the
   ζ-fit at the single momentum `Q` with htransform'd `ψ_{c,k+Q}` on the full
   r-grid → exact, no interpolation error, reuses the validated ζ-fit path. Cost
   = one r-chunk ζ-fit for one `q` + htransform ψ(full-r) at `{k+Q}` (`~1/n_q` of
   a full GW ζ-fit). Head `v(Q+G=0)` stays analytic-separate. **Best when few
   `Q`.** The §II.3 finding makes this the only route with *no uncontrolled
   error*.
2. **Gaussian SR/LR split + interpolate `V^SR_Q`** (§II.6, `coulomb_sr_lr.md`).
   Split `v = v_SR + v_LR`; `V^SR` is smooth (divergence removed), interpolate
   across the coarse grid, re-add `v_LR` analytically at `Q`. Basis-agnostic
   (per-`G` scalar multiply through the existing centroid contract). **Caveat:**
   the ~40% coarse-`q` variation of `ζ` still lives inside `V^SR`, so the
   interpolation error is bounded by coarse-grid ζ-smoothness, not by the split —
   the split cures the *divergence*, not the basis rotation. Viable for **dense
   `Q`** / many `Q`.
3. **Compute-don't-interpolate `W_Q`** (screened exchange / full BSE). htransform
   ψ → pair basis at `Q` → the validated `ω=0` resolvent generates `W_Q` directly
   (no W interpolation). Cost = one resolvent solve per `Q` (finite-`q` `W_q`
   resolvent already on-branch, closes at the GW minimax-quadrature floor
   `~2e-9`–`5e-8`). W-analogue of option 1.

**Direct-eval-from-master-`ζ`: REJECTED** by §II.3.

## II.5 Literature survey + mapping

Three literatures do exactly this problem for Coulomb-mediated quantities; all
converge on the **subtractive** (not multiplicative) convention, and the closest
analogue (exciton-Wannier) is *literally* a rank-1-head factorization.

### II.5.1 Exciton-band interpolation — the direct analogue

**Haber, Qiu, da Jornada, Neaton, "Maximally Localized Exciton Wannier
Functions," PRB 108, 205109 (2023) / arXiv:2308.03012.** Splits the singlet
exciton Hamiltonian and interpolates only the short-range part:

```
H^Xct(Q) = H^SR(Q) + 2 δ_S K^LR(Q)                                      (Eq 32)
K^LR(Q)  = K^{X,Dip}(Q) − K̄^{X,Dip}(0)                                 (Eq 33)  ← SUBTRACT Q=0 dipole
K^{X,Dip}_{MN}(Q) = (4πe²/V_uc) Σ_G [P*_M·(Q+G)][P_N·(Q+G)] / |Q+G|²    (Eq 27)  ← rank-1 in dipole P
K^{NA}_{MN}(Q) ≡ lim_{G=0,Q→0} = (4πe²/V_uc)(P*_M·Q̂)(P_N·Q̂)            (Eq 28-29) ← direction-dep, finite
```

`K^{X,Dip}` is a dyadic in the exciton transition-dipole `P_M`; the `G=0` head
`(P*_M·Q̂)(P_N·Q̂)` is direction-dependent, bounded in 3D, and **is `g0⊗g0`** (`P_M`
= LORRAX's `g0`). The short-range `H^SR(Q)` is FT'd to the exciton-Wannier
lattice `R̄` (decays rapidly ⇒ interpolation-safe); `K^LR(Q)` is re-added
analytically from the smooth dipoles + closed-form `1/|Q+G|²`. **This is exactly
the rank-1-head factorization: interpolate the smooth body, carry the smooth
`g0`, reassemble with analytic `1/|Q|²`.** Earlier precedent: Kammerlander,
Botti, Marques, Marini, Attaccalite, arXiv:1209.1509 (double-grid BSE).

### II.5.2 Polar electron-phonon — the subtractive lineage

**Verdi & Giustino, PRL 115, 176401 (2015) / arXiv:1510.06373** — Fröhlich vertex
`g = g^S + g^L`, `g^L ∝ (q·Z*·e)/(q·ε·q)` (Eq 2, 4), with the explicit recipe:
"(ii) subtract `g^L`; (iii) Wannier-interpolate `g^S`; (iv) add short- and
long-range **after** interpolation." **SUBTRACT → interpolate remainder → ADD
analytic LR.** Parallel: Sjakste, Vast, Calandra, Mauri, PRB 92, 054307 (2015).

**Brunin et al., PRL 125, 136601 (2020) / arXiv:2002.00628** — the *cautionary*
result: dipole-only subtraction is **not enough**; the next order in `q`
(dynamical **quadrupole**) is finite but angular-discontinuous at `q→0`, and if
left inside `g^S` produces unphysical interpolation oscillations near Γ. Fix =
extend `g^L` to the quadrupole. **Lesson for LORRAX: the subtractive split is
only as good as the analytic LR model; if the removed channel misses nonanalytic
structure the "smooth body" is still non-interpolable** — the e-ph mirror of the
§II.3 ζ-rotation caveat.

**2D — Sohier, Calandra, Mauri, Nano Lett. 17, 3758 (2017) / arXiv:1612.07191:**
in 2D `v(q)=2π/(|q| ε_2D(q))`, so the head is **linear in `|q|` with a finite but
direction-discontinuous slope** at `q→0` (nonanalytic first derivative, not a
`1/q²` pole). The analytic re-add must use the 2D-truncated Coulomb.

### II.5.3 ISDF/THC across momentum — two conventions

- **q-independent auxiliary basis** (Lee & Reichman, arXiv:2304.05505; k-point
  RPA-THC, doi:10.1021/acs.jctc.3c00615): the interpolation vectors `ζ_μ(r)` are
  cell-periodic and `k`-independent, all `q`-dependence in Bloch phases. Reusing
  `ζ` across `q` is trivial *because `ζ` was fit to the union span of all pairs
  at once* (larger rank, not tuned per `q`).
- **per-`q` least-squares `ζ_q` (LORRAX's convention):** fit to the *specific*
  pair span at that `q`, which rotates (§II.3, ~40%). LORRAX sits in the
  convention where `ζ` is genuinely `q`-dependent and not directly interpolable.

Two honest routes: (a) recompute `ζ_Q`/`g0(Q)` per `Q` (compute-don't-
interpolate; cheap for the G=0 slice), or (b) migrate to a union-span/global `ζ`
if dense-`Q` dispersion makes per-`Q` refits dominate (larger-rank, separate
design). AFQMC-ISDF (arXiv:1810.00284), complex-k-means ISDF (arXiv:2208.07731)
also do not interpolate `ζ` across `q`.

### II.5.4 Mapping table — literature object ↔ LORRAX object

| literature object | source | LORRAX object |
|---|---|---|
| exciton dipole `P_M` (Haber Eq 27) | 2308.03012 | `g0_μ = ζ̃(q,μ,G=0)` — head channel vector |
| rank-1 dipolar head `K^{X,Dip}` (Eq 27) | 2308.03012 | `V_Q^LR = v(Q)·conj(g0)⊗g0` (`apply_q0_head_rank1`) |
| short-range `H^SR(R̄)` (Eq 32/20) | 2308.03012 | `V_Q^SR = V_qmunu` (G=0-zeroed) — the smooth body |
| subtract `K̄^{X,Dip}(0)` before FT (Eq 33) | 2308.03012 | subtract `V_Q^LR` before interpolating (α→∞ limit) |
| e-ph `g = g_S + g_L` (Verdi Eq 2,4) | 1510.06373 | `v(Q+G) = v_SR + v_LR` per-`G`; `V_Q = V_SR + V_LR` |
| SUBTRACT-interpolate-ADD (Verdi ii–iv) | 1510.06373 | interpolate `V_SR/W_SR`, re-add `v_lr_at_qG` analytically |
| quadrupole for smooth `g_S` (Brunin Eq 3) | 2002.00628 | the ζ-rotation residual (§II.3): body still ~40% coarse-`q` variable |
| 2D `v=2π/(\|q\|ε_2D)`, linear-`\|q\|` head (Sohier) | 1612.07191 | slab-truncated Coulomb `f_2D`; 2D head `~2π/\|Q\|` |
| 2D exchange head `A\|Q\| + A\|Q\|e^{−i2θ}` winding-2 (Qiu Eq 9,10) | 1507.03336 | directional `g0(Q̂)`-carried head |
| k-THC q-independent `ζ_μ(r)`, q in phases | 2304.05505 | the master-ζ hypothesis §II.3 REJECTED for LORRAX |

### II.5.5 2D nonanalytic `q̂` — implication for the MoS₂ fixtures

**Qiu, Cao, Louie, PRL 115, 176801 (2015) / arXiv:1507.03336** is the
load-bearing 2D reference. Their exchange kernel expands (Eqs 9–10) as

```
intravalley:  ⟨S^K_Q|K^x|S^K_Q⟩   = C + A|Q| + βQ²
intervalley:  ⟨S^K_Q|K^x|S^{K'}_Q⟩ = A|Q| e^{−i2θ} + β'Q²        (θ = polar angle of Q)
```

Two consequences for MoS₂ (`sys_dim=2`):

1. **The head is a `|Q|` cusp, not a `1/|Q|²` pole** — the analytic re-add must
   use the 2D-truncated Coulomb `2π/|Q|·f_2D`, not 3D `1/|Q|²`. LORRAX has `f_2D`;
   the rank-1 factorization inherits this **iff** the head's `v(Q)` is evaluated
   through `get_kernel(sys_dim).v_qG`, not a hardwired 3D form.
2. **The head is direction-dependent, winding number 2** (`e^{−i2θ}`). A single
   isotropic scalar averages this away — correct only at the single coarse `Q=0`
   point (Baldereschi–Tosatti). Once `Q` refines toward 0 the isotropic scalar is
   wrong; the rank-1 head `g0(Q)⊗g0*(Q)·v(Q)` carries the winding-2 angular
   structure **naturally, provided `g0(Q)` is the Q̂-dependent `G=0` projection
   (transition-dipole orientation), not a frozen vector.**

## II.6 Owner rulings (settled)

### `|Q|²`-multiply-and-divide: REJECTED for the contracted tile

The BGW-style multiplicative trick (interpolate `|Q|²·(divergent quantity)`,
divide by `|Q|²` after) is only well-posed on an **isolated single divergent
channel** with a common `1/|Q|²` prefactor (a scalar `ε⁻¹₀₀(q)`, a single-`G`
`v(Q+G)`, BGW's head stored as `1.0`). LORRAX's `V_Q[μν]` is a **sum** of a
divergent `G=0` head + a smooth `G≠0` body — there is no common `1/|Q|²` to
factor out of a sum. Multiplying the whole tile by `|Q|²` sends the body
amplitude `→0` as `Q→0`; dividing back is `0/0` on the body — it destroys the
smooth information. BGW never does this to a summed object: it keeps head/wing/
body as separate `G`-indexed channels and multiplies/divides only the isolated
scalar head. LORRAX has already contracted over `G`, so it has no separate
channel to multiply unless it first *reconstructs* one — which is precisely the
rank-1 factorization.

### subtract-analytic-LR: CORRECT, and universal

Verdi–Giustino, Sjakste, Brunin (e-ph) and Haber et al. (excitons) all form
`X_SR = X − X_LR`, interpolate the smooth `X_SR`, add closed-form `X_LR` at the
target. Strictly better-conditioned than any whole-object multiply *because
subtraction respects the additive head+body structure*: it removes only the
singular part and leaves the body amplitude untouched (no `0/0`). The rank-1-head
factorization (parent's hypothesis) is **vindicated — it *is* the subtractive
scheme specialized to LORRAX's known rank-1 head.** The exciton-Wannier LR
exchange is literally rank-1 in a dipole and subtracted/re-added analytically;
LORRAX's head is already stored in exactly this form.

### §9.8 ruling — `g0`-winding kills direct head-vector interpolation; finite-α split promoted

The earlier hedge ("if `g0(Q)` is smooth, interpolate it too") is resolved in the
**negative**: `g0(Q) = ζ̃(Q, G=0)` **winds across the BZ** — the "`G=0`" label is
not periodic (at the zone boundary the `G=0` channel at `Q` maps to a different
`G`-channel at the equivalent `Q+G` point), so componentwise interpolation of
`g0(Q)` chases a multivalued object, on top of the 2D winding-2 `e^{−i2θ}`.
**Direct `g0` interpolation is REJECTED.**

Consequence: the analytic LR channel must be **finite-range**, spanning a shell
of small `|Q+G|` — the finite-α Gaussian split:

$$
v_\text{LR}(Q+G) = v(Q+G)\,\exp\!\big(-|Q+G|^2/4\alpha^2\big)\quad(\text{summed over ALL }G;\text{ periodic in }Q),
$$
$$
v_\text{SR}(Q+G) = v(Q+G) - v_\text{LR}(Q+G)\quad(\text{bounded, smooth}).
$$

`Σ_G ζ̃* v_LR ζ̃` is evaluated **analytically/exactly** at each target `Q` (the
divergence *and* the small-`G` winding live here, handled in closed form); the SR
tile `Σ_G ζ̃* v_SR ζ̃` is the interpolable object (subject to the ζ-rotation
falloff study). The `α→∞` / `G=0`-only rank-1 form is the single-coarse-point
`Q=0` special case in production today. Key values: `v_SR(q+G→0) = 2π/α²`
(finite), `v_LR ~ exp(−G²/4α²) → 0` at large `G` (bare-cutoff regime untouched).
`v = v_SR + v_LR` exactly, per-`G`, for 3D and slab (`f_dim` outer factor). Box
is split-exempt (`v_LR ≡ 0`). Since `W` is nonlinear in `V`, split `W` by its
*physical* singularity: `W^LR_q = ε⁻¹₀₀(q)·V^LR_q`, `W^SR_q = W_q − W^LR_q`.

**α selection remains the one open knob:** `α = c_α·(k-grid spacing)` default vs
an exposed `coulomb_sr_alpha`. Large α → more of `W` rides the exact analytic
`W^LR` (safer interpolation) but `W^SR` carries less physics (more sensitive
`ε⁻¹₀₀` cancellation); small α → `W^SR ≈ W` (interpolation does more work). BGW's
implicit choice is α→∞ (head-only).

---

# PART III — The open problem

## III.1 Statement

> **Produce `V_Q` (bare exchange tile) — and, for full/finite-Q BSE, the exchange
> kernel and optionally `W_Q` — at an arbitrary off-grid exciton momentum `Q`, at
> controlled accuracy and minimal cost per `Q`.** The direct kernel is already
> exact and off-grid-free (§II.2); this problem is the exchange/`V_Q` piece.

The scheme must respect four hard constraints, each established above:

- **(a) Confirmed ingredient smoothness (§I.3).** `C_q, Z_q` — and the smooth
  Coulomb-tile ingredient — are `q`-smooth by density-matrix nearsightedness;
  `C_R, Z_R` are short-ranged (interim: `~10 Bohr` MoS₂ / `~14 Bohr` Si). So an
  interpolant *on the ingredient side of the inversion* is licensed.
- **(b) `C⁻¹Z` conditioning amplification (§I.2, §II.3).** `cond(C) ~ 1e7–1e9`;
  the solved `ζ_q` rotates ~40% between adjacent coarse `q` and is **not**
  directly interpolable (measured 90–340% error). Any scheme that interpolates
  `ζ` or a whole `ζ`-derived tile inherits this rotation as error unless the
  fine grid is dense or the rough part is analytically removed.
- **(c) `g0` / BZ-winding constraint (§II.6/§9.8).** The `G=0` head vector is
  multivalued across the BZ (label non-periodic) + 2D winding-2 — so the
  divergent/near-head channel must be handled **analytically in closed form**
  (finite-α Gaussian LR over a small-`|Q+G|` shell), never by componentwise
  interpolation of `g0(Q)`.
- **(d) 2D directional head (§II.5.5).** For slab fixtures the head is a `|Q|`
  cusp with winding-2 `e^{−i2θ}`; the analytic re-add must use `2π/|Q|·f_2D` and
  a Q̂-dependent `g0(Q̂)`, not an isotropic scalar or a frozen vector.

## III.2 The current front-runner (what the falloff study must adjudicate)

**Rank-1-head factorization = subtract-analytic-LR with LORRAX's finite-α
Gaussian LR channel** (the §II.6 ruling made per-fine-`Q`):

1. Persist the smooth body `V_Q^SR[μν] = V_Q[μν] − Σ_G ζ̃* v_LR ζ̃` — for the
   `α→∞` limit this **is** the already-stored `V_qmunu` (G=0 zeroed); for finite
   α it is the SR contraction. No new body production needed for the α→∞ case.
2. Interpolate `V_Q^SR` across the fine grid (uniform-refinement FFT of the body,
   or the dcc/dvv wfn interpolation — that design's choice).
3. Reassemble at target `Q_fi`:
   `V_{Q_fi} = interp(V^SR)(Q_fi) + Σ_G conj(ζ̃_μ(Q_fi)) v_LR(Q_fi+G) ζ̃_ν(Q_fi)`,
   the LR term **analytic/exact** through `get_kernel(sys_dim).v_qG` (3D `8π/|Q|²`,
   2D `2π/|Q|·f_2D`), with the head/winding living entirely in the LR channel.
4. **Never** multiply the whole tile by `|Q|²` and divide back (§II.6). Do
   removal/re-add on the isolated LR channel only.

The rank-1 factorization is the **correct container either way**: it cleanly
separates the divergence (analytic, exact) from the basis rotation (residual
interpolation error). The falloff study decides only **how the body is produced**
(interpolate `V^SR` vs per-`Q` refit), never whether the head is right. For MoS₂
the honest first cut is **per-`Q` `ζ` refit of the body too** (option 1) until
the study shows the body is smooth enough to interpolate at the target fine-grid
density.

Relationship to existing designs: `coulomb_sr_lr.md` owns the `v_qG_split` seam
(`v_lr_at_qG`, α default, `readd_lr_direct`) and the `W^SR = W − ε⁻¹₀₀V^LR`
partition; the anisotropic winding-2 head machinery is `w_head_wings_interp.md`'s
Q̂-directional `W_head`. htransform (§II.1) supplies the off-grid `ψ(k+Q)` any
per-`Q` refit needs.

## III.3 Evaluation criteria

A candidate scheme is judged on:

**Accuracy targets.**
- Coarse non-regression (**must**, algebraic): with the split ON and
  interpolation OFF, coarse eigenvalues match the no-split run bit-for-bit
  (`V^SR + V^LR = V`, `W^SR + ε⁻¹₀₀V^LR = W`). Anchor: the validated Si-SOC
  `~3 meV` vs BGW ledger. Any drift = re-partition bug.
- Per-`G` split exactness: `v_SR + v_LR == v` to `1e-12`, all `sys_dim`, spanning
  α (box returns `v_LR = None`).
- Head-scalar invariance: `vhead`, `whead=wcoul0` unchanged by the split
  (`vhead ≈ 3303.7` reference) — the split re-partitions, does not re-value.
- Reassembled `V_Q` / kernel at an on-grid `Q` matches the directly-computed tile
  to the ISDF fit tolerance (`ζ_Q` refit round-trip); off-grid: physical exciton
  `E(Q)` smooth with correct `Q→0` limit. The **physical contracted observable**
  (kernel block / Σ^B), not the gauge-dependent tile magnitude, is the right
  accuracy metric (tile magnitude/covariance are gauge artifacts).
- 2D: the reassembled head must reproduce the `A|Q| + A|Q|e^{−i2θ}` structure
  (winding-2 preserved), not an isotropic average.

**Cost per `Q`.**
- Direct-only baseline: one htransform pass (`nk_co` small `O(rank³)` eigh's) +
  one BSE solve; `W` tiles / FFT / valence ψ reused. No new large tensors.
- Exchange add-on: either **one r-chunk `ζ` refit per `Q`** (`~1/n_q` of a full
  GW ζ-fit; exact) **or** one interpolation eval + one analytic LR contraction
  (cheap; amortizes over many `Q` but carries the ζ-rotation as interpolation
  error). The crossover (few-`Q` refit vs dense-`Q` interpolate) is set by how
  many `Q` are needed and by the falloff study's verdict on body smoothness.

**Validating gates (1-GPU, MoS₂/Si fixtures — no 16-GPU gating).**
1. htransform on-grid consistency (`Q=0` reproduces the directly-loaded
   `ψ(r_μ), ε`).
2. Direct-term `Q=0` non-regression (bit-for-bit vs existing TDA `D−W`).
3. `k−k'` on-grid invariance (`W_q` index set / FFT grid byte-identical `Q=0` vs
   `Q≠0`).
4. Per-`G` split round-trip + coarse non-regression + α-independence on the
   coarse grid (eigenvalues invariant to α; α only moves weight between `V^SR` and
   `V^LR`).
5. Fine-grid smoothness: `‖V^SR(q)−V^SR(q')‖ ≪ ‖V(q)−V(q')‖` across adjacent
   coarse `q` (the divergent jump lives in `V^LR`).
6. `ζ_Q` refit round-trip: `ζ_Q` at an on-grid `Q` via htransform-fed refit
   matches the stored `ζ_q` to fit tolerance.
7. (end-to-end) BGW absorption `ε₂(ω)` on a Si coarse-4³ → fine-8³ via the
   Haydock compare harness, once the interpolation design lands.

## III.4 The falloff study — RESOLVED (2026-07-17): ingredient interpolation NOT viable; refit or SR/LR-potential only

The study ran (full section: `arbitrary_q_bse.md` §3.5; scripts+logs
`runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/`). Outcome, in the order
of the causal chain:

**(1) The nearsightedness premise (§I.3) is TRUE.** Per-shell `‖C_R‖_F`
(normalised to R=0) decays Green's-function-like to a `1e-3–1e-4` floor:
MoS₂ (2D) dead by ~10 Bohr; Si (3D) only by ~14 Bohr (slower 3D reach).

**(2) The INGREDIENTS interpolate.** Leave-one-out Fourier interpolation of
`C_q`: MoS₂ 4×4 on-grid median **1.3e-3**, off-grid midpoints (2×2→4×4 truth)
**4.0e-2**; vs 90–340% for the master-ζ strawman. (Si 4×4×4 is still
under-resolved: 33%/72%.)

**(3) But `V_Q` reconstruction is DEFEATED by the `ζ = C⁻¹Z` solve.**
`cond(C_q) ~ 1e7–1e9` amplifies the sub-percent ingredient residual past 100%,
and there is **no regularisation window** (MoS₂ 3×3, nR=7, tile error
`‖ΔV‖_F/‖V‖`): raw 3.7e6 → rankcut 1e-6: 1.2e4 → 1e-4: 11 → 1e-2: **1.00** —
light regularisation explodes, aggressive regularisation has already discarded
the signal. The physical contraction `d*V_q d` (with `d ∈ range(C_q)`) tracks
the tile (0.89 at rankcut 1e-2) — no gauge-artifact escape. Grid density does
NOT rescue it: on 6×6 the error GROWS with more R-vectors.

**(4) Mechanism — the falloff does not transfer to ζ.** `ζ_R` is nearly FLAT
(MoS₂ 3×3: 1.00 → 0.82 → 0.65 to the largest |R|): the `C⁻¹` in `ζ = C⁻¹Z`
**de-localizes** ζ in R (the inverse of a short-ranged operator is
long-ranged). This single fact explains both the master-ζ failure (§II) and the
ingredient-interp failure here; interpolating ζ directly also fails and worsens
with more R-vectors.

**Resolution of the §III.2 decision rule:** the interpolate-the-body route is
REJECTED for anything produced through the ζ-fit. Surviving routes, now the
canonical ranking:
1. **Per-`Q` ζ refit** ("compute-don't-interpolate", exact, no uncontrolled
   error) — the production default for arbitrary-`Q` exchange.
2. **SR/LR interpolation of the divergence-removed potential `V^SR_Q`
   directly** (a *potential-level* object that never passes through `C⁻¹`; its
   smoothness is NOT ruled out by this study and remains the one open
   interpolation question) — with the finite-α analytic LR head fixed by
   constraints (c)–(d) regardless.
3. Screened `W_Q`: compute-don't-interpolate via the validated ω=0 resolvent.

For the first-principles agents this file briefs: any scheme you propose must
either avoid the `C⁻¹Z` solve on interpolated inputs entirely, or explain
precisely why its conditioning analysis escapes the measured no-window result.

---

## Load-bearing citations

1. **Haber, Qiu, da Jornada, Neaton, PRB 108, 205109 (2023), arXiv:2308.03012** —
   exciton-Wannier SR/LR split; `K^{X,Dip}` rank-1 in the dipole, subtracted and
   re-added analytically (Eqs 27–34, 43). The direct proof the rank-1-head
   factorization is the right transfer.
2. **Verdi & Giustino, PRL 115, 176401 (2015), arXiv:1510.06373** — polar e-ph
   `g = g_S + g_L`, explicit SUBTRACT→interpolate→ADD recipe (Eqs 2, 4). The
   subtractive convention.
3. **Qiu, Cao, Louie, PRL 115, 176801 (2015), arXiv:1507.03336** — 2D exchange
   head `A|Q|` + winding-2 `e^{−i2θ}` (Eqs 2, 4, 9, 10). Governs the MoS₂ head.
4. **Brunin et al., PRL 125, 136601 (2020), arXiv:2002.00628** — dipole-only
   subtraction leaves a non-interpolable remainder; quadrupole (next order in q)
   needed (Eq 3). Mirror of the ζ-rotation caveat.
5. **Lee & Reichman, JCTC (2023), arXiv:2304.05505** (+ k-point RPA-THC,
   doi:10.1021/acs.jctc.3c00615) — ISDF/THC across q with a q-**independent**
   auxiliary basis; the precedent-and-contrast for LORRAX's q-dependent per-q ζ.
6. Kammerlander, Botti, Marques, Marini, Attaccalite, arXiv:1209.1509 — double-
   grid BSE (interpolate dipoles, treat the Coulomb singularity analytically).
7. Sjakste, Vast, Calandra, Mauri, PRB 92, 054307 (2015) — GaAs polar-optical
   Wannier interpolation (parallel subtractive split).
8. Sohier, Calandra, Mauri, Nano Lett. 17, 3758 (2017), arXiv:1612.07191 — 2D
   Coulomb `v=2π/(|q|ε_2D)`, linear-`|q|` head.
9. Kohn, PRL 76, 3168 (1996); Prodan & Kohn, PNAS 102, 11635 (2005) —
   nearsightedness / exponential density-matrix decay in gapped systems (the
   §I.3 spine).

## Companion designs (LORRAX-internal, for the owner — not required reading)

`arbitrary_q_bse.md` (the living base doc + incoming falloff study),
`coulomb_sr_lr.md` (the `v_qG_split` seam + `W^SR` partition), `w_head_wings_interp.md`
(Q̂-directional anisotropic head), `fine_grid_interpolation.md` (dcc/dvv wfn
interpolation), `finite_q_bse.md`, `kernel_dataflow_trace.md` (the BSE kernel
spine). LORRAX source: `src/isdf/core.py` (ISDF primitives), `gw/isdf_fitting.py`
(fit orchestrator), `gw/v_q_g_flat.py` (the `V_q` contraction),
`gw/head_correction.py` (`apply_q0_head_rank1`), `bandstructure/htransform.py`
(off-grid ψ/ε).
