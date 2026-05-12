# Bispinor GW Theory — canonical mathematical reference for LORRAX

**Status:** reference document. Written to settle equation/sign/index disputes for the
bispinor pipeline now living on `agent/bispinor-design`-derived branches that have
landed on `lorrax_A/main`.

**Scope:** every equation that the LORRAX bispinor code (Phase-1 DHF + bare-Breit)
claims to evaluate, in the order data flows through the pipeline:

```
ψ_L(G)  →  bispinor lift  →  ψ_4 = (ψ_L, ψ_S)
        →  pair density ρ^{μ_L}_{n_l n_r, k, q}(r)
        →  ζ-fit (open-spin Gram path)  →  ζ^{μ_L}_q(μ_c, r)
        →  V_q^{μ_L, ν_L} via transverse projector
        →  Σ^B (transverse bare-Breit) and Σ^C-bare (charge channel reusing scalar Σ_X)
```

This document is the canonical place to look up index conventions, sign conventions,
γ-matrix conventions, the precise einsums, and the indefinite-Gram justification.
Source-of-truth citations to file/line are inlined throughout.

---

## §0. Source materials and citation key

This report consolidates equations and conventions distributed across:

* Original v1 design: `sources/lorrax_B@9397e35:docs/BISPINOR_DHFB_DESIGN.md`
  (read with `cd sources/lorrax_B && git show 9397e35:docs/BISPINOR_DHFB_DESIGN.md`).
* `reports/v_q_bispinor_plan_2026-05-08/report.md` — Lorentz tensor sectorization
  for V_q^{μν} and the unified-tile orchestrator structure.
* `reports/bispinor_pipeline_2026-05-04/report.md` — agent-B's pipeline notes with
  reference Σ^B per-tile values for MoS2.
* `reports/cri3_sigma_blowup_2026-05-05/report.md` — bug-isolation log; §4.1 of the
  v1 design contradicts that report's recommendation. The design says: Schur form
  is wrong for transverse channels because the spin-pair kernel is indefinite.
  The blowup report's later "scalar-metric" attempt put γ̃ on the RHS only — also
  not the design. Open-spin Gram (§4 below) is the design-intended path.
* Source code on `lorrax_A`:
  * `sources/lorrax_A/src/common/gamma_matrices.py`
  * `sources/lorrax_A/src/common/bispinor_init.py`
  * `sources/lorrax_A/src/common/isdf_fitting.py` (the four open-spin functions
    and the `vertex_mu_L` dispatch)
  * `sources/lorrax_A/src/gw/sigma_x_bispinor.py`
  * `sources/lorrax_A/src/gw/v_q_bispinor.py`
  * `sources/lorrax_A/src/gw/cohsex_sigma.py`

Where this report says "(sign: ?)", the corresponding sign was not derivable from
the code alone without retracing a prior convention; correctness should be
verified against an explicit pen-and-paper reduction before relying on it.

---

## §1. Conventions

### 1.1 Indices

| Symbol                   | Range                  | Meaning                                                     |
|--------------------------|------------------------|-------------------------------------------------------------|
| α, β, γ, δ               | 1..4                   | Bispinor (Dirac) component                                  |
| a, b                     | 1..2                   | Pauli when blocking ψ as ψ = (ψ_L, ψ_S), each 2-component   |
| μ_L, ν_L                 | 0..3                   | Lorentz / 4-vector index ("L" = Lorentz, not "left")        |
| i, j                     | 1..3                   | Spatial subset of Lorentz indices                           |
| μ_c, ν_c, λ_c            | 1..n_rμ (per channel)  | ISDF centroid / interpolation point                         |
| n, m                     | 1..nb                  | Band                                                        |
| k                        | 1..N_k                 | Crystal momentum (full BZ, flat-k convention used in code)  |
| q                        | 1..N_q = N_k           | Momentum transfer                                           |
| K = q + G                | (continuum)            | Reciprocal-space transfer including umklapp G               |
| r                        | continuous / discrete  | Real-space point                                            |
| G                        | reciprocal lattice     | G-vector (integer triplet in LORRAX)                        |
| s, σ                     | 1..2                   | Pauli spin (a synonym for a, b above; "σ" only when the     |
|                          |                        | object is genuinely a Pauli spinor of the L block alone)    |

The bispinor ordering in code is L-then-S, with each block a 2-spinor:

$$\Psi(r) = \begin{pmatrix} \psi_L^{\uparrow}(r) \\ \psi_L^{\downarrow}(r) \\ \psi_S^{\uparrow}(r) \\ \psi_S^{\downarrow}(r) \end{pmatrix} \in \mathbb{C}^4.$$

The Pauli decomposition into L and S blocks follows standard Dirac representation
conventions. `nspinor` in the code refers to bispinor dimension when bispinor=True,
i.e. `nspinor=4`.

### 1.2 γ-matrix convention

LORRAX uses the **absorbed** convention: the γ matrices stored in
`gamma_matrices.py` are not the bare Dirac γ^μ but rather

$$\boxed{ \tilde\gamma^\mu \equiv \gamma^0 \gamma^\mu, \qquad \tilde\gamma^0 = I_4, \quad \tilde\gamma^i = \alpha^i. }$$

This is documented in-source at `sources/lorrax_A/src/common/gamma_matrices.py:14-17`:
the comment reads "I replace gamma0-3 with gamma0*gamma0-3, so that I can use
psidag = conj(psi) rather than psibar = conj(psi) gamma0".

In the standard Dirac representation, the explicit 4×4 forms are:

$$\tilde\gamma^0 = I_4 = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix},$$

$$\tilde\gamma^1 = \alpha^1 = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & \sigma_x \\ \sigma_x & 0 \end{pmatrix},$$

$$\tilde\gamma^2 = \alpha^2 = \begin{pmatrix} 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \\ 0 & -i & 0 & 0 \\ i & 0 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & \sigma_y \\ \sigma_y & 0 \end{pmatrix},$$

$$\tilde\gamma^3 = \alpha^3 = \begin{pmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \\ 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & \sigma_z \\ \sigma_z & 0 \end{pmatrix}.$$

These match the entries in `gamma_matrices.py:17-35` exactly.

**Consequence of the absorbed convention:** the bispinor pair density is

$$\rho^{\mu_L}_{n_l n_r, k, q}(r) = \Psi^\dagger_{l, n_l, k}(r) \, \tilde\gamma^{\mu_L} \, \Psi_{r, n_r, k+q}(r), \quad \text{NO explicit } \bar\Psi.$$

In any expression that resembles a standard relativistic ψ̄γ^μψ, "γ" here means
**γ̃ = γ⁰γ**. This is load-bearing: charge density is `ρ^0 = ψ†ψ` (positive, real),
and current is `ρ^i = ψ†α^iψ` — both have the kinematic factor of γ⁰ already
built in.

### 1.3 Spectral properties of γ̃^μ (load-bearing for §4–5)

Each γ̃^μ is **monomial** — exactly one nonzero entry per row and per column,
with value ∈ {±1, ±i}. For γ̃^0 the matrix is I_4 (perm = identity, phase = 1
on every row). For γ̃^{1,2,3} the matrix is a permutation matrix times a diagonal
phase matrix.

`gamma_matrices.py:60-73` (the function `_to_perm_phase`) computes the (perm, phase)
decomposition such that

$$\boxed{ \tilde\gamma^\mu_{\alpha \beta} = \mathrm{phase}_\mu[\alpha] \cdot \delta_{\beta, \mathrm{perm}_\mu[\alpha]}. }$$

This is what `gammas_perm` and `gammas_phase` in
`gamma_matrices.py:84-86` contain (one (4,) row of integers and one (4,) row of
complex phases per μ ∈ {0,1,2,3}). Eigenvalues of γ̃^0 = I_4 are {+1, +1, +1, +1}.
Eigenvalues of γ̃^i = α^i are **{+1, +1, −1, −1}** (each α^i is unitary and
Hermitian with two-fold degenerate ±1 spectrum). This indefiniteness is the key
fact behind §4.

### 1.4 Sign / normalization conventions (cited from code)

* **k-grid normalization in Σ kernel:** `cohsex_sigma.py:80` defines
  `_inv_sqrt_nk = -1.0 / sqrt(N_k)`. The sign is bundled into this constant,
  so all downstream `_convolve` calls have the kinematic minus already applied.
  The screened-exchange kernel `sigma_sx` then takes `prefactor=1.0` (line 98)
  and uses this constant inside `_convolve`. The Coulomb-hole kernel `sigma_coh`
  uses `prefactor=-0.5` (line 106), giving an overall +0.5/√N_k after the
  internal minus.

* **Fine-structure constant:** `bispinor_init.py:30` stores
  `halfalpha = 0.00364867628215 = α_FS / 2 = 1/(2·137.036)`.

* **Reciprocal-vector units:** the lift uses `(k+G)·b` with `b` in Bohr⁻¹
  (equivalently, the BGW HDF5 `wfn.bvec` stored in 2π/alat units is multiplied
  by 2π/alat upstream of `bispinor_init.py`). This convention is documented in
  `BISPINOR_DHFB_DESIGN.md` §3 and verified by the audit in
  `reports/cri3_sigma_blowup_2026-05-05/report.md` (2026-05-06 small-component
  audit, "PSP code consistently uses B = wfn.blat * wfn.bvec").

* **Hartree atomic units throughout** (energies in Hartree internally, eV at
  output via `RYD_TO_EV`).

### 1.5 Volume / FFT normalization

* **FFTs:** flat-k IFFT/FFT with `norm='forward'` is used in
  `compute_CCT_from_left_right` and the spin-matrix variant
  (`isdf_fitting.py:516-517`). This means IFFT applies the 1/N factor, FFT does
  not, and the convolution-theorem identity
  `C_q = FFT(conj(IFFT(A)) ⊙ IFFT(B))` reproduces the direct sum
  `Σ_k A^*_k B_{k+q}` without an extra normalization factor.
* **Σ kernel FFTs** (`cohsex_sigma.py:77-79`) use `norm='ortho'`. Different
  consumers, different normalizations — the absolute scale of ζ and V is set
  by which path you came down.

---

## §2. Bispinor lift (kinetic balance)

### 2.1 Equation

For a non-relativistic large-component wavefunction ψ_L_{n,k}(G), the small
(lower) component is built by kinetic balance:

$$\boxed{ \psi_{S, n, k}(G) = \frac{\alpha_{\rm FS}}{2}\, \big(\boldsymbol\sigma \cdot (k + G)_{\rm cart}\big)\, \psi_{L, n, k}(G), }$$

where (k+G)_cart is the Bohr⁻¹ momentum and σ are the 2×2 Pauli matrices
acting on the L-block spin index.

**Source:** `sources/lorrax_A/src/common/bispinor_init.py:12-41`. The relevant
einsum at line 40 is

```python
return jnp.multiply(
    halfalpha, jnp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
)
```

with `sigmadotp[i,j,G]` the Pauli contraction `(σ·(k+G))_{ij}` evaluated at each
G-vector. Explicitly (line 34-37):

$$(\sigma\cdot K)_{ij} = \begin{pmatrix} K_z & K_x - i K_y \\ K_x + i K_y & -K_z \end{pmatrix}_{ij},\qquad K = (k+G)_{\rm cart}.$$

The output `psi_small` has shape `(nb, 2, ngk)` — same shape as the L block.
The full bispinor wavefunction is then assembled as
`Ψ = [ψ_L; ψ_S]` in `nspinor=4` order.

### 2.2 Limit α_FS → 0

By construction ψ_S = O(α_FS · ψ_L). In the formal limit α_FS → 0 the small
component vanishes, the four-spinor reduces to (ψ_L, 0), and any γ̃^i = α^i
contraction (which is purely off-block-diagonal in the L/S blocking) gives zero
between two such states. Therefore:

* Σ^C (charge channel) reduces to the scalar Σ_X / Σ_SX result computed on the
  enlarged ns=4 spin axis with the lower-block zero. Identity-vertex regression
  (`BISPINOR_DHFB_DESIGN.md` §7.1) requires bit-identity here.
* Σ^B → 0 because every γ̃^i term sandwiches an L-S off-block-diagonal element
  that is α_FS-suppressed at both vertices.

### 2.3 Caveats on the lift

* The lift is **not** renormalized post-construction. The four-spinor norm is
  ‖Ψ‖² = ‖ψ_L‖² + ‖ψ_S‖² = ‖ψ_L‖² · (1 + O(α_FS²)). For the purposes of
  Σ^B = O(α_FS²) physics this α_FS²-small renormalization is below the
  truncation accuracy and is not applied. (Open question in `BISPINOR_DHFB_DESIGN.md`
  §9 item 2.)

* The lift is **plane-wave-only**: it captures `σ·p` kinetic balance, not
  `σ·(p + r·V_NL + r·Σ)` corrections that would be needed for higher-order
  DKH-style kinematic balance with non-local potentials (DKH4, σ·v).
  Out-of-scope for Phase 1 (`BISPINOR_DHFB_DESIGN.md` §8).

---

## §3. Pair density and the ISDF approximation

### 3.1 Pair density definition

For a vertex γ̃^{μ_L} ∈ {γ̃^0, γ̃^1, γ̃^2, γ̃^3}, the pair density between
states (n_l, k) and (n_r, k+q) at real-space point r is

$$\boxed{ \rho^{\mu_L}_{n_l n_r, k, q}(r) = \sum_{\alpha\beta=1}^{4} \psi^*_{l, n_l, k, \alpha}(r) \, \tilde\gamma^{\mu_L}_{\alpha\beta} \, \psi_{r, n_r, k+q, \beta}(r). }$$

Special cases:

* μ_L = 0 (γ̃^0 = I_4): `ρ^0 = Σ_α |ψ|²_{αα-summed} = Σ_a (|ψ_L^a|² + |ψ_S^a|²)`
  for n_l = n_r — i.e. **scalar charge density** (positive, real).
  Off-diagonal in (n_l, n_r) it is the spin-traced pair-product
  `Σ_α ψ*_l ψ_r` summed over the bispinor index.
* μ_L ∈ {1,2,3} (γ̃^i = α^i): the components of the **Dirac current**
  `j^i = ψ†α^iψ`, off-block-diagonal in L/S blocks. For pair densities
  at non-relativistic order this is

  $$\rho^i_{nn,k,0}(r) = 2\, \mathrm{Re}\big[\psi_L^\dagger \sigma_i \psi_S\big] = \alpha_{\rm FS}\, \mathrm{Re}\big[\psi_L^\dagger \sigma_i (\sigma\cdot K) \psi_L\big] + O(\alpha_{\rm FS}^3),$$

  i.e. order α_FS — the kinematic-balance suppression in action.

### 3.2 The ISDF approximation

The interpolative separable density-fit (ISDF) of Dong–Hu–Lin replaces the
full (n_l, n_r) pair-density product with a centroid expansion **separable in
the two band labels**:

$$\boxed{ \psi^*_{l, n_l, k}(r)\, \psi_{r, n_r, k+q}(r) \approx \sum_{a=1}^{n_{r\mu}} \zeta_{q,a}(r)\, \big[\psi^*_{l, n_l, k}(r_a)\, \psi_{r, n_r, k+q}(r_a)\big], }$$

where {r_a} is the centroid set and ζ is the interpolation kernel (the "S" in
ISDF — single-band-product separable). Crucially, ζ does **not** carry a band
or band-pair index; it is a property of the centroid grid + the band space the
fit is built against, not of any individual band pair.

**Why ζ should not be band-pair-resolved.** Carrying a band-pair (n_l, n_r) index
on ζ would defeat the entire ISDF compression. For phase-1 the bispinor pair
density adds a Lorentz index but keeps the band-pair separability.

### 3.3 Lorentz extension for bispinors

Phase-1 LORRAX fits ζ separately per Lorentz channel μ_L:

$$\rho^{\mu_L}_{n_l n_r, k, q}(r) \;\approx\; \sum_a \zeta^{\mu_L}_{q, a}(r) \, \rho^{\mu_L}_{n_l n_r, k, q}(r_a),$$

with **two centroid files** in production (`BISPINOR_DHFB_DESIGN.md` §4):

* `centroids_frac_<N>.txt` — k-means weighted by occupied charge density
  ρ_charge(r); used for μ_L = 0.
* `centroids_frac_<M>_current.txt` — k-means weighted by Gordon-decomposed
  current density `W_curr(r) = Σ_{n,k,i} |j^Gordon_{n,k,i}(r)|²` with
  `j^Gordon = Im[ψ_L†∇ψ_L] + ½∇×(ψ_L†σψ_L)`. Used for μ_L ∈ {1,2,3}.

N and M may differ; the rest of the pipeline carries this asymmetry as
`n_rmu_C` (charge) and `n_rmu_T` (transverse). On CrI3 the production sizes
have been ~1500 and ~600.

### 3.4 Two flavors of pair-density tensor in code

The ISDF fit needs the pair-product in two indexings to construct C_q (Gram)
and Z_q (RHS for ζ). LORRAX has both a **spin-traced rank-3** form and an
**open-spin rank-5** form, dispatched by `vertex_mu_L`.

#### 3.4.1 Spin-traced rank-3 pair density (charge channel)

For γ̃^0 = I_4 the spin contraction collapses to a sum: the code uses
the rank-3 accumulator

$$P^{\rm CC}_k(\mu_c, \nu_c) = \sum_{n, \alpha} \psi^*_{l, n, k, \alpha}(r_{\mu_c})\, \psi_{r, n, k, \alpha}(r_{\nu_c}),$$

implemented in `compute_pair_density_spin_traced` (used in
`isdf_fitting.py:1865-1868`) with einsum signature `'kmns,knsv->kmv'`:

* k, k-point index (preserved in flat-k);
* m, ν_c centroid indices on left and right (m and v in einsum tags);
* n, band index (summed);
* s, spin index (summed: I_4 contracts over the 4-spinor axis trivially).

This is the historical scalar-GW pair density extended to ns=4. It is fine for
γ̃^0 but loses information needed for γ̃^i.

#### 3.4.2 Open-spin rank-5 pair density (transverse channels)

For γ̃^i, i ∈ {1,2,3}, the spin contraction is **non-trivially mixed** and
must be deferred to the C_q (Gram) reduction step. The four NEW functions in
the bispinor pipeline that implement this are:

1. `compute_pair_density_spin_matrix` (`isdf_fitting.py:396-434`)
2. `accumulate_pair_density_spin_matrix` (`isdf_fitting.py:437-466`)
3. `compute_CCT_from_left_right_spin_matrix` (`isdf_fitting.py:539-629`)
4. `compute_ZCT_from_left_right_zchunk_spin_matrix` (`isdf_fitting.py:770+`)

Tensor signature for the rank-5 P:

$$P^{(\rm spin)}_{k, \alpha, \beta}(\mu_c, \nu_c) = \sum_{n} \psi^*_{l, n, k, \alpha}(r_{\mu_c}) \, \psi_{r, n, k, \beta}(r_{\nu_c}),$$

einsum `'kmna,knbr->kabmr'` (line 430): note the spin axes a, b are kept open
on both sides (no γ̃ applied here — see §4 below). Output sharding
`P(None, None, None, 'x', 'y')`.

Memory cost: rank-5 with ns² = 16 leading spin entries is 16× the rank-3 form.
But it is still O(N_k · n_s² · n_rμ²) — **NOT** the O(N_k · n_l · n_r · n_rμ²)
"proper Gram" form (the literal band-pair Gram of `BISPINOR_DHFB_DESIGN.md`
§4). The 16× factor is the one paid by separating spin from the n-sum;
the band-pair factor is **not paid**.

The dispatch in `isdf_fitting.py:1482-1495` selects between them:
```python
_spin_matrix_path = (vertex_mu_L != 0)
if _spin_matrix_path:
    P_acc_shape = (nk_tot, nspinor, nspinor, n_rmu, actual_n_rchunk)
    P_acc_sharding = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
else:
    P_acc_shape = (nk_tot, n_rmu, actual_n_rchunk)
    P_acc_sharding = P_sharding
```

(also at the outer CCT site, `isdf_fitting.py:1862-1888`). This is the
"open-spin Gram" path described in §4 of the bispinor pipeline report and §4
of this report.

---

## §4. CCT (Gram) and ZCT — open-spin construction

### 4.1 Definitions

The ISDF normal equations (see §5) involve two q-resolved objects:

* **C_q** — the Gram-like interpolation metric over centroids,
  shape (n_q, n_rμ, n_rμ).
* **Z_q** — the right-hand-side over (centroid, real-space-chunk),
  shape (n_q, n_rμ, n_rchunk).

### 4.2 Charge-channel forms (γ̃^0 = I_4)

For μ_L = ν_L = 0, the existing scalar-GW formulas apply:

$$C^{(0,0)}_q(\mu_c, \nu_c) = \sum_k\, P^{*}_{l,k}(\mu_c, \nu_c)\, P_{r,k+q}(\mu_c, \nu_c),$$

evaluated efficiently via convolution-theorem:

$$C^{(0,0)}_q = \mathrm{FFT}_{R\to q}\!\big[ \overline{\mathrm{IFFT}(P_l)}(R)\, \cdot\, \mathrm{IFFT}(P_r)(R) \big],$$

with `norm='forward'` IFFT and FFT (`isdf_fitting.py:516-517`). The product is
**element-wise** in (μ_c, ν_c). Both P_l and P_r here are spin-traced
rank-3.

Similarly for Z_q on (μ_c, r-chunk) instead of (μ_c, ν_c).

### 4.3 Transverse-channel form (γ̃^i, open-spin)

The general bispinor C_q with vertices γ̃^{μ_L} (left) and γ̃^{ν_L} (right) is

$$\boxed{ C^{\mu_L \nu_L}_q(\mu_c, \lambda_c) = \mathrm{FFT}_{R\to q}\!\!\Big[ \sum_{\alpha\beta\alpha'\beta'} \tilde\gamma^{\mu_L}_{\alpha\alpha'} \tilde\gamma^{\nu_L}_{\beta\beta'} \cdot \overline{\mathrm{IFFT}(P_{l, \alpha\beta})}(R; \mu_c, \lambda_c) \cdot \mathrm{IFFT}(P_{r, \alpha'\beta'})(R; \mu_c, \lambda_c) \Big]. }$$

This is implemented in `compute_CCT_from_left_right_spin_matrix`
(`isdf_fitting.py:539-629`). The γ̃·γ̃ contraction is the
`_gamma_double_contract` helper at lines 361-393, which exploits the
monomial structure (perm + phase, no matmul) to reduce the spin axes from
rank 5 to rank 3 inside the JIT.

In monomial form, with `γ̃^μ_{αβ} = phase[α]·δ_{β, perm[α]}`, the inner
spin contraction becomes

$$\sum_{\alpha\beta\alpha'\beta'} \tilde\gamma^{\mu_L}_{\alpha\alpha'} \tilde\gamma^{\nu_L}_{\beta\beta'}\, P^*_{l,\alpha\beta}\, P_{r,\alpha'\beta'} = \sum_{\alpha\beta} \mathrm{phase}_{\mu_L}[\alpha]\, \mathrm{phase}_{\nu_L}[\beta]\, P^*_{l,\alpha\beta}\, P_{r,\,\mathrm{perm}_{\mu_L}[\alpha],\,\mathrm{perm}_{\nu_L}[\beta]}.$$

This is exactly what `_gamma_double_contract` evaluates (gather + element-wise
phase + sum over (a,b) at lines 380-393).

#### 4.3.1 γ̃ = I_4 collapse check

For γ̃^0 = I_4, perm = (0,1,2,3) (identity) and phase = (1,1,1,1). The
double-contraction reduces to

$$C^{(0,0)}_q(\mu_c, \lambda_c) \to \mathrm{FFT}\Big[\sum_{\alpha\beta} \overline{\mathrm{IFFT}(P_{l,\alpha\beta})}\, \mathrm{IFFT}(P_{r,\alpha\beta})\Big],$$

i.e. the **Frobenius (sum-of-squares) reduction** over the rank-5 spin-pair
indices: identical to the scalar charge-channel result up to which P-tensor
shape was used to enter (the answer is the same because `Σ_α ψ_l*_α ψ_r,α` is
the rank-3 spin-traced form, and its CCT product over k gives the same number
either way). Useful identity-vertex regression.

This collapse is the reason the docstring at `isdf_fitting.py:558-559` says:
"For γ̃^{μ_L} = γ̃^{ν_L} = I_4 (perm = identity, phase = 1) this collapses
to the historical Σ_{αβ} |P_{αβ}|² Frobenius reduction."

### 4.4 Why the rank-5 (open-spin) form is the right one for transverse

This is the central numerical claim of the v1 design and the reason §4.1 of
`BISPINOR_DHFB_DESIGN.md` warns against the cheaper "Schur" form
(spin-traced P with γ̃-weighted CCT post-hoc).

**The Schur form** is

$$C^{\rm Schur}_q(\mu_c, \lambda_c) = \sum_k\, P^*_{l,\rm traced}(\mu_c, \lambda_c; k)\,\odot\, P_{r,\rm traced}(\mu_c, \lambda_c; k),$$

with each P_{traced}(μ, λ; k) = Σ_{n,αβ} ψ*\_{n,k,α}(μ) γ̃_{αβ} ψ_{n,k,β}(λ),
i.e. γ̃ folded into the spin-trace at construction time.

**Why this is wrong for γ̃^i:** expanding the quadratic form
⟨v|C^Schur|v⟩ groups spin and spatial indices the wrong way. Concretely,
expanding the spin-pair index gives

$$\langle v | C^{\rm Schur} | v\rangle = \langle v |\, M\, | v\rangle$$

where M = γ̃* ⊗ γ̃ on the 16-dim spin-pair index. The eigenvalues of
γ̃^* ⊗ γ̃ are products of γ̃'s eigenvalues. For γ̃^0 = I_4 the eigenvalues
are all +1 (so M = I_16, PSD: Cholesky works). For γ̃^i with eigenvalues
±1 (each twice), the product spectrum has 8 eigenvalues +1 and 8
eigenvalues −1 — **indefinite**. Cholesky fails (NaN); naive LU produces
catastrophic amplification on null modes (see `cri3_sigma_blowup` 2026-05-05
log: 10⁵–10⁶× blowup from this exact failure mode).

**The open-spin Gram form fixes this** by contracting spin at the same spatial
point everywhere. Concretely, write the bispinor pair density with γ̃ at the
left vertex as

$$\tilde\rho^\mu_{nm,k,q}(r) \equiv \sum_{\alpha\beta} \psi^*_{n,k,\alpha}(r)\, \tilde\gamma^\mu_{\alpha\beta}\, \psi_{m,k+q,\beta}(r).$$

Then

$$\langle v | K_q | v\rangle = \sum_{n_l, n_r, k_l} \big| \sum_{\mu_c} v_{\mu_c} \tilde\rho^\mu_{n_l n_r, k_l, q}(r_{\mu_c}) \big|^2 \;\geq\; 0,$$

i.e. the proper Gram of band-pair vectors, manifestly PSD. The open-spin
construction in LORRAX implements exactly this: γ̃ contracted only **inside**
the spatial-point reduction (CCT/ZCT post-IFFT), never spread across two
band-pair-summed factors.

### 4.5 PSD vs indefinite — what LORRAX actually constructs

Reading `isdf_fitting.py:1862-1888` carefully:

* For μ_L = 0, P_l and P_r are spin-traced rank-3 (line 1865-1868) and
  C_q is built from `compute_CCT_from_left_right` (line 1871). This is
  the historical PSD path → Cholesky works.
* For μ_L ∈ {1,2,3}, P_l and P_r are open-spin rank-5 (line 1878-1881)
  and C_q is built from `compute_CCT_from_left_right_spin_matrix` with
  γ_L = γ_R = γ̃^{μ_L} (line 1887-1888).

**Now: is the open-spin C_q^{i,i} (left = right = γ̃^i) PSD?** In principle,
yes — the construction matches the proper Gram in §4.4 above. In practice,
the dispatch at `isdf_fitting.py:1006-1028` (`compute_L_q_from_CCT`) treats
**all** μ_L ≠ 0 channels as "Hermitian indefinite" and skips Cholesky,
returning the raw C_q for downstream pivoted-LU back-solve. From the
docstring at 937-943:

> For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
> the CCT is Hermitian but **indefinite** — Cholesky NaNs and the LU
> fallback in :func:`solve_zeta_from_L_q` is required.

This is in tension with the proper-Gram analysis in `BISPINOR_DHFB_DESIGN.md`
§4. There are two ways to read the source:

1. The code's "indefinite" comment is a holdover from when γ̃ was being
   applied at the spin-trace step (the Schur form), not at the rank-5
   reduction step. Under the open-spin path γ̃ is at the *Frobenius reduction*
   of the rank-5 P — which preserves PSD. In this reading the LU branch
   is overcautious but not wrong: pivoted LU on a (ridge-regularized) PSD
   matrix is fine.

2. The reduction from rank 5 to rank 3 *via γ̃·γ̃* introduces an indefinite
   weighting that breaks PSD even in the open-spin form. This would happen
   if the FFT/IFFT step doesn't commute with the γ̃ contraction in a way
   that preserves the band-pair-sum-of-magnitudes structure.

**Best current evidence:** the post-fix MoS2 V_q^{1,1}[q=0] ≈ 1e6 and the
post-fix CrI3 V_q^{1,1}[q=0] ≈ 1.78e13 (`cri3_sigma_blowup_2026-05-06`,
"after scalar metric"). The blowup is much smaller than the LU-on-Schur
disaster (1e17), but transverse traces are still ~10× charge — too large
for the α_FS²-suppressed expectation. Whether this remaining factor is
(i) a still-indefinite open-spin C_q, (ii) a current-centroid conditioning
issue, or (iii) something downstream in V_q^{i,j} construction is *not yet
established*. See §10 and the open questions appendix.

(Sign of γ̃·γ̃ contribution to the Gram quadratic form for γ̃^i: ?
A pen-and-paper expansion of `Σ_{αβα'β'} γ̃^μ_{αα'} γ̃^μ_{ββ'} P*_{αβ} P_{α'β'}`
with P_l = P_r = P would settle this. Done correctly the answer is
`Σ_{αβ} |γ̃^μ_α' P_α'β γ̃^μ_β'|² ≥ 0` if γ̃^μ acts as a similarity
transform on each of (α, β) independently. But the FFT in between makes
the band-pair vectors at different k-points combine through a phase — and
this is where the careful sign analysis is needed.)

### 4.6 Z_q — analogous construction

`compute_ZCT_from_left_right_zchunk_spin_matrix` (`isdf_fitting.py:770+`)
mirrors the C_q construction with the right axis being a real-space chunk
of size n_rchunk instead of n_rμ. The same γ̃·γ̃ reduction at the
post-IFFT step (line ~810ish in code, same pattern as CCT) produces

$$Z^{\mu_L \nu_L}_q(\mu_c, r) = \mathrm{FFT}_{R\to q}\!\!\Big[\sum_{\alpha\beta\alpha'\beta'} \tilde\gamma^{\mu_L}_{\alpha\alpha'}\, \tilde\gamma^{\nu_L}_{\beta\beta'} \cdot \overline{\mathrm{IFFT}(P_{l,\alpha\beta})}(R; \mu_c, r)\, \mathrm{IFFT}(P_{r,\alpha'\beta'})(R; \mu_c, r)\Big],$$

shape (n_q, n_rμ, n_rchunk).

For consistency with C_q: when γ_L = γ_R = γ̃^{μ_L} (single channel),
the C_q and Z_q live on the same (n_q, n_rμ, ·) grid and are passed to
the same `solve_zeta_from_L_q` per channel.

---

## §5. ζ from C_q and Z_q — least-squares back-solve

### 5.1 LSQ derivation

The ISDF approximation (§3.2) defines ζ as the minimizer of the L² error
in the pair-density fit:

$$\zeta^{\mu_L}_{q} = \arg\min_{\zeta} \sum_{n_l, n_r, k_l, r}\Big| \rho^{\mu_L}_{n_l n_r, k_l, q}(r) - \sum_a \zeta_a(r)\, \rho^{\mu_L}_{n_l n_r, k_l, q}(r_a) \Big|^2.$$

Setting ∂/∂ζ_a^* = 0 gives the normal equations:

$$\sum_b\, K^{\mu_L}_q(a, b)\, \zeta_b(r) = Y^{\mu_L}_q(a, r),$$

where (using the open-spin form for transverse, scalar form for charge):

$$K^{\mu_L}_q(a, b) = \sum_{n_l, n_r, k_l} \overline{\rho^{\mu_L}_{n_l n_r, k_l, q}(r_a)}\, \rho^{\mu_L}_{n_l n_r, k_l, q}(r_b),$$

$$Y^{\mu_L}_q(a, r) = \sum_{n_l, n_r, k_l} \overline{\rho^{\mu_L}_{n_l n_r, k_l, q}(r_a)}\, \rho^{\mu_L}_{n_l n_r, k_l, q}(r).$$

In LORRAX, K^{μ_L}_q ≡ C^{μ_L}_q and Y^{μ_L}_q ≡ Z^{μ_L}_q in the per-channel
contraction described in §4.

### 5.2 Charge channel — Cholesky back-solve

For μ_L = 0, C_q is PSD by §4.4 and §4.5(reading 1). The back-solve
factorizes C_q = L L^H once per q (in `compute_L_q_from_CCT`,
`isdf_fitting.py:1044, 1063, 1087`), then solves
ζ = (L^H)⁻¹ L⁻¹ Z_q via two triangular substitutions in
`solve_zeta_from_L_q` (`isdf_fitting.py:1212-1213`):

```python
y    = solve_triangular(L, Z_cols, lower=True)
zeta = solve_triangular(L.conj().T, y, lower=False)
```

A small ridge `1e-14·|trace|/n` is added on the 1×1-mesh path
(`isdf_fitting.py:1060-1062`) for safety.

### 5.3 Transverse channels — pivoted LU + ridge

For μ_L ∈ {1,2,3}, the code (correctly or overcautiously — see §4.5) treats
C_q as Hermitian indefinite and uses pivoted LU. The diagonal ridge is

$$\boxed{ \text{LU\_RIDGE} = 10^{-12} \cdot \frac{|\mathrm{tr}(C_q)|}{n_{r\mu}}, }$$

added as `ridge·I_{n_rμ}` (`isdf_fitting.py:1183, 1199-1202`):

```python
n = L.shape[-1]
ridge = LU_RIDGE * jnp.abs(jnp.trace(L)) / n
L_reg = L + ridge * jnp.eye(n, dtype=L.dtype)
return jnp.linalg.solve(L_reg, Z)
```

The choice 1e-12 is calibrated to sit "well below any physically meaningful
eigenvalue but well above the partial-pivoting floor"
(`isdf_fitting.py:1196-1197`). It lifts TRS-paired near-zero modes safely
above the LU stability floor without perturbing well-conditioned modes.

**Note on Bunch-Kaufman:** the natural factorization for Hermitian indefinite
is LDL^T (Bunch-Kaufman). JAX does not expose it, so pivoted LU
(`jnp.linalg.solve` defaults to `lu_solve`) is used — numerically equivalent
for non-singular cases.

### 5.4 Phase 3a padded path

When n_rmu_logical is not divisible by the mesh product, C_q is computed at
a padded extent with zero rows/cols in the pad block (Phase 3a invariant),
and `compute_L_q_from_CCT` slices to logical, factorizes, and embeds the
factor back into a padded matrix with **identity in the pad block**
(`isdf_fitting.py:1024-1028, 1044-1047`). This makes the back-solve see
its committed shape while the logical solution stays bit-identical.

---

## §6. V_q^{μ_L, ν_L} — Coulomb tensor in Lorentz gauge

### 6.1 Bare 4×4 photon propagator in Coulomb gauge

In Coulomb gauge, the bare photon (Coulomb) propagator is **block-diagonal**
in Lorentz indices — the (0, i) and (i, 0) cross terms vanish identically:

$$D^{\mu_L \nu_L}(K) = \begin{pmatrix} v(K) & 0 \\ 0 & v(K)\, t^{ij}(K) \end{pmatrix}, \qquad K = q + G,$$

with

$$v(K) = 4\pi/|K|^2 \quad (\text{3-D, untruncated}), \qquad t^{ij}(K) = \delta^{ij} - \hat K_i \hat K_j, \quad \hat K_i = K_i/|K|.$$

(2-D / slab Coulomb cutoff modifies v(K) but leaves the projector formula
unchanged.)

### 6.2 ISDF-basis V_q

Substituting the centroid expansion, V_q in the ISDF basis is

$$\boxed{ V^{\mu_L \nu_L}_q(\mu_c, \lambda_c) = \sum_{G \in \rm sphere}\, \overline{\zeta^{\mu_L}_{q}(K, \mu_c)}\, v(K)\, t^{\mu_L \nu_L}(K)\, \zeta^{\nu_L}_{q}(K, \lambda_c), }$$

with `ζ^μ(K, μ_c)` the discrete Fourier transform of `ζ^μ(r, μ_c)` over the
Coulomb-cut sphere of G-vectors.

The transverse projector is implemented in `v_q_bispinor.py:161-172`:

```python
def v_per_G_fn(qvec_np_batch):
    qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
    v = base_v_per_G_fn(qvec_arr)                  # (Q, n_G_sph) c128
    K_cart = K_cart_batch_fn(qvec_arr)             # (Q, n_G_sph, 3) f64
    K2 = jnp.sum(K_cart * K_cart, axis=-1)
    K2_safe = jnp.where(K2 > eps_K2, K2, 1.0)
    Khat_ij = K_cart[..., i] * K_cart[..., j] / K2_safe
    if i == j:
        t = (1.0 - Khat_ij)
    else:
        t = -Khat_ij
    return (v * t.astype(v.dtype))
```

Here `i = mu_L - 1`, `j = nu_L - 1` (zero-indexed in the spatial subset), and
`base_v_per_G_fn` returns the standard scalar Coulomb v(K) (line 276-278, the
square of `sqrt_v_batch`). The `eps_K2` guard prevents 0/0 at K = 0.

### 6.3 The 16 (μ_L, ν_L) blocks

The 16 = 4 × 4 Lorentz-tensor blocks decompose by Coulomb gauge as
(`v_q_bispinor.py:55-74`, `BISPINOR_DHFB_DESIGN.md` §2):

| Count | Sector                       | Status                                     | Stored as                    |
|------:|------------------------------|--------------------------------------------|------------------------------|
| 6     | (0,i) and (i,0)              | Zero by Coulomb gauge — never computed     | `_zero_tile()` on read       |
| 1     | (0,0) charge-charge (CC)     | Scalar Coulomb tile                        | `V_qmunu_CC` dataset         |
| 3     | (i,i) transverse-diagonal    | weight 1 − K̂_i², `same_zeta=True`         | `V_qmunu_TT_ii`              |
| 6     | (i,j), i ≠ j                 | 3 unique upper (i<j) + 3 Hermitian fills   | `V_qmunu_TT_ij` for i<j      |

Hermitian-fill rule (`v_q_bispinor.py:530-536`):

$$V^{j, i}_q(\mu_c, \nu_c) = \overline{V^{i, j}_q(\nu_c, \mu_c)}.$$

This is implemented as `jnp.conj(jnp.swapaxes(V_companion, -1, -2))` on read
— the redundant tiles are not stored.

### 6.4 Per-tile orchestrator

`compute_V_q_bispinor_to_h5` (`v_q_bispinor.py:182-442`) loops over the 7
unique tiles (UNIQUE_TILES at line 56-61):

```python
UNIQUE_TILES: tuple[tuple[int, int], ...] = (
    (0, 0),                              # CC
    (1, 1), (2, 2), (3, 3),              # TT diagonal
    (1, 2), (1, 3), (2, 3),              # TT off-diagonal upper triangular
)
```

Each tile is computed by exactly the same `compute_V_q_tile` primitive that
drives the scalar charge-only V_q (`v_q_bispinor.py:349`), with a closure
`v_per_G_fn` from `_make_v_per_G_for_tile` that bakes in the projector for
this tile. The CC tile uses base_v(K); TT tiles use v(K) · t^{i,j}(K).

The q→0 head (`g0_acc`) is materialized only on the CC tile
(`v_q_bispinor.py:319-324`); the TT projector kills the head when K aligns
with the spatial axis (and the residual finite-q head is captured in the
body integral via v(K)·t).

### 6.5 BGW v(q+G) overlay

The BGW v(q+G) overlay (matching the BerkeleyGW Coulomb-cutoff convention)
is applied **only to the (0,0) tile** (`v_q_bispinor.py:323-324`) — the
transverse tiles are pure implementation-side projector applications.
This matches the `bare_coulomb_cutoff` convention discrepancy noted in
the user's MEMORY.md ("LORRAX default = 4·ecutwfc; BGW = ecutwfc — always
set explicitly when comparing").

---

## §7. Σ^B — bispinor bare-Breit self-energy

### 7.1 Equation

Per the v1 design (`BISPINOR_DHFB_DESIGN.md` §3),

$$\boxed{ \Sigma^B_{\alpha\beta}(1, 2) = -\sum_{i, j \in \{1, 2, 3\}}\, \tilde\gamma^i_{\alpha\gamma}\, G^0_{\gamma\delta}(1, 2)\, \tilde\gamma^j_{\delta\beta}\, D^{i j}_{\rm bare}(1, 2), }$$

with G^0 the bispinor 4×4 noninteracting Green's function and
D^{ij}_bare = V^{i,j}_q (the transverse block of the bare photon
propagator) — no transverse screening, no retardation. In ISDF
form, this resembles the scalar Σ_X with γ̃ insertions at the two vertices
and V^{i,j} replacing v.

### 7.2 Implementation strategy: γ̃-insertion via ψ-rewrite

The clever trick in `sigma_x_bispinor.py:62-111` is to fold each γ̃ into
the wavefunction at the appropriate vertex, then call the **unmodified**
scalar Σ_X kernel. Concretely:

* **Left vertex γ̃^i:** rewrite `psi_xn` (the "internal-band" wavefunction
  at the left vertex) as `γ̃^i ψ_xn` on its spin axis — see
  `_apply_gamma_left_to_xn` (line 62-70):

  ```python
  return jnp.einsum('bs,ksxn->kbxn', gamma, psi_xn, optimize=True)
  ```

  This computes `out[k, β, x, n] = Σ_s γ̃[β, s] · ψ_xn[k, s, x, n]`,
  exactly `(γ̃ ψ)_β`. ψ_xn shape is (n_k, n_s=4, n_rμ_T, n_band).

* **Right vertex γ̃^j:** rewrite `psi_yr` (the corresponding right-vertex
  wavefunction, conjugated downstream by `build_G`) as `γ̃^j ψ_yr` on its
  spin axis — see `_apply_gamma_left_to_yr` (line 73-88):

  ```python
  return jnp.einsum('bs,knsx->knbx', gamma, psi_yr, optimize=True)
  ```

  Note the docstring's identity at line 81-86: `build_G` internally applies
  `jnp.conj(psi_yr)` and then contracts on the spin axis t. Folding γ̃^j
  (Hermitian) into psi_yr here yields `conj(γ̃^j ψ) = γ̃^j^T ψ*` which,
  contracted with `psi_yn[k, t, μ, n]` on axis t, gives the correct
  `ψ†_{yr} γ̃^j ψ_{yn}` Hermitian sandwich. **γ̃^j must be Hermitian**
  for this identity — α^i and I_4 both qualify, so the trick is valid for
  every vertex in this code path.

After the ψ rewrite, `_make_cohsex_kernels` (`cohsex_sigma.py`) builds the
G^0 from `wfns_ij.psi_xn / wfns_ij.psi_yr` (now γ̃-folded), `_convolve` with
V^{i,j}_q, and `_project` against the unchanged sigma-band wavefunctions.
The output is the (i, j) tile contribution to Σ^B.

### 7.3 Sum over (i, j) tiles

`compute_sigma_x_bispinor` (`sigma_x_bispinor.py:114-210`) loops the 9
transverse pairs and accumulates:

```python
for i in _TRANSVERSE_INDICES:           # (1, 2, 3)
    for j in _TRANSVERSE_INDICES:       # (1, 2, 3)
        wfns_ij = _wfns_with_lorentz_vertices(wfns_transverse, i, j)
        V_ij = _pad_V_to_padded(reader.get_tile(i, j))
        contrib = sigma_sx_k(wfns_ij, Gij, V_ij)
        sig_x_b = contrib if sig_x_b is None else sig_x_b + contrib
```

The `BispinorVqReader` returns the 6 unique TT tiles directly and the 3
Hermitian-redundant (j, i), j > i, tiles via the conj-swapaxes operation
described in §6.3.

### 7.4 Reduction to scalar Σ_X for γ̃^0 = I

If we replaced the inner-(i, j) loop with the single (0, 0) term and
γ̃ = I_4, `_apply_gamma_left_to_xn` and `_apply_gamma_left_to_yr` are
identity (line 106-109: `(wfns.psi_xn if mu_L == 0 else _apply_...)`),
and the V used is V^{0,0} = scalar Coulomb. So Σ^B with all 9 transverse
loops replaced by (0,0)-only is **byte-identical to the scalar Σ_X**.
This is the identity-vertex regression test in `BISPINOR_DHFB_DESIGN.md`
§7.1.

### 7.5 α_FS → 0 limit

Each Σ^B(i, j) tile factor structure:
* `wfns_ij.psi_xn` carries γ̃^i, which has support **only off-block-diagonal**
  in L/S blocks (since γ̃^i = α^i = block-anti-diagonal).
* `wfns_ij.psi_xn[k, β, x, n]` = (γ̃^i ψ_xn)_β — has nonzero output in the
  L block only when the input had nonzero S block, and vice versa.
* As α_FS → 0, ψ_S → 0 (§2.2), so `(γ̃^i ψ_xn)_β` → 0 in the L block and
  trivially zero in the S block where ψ_S already vanishes.

Therefore `psi_xn_new` after γ̃^i rewrite is O(α_FS) entrywise. The same
applies at the right vertex. The kernel then evaluates a bilinear in
ψ-with-γ̃ folded on both sides, so the result is **O(α_FS²)** —
recovering the standard non-relativistic GW as α_FS → 0. This matches the
v1 design's prediction (`BISPINOR_DHFB_DESIGN.md` §3, last paragraph) and
the "α × σ_X^{0,0}" expectation in `cri3_sigma_blowup_2026-05-05` §"Theory
check".

### 7.6 Symmetry expectations (in-plane MoS2)

For a non-magnetic system with point-group symmetry containing C_3v or higher
(e.g. monolayer MoS2, with full PBE relaxed lattice), the rank-2 tensor
σ^B(i, j) (rows / columns labeled by Cartesian indices i, j ∈ {1, 2, 3})
must transform as a symmetric rank-2 Cartesian tensor under the point group.
For C_3v acting in-plane on (i, j) ∈ {1, 2}, the only invariants are

$$\sigma^B(i, j) = c_\parallel \delta^{ij}_{\rm in-plane} + c_z\, \delta_{i,3}\delta_{j,3},$$

i.e. σ^B(1,1) = σ^B(2,2) and σ^B(1,2) = σ^B(2,1) = 0. The (3,3) component
(out-of-plane) is independently fixed.

agent-B's MoS2 reference values (`bispinor_pipeline_2026-05-04` and
`cri3_sigma_blowup_2026-05-05` "MoS2 80-band run"):
* (1,1) trace: −14.68 eV
* (2,2) trace: −34.29 eV
* (3,3) trace: −70.61 eV
* off-diagonal: < 0.01 eV

The relative spread between (1,1) and (2,2) — factor ~2.3× — is **larger
than C_3 invariance allows**. This indicates that the WFN.h5 used (or the
Σ_X integration grid) does not respect C_3 in MoS2 — most likely because
QE's symmetry detection found only id + σ_h (ntran = 2), enforcing only
the z-mirror but not the threefold rotation. Without C_3 in the discrete
symmetry that LORRAX uses to unfold its k-grid and weight band-pair
contributions, σ^B(1,1) ≠ σ^B(2,2) is allowed.

**Input-side requirement for clean validation:** force QE to include the
C_3 symmetry in `WFN.h5` (`ntran ≥ 6` for C_3v), e.g. by tightening
`scf.in` symmetry tolerance or running with explicit symmetry enforcement.
For 6×6×1 k-grids on MoS2, this gives ntran = 12 (C_6v) and σ^B(1,1) =
σ^B(2,2) to FFT-grid precision.

### 7.7 Σ^C-bare (charge channel)

The bispinor bare exchange has a charge channel **and** a transverse
channel. The charge channel (μ_L = 0, ν_L = 0) is

$$\Sigma^{C{\rm-bare}}_{\alpha\beta}(1, 2) = -G^0_{\alpha\beta}(1, 2)\, V^{0, 0}_{\rm bare}(1, 2),$$

i.e. `sigma_sx_k(wfns, Gij, V_q[CC])` with the un-rewritten ψ — the same
call as the scalar Σ_X with V_q = V_qmunu_CC. The bispinor pipeline reuses
`compute_cohsex_sigma(..., compute_bare_x=True)` for this; no code changes
beyond running on a 4-spinor wavefunction (`cohsex_sigma.py:215-217`).

The total bare exchange in `cohsex_sigma.py:229-240` is

```python
if wfns_transverse is not None and bispinor_v_q_path is not None:
    sig_x_b = compute_sigma_x_bispinor(...)
    sig_x = sig_x + sig_x_b
```

i.e. `Σ_X_total = Σ_X[CC] + Σ_X[Breit-9-tiles]`. Σ_X[CC] is the dominant
piece; Σ^B is the α_FS²-suppressed correction.

---

## §8. Permutation γ̃ optimization (gather + phase, no matmul)

### 8.1 Motivation

Every γ̃^μ in the standard Dirac representation is monomial (one nonzero
entry per row, value ∈ {±1, ±i}). The full 4×4 matmul `Σ_β γ̃^μ_{αβ} X[β]`
amounts to gathering one element of X per output row α, multiplied by a
single phase. This is implemented at hot-loop sites to avoid 4×4 matmul
fan-out in the JIT-compiled inner kernels.

### 8.2 (perm, phase) decomposition

`gamma_matrices.py:60-73` builds, for each γ̃^μ, two arrays:

* `gammas_perm[μ]` — int32 shape (4,): for each row α, the column β such that
  γ̃^μ_{α, β} ≠ 0.
* `gammas_phase[μ]` — complex128 shape (4,): the value of γ̃^μ_{α, perm[α]}.

Such that `γ̃^μ_{αβ} = phase[α] · δ_{β, perm[α]}`.

**Examples** (from inspection of the matrices in §1.2):

| μ | perm    | phase       |
|---|---------|-------------|
| 0 | (0,1,2,3) | (+1,+1,+1,+1) |
| 1 | (3,2,1,0) | (+1,+1,+1,+1) |
| 2 | (3,2,1,0) | (−i,+i,−i,+i) |
| 3 | (2,3,0,1) | (+1,−1,+1,−1) |

(Read columns of gamma_matrices.py:22-35 directly; the permutation reverses
the order for γ̃^1, γ̃^2; γ̃^3 swaps L-S blocks within each spin.)

### 8.3 Helper: `gamma_apply` style rewrite

For a vector X[β] of any shape with leading spin axis,

$$(\gamma\, X)_\alpha = \sum_\beta \tilde\gamma^\mu_{\alpha\beta}\, X[\beta] = \mathrm{phase}[\alpha] \cdot X[\mathrm{perm}[\alpha]].$$

For the **right-multiply form** Σ_α X[α] γ̃^μ_{αβ}, expand using the same
identity:

$$\sum_\alpha X[\alpha]\, \tilde\gamma^\mu_{\alpha\beta} = \sum_\alpha X[\alpha] \cdot \mathrm{phase}[\alpha] \cdot \delta_{\beta, \mathrm{perm}[\alpha]} = X[\mathrm{perm}^{-1}[\beta]] \cdot \mathrm{phase}[\mathrm{perm}^{-1}[\beta]].$$

These identities are used in `_gamma_double_contract`
(`isdf_fitting.py:361-393`) to do the γ̃·γ̃ contraction on the rank-5 P
tensor without 4×4 matmul:

```python
P_r_p = jnp.take(P_r,   perm_L, axis=a_axis)
P_r_p = jnp.take(P_r_p, perm_R, axis=b_axis)
phase_2d = phase_L[:, None] * phase_R[None, :]
phase_bcast = phase_2d.reshape(...)
return jnp.sum(P_l_conj * P_r_p * phase_bcast, axis=spin_axes)
```

### 8.4 Sigma^B left/right vertex application

In `sigma_x_bispinor.py:62-88`, the bispinor σ^B kernel uses the explicit
4×4 matmul rather than the perm/phase form:

```python
return jnp.einsum('bs,ksxn->kbxn', gamma, psi_xn, optimize=True)
```

This is intentional — XLA's matmul fan-out for a (4,4) matrix is a single
small dense kernel, the same compiled cost as the gather+phase form. The
perm/phase optimization matters more inside the inner ζ-fit (rank-5 spin
contraction inside an FFT loop) than at this outer-vertex application.

---

## §9. Symmetry expectations and bispinor-aware unfolding

### 9.1 Symmetry of σ^B as a rank-2 Cartesian tensor

For a non-magnetic ground state with point group G containing G_p (the
point-group subgroup), σ^B(i, j) is a rank-2 Cartesian tensor that must
transform as G_p permits. Specifically, for any g ∈ G_p with rotation
matrix R_g acting on Cartesian (i,j),

$$\sigma^B(i, j) = \sum_{i', j'} R_g(i, i')\, R_g(j, j')\, \sigma^B(i', j').$$

The number of independent components is

* 6 (full 3×3 symmetric) for trivial G_p.
* 4 (σ_∥, σ_z, off-diagonal in-plane = 0 by σ_h, no constraint between σ_xx
  and σ_yy) for G_p ⊇ {id, σ_h}.
* 2 (σ_∥ = σ^B_xx = σ^B_yy and σ_z = σ^B_zz, all off-diag = 0) for
  G_p ⊇ C_3v.
* 2 with σ_∥ = σ_z (single c·δ_{ij}) for G_p ⊇ O_h or T_d.

### 9.2 What ntran=2 (z-mirror only) actually means

If QE's symmetry detection in `scf.in` finds only ntran=2 (id + σ_h), the
WFN.h5 produced has only those two operations in the symmetry block. LORRAX's
unfolding (full BZ from IBZ) and band-pair averaging then enforce only that
σ^B has σ^B_xz = σ^B_yz = 0 (by σ_h flipping z), but **does not** enforce
σ^B_xx = σ^B_yy. The agent-B MoS2 numbers in §7.6 are consistent with this:
substantial (1,1) vs (2,2) mismatch, very small off-diagonals.

### 9.3 Practical consequence

For clean σ^B validation against an analytic α_FS² expectation, the input
WFN.h5 **must** carry the full point group of the relaxed structure. This is
not a bispinor-pipeline issue — it's an upstream QE / pw2bgw configuration
issue. Document this in the Σ^B regression script: "ntran ≥ 6 required
for C_3-respecting σ^B(1,1) = σ^B(2,2)."

### 9.4 4×4 spinor symmetry maps

`BISPINOR_DHFB_DESIGN.md` §5 calls out an outstanding piece:
`src/common/symmetry_maps.py` needs 4×4 spinor rotations (extending the
current 2×2 Pauli ones) to do symmetry-aware unfolding correctly for
bispinor wavefunctions. This is "not yet implemented" as of the v1 design;
the current pipeline runs on full-zone WFN.h5 produced by QE without
symmetry-aware unfolding inside LORRAX.

---

## §10. Numerical regime

### 10.1 Per-band magnitudes — order-of-magnitude estimate

Bare exchange Σ_X^scalar per band ∼ −several × 10 eV for typical valence
states (light-element solids: −37 eV for MoS2 valence; −38 eV for MoS2
80-band).

Σ^B per band ∼ α_FS² · Σ_X^scalar:
* α_FS = 1/137.036, α_FS² ≈ 5.3 × 10⁻⁵.
* For MoS2 valence (−37 eV): expected Σ^B ∼ −0.002 eV per band ∼ **−2 meV per band**.

With the larger band counts in the agent-B MoS2 80-band run (Σ_X^scalar trace
≈ −38 eV at k=0), per-tile traces in the **mV range** are expected.

### 10.2 Comparison to agent-B MoS2 reference (light elements)

From `bispinor_pipeline_2026-05-04` and the MoS2 80-band data in
`cri3_sigma_blowup_2026-05-05`:

| System          | Σ_X[CC] / band | Σ^B(1,1) trace | Σ^B(2,2) trace | Σ^B(3,3) trace | scale       |
|-----------------|----------------|----------------|----------------|----------------|-------------|
| MoS2 32-band    | −37 eV         | −0.019 eV      | (small)        | (small)        | ~mV/band    |
| MoS2 80-band    | −38.93 eV      | −14.68 eV trace | −34.29 eV trace | −70.61 eV trace | ~10–50 mV/(k,n) |
| CrI3 80-band    | −1 eV / (k,n)  | −110 keV / (k,n) | (broken — see §10.4) | | not credible |

The MoS2 numbers are **plausible** at ~9-50 meV per (k,n), which is
α_FS² × Σ_X within an order of magnitude. CrI3 numbers are still
post-LU-fix and contaminated by the residual ζ_current conditioning issue
(§10.4).

### 10.3 Heavy-element scaling

For heavy elements, the relevant α_FS expansion parameter is Zα where Z
is the nuclear charge. For Cr (Z=24), Zα ≈ 0.18, and so α_FS² Z² ≈
3 × 10⁻³ — i.e. ~50× the light-element α_FS² scale. For I (Z=53),
Zα ≈ 0.39, scale ≈ 1.5 × 10⁻². So for CrI3 valence (Σ_X^scalar ∼ −1 to
−5 eV per band), expected Σ^B ∼ −10 to −50 meV per band — substantially
larger than MoS2 but nowhere near the keV/MeV/GeV blowups seen in the
2026-05-05/06 logs.

### 10.4 Open numerical bug status

As of 2026-05-08 (the most recent dated note in the source materials), the
post-LU-fix open-spin Gram path produces:

| Quantity | Pre-fix (Schur-LU) | Post-fix (open-spin Gram + LU) | Theory (α²·Σ_X) |
|----------|-------------------:|-------------------------------:|----------------:|
| MoS2 V^{1,1}[q=0]  | ~10⁵            | ~10⁶                           | (consistent)     |
| CrI3 V^{1,1}[q=0]  | 8.7 × 10¹⁷       | 1.78 × 10¹³                    | ~10¹⁰?           |
| MoS2 Σ^B/band      | ~mV              | ~mV (consistent)                | ~mV              |
| CrI3 Σ^B/band      | ~MeV (broken)    | ~keV (still broken by ~10⁵×)    | ~10 meV          |

The CrI3 Σ^B is still off by ~10⁵× even after the open-spin Gram fix.
The 2026-05-06 audit narrowed the issue to the **current-centroid ISDF
basis conditioning**, not to the small-component lift, the transverse
projector, or the LU stability:

> The remaining blowup is not a large physical Dirac current. It looks more
> like the current-channel ISDF representation is numerically unstable on the
> Gordon-current centroid set.

Same-count charge-derived centroid bases (600 charge centroids subset,
2026-05-06 probe) showed the near-null tail of C_q drops from ~50 eigenvalues
< 1e-10 · trace/n (Gordon-current centroids) to 0 (charge subset). This
suggests that for heavy elements, a **single centroid set** based on
(charge + current) joint weighting may be needed.

---

## §11. End-to-end flow recap

Putting §2–7 together, the data flow is:

1. **Read ψ_L(G)** from QE/BGW WFN.h5 (PHDF5 reader, optionally bispinor=True).
2. **Lift** ψ_L → ψ_4 = (ψ_L, ψ_S) per §2.
3. **K-means weights:** build ρ_charge(r) and W_curr(r) per §3.3; run k-means
   to produce charge centroids (n_rmu_C) and current centroids (n_rmu_T).
4. **For each μ_L ∈ {0, 1, 2, 3}:**
   1. Stream band chunks; build P_l(k, μ, ν) and P_r(k, μ, r) per §3.4.
      For μ_L = 0: rank-3 spin-traced. For μ_L ≠ 0: rank-5 open-spin.
   2. Build C_q (CCT) and Z_q (ZCT) per §4. For μ_L ≠ 0, γ̃·γ̃ contraction
      reduces rank 5 → rank 3 at the post-IFFT step.
   3. Solve C_q ζ = Z_q per §5. For μ_L = 0: Cholesky + triangular solves.
      For μ_L ≠ 0: pivoted LU + ridge.
   4. Stream ζ^{μ_L}_q to disk (`zeta_q.h5` for μ_L=0,
      `zeta_q_mu1/2/3.h5` for μ_L=1,2,3).
5. **For each (μ_L, ν_L) tile in UNIQUE_TILES (§6.4):**
   1. Build v_per_G_fn closure with v(K) for CC, v(K)·t^{i,j}(K) for TT
      (`v_q_bispinor.py:127-174`).
   2. Stream V^{μ_L, ν_L}_q to disk
      (`V_qmunu_CC` for (0,0), `V_qmunu_TT_ij` for transverse).
6. **Σ_X[CC]:** call `compute_cohsex_sigma(..., compute_bare_x=True)` with
   V_q = V_qmunu_CC and the un-modified bispinor wavefunctions. This is
   the dominant piece of the bare exchange.
7. **Σ^B[Breit]:** loop over the 9 transverse (i, j) tiles
   (`sigma_x_bispinor.py:189-205`), for each:
   1. Rewrite `wfns.psi_xn → γ̃^i ψ_xn` and `wfns.psi_yr → γ̃^j ψ_yr` per §7.2.
   2. Read V^{i,j}_q from disk (with Hermitian-fill on i > j).
   3. Call the unmodified scalar `sigma_sx_k(wfns_ij, Gij, V_ij)` and
      accumulate.
8. **Σ_X_total = Σ_X[CC] + Σ^B[Breit]** (`cohsex_sigma.py:240`).

For full SC-COHSEX, this Σ_X feeds back into the QP fixed-point as in the
non-bispinor pipeline.

---

## §12. Open questions / known weak points

1. **Open-spin C_q definiteness.** §4.5 flags an unresolved tension: the
   v1 design says the open-spin Gram is PSD (literal sum of squares of
   band-pair vectors), but the code dispatches all μ_L ≠ 0 channels to the
   pivoted-LU branch in `compute_L_q_from_CCT`. Either the LU branch is
   over-cautious (PSD matrices are also LU-solvable) or the FFT-mediated
   reduction breaks PSD in subtle ways. A deterministic test: build C_q
   for a small system (MoS2 3×3, μ_L = 1), compute eigenvalues, and
   verify they are all ≥ 0 (modulo float roundoff). If yes, the LU branch
   could be replaced by Cholesky for transverse channels too — saving
   ~10% on factorization cost.

2. **Open-spin σ^B reproduction of agent-B MoS2 reference.** The agent-A
   open-spin path (the "spin_matrix" path in `isdf_fitting.py`) has
   not been benchmarked against agent-B's MoS2 reference values for
   σ^B[1,1], σ^B[2,2], σ^B[3,3] traces (cited in §10.2). Until that
   regression passes byte-for-byte (or to ~1% on the per-tile traces),
   the open-spin path's correctness is conjectural.

3. **CrI3 transverse blowup remaining factor.** Even after the LU-on-Schur
   fix, CrI3 σ^B is ~10⁵× too large (§10.4). The narrowed hypothesis is
   current-centroid conditioning. Untested fixes: (a) rebuild current
   centroids from ρ_charge + W_curr joint weighting; (b) use the same
   centroid set for all 4 channels; (c) replace pivoted LU with explicit
   Bunch-Kaufman LDL^T (would require an external linalg library, since
   JAX doesn't expose it).

4. **σ^B(1,1) = σ^B(2,2) in MoS2 disagreement.** §7.6 attributes this to
   ntran = 2 in the WFN.h5 (only z-mirror, no C_3). Verify by running the
   same MoS2 input with QE forced to ntran ≥ 6 (full C_3v) and check
   that σ^B(1,1) = σ^B(2,2) within FFT-grid precision. If they still
   differ, there is a deeper symmetry breakage in the bispinor pipeline
   (most likely the 4×4 spinor symmetry maps in `symmetry_maps.py` —
   noted as "not yet implemented" in `BISPINOR_DHFB_DESIGN.md` §5).

5. **Sign of the Σ^B prefactor.** The v1 design writes
   Σ^B = − Σ_{ij} γ̃^i G^0 γ̃^j D^{ij} (line 13 of
   `BISPINOR_DHFB_DESIGN.md` §3, and again in `sigma_x_bispinor.py`
   line 13). The implementation reuses `sigma_sx_k` which has prefactor
   +1.0 in `_convolve` (not −1.0); the minus is in `_inv_sqrt_nk`
   (`cohsex_sigma.py:80`). So the explicit `-` in the design equation is
   absorbed into the kernel's standing convention. (Sign: confirmed by
   reading `cohsex_sigma.py:80, 87, 98`; this matches the scalar Σ_X
   sign convention. No additional sign flip is applied at the bispinor
   ψ-rewrite step.)

6. **q=0 head correction for transverse channels.** `v_q_bispinor.py:319-324`
   only writes the q=0 head for the CC tile. The 2026-05-08 v_q_bispinor_plan
   §7 item 2 questions whether this is correct: "the transverse projector
   kills G=0 in the limit K̂ → q̂ on the diagonal (i,i) tile, but only
   when K aligns with the i-axis. At generic q→0 the head is finite for
   off-diagonal (i,j)." The current implementation captures the residual
   finite-q head in the body integral via v(K)·t — should be checked
   that this is actually adequate for small q→0 ε⁻¹ behavior.

7. **Fine-structure renormalization of ‖Ψ‖.** §2.3 — ψ_S adds
   O(α_FS²) to the norm, not currently corrected. Below the truncation
   accuracy of Phase 1 but a systematic effect for any future
   self-consistent DHFB iteration that uses the wavefunction norm
   explicitly.

---

## §13. References (file:line)

* `sources/lorrax_A/src/common/gamma_matrices.py:14-86` — γ̃^μ, perm/phase decomp.
* `sources/lorrax_A/src/common/bispinor_init.py:12-41` — kinetic-balance lift,
  `halfalpha = 0.00364867628215` at line 30.
* `sources/lorrax_A/src/common/isdf_fitting.py:344-466` —
  `compute_pair_density_spin_matrix`, `accumulate_pair_density_spin_matrix`.
* `sources/lorrax_A/src/common/isdf_fitting.py:539-629` —
  `compute_CCT_from_left_right_spin_matrix`.
* `sources/lorrax_A/src/common/isdf_fitting.py:770+` —
  `compute_ZCT_from_left_right_zchunk_spin_matrix`.
* `sources/lorrax_A/src/common/isdf_fitting.py:921-1103` —
  `compute_L_q_from_CCT` (Cholesky / passthrough dispatch).
* `sources/lorrax_A/src/common/isdf_fitting.py:1110-1230` —
  `solve_zeta_from_L_q` (Cholesky-back-solve / pivoted-LU dispatch).
* `sources/lorrax_A/src/common/isdf_fitting.py:1370-1572` —
  `_make_fit_one_rchunk_kernel` (per-r-chunk dispatch site).
* `sources/lorrax_A/src/common/isdf_fitting.py:1862-1888` —
  outer dispatch site for spin_matrix path.
* `sources/lorrax_A/src/gw/cohsex_sigma.py:80, 87, 98` — sign / normalization
  constants for Σ_X.
* `sources/lorrax_A/src/gw/cohsex_sigma.py:229-240` — bispinor Σ_X total.
* `sources/lorrax_A/src/gw/v_q_bispinor.py:55-74` — UNIQUE_TILES, ZERO_TILES,
  HERMITIAN_PAIRS.
* `sources/lorrax_A/src/gw/v_q_bispinor.py:127-174` — per-tile transverse
  projector closure.
* `sources/lorrax_A/src/gw/v_q_bispinor.py:182-442` —
  `compute_V_q_bispinor_to_h5` orchestrator.
* `sources/lorrax_A/src/gw/v_q_bispinor.py:450-555` — `BispinorVqReader`.
* `sources/lorrax_A/src/gw/sigma_x_bispinor.py:62-111` — γ̃-folded ψ rewrites.
* `sources/lorrax_A/src/gw/sigma_x_bispinor.py:114-210` —
  `compute_sigma_x_bispinor` orchestrator.
* `sources/lorrax_B@9397e35:docs/BISPINOR_DHFB_DESIGN.md` — v1 design,
  particularly §3 (equations) and §4.1 (Schur vs proper Gram).
* `reports/bispinor_pipeline_2026-05-04/report.md` — implementation milestones.
* `reports/cri3_sigma_blowup_2026-05-05/report.md` — bug isolation history,
  current state of the open numerical issues.
* `reports/v_q_bispinor_plan_2026-05-08/report.md` — Lorentz tensor sectors.
