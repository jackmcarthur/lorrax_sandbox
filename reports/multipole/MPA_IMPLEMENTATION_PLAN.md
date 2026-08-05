# Multipole full-frequency GW in LORRAX — implementation plan

> **Status (2026-07-21): design only; no MPA source implementation exists.**
> This plan was written against `sources/lorrax_D` before the July GW branch split.
> Its MPA-W architecture remains the preferred direction, but its executable order,
> source locations, validation baseline, and several edge contracts are stale. See
> [`CURRENT_STATE_2026-07-21.md`](CURRENT_STATE_2026-07-21.md) before starting work.

Date: 2026-07-06
Checkout: `sources/lorrax_D` @ `agent/memplanner-cleanup`
Source papers (this folder): `multipole-sigma-2025.md` (Leon–Berland–Cardoso, MPA-Σ/MPA-G),
`Multipole-W-metals.md` (Leon et al. PRB 107, 155130, MPA-W).

---

## 0. TL;DR — the lazy reading

LORRAX **already has** a full-frequency dynamic-Σ engine. `gw.ppm_sigma` fits a *single*
plasmon pole `{Ω(q,μ,ν), B(q,μ,ν)}` per ISDF matrix element (Godby–Needs or Hybertsen–Louie)
and analytically convolves it against `G` on a real-ω grid via the CTSP τ-loop. **MPA-W is
that exact machinery with the pole index `p` promoted from 1 to `n_p`.**

So the multipole approach is **not** a new engine. It is:

1. Evaluate `W_c(q, z_i)` at `2·n_p` complex sample frequencies instead of 2 (reuse
   `compute_screening`, one frequency at a time, **spill each to disk**).
2. Replace the 2-point GN/HL fit with an `n_p`-pole Padé fit per `(q,μ,ν)` element
   (**the one genuinely new kernel**), streamed over `(μ,ν)` chunks, poles spilled to disk.
3. Run the existing sigma branch/τ machinery **once per pole**, loading `{B_p, Ω_p}` from
   disk one pole at a time and accumulating `Σ_c += contribution(p)`.

Everything memory-critical the user asked for falls out of "loop the index that already
exists." **Recommendation: implement MPA-W (fit `W`, analytic G·W convolution). Skip MPA-Σ
and MPA-G** (fit Σ directly, Padé Green's function, satellites/spectral functions) as YAGNI
until spectral functions are actually requested — they are a separate, additive layer on top
(§7).

> **Reviewed 2026-07-06 (four-lens Fable panel, §10).** Architecture and MPA-W-only scope upheld;
> four correctness blockers fixed inline (Σ residue `+R_p` not `−2R_p`; mandatory residue refit
> after time-order flip; rescaled/host-side Padé solve; proper complex-target off-axis quadrature
> — the α-refold was unsound) plus the missing q→0 head and a CrI3-is-metallic gate. Honest effort
> is **~1.5–2× a "small port"**, with the interior-crossing χ₀ kernel deferred behind a sampling
> A/B. Read §10 before implementing.

---

## 1. Where LORRAX sits vs. the papers

The papers are plane-wave (`GG'`) codes (yambo). LORRAX differs in two ways that *simplify*
the port:

| Aspect | Papers (yambo) | LORRAX |
|--------|----------------|--------|
| Basis for `W` | reciprocal `W_{GG'}(q,ω)` | **real-space ISDF** `W_{μν}(q,ω)` on interpolation points `r_μ` |
| Σ frequency integral | analytic MPA-W (Eq. 10) *or* MPA-Σ Padé | **real-axis CTSP** time-propagator (`ppm_sigma`), already analytic per pole |
| Current pole count | `n_W ≈ 8–12` | **1** (GN/HL PPM) |

The ISDF index `(μ,ν)` plays the exact role of `(G,G')`: `W_{μν}(q,ω)` is a dense
`n_μ × n_μ` matrix per `q`, fit element-wise. The pair-density `ρ_{nm}(k,q,G)` of Eq. (11)
is, in LORRAX, the ζ/ψ real-space product already contracted inside `_get_sigma_kij_kernel`.
**We do not touch the pair-density path at all** — it is pole-count agnostic.

### 1.1 The single-pole model LORRAX has today

GN-PPM (`minimax_screening.fit_gn_ppm_from_wc_pair`) fits, per element, from `W_c(0)` and
`W_c(iω_p)`:

```
Ω²(q,μ,ν) = -z_p² · W_c(z_p) / [W_c(0) - W_c(z_p)]        (z_p = iω_p)
B(q,μ,ν)  = -½ · W_c(0) · Ω
```

giving the time-ordered single-pole correlation screening

$$
W_c(q,\omega;\mu,\nu) \;=\; B\left[\frac{1}{\omega-\Omega} - \frac{1}{\omega+\Omega}\right]
\;=\; \frac{2B\Omega}{\omega^2-\Omega^2}. \tag{1}
$$

Check: at `ω=0`, `(1) = -2B/Ω = W_c(0)`. ✔ (matches the residue form of MPA-W Eq. 9 at `n_W=1`.)

The sigma τ-kernel (`ppm_sigma._build_W_t_q`) turns `{B,Ω}` into a time propagator

$$
W_c(q,\tau;\mu,\nu) \;=\; B\,e^{-i(\Omega - E_{\text{ref}})\tau}\cdot \mathbb{1}[\text{valid}], \tag{2}
$$

which the CTSP kernel convolves with `G(τ)` and projects onto the real-ω output grid.

---

## 2. MPA-W: the multipole generalization

### 2.1 The model

Promote Eq. (1) to `n_p` complex poles per element (MPA-W Eq. 7 / Eq. 9, ISDF form):

$$
\boxed{\;W_c^{\mathrm{MPA}}(q,\omega;\mu,\nu) \;=\; \sum_{p=1}^{n_p}
R_p(q,\mu,\nu)\left[\frac{1}{\omega-\Omega_p(q,\mu,\nu)}-\frac{1}{\omega+\Omega_p(q,\mu,\nu)}\right]\;} \tag{3}
$$

with complex poles `Ω_p` and residues `R_p`, **both per `(q,μ,ν)` element**. Time ordering
requires `Re[Ω_p]·Im[Ω_p] < 0` (poles in the 2nd/4th quadrant). Setting `n_p=1`,
`R_1 = B`, recovers Eq. (1) exactly — GN-PPM is the `n_p=1` special case, which is the
regression anchor (§6).

Equivalently, using `x ≡ ω²` and `2R_pΩ_p ≡ a_p`, `Ω_p² ≡ b_p`, Eq. (3) is a rational
function of `x` with `n_p` poles:

$$
W_c^{\mathrm{MPA}}(x) \;=\; \sum_{p=1}^{n_p}\frac{a_p}{x-b_p}
\;=\;\frac{P_{n_p-1}(x)}{Q_{n_p}(x)}. \tag{4}
$$

This even-in-ω structure is what makes the fit a small rational-interpolation problem.

### 2.2 Self-energy — reuse the existing convolution, sum over `p`

The MPA-W self-energy (multipole-sigma-2025 Eq. 10) in ISDF form:

$$
\Sigma_c(k,\omega)_{nn} = \sum_m \sum_{\mu\nu}\sum_{p=1}^{n_p}
\tilde\rho_{nm}(\mu)\,\big[+R_p\big]\,\tilde\rho_{nm}^{*}(\nu)
\left[\frac{f_m}{\omega-\varepsilon_m+\Omega_p}+\frac{1-f_m}{\omega-\varepsilon_m-\Omega_p}\right]. \tag{5}
$$

> **Review correction (R1/R4, blocker).** The residue is `+R_p` **under Eq. (3)'s
> convention**, *not* the `−2R_p` of multipole-sigma-2025 Eq. (11) — that paper's `−2`
> absorbs a different `R`/spin normalization and is inconsistent with the metals paper's
> Eq. (8) (coefficient `+vR_nv`). Two independent derivations (contour convolution both
> half-plane closures; COHSEX static-limit referee `Σ_m(½−f_m)W_c(0)` with
> `W_c(0)=−2Σ_p R_p/Ω_p`) give `+R_p`. `−2R_p` yields −2×COHSEX. The reuse-the-GN-driver
> code path is already correct (it convolves with `+B`); the danger is only that a test
> written from the *printed* equation would be off by −2. See §10 adjudication F1.

The `p`-sum is **outside** everything else. Each term is structurally identical to the
`n_p=1` term LORRAX evaluates now via Eq. (2) + the CTSP kernel. Therefore:

> **`Σ_c = Σ_p Σ_c^{(p)}`, where `Σ_c^{(p)}` is one call to the existing branch/τ driver with
> `{B_p ≡ R_p, Ω_p}` in place of `{B, Ω}`.**

This is the crux and the reason the port is small: the frequency convolution, the 4-branch
`ω∈ℝ` split, the minimax windows, the `k·i·j` streaming accumulator — none of it changes.

### 2.3 Complex poles and windowing (the one real gotcha)

GN gives *real* `Ω` (`√` of a positive `Ω²`). MPA gives *complex* `Ω_p`. The τ-phase Eq. (2)
already runs in `complex128`, so complex `Ω_p` flows through the kernel untouched — `Im[Ω_p]`
is the intrinsic plasmon broadening and is physically wanted.

But the **minimax window planner** (`_build_windows_for_branch`) places τ-nodes using pole/band
*energies* assumed real (it partitions `E_A ± E_ref` intervals). Rule:

- **Window classification and `E_ref` use `Re[Ω_p]`.**
- **The τ-phase uses the full complex `Ω_p`.**

`Re[Ω_p]` sets which of the 4 ω-branches / 3 conduction windows a pole lands in; `Im[Ω_p]`
only damps.

> **Review correction (R3, major): this is NOT a 1-line change — it is a two-array threading
> job through ~6 signatures.** The pipeline never delivers complex `Ω` to the kernel today:
> `_prepare_sigma_state` (`ppm_sigma.py:343`) computes `Omega_abs = maximum(Re(Ω_q), 0)` and the
> driver passes *that* everywhere; `_materialize_window_mask_B` (`:278`), `_masked_stats_device`
> (`:261`), and `_build_windows_for_branch` (`:1128`) all `min/max/compare` `Ω` and break on
> complex dtype. Fix: thread **two** arrays — a real windowing-`Ω` (unchanged) and a complex
> phase-`Ω` — through `_prepare_sigma_state → _run_sigma_branch → _integrate_tau_windows_for_branch
> → _tau_kernel/_build_W_t_q`, plus `precompile_sigma`. Mechanically simple, gated by the GN
> regression, but ~6 signatures, not one line. Also: `B_mask_raw = Omega_abs > 1e-14` (`:345`)
> drops MPA poles with tiny `Re Ω` but finite `Im Ω` — switch the validity mask to `|Ω_p|`.

---

## 3. The MPA fit (the only new physics kernel)

Given `W_c(q, z_i; μ, ν)` at `2n_p` complex samples `{z_i}`, solve Eq. (4) per element for
`{b_p = Ω_p², a_p = 2R_pΩ_p}`. Use the **linear-algebra (Padé-in-`x`) method** — it is fully
batched (vmap over `(q,μ,ν)`), needs no per-element Newton iteration, and is the method the
papers cite for robustness.

### 3.1 Sampling grid

Double-parallel sampling in the *complex* plane (Leon 2021 / metals paper §II C):

- Two lines parallel to the real axis: `Im z = ϖ₁` (near, `≈0.1` Ha) and `Im z = ϖ₂` (far,
  `≈1` Ha). `n_p` points on each → `2n_p` samples.
- Real parts on a semi-homogeneous power-of-two partition on `[0, ω_m]`, denser near 0
  (metals paper Eq. 11, exponent `α=1` semiconductors, `α=2` for low-ω structure).
- `ω_m` ≈ max valence→conduction transition, or the classical plasmon energy.
- For the intraband/metal case, shift the origin sample to `z = iϖ₁, ϖ₁=10⁻⁵` Ha (metals
  paper §II C). **Skip for the insulating gate systems (MoS2, Si, CrI3) — YAGNI until a metal
  is on the table.**

Because `W` is even in `z`, sample only `Re z ≥ 0`; the fit variable is `x = z²`.

#### 3.1.1 There is NO separate broadening — ϖ *is* the regularization

Worth stating explicitly, because it is the whole reason MPA is affordable (and it is what §9's
cost law keys on):

- **The papers use no finite broadening η anywhere in the χ/W evaluation.** `X₀` is the bare
  Lehmann sum (metals Eq. 6) with `Im[Ω^KS] → 0⁻` — a *time-ordering infinitesimal*, not a
  smearing. Likewise `i0⁺` in `G₀` (sigma Eqs. 1–2) and the `iη` in the analytic Σ (metals Eq. 8).
  **The imaginary part `ϖ` of the sampling point is the only thing keeping the evaluation off the
  poles.** You never evaluate on the real axis, so you never need to broaden.
- **The `ϖ` used for `W` is LARGE — Hartree-scale, comparable to or above the plasmon itself:**
  `ϖ₁ = 0.1 Ha = 2.72 eV` (near), `ϖ₂ = 1 Ha = 27.2 eV` (far) — metals §II C. Compare the paper's
  own plasmon energies: Na 5.8, MoS₂ 11.0, Si 16.6, Cu 26.5 eV. **The far line sits at/above `ω_pl`
  for every system they run** — that is the design (W is smooth there, so few poles describe it),
  and it is exactly why §9 lever 1 is cheap.
- **⚠ Unit trap (compounds R1's Ry/Ha trap):** the *sigma* paper quotes MPA-**Σ** sampling in **eV**
  (`ϖ = ±0.1 eV` near; ±1 / ±20 eV in Fig. 1) — **27× smaller** than the MPA-**W** values. These are
  different quantities with different sampling scales. Reading "0.1 eV" and applying it to `W`
  sampling gives a `ϖ` 27× too small ⇒ ~27× the crossing nodes (Eq. 7) *and* a badly-conditioned
  fit. **W sampling is Hartree-scale; Σ sampling is eV-scale.** In LORRAX's Ry-native units:
  `ϖ₁ = 0.2 Ry`, `ϖ₂ = 2.0 Ry` (§5.4).
- The only genuinely tiny imaginary part in either paper is the **metals** origin shift
  `z₁ = iϖ, ϖ = 10⁻⁵ Ha` — a stability dodge around zero-energy *intraband* transitions, not a
  physical broadening. Insulators keep `z₁ = 0` exactly.

Consequence for §9: with `ϖ_near = 0.1 Ha` against MoS₂'s `E_bw ≈ 2–3 Ha`, an interior near-line
sample costs `N ≈ 1.2·E_bw/ϖ ≈ 28–45` nodes — bounded, and independently confirmed by R2 against
the crossing-solver error model.

### 3.2 Solve, per element (batched)

Model `P(x)/Q(x)` with `deg Q = n_p`, `deg P = n_p−1`, `Q` monic. Cross-multiplying the
interpolation conditions `W(x_i)·Q(x_i) = P(x_i)` at the `2n_p` samples gives a **linear**
system in the `2n_p` unknowns `{c₀..c_{n_p−1}}` (coeffs of `Q` below the monic term) and
`{d₀..d_{n_p−1}}` (coeffs of `P`):

$$
\sum_{k=0}^{n_p-1} d_k x_i^k \;-\; W(x_i)\sum_{k=0}^{n_p-1} c_k x_i^k \;=\; W(x_i)\,x_i^{n_p},
\qquad i=1,\dots,2n_p. \tag{6}
$$

**Rescale first (R1/R4, major).** `x=z²` spans `[0, ω_m²+ϖ²]` with `ω_m` several Ry, so the
raw Vandermonde `x⁰…x^{n_p}` is catastrophically ill-conditioned at `n_p≈8` (κ ~ 10¹⁴–10³²) and
`jnp.linalg.solve` returns garbage poles in float64. Nondimensionalize `x̂ = x/x_max` inside the
fit (undo `b_p ← x_max·b̂_p`, `a_p ← x_max·â_p`); use `lstsq` on the `2n_p` system. Keep Thiele
continued-fraction interpolation (the papers' 2nd method) in reserve. **Make the §3.3 analytic
test use realistic Ry-scale frequencies** or it passes while production fails.

Then:

1. **Poles**: `b_p = Ω_p²` are the roots of `Q(x̂) = x̂^{n_p} + Σ c_k x̂^k` → companion-matrix
   eigenvalues. **Non-symmetric `eigvals` has no GPU lowering in JAX (R3, major)** — Stage B is
   disk-staged and cheap, so run the fit on **host numpy** (`np.linalg.eigvals`/`np.roots`) over
   `(μ,ν)` chunks; keep only the linear solve in JAX if desired.
2. **Residues**: `a_p = P(b_p)/Q'(b_p)` (residue theorem). Then `Ω_p = √b_p`
   (branch: `Re Ω_p ≥ 0`, WLOG — the model is invariant under `(Ω,R)→(−Ω,−R)`), `R_p = a_p/(2Ω_p)`.
3. **Physical constraints (per-pole, not per-element — R1/R4, major).** For each pole,
   enforce time ordering `Re·Im < 0` (conjugate `Ω_p` if `Im Ω_p > 0`) — this is the papers'
   "poles near the real axis" condition, equivalently `Re[Ω_p²] > Im[Ω_p²]`. **A flip changes
   `b_p=Ω_p²`, so the residues MUST be re-solved** by linear least-squares
   `Σ_p a_p/(x̂_i−b̂_p)=W(x̂_i)` with the constrained poles fixed — the "or" in the old text is
   now mandatory whenever any pole moved. A leaked `Im Ω>0` pole enters Stage C as `e^{+|ImΩ|τ}`
   → exponential blowup, so this must be watertight. Constrain/repair the offending pole and
   refit; **do not nuke the whole element** because one of `n_p` poles violates — that reproduces
   the metals-paper Cu pathology (48% GN "unfulfilled modes" wrecking Σ). Report the
   constrained-pole fraction as the method's health metric. Element-level `ppm_invalid_mode`
   (`static_limit` preferred; note `2ry` is meaningless for `n_p>1`; `static_limit` currently
   raises `NotImplementedError` — see §10 F-arch) only as last resort, plus a `|W|` magnitude
   guard so near-zero ISDF pairs don't make the solve singular.

This is the direct analog of `fit_gn_ppm_from_wc_pair`, just `n_p` poles via a rescaled linear
solve + host eigvals instead of a closed form. Cheap relative to the χ₀/W-solve.

### 3.3 Validation of the fit in isolation

- **Analytic**: synthesize `W` from known `{Ω_p, R_p}`, sample, refit, assert recovery to
  `~1e-8`.
- **GN limit**: `n_p=1` fit from `{z=0⁺, z=iω_p}` must reproduce `fit_gn_ppm_from_wc_pair`
  bit-for-bit (shared code path if we let GN be `MPA(n_p=1)` — see §5 note).

---

## 4. Memory-staged pipeline (exactly the user's spec)

Three disk-staged stages, mirroring the user's three sentences. All artifacts under the run's
`00_lorrax_mpa/mpa_cache/`.

```
Stage A — W(z_i), one frequency at a time → disk
────────────────────────────────────────────────
for i, z_i in enumerate(sample_grid):          # 2·n_p complex freqs
    chi0 = compute_chi0(wfns, quad(z_i), ...)  # existing
    W_i  = solve_w(V_q, chi0, ...)             # existing, (nq, μ, ν)
    Wc_i = W_i - V_q
    write_h5(cache/f"Wc_z{i:03d}.h5", Wc_i)    # ~50–160 MB each
    del chi0, W_i, Wc_i                        # never hold >1 in device mem

Stage B — fit poles over (μ,ν) chunks → disk
────────────────────────────────────────────────
for (mu_slice, nu_slice) in chunks(n_μ, n_μ):  # sized to device budget
    Wc_stack = stack([read_h5(cache/f"Wc_z{i:03d}.h5")[:, mu_slice, nu_slice]
                      for i in range(2*n_p)])   # (2n_p, nq, dμ, dν)
    Omega, R, valid = mpa_fit(Wc_stack, sample_grid)   # NEW kernel, §3
    write_h5(cache/f"poles_{mu_slice}_{nu_slice}.h5", Omega, R, valid)
    del Wc_stack, Omega, R
# → assemble Omega[p], R[p] as (n_p, nq, μ, ν); can restage per-pole:
#   repack to cache/pole_{p:03d}.h5 holding one pole's (nq, μ, ν) B,Ω,mask.

Stage C — integrate one pole at a time → Σ_c
────────────────────────────────────────────────
Sigma_c = 0
for p in range(n_p):                           # ONE pole's B,Ω live at a time
    B_p, Om_p, mask_p = read_h5(cache/f"pole_{p:03d}.h5")   # (nq, μ, ν)
    Sigma_c += run_sigma_all_branches(          # == today's whole ppm driver
        wfns, B_p, Om_p, mask_p, omega_grid, ...)   # reuses _run_sigma_branch
    del B_p, Om_p
write_sigma_omega_h5(Sigma_c)                   # existing sigma_mnk.h5 writer
```

Peak device memory over the whole thing ≈ `max(one W(z_i), one (μ,ν) fit chunk,
one pole's Σ branch)` — i.e. the same footprint as a single GN-PPM run, independent of
`n_p`. That is the entire point of the staging.

### 4.1 Cost knobs / sizing

- Stage A: `2n_p` χ₀+W solves. `n_p=8` → 16 solves vs GN's 2. This is the dominant new cost
  (linear in `n_p`), but each is an existing, already-optimized solve.
- Stage C: `n_p ×` the current dynamic-Σ time. Linear in `n_p`. The whole reason to loop poles
  (not batch them) is to keep memory flat; the compute is unavoidable and identical to running
  GN `n_p` times.
- Disk: `2n_p` W-files + `n_p` pole-files, `~50–160 MB` each → ≤ a few GB. Trivial on `$SCRATCH`.

---

## 5. Code scaffold (minimal, reuse-first)

New surface is small. **No new abstractions, no new classes beyond one fit result;** the pipeline
is procedural on plain arrays (memory `no-new-api-layers`, `minimal-signatures`).

### 5.1 New: `src/gw/mpa_fit.py` (~120 lines)

```python
"""Multipole (MPA-W) pole fit: W_c(q,z;μ,ν) at 2·n_p complex samples → n_p poles.

Generalizes gw.minimax_screening.fit_gn_ppm_from_wc_pair (n_p=1) to n_p poles via a
batched Padé-in-x=z² solve.  Pure local algebra over (q,μ,ν); vmap-friendly, no gathers.
"""
import jax, jax.numpy as jnp

def mpa_sample_grid(n_p, omega_m_ry, *, im_near=0.1, im_far=1.0, alpha=1):
    """2·n_p complex z samples (Ry): double-parallel, semi-homogeneous Re-partition.
    Returns z (complex, (2n_p,)).  Even-in-z ⇒ Re z ≥ 0 only.  See metals paper Eq. 11."""
    ...

def _fit_one(x, w):                 # x,(2n_p,) complex ; w=W_c(x), (2n_p,) complex
    """Solve W(x_i)Q(x_i)=P(x_i), deg Q=n_p monic, deg P=n_p-1.
    Returns (Omega,(n_p,) complex, R,(n_p,) complex, valid bool)."""
    n = x.shape[0] // 2
    Vx = jnp.vander(x, n, increasing=True)          # (2n_p, n_p)
    A  = jnp.concatenate([Vx, -w[:, None] * Vx], axis=1)   # [P | -W·Q_low]
    rhs = w * x**n                                   # W·x^{n_p}
    coef = jnp.linalg.solve(A, rhs)                  # 2n_p unknowns: d[:n], c[n:]
    d, c = coef[:n], coef[n:]
    Q = jnp.concatenate([c, jnp.ones((1,), coef.dtype)])   # monic x^{n_p}+Σ c_k x^k
    b = np.roots(Q[::-1])                            # b_p = Ω_p² — HOST numpy (see note)
    Pb = np.polyval(d[::-1], b); Qp = np.polyval(np.polyder(Q[::-1]), b)
    Omega = np.sqrt(b); Omega[Omega.real < 0] *= -1
    flipped = Omega.imag > 0; Omega[flipped] = np.conj(Omega[flipped])   # time-order
    a = _refit_residues(x_hat, w, b_of(Omega)) if flipped.any() else Pb/Qp   # MANDATORY refit
    R = a / (2 * Omega)
    ok = np.abs(Omega) > 1e-14                       # per-pole validity, |Ω| not Re Ω
    return Omega, R, ok
```

**Review corrections applied (R1/R3/R4):** (1) the raw `x=z²` Vandermonde is solved on the
**rescaled** `x̂=x/x_max` (undo on output) — see §3.2. (2) Roots run on **host numpy**
(`np.roots`/`np.linalg.eigvals`): non-symmetric `eigvals` has **no GPU lowering in JAX**, and no
poly-roots helper exists in `common/` (only `eigh`/`eigvalsh` are used in-tree), so Stage B is a
host loop over disk-staged `(μ,ν)` chunks — cheap, and it sidesteps the GPU-eig gap entirely. (3)
residues are **re-fit** whenever a pole was time-order-flipped (`_refit_residues` = linear LSQ with
poles fixed); the pre-flip `P(b)/Q'(b)` is stale otherwise. (4) validity is **per pole on `|Ω|`**,
not an all-or-nothing element kill on `Re[Ω²]>0`.

### 5.2 Extend the screening planner — `2n_p` complex requests

`ScreeningRequest` / `compute_screening` already support pure-real and pure-imag ω via
`build_real_quadrature` / `build_imag_quadrature`, and reject fully complex ω. The MPA grid
is off-axis (`Re≠0` **and** `Im≠0`). The quadrature that makes this cheap — and the physics of
when it *isn't* cheap — is **§9** (read it before implementing). In short: add one
`build_offaxis_quadrature(quad, z, windows)` helper (real τ + complex α for the noncrossing
branches — **no** kernel change; the interior-crossing branch needs a complex-time χ₀-kernel
extension, §9.4), and extend `screening_requests_for(MPA)` to emit `role=f"mpa_z{i:03d}"` for
each sample. **Note (R3):** `compute_screening` accumulates all W in one dict — for the `2n_p`
grid that is the replicated-buffer pile Stage A exists to kill, so Stage A loops
`compute_chi0`/`solve_w` directly (§4) and spills, rather than calling `compute_screening`. Gate
on a new `ComputeMode.MPA` (§5.4).

### 5.3 Stage C driver — pole loop around the existing branch driver

In the MPA dispatch case (parallel to `ppm_pipeline.compute_ppm_sigma_pipeline`):

```python
def compute_mpa_sigma_pipeline(wfns, pole_cache, V_q, meta, mesh_xy, config, ...):
    """Σ_c(ω) = Σ_p (existing 4-branch dynamic-Σ driver with pole p's {R_p,Ω_p})."""
    sigma_c = None
    for p in range(config.mpa.n_pole):
        B_p, Om_p, mask_p = read_pole_h5(pole_cache, p)      # (nq,μ,ν), device_put sharded
        sig_p = _run_all_branches(wfns, E_A=..., B_q=B_p, Omega_q=Om_p,
                                  base_mask_B=mask_p, omega_grid=..., meta=meta,
                                  mesh_xy=mesh_xy, ...)        # == today's ppm branch loop
        sigma_c = sig_p if sigma_c is None else sigma_c + sig_p
        del B_p, Om_p, sig_p
    return sigma_c
```

`_run_all_branches` is the existing `_iter_branches` → `_run_sigma_branch` loop lifted out of
`ppm_sigma` so both PPM (`n_p=1`) and MPA (`n_p` loop) call it. Per-pole reuse is verified sound
(R3): the driver takes `B_q/Ω_q/mask` as plain `(nq,μ,ν)` args, module caches are keyed only on
`(id(mesh_xy), kgrid)` (constant across the loop → no recompiles), and `Σ_p` is a plain add of
sharded arrays. **Caveats the "refactor not new logic" framing must own (R3):**
- The lift is a real ~150-line driver refactor: the streaming-h5 file opens `mode="w"` (truncates)
  **inside** the driver, so the pole loop must wrap the *extracted* `_run_all_branches`, never the
  whole driver, or each pole clobbers the last pole's Σ file. Windows are rebuilt per pole per
  branch (host minimax, cheap with shipped tables).
- **The analytic q→0 head correction is MISSING from this sketch (R3, major).** GN-PPM fits a
  scalar single-pole head (`ppm_pipeline._fit_head_correction` → `compute_ppm_head_sigma_kij`) and
  injects it into Σ_c; on the 2D MoS2 gate system the head was its own bug-fix initiative, and a
  head-less MPA Σ_c is silently wrong by the head shift — the BGW cross-check would then chase a
  known artifact. Needs an MPA head story: sample the head at the same `2n_p` `z_i` (check
  `HeadResolver` supports off-axis complex ω) and reuse `mpa_fit` on the scalar head, or an
  explicit, documented single-pole-head approximation. Budget this — it is not free.

### 5.4 Config: one enum value + one sub-config

```python
# gw_config.py
class ComputeMode: ... ; MPA = "mpa"          # dynamic Σ_c(ω) via n_p-pole MPA-W

@dataclass(frozen=True)
class MPAConfig:
    n_pole: int          # n_p — 8/α=1 for Si/MoS2; 12/α=2 for CrI3 (Cu-like flat d bands, R4)
    omega_m_ry: float    # sampling upper edge ≈ (2–3)·ω_pl (classical plasmon), NOT max-transition
    im_near_ry: float = 0.2   # ≈0.1 Ha — UNITS: fields are Ry (LORRAX-native); 0.1 Ha = 0.2 Ry (R1)
    im_far_ry:  float = 2.0   # ≈1 Ha = GN's default probe (2 Ry) → free n_p=1 anchor alignment
    alpha: int = 1       # 1 semiconductors, 2 low-ω structure / metals
    intraband: bool = False   # metals CA/Drude head — deferred, YAGNI (see §6 gate)
    # σ-grid + invalid_mode reused from PPMConfig (identical output stage)
```

**Units trap (R1):** the papers quote `ϖ` in **Ha**; LORRAX is Ry-native (GN probe `2j` Ry = 1 Ha).
Name the fields `_ry` and set `0.2`/`2.0`, or the far line lands at half the paper's value. The
happy coincidence `im_far = 2 Ry = 1 Ha = GN probe` lets the `n_p=1` anchor share the exact grid.

`screening_requests_for(MPA)` → the `2n_p` requests; `sigma_dispatch.compute_sigma_xc` gets
one `MPA` case delegating to `compute_mpa_sigma_pipeline`.

### 5.5 What is explicitly *not* built

- **MPA-Σ / MPA-G** (fit Σ directly, Padé Green's function, satellites, `Z` beyond linearized,
  spectral functions `A(k,ω)`): a *separate additive layer* (§7). YAGNI until spectral
  functions are requested. When they are, MPA-Σ reuses `mpa_fit` on `Σ_c(z_i)` samples
  (fit Σ instead of W) — the fit kernel is the same; only the sampling target changes.
- **Intraband/Drude head for metals**: only matters for partially-filled bands; all sandbox
  gate systems are insulators. Leave `MPAConfig.intraband=False` stub.
- Any new `PoleBlock`/`DenomType` dataclasses from `FREQ_INTEGRATION_REWRITE_PLAN.md`: that
  design targets a *different* (CTSP G·W with rank-1 residues) engine and is **not** wired in
  `lorrax_D`. MPA-W rides the production `ppm_sigma` path instead — do not resurrect freqint
  for this.

---

## 6. Validation plan

Anchor everything on GN-PPM as the `n_p=1` limit, then converge in `n_p`, then cross-check BGW.

| Test | Assertion |
|------|-----------|
| **Fit analytic** (§3.3) | synth `{Ω_p,R_p}` at **realistic Ry-scale** ω → sample → refit recovers `~1e-8` (O(1) synthetic ω hides the Vandermonde blow-up — R1/R4) |
| **Complex-pole Σ kernel** | single complex pole → closed-form `Σ_c` vs CTSP-kernel `Σ_c` (validates minimax nodes under strong `Im Ω` damping — R4) |
| **GN limit** | `mpa_fit(n_p=1)` on the pinned `{0, i·ω_p}` grid == `fit_gn_ppm_from_wc_pair` to `<1e-12` — **requires** `n_p=1` share GN's `Re[Ω²]`+sqrt projection & invalid path (not "bitwise" — R1/R3) |
| **Σ GN limit** | `ComputeMode.MPA, n_pole=1` reproduces `ComputeMode.GN_PPM` `sigma_diag.dat` to `<1e-4` (chunk-order tolerance) |
| **`n_p` convergence** | MoS2 3×3: `Re Σ_c(E_dft)` at Γ satisfies **Cauchy-plateau** `|Σ(8)−Σ(4)| < tol` over `n_p∈{1,2,4,8}` — NOT monotonic (Padé is non-variational; monotonicity invites tuning-to-pass — R4) |
| **Sampling adequacy** | A/B of {far-only, far+below-gap, full double-parallel} vs BGW FF — a plateau at the *wrong* value is a sampling failure the convergence gate can't see (R2/R4) |
| **Sum-rule asserts** | (a) `Σ_p 2R_pΩ_p` vs HL-GPP `Ω²_{μν}` (f-sum, machinery in `hl-gpp-derivation.md`); (b) `−Σ_p 2R_p/Ω_p` vs sampled `W_c(0)` (static, post-constraint); (c) `−Im W_c(q;μμ;ω>0)>0` on diagonals (R4) |
| **BGW FF cross-check** | vs BGW `frequency_dependence 2` (real-axis FF) on the frozen MoS2 3×3 WFN. **Include the MPA head (§5.3)** or the ~const 2D head/convention offset reappears identically and reads as an MPA bug (R3/R4; cf. `no-isdf-rank-excuse` plateau⇒convention memory) |
| **Distributed** | 1-GPU vs 4-GPU identical `Σ_c` (MoS2, per `no-16gpu-gating`) |
| **Disk staging** | Stage-C `Σ_c` from disk-loaded poles == in-memory `Σ_c` (parity) |

Gate system: **MoS2 3×3 on 1 GPU** is the portable anchor (insulator deferral of intraband is
unconditionally safe here). **CrI3 gap gate (R4, major):** the sandbox CrI3 WFNs are
non-magnetic zero-net-spin (memory `cri3-wfns-nonmagnetic`) → partially-filled Cr-d → **metallic
or near-metallic**, exactly where the YAGNI-deferred intraband/Drude + `z=iϖ,ϖ=10⁻⁵` origin-shift
+ fractional occupations are needed, and the failure is *silent* (bad q→0 screening, spurious
Fermi gap). So: **gate the MPA pipeline on `min gap > 0`** read from WFN occupations, and run CrI3
only on the **FM variant** (gap ≳0.8 eV; needs no intraband) with 16-GPU per `cri3-always-16-gpus`.

Use the existing parsers (`skills/compare/SKILL.md`; `sigma_freq_debug.dat` `sig_c(Edft).Re` at
col 8 per `KNOWN_SANDBOX_ERRORS.md`).

---

## 7. Optional follow-on: MPA-Σ / MPA-G (spectral functions)

If/when `A(k,ω)`, satellites, or `Z` beyond the linearized factor are needed
(multipole-sigma-2025 §II C–D), add a thin layer — **do not** rebuild:

1. Sample `Σ_c(z_i)` on the complex grid from the MPA-W run above (it already produces
   `Σ_c(ω)` on a grid; extend to complex `z` or fit the real-axis grid).
2. Reuse `mpa_fit` on `Σ_c` samples → `{ξ_p, S_p}` (Eq. 13–17). Same solver.
3. Dyson from the Padé (Eq. 18–22): `G` poles = roots of `C_{n_Σ+1}(z)=zB−Σ_xB−A`
   (companion matrix again), residues by Eq. 22. `Σ_p Z_p = 1` sum-rule check (Eq. 24).

That is the *entire* MPA-Σ/MPA-G addition — one fit reuse + one polynomial Dyson. Kept out of
scope now because QP energies + `Z_lin` (what LORRAX reports today) need only Eq. (5)'s
`Σ_c(ω)`, which MPA-W already delivers.

---

## 8. Implementation order (revised after review)

1. `mpa_fit.py` (host roots, x-rescaled, per-pole constraint + mandatory residue refit) + fit
   tests: analytic recovery at **Ry-scale** ω, and the pinned-grid GN-limit. *Standalone.*
2. `build_offaxis_quadrature` **noncrossing branches only** (proper complex-target minimax — real
   τ, complex α, **no** kernel change) + `screening_requests_for(MPA)` + Stage-A disk spill (loop
   `compute_chi0`/`solve_w` directly, not `compute_screening`). **Fix `compute_chi0`'s
   `w_isdf.py:629` host fold to keep complex α.** Milestone: far + below-gap sampling only.
3. Lift `_iter_branches`→`_run_sigma_branch` into `_run_all_branches` (own the ~150-line
   streaming-h5/accum seam; pole loop wraps the *extracted* fn); thread the **two-array complex-Ω**
   through the branch stack (§2.3). GN-PPM gates stay green.
4. `compute_mpa_sigma_pipeline` (pole loop) + `MPAConfig` + dispatch case + **MPA q→0 head**
   (§5.3) + `min gap > 0` CrI3 gate (§6).
5. `ComputeMode.MPA n_pole=1` == GN-PPM gate; then `n_p` Cauchy-plateau + sampling-adequacy A/B +
   BGW FF cross-check (with head).
6. **Only if step-5 sampling A/B demands it:** the interior-crossing χ₀ path — complex-time
   `build_G_tau` / Euler split (§9.4). This is the one genuinely new *kernel*; defer until proven
   necessary.
7. Checkpoint: pytest, commit on a feature branch (`agent/mpa-w`), report, CHANGELOG.

Branch discipline: **feature branch before any source edit** (`sources/lorrax_D/AGENTS.md`). Net
shared-code edits (all GN-regression-gated): the two-array complex-Ω threading (§2.3), the
branch-driver lift (step 3), the `compute_chi0` host-fold dtype fix (step 2), and — only at step 6
— the complex-time χ₀ kernel. Honest effort: **~1.5–2× the original "small diff" framing** (R3).

---

## 9. Computing χ₀(z) off ω=0 — the crossing corner and the cheapest sampling

This is the corner that decides whether MPA sampling is affordable. The user's cost dichotomy
is exactly right and is baked into the LORRAX solvers:

| Solver | Fits | Node key | Cost |
|--------|------|----------|------|
| `solve_laplace_minimax_interval` (noncrossing) | `1/x` on `[x_min,x_max]` | `logR = log(x_max/x_min)` | **`log(E_dynrange)`** |
| `solve_laplace_minimax_imag_interval` | `x/(x²+ϖ²)` (χ₀ at `z=iϖ`) | `logR`, `ϖ/x_min` | **`log`-ish** |
| `solve_phase_minimax_bandwidth` (crossing/HGL) | `sin`-regularized on `[0,A_dim]` | `A_dim` (abs. bandwidth) | **`O(E_bw)`** (max_nodes 500 vs 64) |

### 9.1 What already exists (and what doesn't)

χ₀ off the static point is representable as `χ₀(z) = Σ_cv A_cv[1/(z−Δ_cv) − 1/(z+Δ_cv)]`,
`Δ_cv = E_c−E_v ∈ [x_min, x_max]`. LORRAX evaluates it by the **separable** CTSP trick: fit
the denominator as `Σ_l α_l e^{−τ_l Δ_cv}`, and since `e^{−τ_l(E_c−E_v)} = e^{−τ_l E_c}·e^{+τ_l E_v}`
it factors into `G_c(τ_l)·G_v(τ_l)` — cheap. Existing builders:

- **`z = 0`** — `build_static_quadrature`. Noncrossing (`Δ>0`), `log`.
- **`z = iϖ` (pure imag)** — `build_imag_quadrature`. Noncrossing (`|iϖ−Δ|≥Δ>0`, no real crossing),
  `log`-ish. **τ real, α real** (it fits the real combined kernel `x/(x²+ϖ²)`).
- **`z = Ω real, Ω > x_max` (above all transitions)** — `build_real_quadrature`. Splits into
  `1/(Ω±x)`, both fixed-sign → two noncrossing `1/y` fits, `log`. Folds `e^{−τΩ}` into α.

**Missing (and the hard case): `z = ω + iϖ` with `Re z = ω ∈ (x_min, x_max)`** — the resonant
denominator `Re(z−Δ) = ω−Δ` changes sign across the transition range. No production w_isdf
builder covers it; χ₀-inside-the-continuum is currently unsupported.

### 9.2 Why the interior case is genuinely `O(bandwidth/ϖ)`, not `log`

Tempting idea: split the transition sum at `Δ = ω` into two fixed-real-sign halves, each a
convergent complex Laplace branch → `log` each. **This fails**: the split is a mask on the
composite `Δ_cv = E_c − E_v`, which is *not separable* into a valence mask × conduction mask,
so it breaks the `G_c·G_v` factorization that makes the kernel cheap. Any CTSP-compatible
(separable) representation must fit `1/(z−Δ)` **globally** over `[x_min, x_max]` as
`Σ_l α_l e^{−τ_l Δ}`. Globally, `1/(z−Δ)` has a pole at `Δ = ω+iϖ`, a distance `ϖ` off the real
axis — a Lorentzian of width `ϖ`. Resolving a width-`ϖ` feature across a bandwidth `E_bw` in an
exponential/Fourier basis needs

$$
N \sim E_{\text{bw}}/\varpi \quad\text{(linear)}, \tag{7}
$$

which is the crossing/HGL regime. **The finite `ϖ` caps it (no true singularity), but does not
buy back `log`.** So: `Re z` outside `(x_min,x_max)` **or** large `ϖ` → `log`; `Re z` inside with
small `ϖ` → linear. This is the whole corner.

### 9.3 The cheapest sampling strategy (three levers)

Do not try to make the interior sample cheap — **arrange the sampling so few samples are
interior-and-near-real**, and localize the cost of the ones that are:

1. **Lean on the far line.** At `ϖ_far ≈ 1 Ha ~ O(E_bw)`, Eq. (7) gives `N ~ O(1)` even for
   interior `ω`. The far line is cheap *everywhere*; it carries the global pole structure (this
   is exactly why the papers' double-parallel sampling puts a line out at ~1 Ha). **Put the bulk
   of the MPA information on the far line.**

2. **Place near-line points where they pin the poles — which for these systems IS partly inside
   the continuum (R2/R4 correction to the original "keep it out").** The original lever 2 ("keep
   the near line below the gap") is *wrong for the gate systems*: MoS2/Si/CrI3 gaps (0.05–0.15 Ha)
   sit far below the plasmon (0.4–1 Ha), so essentially all of `W`'s pole structure lies *inside*
   the continuum. Below-gap-only near-line sampling starves the fit exactly where the poles live
   and leaves them to ill-conditioned analytic continuation from the `ϖ=1 Ha` line. So: **budget
   `~n_p/2` interior near-line samples as the expected production path** (the papers' double-parallel
   sampling deliberately runs the `ϖ₁=0.1 Ha` line *across* the range for this reason). Use `ω_m ≈
   (2–3)×ω_pl` (classical plasmon from the density), not "max transition" (which is tens of Ha at
   production band counts and starves the ≤2 Ha region — R4). The MPA fit still *extrapolates*
   detail between samples, but it must *see* near-axis structure to place poles.

3. **For an interior sample, use ONE global crossing fit — do NOT window-pair the χ₀ path
   (R2 correction).** Window-pairs are separable and would localize the crossing, but for **χ₀**
   the per-τ-node cost is dominated by the `n_μ²` FFT/einsum and is nearly band-count-independent
   (bands are pre-summed in `build_G_tau`). Splitting into `n²` pair-kernels therefore *adds*
   `(n²−n)` noncrossing pair-kernels at full per-node cost → ~5–8× **more** total work than one
   global HGL fit per sample. Window-pairing's only benefit here is bounded `A_dim` (static shapes,
   `max_nodes` cap) and conditioning — not total cost. **Drop the `get_windows` dependency from the
   χ₀ path**; keep it only if a single global crossing kernel's shape/conditioning misbehaves.
   (This differs from the *sigma* side, where window-pairing does pay — the Σ kernel's cost scales
   with the band/pole content per window.)

### 9.4 Implementation: `build_offaxis_quadrature(quad, z, *, windows=None)`

> **Review correction (R2, major): the "reuse `quad`'s τ, fold the shift into α" recipe is
> unsound. You must SOLVE the minimax against the true complex target.** Reusing the *static*
> τ-nodes and setting `α_l ← w_l e^{−τ_l z}` is only an exact algebraic identity for the `ϖ=0`
> real shift `build_real_quadrature` does; off-axis the minimax error `E(y)=1/y−Σw_l e^{−τ_l y}`
> was controlled only on the *real* interval, and the fold gives no error control. Worse, on the
> far line `ϖ=1 Ha=2 Ry`, `e^{−iτϖ}` oscillates with period `π Ry⁻¹` while static τ-nodes run to
> `~30–60 Ry⁻¹` → many periods between nodes → **O(1) error in every `W_c(z_i)` sample**, i.e.
> garbage fed to the Padé fit. Correct construction of `build_offaxis_quadrature`:

- **`1/(z±Δ)` noncrossing branches** (`Re z≤x_min`, `Re z≥x_max`, and the `z+Δ` branch always):
  *solve* a minimax fit against the true complex target on the real `Δ`-interval — real τ,
  **complex weights**. Cleanest: split into two real fits — the Re part `y/(y²+ϖ²)` is exactly
  the existing `solve_laplace_minimax_imag_interval` target on the shifted interval; the Im part
  `−ϖ/(y²+ϖ²)` is a **new** Lorentzian target (reuse the VarPro/Lawson scaffolding in
  `common/minimax.py`). Node count stays `log` — the §9 *cost* conclusion holds; only the *recipe*
  changes. This keeps τ real / α complex → **no χ₀-kernel change** for these branches.
- **`1/(z−Δ)` interior branch** (`x_min<Re z<x_max`): genuine **crossing**, one global fit,
  `O(E_bw/ϖ)`. **This branch DOES need a χ₀-kernel change (R2, major):** the crossing basis is
  `sin(τu)` = imaginary-time phases `e^{∓iτE}`, and `minimax_tau_integrate_chi` casts τ→real
  (`w_isdf.py:174`) with no Im-projection path. Either extend `build_G_tau` to complex time or use
  the Euler `P_plus/P_cross` split (the machinery already exists on the *sigma* side —
  `_laplace_to_minimax_nodes` supports `t=−iτ`). Real-τ+complex-α cannot substitute here: real
  `e^{±τΔ}` span `e^{E_bw/ϖ}` ≈ 10–13 digits at `ϖ=0.1 Ha` → float64 cancellation death.

> **Second silent trap (R3, major):** `compute_chi0`'s host-side prefactor fold
> (`w_isdf.py:629`, `alpha_chi = -2.0*np.asarray(quad.alpha, dtype=np.float64)*…`) casts complex
> α → float64, **silently discarding Im(α)** (numpy `ComplexWarning`). A complex-α quadrature
> through this line gives wrong `χ₀(z)` with no crash and a green static regression. One-line
> dtype fix, but it is the *host* fold, not the kernel, that must change — verify explicitly.

### 9.5 Practical takeaway for the MPA plan (revised after review)

- **De-risk with the far line + below-gap near line**, using the *proper complex-target*
  quadrature (§9.4, not the refold) — all `log`, all noncrossing, **no χ₀-kernel change**. Fit
  MPA and check vs a BGW FF reference. This is the first milestone.
- **Expect to add interior near-line samples** (`~n_p/2`) to pin the poles — the gate systems
  put `W`'s structure inside the continuum (R2/R4). Each interior sample is one global crossing
  fit, `O(E_bw/ϖ)` ≈ 30–45 nodes at `ϖ=0.1 Ha` (bounded, ~one static+imag solve-pair), and
  **requires the complex-time χ₀-kernel extension** (§9.4). Validate sampling adequacy by an A/B
  of {far-only, far+below-gap, full double-parallel} against BGW FF — the `n_p`-convergence gate
  alone cannot certify it (a poorly-sampled fit plateaus at the *wrong* answer, R4).

---

## 10. Physics review — four-lens Fable panel + adjudication (2026-07-06)

Four independent reviewers (model: fable), each on one lens, each instructed to do real
index/formula math (not pattern-match — per the `agent-audit-failure-modes` memory). Verdicts,
then the consolidated round-of-discussion. **All corrections above are already folded into the
plan body**; this section is the durable record of what changed and why, plus the one live
disagreement.

### 10.1 Per-lens verdicts

| Lens | Verdict | Headline |
|------|---------|----------|
| **R1 — fit algebra** | sound *after* corrections | `−2R_p` is a blocker; Padé system, residue extraction, GN-reduction all re-derived and confirmed |
| **R2 — freq-int numerics** | cost analysis correct; recipe unsafe | §9 log/linear dichotomy independently confirmed (incl. a basis-independent ε-rank bound); the α-refold and "no kernel change" are wrong |
| **R3 — architecture** | core sound; ~1.5–2× the implied work | per-pole reuse verified TRUE; but complex-Ω is 6-signature, GPU-eigvals won't run, head correction missing, host-fold kills complex α |
| **R4 — physics & scope** | physics sound; **MPA-W-only scope SOUND** | 4 majors: −2 convention, per-pole constraint+refit, restore in-continuum sampling, verify CrI3 is gapped |

### 10.2 Round of discussion — consensus findings (adopted)

**Strong convergence (≥2 reviewers, independently derived):**

- **F1 [BLOCKER] Eq. (5) residue `−2R_p` → `+R_p`.** R1 and R4 both derived `+R_p` independently
  (contour convolution; COHSEX static-limit referee). The `−2` is a cross-paper normalization
  clash, not physics. *Adopted — §2.2 corrected, boxed.* The reuse-the-driver code was already
  right; only a test written from the printed eq. would have been −2× off.
- **F2 [MAJOR] Time-order flip leaves stale residues → residue refit MANDATORY.** R1+R4. A flipped
  pole changes `b=Ω²`; keeping `a=P(b)/Q'(b)` de-interpolates, and a leaked `Im Ω>0` pole blows up
  the τ-kernel as `e^{+|ImΩ|τ}`. *Adopted — §3.2 step 3 now mandatory refit.*
- **F3 [MAJOR] Vandermonde-in-`x=z²` catastrophically ill-conditioned at `n_p≈8`.** R1+R4.
  *Adopted — rescale `x̂=x/x_max`, `lstsq`, Ry-scale analytic test.*
- **F4 [MAJOR] Per-element all-or-nothing invalidation discards MPA's advantage.** R1+R4 (the Cu
  48%-unfulfilled lesson). *Adopted — per-pole constraint + refit; report constrained-pole
  fraction as the health metric; element `invalid_mode` last-resort only.*
- **F5 [MAJOR] "n_p=1 == GN bitwise" is false** (GN takes `Re[Ω²]`+real sqrt; MPA keeps complex).
  R1+R3+R4. *Adopted — anchor requires pinned `{0,iω_p}` grid + shared Re-projection, `<1e-12`.*
- **F6 [MAJOR] In-continuum near-line sampling is the EXPECTED path, not a rare fallback** — gate
  systems put `W`'s poles inside the continuum; below-gap-only starves the fit. R2+R4. *Adopted —
  §9.3 lever 2 rewritten, §9.5 revised, sampling-adequacy A/B gate added.*

**Single-reviewer, high-confidence (adopted):**

- **F7 [MAJOR, R2] The α-refold of static τ-nodes is unsound** (uncontrolled off-axis error;
  far-line `e^{−iτϖ}` oscillates many periods between nodes → garbage samples). *Adopted — §9.4
  now specifies solving the true complex-target minimax (real τ, complex weights via Re=imag-solver
  + new Im-Lorentzian target).*
- **F8 [MAJOR, R2] Interior-crossing branch needs a χ₀-kernel change** (sin-basis = imaginary-time
  phases; real-τ+complex-α dies by `e^{E_bw/ϖ}` cancellation). *Adopted — "no kernel change"
  retracted for the interior branch; deferred to step 6, gated on the sampling A/B.*
- **F9 [MAJOR, R3] Complex-Ω is a two-array, ~6-signature threading job, not 1 line** (`_prepare_sigma_state`
  discards `Im Ω` via `Omega_abs`; masks/stats compare `Ω`). *Adopted — §2.3 rewritten.*
- **F10 [MAJOR, R3] `jnp.linalg.eigvals` has no GPU lowering** and no `common/` roots helper
  exists. *Adopted — Stage B fit runs on host numpy over disk-staged chunks.*
- **F11 [MAJOR, R3] The analytic q→0 head correction is missing** — on 2D MoS2 it silently shifts
  Σ_c and would corrupt the BGW check. *Adopted — §5.3 MPA-head story + §6 test note.*
- **F12 [MAJOR, R3] `compute_chi0` host fold (`w_isdf.py:629`) silently casts complex α → float64.**
  *Adopted — §9.4 flag + §8 step 2.*
- **F13 [MAJOR, R4] Sandbox CrI3 WFNs are non-magnetic ⇒ metallic** ⇒ the deferred intraband
  machinery is exactly what breaks, silently. *Adopted — §6 `min gap>0` gate + FM-variant rule.*
- **F14 [MINOR, R3] `compute_screening` accumulates all W in a dict** (defeats staging). *Adopted —
  Stage A loops solvers directly.* **F15 [MINOR, R4] ω_m="max transition" starves the fit** at
  production band counts. *Adopted — `ω_m≈(2–3)ω_pl`.* **F16 [MINOR, R4] n_p=8/α=1 not universal;
  CrI3 is Cu-like** → 12/α=2. *Adopted — §5.4.* **F17 [MINOR, R4] monotonic-convergence gate is
  wrong** (Padé non-variational) → Cauchy-plateau. *Adopted — §6.* **F18 [MINOR, R4] add
  f-sum/static/loss-sign asserts.** *Adopted — §6.* **F19 [MINOR, R1] Ry/Ha units trap.** *Adopted —
  §5.4.* **F20 [NIT, R1/R4] W even-in-z is exact only under TRS** (magnetic off-diagonals inherit a
  model error, same as GN). *Noted — §3.1 assumption.* **F21 [NIT, R4] complex-Ω through the minimax
  τ-quadrature is asserted, not shown** → added the single-complex-pole Σ kernel test (§6).

### 10.3 The one live disagreement — window-pairing the χ₀ interior samples

R4 endorsed lever 3 (window-paired HGL) as the *expected production path* for interior samples;
R2 argued to **drop window-pairing from the χ₀ path** because χ₀'s per-τ-node cost is dominated by
the `n_μ²` FFT/einsum and is nearly band-count-independent, so splitting into `n²` pair-kernels
*adds* ~5–8× full-cost work vs one global crossing fit.

**Adjudication: R2 wins on the mechanism; R4 wins on the conclusion that interior samples are
needed.** These aren't actually in conflict — R4's core point (you *must* sample inside the
continuum) stands, but the *how* is R2's: **one global crossing fit per interior sample, not
window-pairs.** R2's cost argument is specific and correct about LORRAX's χ₀ kernel (bands
pre-summed in `build_G_tau`); R4 was reasoning from the sigma side, where window-pairing does pay
because the Σ kernel's cost scales with per-window band/pole content. *Resolution folded into §9.3
lever 3 (drop `get_windows` from the χ₀ path) and §9.4.*

### 10.4 What survived unscathed (re-derived and confirmed)

- The **MPA-W-only scope** (skip MPA-Σ/MPA-G) — R4 explicitly SOUND: `Σ_c(ω)` + `Z_lin` need only
  Eq. (5); LORRAX's CTSP already outputs Σ_c on a real-ω grid, so it buys even less than yambo
  gains. MPA-Σ deferral (§7) is genuinely additive.
- The **pole-promotion architecture** — R3 verified per-pole driver reuse TRUE (no shape-keyed
  caches, plain `Σ_p` add), pair-density path untouched, `FREQ_INTEGRATION_REWRITE_PLAN` correctly
  not resurrected.
- The **§9 cost dichotomy** — R2 independently re-derived noncrossing `≈lnR·ln(1/ε)/3.55` and
  crossing `N≈1.2·E_bw/ϖ`, checked against real MoS2 logs, and proved a *basis-independent* ε-rank
  bound closing the "clever smooth-split" loophole. The time-ordering condition `Re·Im<0` matches
  paper Eq. 9 and is correctly inherited by Σ.

### 10.5 Net effect on the plan

No architectural change: MPA-W = GN-PPM pole-promoted + disk-staged pole loop still stands. The
review converted a "small, reuse-first port" into an **honest ~1.5–2× effort** with four concrete
correctness fixes that would each have produced silently-wrong physics (−2× Σ_c, blown-up poles,
garbage off-axis samples, head-shift artifact) and one deferred new kernel (interior-crossing χ₀)
that is now gated on a sampling A/B rather than assumed unnecessary.

---

### Key references
- `multipole-sigma-2025.md` Eqs. 9–22 (MPA-W self-energy, MPA-Σ, MPA-G).
- `Multipole-W-metals.md` Eqs. 6–14 (X/W multipole, sampling Eq. 10–11, intraband).
- LORRAX: `gw/ppm_sigma.py` (τ-kernel, branch driver), `gw/minimax_screening.py`
  (`fit_gn_ppm_from_wc_pair` — the `n_p=1` anchor), `gw/screening.py` (request planner),
  `gw/w_isdf.py` (`compute_chi0`/`solve_w`), `docs/theory/hl-gpp-derivation.md` (moment
  conditions), `docs/dev/plans/FREQ_INTEGRATION_REWRITE_PLAN.md` (the *alternate* CTSP engine —
  not used here), `get_windows`/`hgl_quadrature` (window-pair + crossing machinery, §9.3).
