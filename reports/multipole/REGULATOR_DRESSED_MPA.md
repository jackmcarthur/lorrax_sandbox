# Regulator-dressed multipole screening

Date: 2026-07-21

Status: design note only. No implementation or numerical validation exists yet.

This note separates two procedures that can look similar computationally but
represent different mathematical objects:

1. ordinary multipole approximation (MPA) fitted from complex-frequency samples
   at $z=\omega+i\bar{\omega}$; and
2. a regulator-dressed multipole approximation (RegMPA) fitted from real-frequency
   values produced by LORRAX's excluded-range minimax regularizer.

The second route deliberately permits regulator-dependent mode positions. It is
not a way to reproduce the first route without computing complex frequencies.

## 1. The existing excluded-range construction

The experimental builder in
[`common/minimax.py`](../../sources/lorrax_A/src/common/minimax.py) fits
$1/u$ only outside a central interval,

$$
\frac{1}{u}\approx \sum_{\ell=1}^{N_Q}w_\ell\sin(\tau_\ell u),
\qquad u\in[u_{\min},A].
$$

It then uses the missing first-moment area to associate the fitted kernel with an
effective Lorentzian width. In the current code,

$$
A=\frac{E_{\mathrm{bw}}}{\xi_0},
\qquad
\xi_{\mathrm{eff}}=a_{\mathrm{eff}}\xi_0.
$$

The freedom inside $|u|<u_{\min}$ is what permits fewer nodes than a uniform
minimax approximation to

$$
\operatorname{Re}\frac{1}{x+i\eta}
=\frac{x}{x^2+\eta^2}
$$

over the entire crossing interval.

The relevant functions are presently retained but not connected to the live GW
path:

- `_cr_solve_1overx` fits on $[u_{\min},A]$;
- `build_crossing_quadrature` calibrates $\xi_{\mathrm{eff}}$;
- `evaluate_crossing` evaluates the resulting sine sum.

The live crossing path instead requests the fixed `target_kind="hgl"`. RegMPA
would therefore be a deliberate new use of the excluded-range construction, not a
small extension of the current production call graph.

## 2. Procedure A: ordinary $+i\bar{\omega}$ MPA

### 2.1 Samples

Ordinary double-parallel MPA evaluates one analytic response function at complex
coordinates

$$
z_{jn}=\omega_n+i\bar{\omega}_j,
\qquad j\in\{1,2\}.
$$

The usual two branches use a near and a far displacement. For example,

$$
\bar{\omega}_1=0.1\ \mathrm{Ha},
\qquad
\bar{\omega}_2=1.0\ \mathrm{Ha},
$$

with the first point on the near branch often replaced by the exact static value
$W(0)$.

A real-frequency implementation with Cauchy damping is mathematically the same
procedure because

$$
\frac{1}{x+i\eta}
=-i\int_0^\infty dt\,e^{-\eta t}e^{ixt}.
$$

Thus Cauchy-regularized evaluation at the real center $\omega_n$ is still an
evaluation of the original analytic function at $z=\omega_n+i\eta$.

### 2.2 Model and fit

The ordinary MPA model is rational:

$$
W^c(z)
\approx
\sum_{p=1}^{n_p}
\frac{2\Omega_pR_p}{z^2-\Omega_p^2}.
$$

The $2n_p$ complex samples determine numerator and denominator coefficients of a
Pade interpolant. The pole squares are roots of the denominator polynomial and can
be obtained from a companion matrix. Residues are then refitted linearly.

The sampling displacement $\bar{\omega}$ is **not part of the final model**. It is
only a coordinate used to obtain stable information about the same analytic
$W^c(z)$. In an exact and sufficiently expressive fit, changing
$\bar{\omega}$ does not define a new response function.

### 2.3 Sigma

Stage C uses the recovered pole pairs in the physical pole model. Any HGL or other
crossing regularization applied while evaluating $\Sigma(\omega)$ is a separate
numerical choice, with a separate width. It must not be confused with the MPA
sampling displacement.

## 3. Procedure B: regulator-dressed MPA

### 3.1 The regulator is part of the modeled object

Let $Q$ denote the complete quadrature specification:

$$
Q=(A,u_{\min},N_Q,\epsilon_Q,\{\tau_\ell,w_\ell\}).
$$

Define the physical-scale odd regularizer

$$
r_{\xi_0,Q}(x)
=\frac{1}{\xi_0}
\sum_{\ell=1}^{N_Q}
w_\ell\sin\left(\tau_\ell\frac{x}{\xi_0}\right).
$$

The sine sum itself supplies values inside the excluded interval even though those
values were not constrained to approximate $1/x$. That continuation through the
excluded region is part of the regulator definition.

A positive-time complex completion is

$$
d_{\xi_0,Q}(x)
=-\frac{i}{\xi_0}
\sum_{\ell=1}^{N_Q}
w_\ell
\exp\left(i\tau_\ell\frac{x}{\xi_0}\right),
$$

for which

$$
\operatorname{Re}d_{\xi_0,Q}(x)=r_{\xi_0,Q}(x).
$$

This completion is analytic in the upper half-plane because it contains only
positive-time exponentials. Signed minimax weights can nevertheless violate
passivity or the expected diagonal loss sign, so causality and passivity must be
tested separately.

### 3.2 Regulator-dressed paired modes

Replace the bare paired resolvent by

$$
\Phi_{\xi_0,Q}(\omega;\Omega_p)
=d_{\xi_0,Q}(\omega-\Omega_p)
-d_{\xi_0,Q}(\omega+\Omega_p).
$$

The RegMPA model is

$$
W^c_{\xi_0,Q}(\omega)
\approx
\sum_{p=1}^{n_p}
R_p(\xi_0,Q)\,
\Phi_{\xi_0,Q}
\left(\omega;\Omega_p(\xi_0,Q)\right).
$$

Here $\xi_0$ and $Q$ remain part of the model. Both the centers and residues are
allowed to depend on the regularizer. They should be called **mode centers** rather
than literal poles because the finite sine-sum kernel has no rational singularity.

For comparison, choosing

$$
d_\eta(x)=\frac{1}{x+i\eta}
$$

turns the same expression into a Lorentzian-broadened rational pole pair. The
excluded-range kernel is a more general basis.

### 3.3 Fit algorithm

RegMPA is not the Pade interpolation problem used by ordinary MPA. For fixed mode
centers, the residues remain linear, but the centers enter nonlinearly:

$$
\min_{\{\Omega_p,R_p\}}
\sum_j
\left\lVert
W^c_{\xi_0,Q}(\omega_j)
-\sum_p R_p\Phi_{\xi_0,Q}(\omega_j;\Omega_p)
\right\rVert^2.
$$

The natural solver is variable projection:

1. propose the mode centers $\{\Omega_p\}$;
2. solve the linear least-squares problem for $\{R_p\}$;
3. optimize only the center variables;
4. refit residues after every constraint or center repair.

Initially, the mode centers should be real. The regulator already supplies the
width, so allowing arbitrary $\operatorname{Im}\Omega_p$ at the same time creates a
strong width-identifiability problem. Complex residues remain necessary for
off-diagonal matrix elements.

### 3.4 Identifiability limit of the sine basis

The paired kernel can be written as

$$
\Phi_{\xi_0,Q}(\omega;\Omega_p)
=-\frac{2}{\xi_0}
\sum_\ell w_\ell
e^{i\tau_\ell\omega/\xi_0}
\sin\left(\tau_\ell\frac{\Omega_p}{\xi_0}\right).
$$

Consequently, all information about the mode centers enters through the finite
vectors

$$
\left\{
\sin\left(\tau_\ell\Omega_p/\xi_0\right)
\right\}_{\ell=1}^{N_Q}.
$$

The number of stably identifiable modes cannot exceed the numerical rank of this
basis. If center extraction is unstable, the more natural compressed object is the
time-node coefficient

$$
C_\ell=\sum_pR_p
\sin\left(\tau_\ell\frac{\Omega_p}{\xi_0}\right),
$$

rather than a set of individually meaningful poles.

## 4. Why RegMPA is not an off-axis MPA in disguise

Screening is obtained from

$$
W[\chi^0]=(1-V\chi^0)^{-1}V.
$$

Complex evaluation is an algebra homomorphism: evaluating every quantity at one
$z$ commutes with products and inverses. Therefore

$$
W(z)=(1-V\chi^0(z))^{-1}V
$$

is unambiguous.

A general smoothing operator $L_{\xi,Q}$ does not have that property:

$$
(1-VL_{\xi,Q}[\chi^0])^{-1}V
\ne
L_{\xi,Q}\left[(1-V\chi^0)^{-1}V\right]
$$

in general. Thus a RegMPA fit to the first expression is an empirical modal
representation of the **regulated screened interaction**. It is not an exact
deconvolution of physical $W$ and is not expected to produce width-independent
centers.

This is acceptable if the target quantity is explicitly a regulated GW result and
the same regulator is used consistently through the self-energy.

## 5. Sigma procedure for RegMPA

The regulator is already present in the time representation of each mode. Stage C
must evaluate the $G\,W$ contraction using the defining regulated kernel or its
time-node representation.

It must **not** perform the ordinary MPA sequence

1. interpret $\Omega_p$ as a bare rational pole;
2. analytically form the unregulated pole denominator; and
3. apply an independent HGL crossing regularizer.

That sequence changes the fitted model and generally regularizes it twice.

Instead, Stage C should use

$$
W^c_{\xi_0,Q}(t)
\longleftrightarrow
\sum_pR_p\Phi_{\xi_0,Q}(\omega;\Omega_p)
$$

directly. For a finite sine-sum regulator, the existing positive time nodes are
already the natural evaluation points for the $G(t)W(t)$ contraction. A separate
crossing kernel is required only if it is deliberately defined as an additional
physical broadening, in which case the two widths must be reported separately.

## 6. Quadrature reuse across real frequencies

RegMPA does not require a custom quadrature for every real sample frequency.
Frequency translation appears only in

$$
e^{i\tau_\ell\omega_j/\xi_0}.
$$

For a fixed regulator and transition-window class:

- all $\omega_j$ reuse the same $\{\tau_\ell,w_\ell\}$;
- propagators at the time nodes are built once;
- all requested real frequencies are formed by a batched scalar phase mix;
- only the Dyson solve and output storage retain a frequency dimension.

The grid still depends on the dimensionless bandwidth

$$
A=\frac{E_{\mathrm{bw}}}{\xi_0},
$$

but this can be handled with conservative precomputed buckets. Select the smallest
table with $A_{\mathrm{table}}\ge A_{\mathrm{requested}}$. Two regulator widths then
require approximately two grids per window class, not one grid per $\omega_j$.

If the minimax optimization is allowed to discover a different excluded-region
shape for every $A$, then $A$ and the table identity are part of the model, not
mere numerical metadata. A cleaner production design would keep one normalized
target kernel and vary only how accurately and over how large an interval each
table approximates it.

## 7. Multiple-width procedures

Using two widths in RegMPA is not the same as double-parallel sampling.

### 7.1 Independent-width models

Fit

$$
\{\Omega_p(\xi_1),R_p(\xi_1)\}
\quad\text{and}\quad
\{\Omega_p(\xi_2),R_p(\xi_2)\}
$$

independently. This is the simplest and most honest procedure when
regulator-dependent poles are acceptable. Each model produces a different
regulated Sigma and must retain its own metadata.

### 7.2 Shared-center diagnostic

Fit both widths jointly with common real centers and width-specific residues:

$$
W^c_{\xi_k,Q_k}(\omega_j)
\approx
\sum_pR_{p,k}\Phi_{\xi_k,Q_k}(\omega_j;\Omega_p).
$$

This tests whether the regulator mainly changes mode amplitudes. Failure is useful
evidence that the Dyson non-closure or the unconstrained central region moves the
mode centers substantially.

### 7.3 Width-dependent center model

Only after the shared-center test, introduce a low-order width dependence, for
example

$$
\Omega_p(\xi)=\Omega_p^{(0)}+c_p\xi^2+\cdots.
$$

At least three widths are needed before such an extrapolation is credible. The
expansion is not uniform near a resonance, so a zero-width extrapolation is a
separate numerical hypothesis, not an automatic consequence of RegMPA.

## 8. Side-by-side procedure summary

| Stage | Ordinary $+i\bar{\omega}$ MPA | Regulator-dressed MPA |
|---|---|---|
| Sample coordinates | Complex $z_j=\omega_j+i\bar{\omega}_k$ | Real $\omega_j$ plus regulator $(\xi_0,Q)$ |
| Object sampled | One analytic $W(z)$ | A family of regulated functions $W_{\xi_0,Q}(\omega)$ |
| Meaning of width | Sampling coordinate only | Permanent model parameter |
| Expected pole dependence on width | None in the exact/full-rank limit | Allowed and generally expected |
| Kernel | $2\Omega R/(z^2-\Omega^2)$ | $R[d_{\xi,Q}(\omega-\Omega)-d_{\xi,Q}(\omega+\Omega)]$ |
| Fit | Pade linear system, companion roots, residue refit | Nonlinear center fit plus linear residue solve |
| Exact $2n_p$ interpolation | Natural algebraic construction | Not generally available |
| Preferred sampling count | $2n_p$ minimum; extra points improve robustness | Overdetermined real grid strongly preferred |
| Final model | Unregularized rational pole model | Regulated finite-band modal model |
| Sigma crossing treatment | Separate HGL/minimax choice | Reuse the model regulator; do not apply it twice |
| Two widths | Two lines sampling the same analytic function | Two different regulated functions unless jointly constrained |
| Cache provenance | Store $z_j$ and fit settings | Store $\xi_0$, $\xi_{\mathrm{eff}}$, $Q$, valid range, and fit settings |

## 9. Proposed first numerical experiment

No Lanczos-chain data are required for this experiment.

1. Select one small gapped fixture and one fixed excluded-range regulator table.
2. Compute $W^c_{\xi,Q}(\omega_j)$ at 20 nonuniform real frequencies.
3. Reserve at least four additional frequencies as held-out validation points, or
   fit only 16 of the 20 initially.
4. Fit $n_p=4,6,8$ real mode centers with complex residues by variable projection.
5. Compare held-out $W$ error with a direct time-coefficient fit using
   $\{C_\ell\}$ and with an ordinary rational fit to the same regulated data.
6. Check static reconstruction, Hermitian/conjugacy relations, diagonal loss sign,
   condition number, and mode stability under changes in the fit grid.
7. Repeat with a second width. Run the independent-width and shared-center fits.
8. On a small explicit band/pair sum, compare the RegMPA $G\,W$ contraction with a
   direct calculation using the same regulator. This is the decisive Stage-C gate.

For 20 samples, an overdetermined $n_p=6$ or $8$ fit is safer than an exactly
determined ten-mode fit. Quadrature, ISDF, and Dyson errors otherwise have enough
freedom to become spurious mode pairs.

## 10. Decision criterion

RegMPA is worth pursuing if it simultaneously demonstrates:

- materially fewer time nodes than the Cauchy/Lorentzian sampler at matched
  regulated-$W$ error;
- stable held-out reconstruction of $W^c_{\xi,Q}(\omega)$;
- a low-rank mode representation that improves materially over storing the direct
  time coefficients $C_\ell$;
- a regulator-consistent Sigma matching the direct small-system sum; and
- a useful QP plateau over a reasonable range of regulator widths.

If the time-coefficient representation is both smaller and more stable than the
mode fit, forcing multipoles adds no compression. In that case the correct
low-scaling model is a regulator-specific time expansion rather than MPA.

## References

- D. A. Leon et al., "Frequency dependence in GW made simple using a multi-pole
  approximation," Phys. Rev. B 104, 115157 (2021),
  <https://arxiv.org/abs/2109.01532>.
- M. Kim, G. J. Martyna, and S. Ismail-Beigi, "Complex-time shredded propagator
  method for large-scale GW calculations," Phys. Rev. B 101, 035139 (2020),
  <https://arxiv.org/abs/1904.10512>.

