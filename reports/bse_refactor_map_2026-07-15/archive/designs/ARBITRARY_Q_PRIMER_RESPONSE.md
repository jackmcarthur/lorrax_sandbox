Yes. I think your objection is correct, and I would weaken my previous negative conclusion substantially.

The observation that raw (\zeta_q=C_q^{-1}Z_q) is rough—or that its naïve (q)-Fourier transform appears long-ranged—does **not** rule out interpolation after identifying and parallel-transporting the physical subspace. In particular:

1. (\zeta_q) is a dual basis in a badly conditioned point-value coordinate system.
2. Its Fourier transform is meaningless as a locality diagnostic unless every (q) uses the same smooth, BZ-periodic frame and the (G)-channel sewing at the zone boundary is handled correctly.
3. The natural smooth object contains only **one half-inverse** of (C_q), not the full inverse.

The earlier physical-contraction failure is still a warning, but it tested a reconstruction that passed interpolation error through (C^{-1}). It does not test the construction below. Your measured decay of (C_R) and (Z_R) is exactly the reason this construction remains plausible. 

Benzi–Boito–Razouk rigorously establish decay of spectral projectors under spectral-gap/locality hypotheses, including nonorthogonal representations. Their paper is not directly a theorem about your (q)-dependent ISDF Gram, but it supports the underlying principle: locality belongs naturally to projectors and matrix functions associated with separated spectral subspaces, not necessarily to individual ill-conditioned dual vectors. ([arXiv][1])

# 1. Rewrite the ISDF fit in its physical orthonormal frame

Let (A_q) be the matrix of pair densities:

[
[A_q]*{p,r}=\rho*{p,q}(r),
]

where (p) abbreviates the band, (k), and spin labels. Let (X_q) be the restriction to the interpolation points:

[
[X_q]*{p\mu}=\rho*{p,q}(r_\mu).
]

Then

[
C_q=X_q^\dagger X_q,
\qquad
Z_q=X_q^\dagger A_q,
\qquad
\zeta_q=C_q^+Z_q.
]

Take a thin SVD of the centroid-value matrix:

[
X_q=L_q S_q R_q^\dagger,
]

with

[
L_q^\dagger L_q=I_r,
\qquad
R_q^\dagger R_q=I_r,
\qquad
S_q=\operatorname{diag}(\sigma_{1q},\ldots,\sigma_{rq}).
]

Thus

[
C_q=R_qS_q^2R_q^\dagger.
]

Now define

[
\boxed{
\Phi_q
\equiv
S_q^{-1}R_q^\dagger Z_q.
}
]

Because

[
Z_q=X_q^\dagger A_q
===================

R_qS_qL_q^\dagger A_q,
]

we get the crucial identity

[
\boxed{
\Phi_q=L_q^\dagger A_q.
}
]

So (\Phi_q) is not fundamentally an inverse-Gram object. It is simply the representation of the pair densities in the orthonormal pair-feature frame (L_q).

The fitted pair-density matrix becomes

[
A_q^{\rm fit}=L_q\Phi_q.
]

Meanwhile,

[
\boxed{
\zeta_q=R_qS_q^{-1}\Phi_q.
}
]

This final factor (R_qS_q^{-1}) maps the physical orthonormal frame back into the highly redundant point-value dual coordinates. That is the part I strongly suspect destroys apparent locality.

In terms of powers of conditioning:

[
\Phi_q=S_q^{-1}R_q^\dagger Z_q
]

contains (1/\sigma), whereas

[
\zeta_q=R_qS_q^{-2}R_q^\dagger Z_q
]

contains (1/\sigma^2). Since

[
\kappa(C)=\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^2,
]

the difference is roughly (\sqrt{\kappa(C)}) versus (\kappa(C)).

For (\kappa(C)\sim10^9), (\sqrt{\kappa}\sim3\times10^4). That is not pleasant, but it is qualitatively different from (10^9), and the structured errors in (Z) may be much less harmful than a worst-case bound suggests.

## Recovering (\Phi_q) from existing data

You do not need stored (Z_q). From

[
\zeta_q=R_qS_q^{-1}\Phi_q
]

it follows that

[
\boxed{
\Phi_q=S_qR_q^\dagger\zeta_q.
}
]

Thus, on your existing reciprocal representation,

[
\Phi_q(G)
=========

S_qR_q^\dagger\widetilde\zeta_q(G).
]

Multiplication by (S_q) suppresses precisely the near-null directions that are enormous in the raw (\zeta_q).

# 2. Never interpolate the raw (V_{\mu\nu}) if you can avoid it

Define the Coulomb matrix in the orthonormal pair-feature frame:

[
\boxed{
K_q
===

\Phi_q,v_q,\Phi_q^\dagger.
}
]

It is related to your existing centroid tile by

[
\boxed{
K_q=S_qR_q^\dagger V_qR_qS_q.
}
]

For a pair whose centroid-value row is (x_{p,q}), define

[
a_{p,q}=x_{p,q}R_qS_q^{-1}.
]

For the training pair densities, these are exactly the corresponding rows of (L_q). Then

[
\boxed{
x_{p,q}V_qx_{p',q}^\dagger
==========================

a_{p,q}K_qa_{p',q}^\dagger.
}
]

This is the same physical interaction, but represented in an orthonormal physical pair frame.

At arbitrary (Q), you can obtain (x_{p,Q}) from the htransform wavefunctions at the centroids. You can therefore build (C_Q=X_Q^\dagger X_Q), diagonalize it, and obtain (a_{p,Q}), without ever generating wavefunctions on the full real-space grid.

So the off-grid workflow should output either:

[
\Phi_Q
\quad\text{or}\quad
K_Q,
]

not necessarily a raw (V_Q).

Reconstructing

[
V_Q
===

R_QS_Q^{-1}K_QS_Q^{-1}R_Q^\dagger
]

puts the bad conditioning back. The BSE exchange application should ideally be changed to contract (a_QK_Qa_Q^\dagger) directly.

# 3. The gauge freedom that parallel transport fixes

The SVD is not unique. At every (q),

[
L_q\rightarrow L_qU_q
]

for any (r\times r) unitary (U_q), provided

[
\Phi_q\rightarrow U_q^\dagger\Phi_q.
]

Correspondingly,

[
K_q\rightarrow U_q^\dagger K_qU_q,
\qquad
a_q\rightarrow a_qU_q.
]

The physical contraction remains invariant:

[
a_qK_qa_q^\dagger
\rightarrow
a_qU_q
U_q^\dagger K_qU_q
U_q^\dagger a_q^\dagger
=======================

a_qK_qa_q^\dagger.
]

So ({L_q}) is a frame for an (r)-dimensional vector bundle over the BZ. A raw eigendecomposition of (C_q) chooses an arbitrary frame at every point. Fourier transforming (\Phi_q) or (\zeta_q) before aligning those frames can easily give a completely flat or nonsensical (R)-space tail even when the underlying projector is analytic.

The standard multiband parallel-transport construction uses overlap matrices between neighboring frames, with an SVD/polar decomposition to choose the maximally aligned unitary. This is exactly the machinery used to smooth composite Bloch manifolds and construct periodic gauges for Wannier interpolation. ([arXiv][2])

# 4. The auxiliary-frame overlap

Suppose first that the pair-row spaces at (q) and (q') have already been identified in a common band gauge. Then form

[
M_{qq'}=L_q^\dagger L_{q'}.
]

Using centroid data, you do not have to materialize (L_q). Define the cross Gram

[
H_{qq'}=X_q^\dagger X_{q'}.
]

Then

[
\boxed{
M_{qq'}
=======

S_q^{-1}
R_q^\dagger
H_{qq'}
R_{q'}
S_{q'}^{-1}.
}
]

If the two physical auxiliary subspaces are nearly the same, the singular values of (M_{qq'}) will be close to one. They are the cosines of the principal angles between the two subspaces.

Take

[
M_{qq'}=A\Lambda B^\dagger.
]

The unitary polar factor is

[
\boxed{
T_{q\leftarrow q'}=AB^\dagger.
}
]

This maps internal coordinates at (q') into the frame at (q).

Transport rules are then

[
\boxed{
\Phi_{q'\to q}
==============

T_{q\leftarrow q'}\Phi_{q'},
}
]

[
\boxed{
K_{q'\to q}
===========

T_{q\leftarrow q'}K_{q'}T_{q\leftarrow q'}^\dagger,
}
]

and

[
a_{q'\to q}
===========

a_{q'}T_{q\leftarrow q'}^\dagger.
]

These formulas are gauge covariant. Under independent frame changes

[
L_q\to L_qG_q,
\qquad
L_{q'}\to L_{q'}G_{q'},
]

the polar link transforms as

[
T_{q\leftarrow q'}
\to
G_q^\dagger
T_{q\leftarrow q'}
G_{q'},
]

so the transported (\Phi_{q'\to q}) transforms only in the target frame:

[
\Phi_{q'\to q}\to G_q^\dagger\Phi_{q'\to q}.
]

The polar factor is also the Procrustes-optimal unitary: it gives the rotation of the (q') frame that maximizes its overlap with the (q) frame. The gauge covariance of polar-overlap transport can be proved directly from these equations and is standard in discrete non-Abelian parallel transport. ([arXiv][3])

# 5. You first need to identify the pair-row Hilbert spaces

There is a subtlety: the row label

[
p=(m,n,k)
]

does not automatically have the same gauge at (q) and (q'), because the left orbital is evaluated at (k-q) versus (k-q').

You therefore need single-particle parallel transport before constructing (H_{qq'}).

For each (k), let (U_\kappa) be the selected band-frame matrix at

[
\kappa=k-q,
\qquad
\kappa'=k-q'.
]

The columns can be represented in your fixed htransform (\alpha)-basis, so their overlaps should be available without full real-space wavefunctions:

[
S_{\kappa\kappa'}
=================

U_\kappa^\dagger U_{\kappa'}.
]

Take its polar factor:

[
t_{\kappa\leftarrow\kappa'}
===========================

\operatorname{polar}
\left(
U_\kappa^\dagger U_{\kappa'}
\right).
]

For pair densities with the left orbital conjugated and the right orbital fixed, the induced map in pair-index space is

[
\mathcal B_{q\leftarrow q'}
===========================

\bigoplus_k
\left(
t_{k-q\leftarrow k-q'}^*
\otimes I_{\rm right}
\right).
]

If both legs move, it becomes

[
t_L^*\otimes t_R.
]

The actual cross Gram is therefore

[
\boxed{
H_{qq'}
=======

X_q^\dagger
\mathcal B_{q\leftarrow q'}
X_{q'}.
}
]

You can implement this without forming the enormous pair-space matrix (\mathcal B): rotate the shifted-band index of the (q') centroid pair products by (t^*) before performing the band and (k) sums.

The resulting (H_{qq'}) should retain the two-sums structure. Schematically, its entries are products of:

* a cross-density matrix between the transported left-band frames at (k-q) and (k-q');
* the ordinary same-(k) density matrix on the right leg.

This is all centroid-only work.

## Degenerate bands

All transport must be performed on complete band subspaces, never individual eigenvectors. If a degeneracy straddles the band-window cutoff, the pair subspace itself is noncovariant, and no auxiliary transport will repair it. Your existing closed-shell degeneracy-window rule remains mandatory.

# 6. BZ boundary sewing is absolutely capable of ruining the previous test

This is one of the first things I would investigate.

For Bloch-periodic orbitals,

[
|u_{n,k+G}\rangle
=================

e^{-iG\cdot r}
\sum_m |u_{m,k}\rangle
,B^G_{mn}(k),
]

where (B^G(k)) is the band sewing matrix for the selected composite subspace.

When a transport edge crosses the BZ boundary, the single-particle overlap must therefore be

[
S_{k,k'+G}
==========

U_k^\dagger
e^{-iG\cdot r}
U_{k'_{\rm wrapped}},
]

with the appropriate band sewing included.

In the htransform basis, (e^{-iG\cdot r}) should be represented as a fixed small matrix. At the centroids, it is simply

[
D_G^{(\mu)}
===========

\operatorname{diag}
\left(
e^{-iG\cdot r_\mu}
\right).
]

## Fixed (G)-slot interpolation is not periodic

For a Bloch-stripped quantity,

[
\bar\Phi_q(r)=e^{-iq\cdot r}\Phi_q(r),
]

BZ sewing gives

[
\bar\Phi_{q+G_0}(r)
===================

e^{-iG_0\cdot r}\bar\Phi_q(r),
]

and hence, in reciprocal space,

[
\boxed{
\bar\Phi_{q+G_0}(G)
===================

\bar\Phi_q(G+G_0)
}
]

up to the auxiliary and band sewing matrices.

Thus, the fixed integer label (G=0) is not a periodic component. At the BZ boundary, it becomes a different reciprocal channel. Componentwise interpolation of padded arrays at fixed (G)-slot can produce an apparent discontinuity even for a perfectly smooth physical function.

The safest representation for the (q)-interpolation is therefore the **full Bloch pair function in real space**:

[
\Phi_q(r)
=========

e^{iq\cdot r}
\sum_G
\bar\Phi_q(G)e^{iG\cdot r}.
]

After all band sewing is applied, (\Phi_q(r)) should be periodic under (q\to q+G_0). Interpolate this object in (q). Only at the target (Q) should you strip (e^{iQ\cdot r}) and Fourier transform back to (G).

This does not require full-grid off-grid wavefunctions. You are only reconstructing the already-stored coarse-(q) auxiliary functions on the real-space FFT grid.

# 7. Two ways to implement the transport

## Method A: target-centered local transport

This is the simplest and probably the best first experiment.

For a target (Q):

1. Generate the shifted wavefunctions only at the centroids.
2. Build (X_Q) and therefore
   [
   C_Q=X_Q^\dagger X_Q.
   ]
3. Compute
   [
   X_Q=L_QS_QR_Q^\dagger.
   ]
4. Select a local coarse-(q) stencil around (Q), respecting the BZ torus.
5. For each stencil point (q_i), form
   [
   M_{Qq_i}
   ========

   L_Q^\dagger
   \mathcal B_{Q\leftarrow q_i}
   L_{q_i}.
   ]
6. Take
   [
   T_{Q\leftarrow q_i}
   ===================

   \operatorname{polar}(M_{Qq_i}).
   ]
7. Transport the coarse auxiliary functions into the target frame:
   [
   \widehat\Phi_{q_i}^{(Q)}
   ========================

   T_{Q\leftarrow q_i}\Phi_{q_i}.
   ]
8. Apply ordinary scalar interpolation:
   [
   \boxed{
   \Phi_Q^{\rm interp}
   ===================

   \sum_i w_i(Q),
   \widehat\Phi_{q_i}^{(Q)}.
   }
   ]

Because all terms now live in the target frame, this sum is gauge covariant.

Then compute

[
K_Q
===

\Phi_Q^{\rm interp}
v_Q
\Phi_Q^{{\rm interp}\dagger}.
]

Finally,

[
a_{p,Q}
=======

x_{p,Q}R_QS_Q^{-1},
]

and apply exchange as

[
a_{p,Q}K_Qa_{p',Q}^\dagger.
]

This avoids:

* interpolating (C^{-1}Z);
* constructing off-grid full-grid wavefunctions;
* a global periodic gauge;
* global topological obstructions.

The stencil can start with (2^d) cell corners and multilinear weights, but I would probably use a periodic tensor-product cubic stencil once the basic construction works.

### Direct versus path transport

A direct overlap (M_{Qq_i}) is preferable when its smallest singular value is reasonably large. If a stencil point is far enough that the subspaces have large principal angles, transport through neighboring mesh points:

[
T_{Q\leftarrow q_i}
===================

T_{Q\leftarrow q_{n}}
T_{q_n\leftarrow q_{n-1}}
\cdots
T_{q_1\leftarrow q_i}.
]

The product is a discrete path-ordered parallel transporter.

Different paths will differ by the subspace curvature. That difference is diagnostic, not necessarily an error.

## Method B: construct a global smooth periodic frame and FFT

This is the version that most directly exploits your (R)-space decay argument.

Build nearest-neighbor links

[
T_j(q)
======

T_{q\leftarrow q+b_j}
]

on the entire coarse (q)-mesh.

Choose a reference point (q_0), set its gauge matrix (G(q_0)=I), and propagate:

[
G(q+b_j)
========

T_j(q)^\dagger G(q).
]

The transported frame and auxiliary functions are

[
\widetilde L_q=L_qG(q),
\qquad
\widetilde\Phi_q=G(q)^\dagger\Phi_q.
]

This maximally aligns adjacent frames.

### Restoring BZ periodicity in one dimension

After transporting around a reciprocal-space loop, the endpoint will generally differ from the reciprocal-sewn starting frame by a unitary obstruction or Wilson matrix (W):

[
\widetilde L_{q_0+G}
====================

\mathcal S_G\widetilde L_{q_0}W.
]

Choose a matrix logarithm

[
W=e^{\mathcal L}.
]

At fractional position (t\in[0,1]), distribute the mismatch:

[
\widetilde L_q
\rightarrow
\widetilde L_q e^{-t\mathcal L},
]

[
\widetilde\Phi_q
\rightarrow
e^{t\mathcal L}\widetilde\Phi_q.
]

At (t=1), the obstruction is removed. This is the non-Abelian analogue of distributing a Berry phase uniformly along a loop.

In two and three dimensions, perform this sequentially:

1. parallel transport and periodic correction along (q_1);
2. transport along (q_2) for each (q_1), then correct the (q_2) obstruction as a continuous function of (q_1);
3. repeat along (q_3).

This is essentially the constructive periodic-gauge procedure used in Wannier methods. Nonzero winding of the obstruction matrices signals a topological obstruction to a single global smooth periodic frame; in that case use local patches rather than forcing a global gauge. ([arXiv][2])

Once the periodic frame is established, take

[
\widetilde\Phi_R(r)
===================

\frac{1}{N_q}
\sum_q
e^{-iq\cdot R}
\widetilde\Phi_q(r).
]

The decisive test is whether

[
|\widetilde\Phi_R|
]

decays.

If it does, interpolate by

[
\widetilde\Phi_Q(r)
===================

\sum_{R\in R_c}
e^{iQ\cdot R}
\widetilde\Phi_R(r).
]

Then align the target raw frame (L_Q) with the nearby global transported frame, obtain the corresponding target gauge matrix (G(Q)), and use

[
a_Q^{\rm PT}=a_QG(Q)
]

with the interpolated (\widetilde K_Q).

# 8. Why decay of (C_R) and (Z_R) gives a real mathematical basis for this

Suppose

[
C(q)=\sum_R e^{iq\cdot R}C_R,
\qquad
Z(q)=\sum_R e^{iq\cdot R}Z_R,
]

and (C_R,Z_R) decay exponentially. Then (C(q)) and (Z(q)) extend analytically into a strip of complex (q).

Now assume either:

[
C(q)\ge \lambda_* I
]

uniformly on the relevant subspace, or there is a fixed spectral cluster separated from the discarded spectrum:

[
\lambda_r(q)\ge a>b\ge\lambda_{r+1}(q)
]

for every (q).

Then the spectral projector onto the active cluster,

[
P(q)
====

\frac{1}{2\pi i}
\oint_\Gamma
(z-C(q))^{-1},dz,
]

is analytic. The restricted inverse square root

[
C(q)^{-1/2}P(q)
]

is also analytic.

Therefore the half-whitened object

[
\Phi(q)
=======

C(q)^{-1/2}Z(q)
]

is locally analytic as a section of the active subspace bundle. After choosing a smooth parallel-transport frame, its Fourier coefficients should decay exponentially, with a rate controlled by:

* the decay lengths of (C_R) and (Z_R);
* the distance to complex-(q) singularities;
* the minimum retained singular value;
* the gap separating retained and discarded singular subspaces.

That argument does **not** apply as cleanly to

[
\zeta(q)=C(q)^{-1}Z(q),
]

because it contains the additional inverse square root mapping from the orthonormal physical frame into the centroid dual frame. The constants in any decay bound can deteriorate dramatically with (\kappa(C)).

So I agree with your core point: the decay result should motivate trying a projector/frame construction, not abandoning interpolation merely because the dual basis itself looks bad.

# 9. Fixed rank is important

Parallel transport is simplest when the active rank (r) is constant over the BZ.

I would inspect the full spectrum

[
\sigma_j(q)^2=\lambda_j(C_q)
]

and look for a **uniform** gap. Do not choose the rank separately at every (q) using a relative threshold; that creates rank jumps and makes the bundle discontinuous by construction.

Possible choices:

### Use all numerically resolved directions

With

[
\kappa(C)\sim10^9,
]

the condition number of the square root is only

[
\sqrt{\kappa(C)}\sim3\times10^4.
]

That may be entirely manageable in complex128. It is worth trying the full nominal rank before imposing aggressive truncation.

### Fixed-rank truncation

Choose one (r) globally based on:

* the worst (q);
* direct ISDF residual convergence;
* physical exchange-kernel convergence.

Then use the top (r) directions everywhere.

### Soft filtering

If there is no clean gap, define a smooth filter such as

[
f_\tau(\lambda)
===============

\frac{1}{\sqrt{\lambda+\tau}}
]

or a smoother spectral window, rather than a discontinuous rank cut. The corresponding fit is no longer exactly the original least-squares fit, but the variation with (q) is controlled. I would first test fixed full rank, however.

# 10. What I would calculate immediately

The following sequence can determine quite quickly whether this is real.

## A. Exact BZ sewing test

For equivalent (q) and (q+G), verify the expected covariant identities for:

[
C_q,\qquad
X_q,\qquad
\Phi_q(r),
\qquad
\Phi_q(G).
]

In particular, verify reciprocal-channel relabeling rather than fixed-slot equality:

[
\bar\Phi_{q+G_0}(G)
\stackrel{?}{=}
\bar\Phi_q(G+G_0).
]

I would not trust any (q)-FFT result until this closes near machine precision.

## B. Neighboring-subspace principal angles

For every nearest-neighbor edge, compute the singular values of

[
M_{qq'}.
]

Plot

[
1-s_{\min}(M_{qq'})
]

and the full distribution.

Interpretation:

* all singular values near one: roughness is mostly gauge;
* a few smaller singular values: a small tail of the auxiliary subspace changes rapidly;
* many small singular values: the physical pair-feature subspace itself is underresolved by the coarse grid.

## C. Compare four (R)-space tails

Calculate:

1. raw (\zeta_R);
2. half-whitened (\Phi_R) without transport;
3. parallel-transported (\widetilde\Phi_R);
4. transported short-range kernel (\widetilde K_R^{\rm SR}).

That comparison will tell you exactly which operation restores locality.

My prior claim would only be justified if **(3)** remains nondecaying after exact seam treatment.

## D. Plaquette holonomy

For every elementary (q)-mesh plaquette, compute

[
W_\square
=========

T_{1}T_{2}T_{3}T_{4}.
]

Measure

[
|W_\square-I|.
]

This measures the non-Abelian curvature over one coarse cell. If it is small, path dependence is negligible and local interpolation should work well. If it is large, the grid is too coarse for a single low-order transported interpolant, even if the underlying bundle is smooth.

## E. Leave-one-out in the transported representation

At a held-out coarse point (q_0):

1. construct (L_{q_0},S_{q_0},R_{q_0}) from exact centroid data;
2. transport neighboring coarse (\Phi_q)'s into the (q_0) frame;
3. interpolate (\Phi_{q_0});
4. form (K_{q_0}) using the exact (v(q_0+G));
5. compare the physical action
   [
   a_{q_0}K_{q_0}a_{q_0}^\dagger.
   ]

This is the apples-to-apples test that the previous (C^{-1}Z) reconstruction did not perform.

# 11. Minimal pseudocode

For each coarse (q):

```python
# X_q: pair-density values at centroids, shape (npair, nmu)
C = X.conj().T @ X

lam, R = eigh(C)
idx = fixed_active_indices
s = sqrt(lam[idx])
R = R[:, idx]

# Orthonormal pair-feature frame, used only for links.
L = (X @ R) / s[None, :]

# zeta_qG shape: (nmu, nG)
# Phi_qG shape: (rank, nG)
Phi_qG = s[:, None] * (R.conj().T @ zeta_qG)

# Existing V tile can be converted without using zeta explicitly.
K_q = (
    s[:, None]
    * (R.conj().T @ V_q @ R)
    * s[None, :]
)
```

For an edge (q\to q'):

```python
# H = X_q^H B_pair(q <- q') X_qp
# B_pair contains single-particle band parallel transport.
M = (
    (R_q.conj().T @ H @ R_qp)
    / s_q[:, None]
    / s_qp[None, :]
)

U, principal_cosines, Vh = svd(M)
T_q_from_qp = U @ Vh

Phi_qp_in_q = T_q_from_qp @ Phi_qp
K_qp_in_q = T_q_from_qp @ K_qp @ T_q_from_qp.conj().T
```

At arbitrary (Q), with target centroid samples:

```python
C_Q = X_Q.conj().T @ X_Q
lam_Q, R_Q = eigh(C_Q)
s_Q = sqrt(lam_Q[idx])
R_Q = R_Q[:, idx]

a_Q = (x_pair_Q @ R_Q) / s_Q[None, :]

# Transport/interpolate coarse Phi into target frame.
Phi_Q = sum(
    weight_i
    * T_Q_from_qi
    @ Phi_qi
    for i in stencil
)

K_Q = contract_Phi_v_Phi(Phi_Q, v_at_Q_plus_G)

exchange = a_Q @ K_Q @ a_Q.conj().T
```

# Bottom line

I now think the most promising interpolation is:

[
\boxed{
\text{parallel-transport and interpolate }
\Phi_q=L_q^\dagger A_q
======================

S_qR_q^\dagger\zeta_q,
\text{ not }\zeta_q.
}
]

Then:

[
\boxed{
K_Q=\Phi_Qv_Q\Phi_Q^\dagger
}
]

is formed using the exact Coulomb kernel at the target, and the off-grid centroid pair samples are mapped into the target orthonormal frame through

[
a_Q=x_QR_QS_Q^{-1}.
]

This construction needs off-grid wavefunctions only at the centroids. It is explicitly gauge covariant, can enforce exact BZ sewing, reduces the conditioning exposure from (C^{-1}) to (C^{-1/2}), and gives your observed decay of (C_R,Z_R) a mathematically coherent path to imply decay of a transported physical object.

The most decisive first result would be a plot of the (R)-shell norm of **parallel-transported (\Phi_R)**. If that decays while raw (\zeta_R) does not, the entire earlier negative interpretation was mostly a coordinate/gauge artifact.

[1]: https://arxiv.org/abs/1203.3953 "https://arxiv.org/abs/1203.3953"
[2]: https://arxiv.org/pdf/2306.00678 "https://arxiv.org/pdf/2306.00678"
[3]: https://arxiv.org/pdf/2605.26697 "https://arxiv.org/pdf/2605.26697"






I (Jack) then asked what it thought you needed to know about long/short range separation, and it said



Yes. The main additional instruction is:

> **Parallel transport acts on the auxiliary frame; the SR/LR split acts on the Coulomb operator. Do not split or interpolate the raw dual vectors.**

Those two operations commute cleanly, but several superficially similar implementations do not.

## 1. Use this exact decomposition

With the half-whitened auxiliary functions

[
\Phi_q=S_qR_q^\dagger\zeta_q,
]

define the interaction in the orthonormal auxiliary frame:

[
\mathcal V_q
============

# \Phi_q,v_q,\Phi_q^\dagger

S_qR_q^\dagger V_qR_qS_q.
]

Split the **scalar reciprocal-space operator**, not (\Phi):

[
v_q(G)
======

v_q^{\mathrm{SR}}(G)+v_q^{\mathrm{LR}}(G),
]

[
v_q^{\mathrm{LR}}(G)
====================

v_{\mathrm{dim}}(q+G)
e^{-|q+G|^2/(4\alpha^2)},
]

[
v_q^{\mathrm{SR}}(G)
====================

v_{\mathrm{dim}}(q+G)
\left[
1-e^{-|q+G|^2/(4\alpha^2)}
\right].
]

Then

[
\boxed{
\mathcal V_q
============

\mathcal V_q^{\mathrm{SR}}
+
\mathcal V_q^{\mathrm{LR}}
}
]

with

[
\mathcal V_q^s
==============

\Phi_qv_q^s\Phi_q^\dagger,
\qquad s\in{\mathrm{SR},\mathrm{LR}}.
]

There are **no cross terms**, because the decomposition is additive in the operator (v). Cross terms would arise if one instead tried to write

[
\Phi=\Phi^{\mathrm{SR}}+\Phi^{\mathrm{LR}},
]

which is not the desired decomposition.

The subtract–interpolate–add strategy in the exciton-Wannier and polar electron–phonon literature is likewise a split of the physical interaction or vertex into an analytic long-range contribution and a smooth remainder. ([arXiv][1])

## 2. Transport and splitting commute

If (T_{Q\leftarrow q}) transports auxiliary coordinates from the (q) frame into the target (Q) frame, then

[
\Phi_q\rightarrow
\widehat\Phi_q^{(Q)}
====================

T_{Q\leftarrow q}\Phi_q,
]

and

[
\mathcal V_q^s
\rightarrow
\widehat{\mathcal V}_q^{s,(Q)}
==============================

T_{Q\leftarrow q}
\mathcal V_q^s
T_{Q\leftarrow q}^\dagger.
]

Because (T) acts on the auxiliary index and (v^s) acts on the reciprocal-space index,

[
T\left(\Phi_qv_q^s\Phi_q^\dagger\right)T^\dagger
================================================

(T\Phi_q)v_q^s(T\Phi_q)^\dagger.
]

So the implementation can choose between two strategies:

### Strategy A: interpolate transported (\Phi_q)

[
\Phi_Q^{\mathrm{interp}}
========================

\sum_i w_i(Q)
T_{Q\leftarrow q_i}\Phi_{q_i},
]

then evaluate both pieces at the exact target:

[
\mathcal V_Q^s
==============

\Phi_Q^{\mathrm{interp}}
v_Q^s
\Phi_Q^{{\mathrm{interp}}\dagger}.
]

Advantages:

* positivity is automatic;
* the exact target (v(Q+G)) is used;
* SR and LR come from one consistent fitted pair-density representation.

Disadvantages:

* reciprocal-channel sewing and cutoff consistency must be handled;
* (\Phi) is a larger object than the (r\times r) interaction matrix.

### Strategy B: interpolate only transported (\mathcal V_q^{\mathrm{SR}})

[
\boxed{
\mathcal V_Q^{\mathrm{SR,interp}}
=================================

\sum_i w_i(Q)
T_{Q\leftarrow q_i}
\mathcal V_{q_i}^{\mathrm{SR}}
T_{Q\leftarrow q_i}^\dagger
}
]

and add the LR kernel separately.

This is probably the best first production implementation because:

* only (r\times r) matrices are interpolated;
* the reciprocal (G)-index has already been contracted away;
* BZ-boundary (G)-relabeling cannot directly contaminate the interpolation;
* the object is a physical operator in an orthonormal frame.

At the target, with

[
A_Q=X_QR_QS_Q^{-1},
]

the short-range physical pair kernel is

[
K_Q^{\mathrm{SR}}
=================

A_Q
\mathcal V_Q^{\mathrm{SR,interp}}
A_Q^\dagger.
]

Then add (K_Q^{\mathrm{LR}}) directly.

I would implement Strategy B first and Strategy A as the more stringent diagnostic.

## 3. The ideal LR implementation is a separate low-rank physical kernel

Let

[
F_{pK}(Q)
=========

\int d r,
\rho_{pQ}(r)e^{-iK\cdot r},
\qquad
K=Q+G,
]

for the small set of reciprocal vectors retained by the Gaussian LR factor.

Then

[
\boxed{
K_{pp'}^{\mathrm{LR}}(Q)
========================

\sum_{G}
F_{p,Q+G}^*(Q)
v^{\mathrm{LR}}(Q+G)
F_{p',Q+G}(Q).
}
]

In matrix form,

[
K_Q^{\mathrm{LR}}
=================

F_QD_Q^{\mathrm{LR}}F_Q^\dagger.
]

Because the Gaussian suppresses large (K), this has only a relatively small number of reciprocal channels and can be applied as a low-rank operation:

[
x\longmapsto
F_Q
\left[
D_Q^{\mathrm{LR}}
(F_Q^\dagger x)
\right].
]

The final exchange action is therefore

[
\boxed{
K_Qx
====

A_Q\mathcal V_Q^{\mathrm{SR,interp}}A_Q^\dagger x
+
F_QD_Q^{\mathrm{LR}}F_Q^\dagger x.
}
]

This is preferable to forcing the LR piece back into the auxiliary representation. The LR term naturally has a different, extremely low-rank representation.

### Where to obtain (F_Q)

In descending order of preference:

1. Evaluate small-(K) matrix elements using the htransform coefficient basis and precomputed projected plane-wave operators.
2. Use velocity/dipole and, where required, quadrupole matrix elements near (K=0).
3. Approximate them from the transported auxiliary representation:
   [
   F_Q\approx A_Q\Phi_Q^{\mathrm{interp}}.
   ]

The first option still needs no full-grid off-grid wavefunctions. If the htransform orbitals are represented as

[
|u_{nQ}\rangle
==============

\sum_a c_{an}(Q)|B_a\rangle,
]

precompute

[
M_{ab}(G)
=========

\langle B_a|e^{-iG\cdot r}|B_b\rangle.
]

Then arbitrary-(Q) form factors are small dense contractions of (c(Q)), (c(k)), and (M(G)).

## 4. Do not mix incompatible LR definitions

A common implementation error would be:

* subtract an exact finite-(\alpha), all-(G) LR contribution on the coarse grid;
* add back only a (G=0) dipole model at the target.

Then the interpolated remainder and added LR model do not sum to the original interaction.

The same mathematical LR model must be used on both sides:

[
K_q^{\mathrm{SR}}
=================

## K_q^{\mathrm{full}}

K_q^{\mathrm{LR,model}},
]

followed by

[
K_Q
===

K_Q^{\mathrm{SR,interp}}
+
K_Q^{\mathrm{LR,model}}.
]

There are two consistent choices:

* finite-(\alpha) Gaussian LR on both coarse and target points;
* a dipole-plus-quadrupole analytic LR model on both coarse and target points.

I strongly favor the finite-(\alpha) Gaussian form, using multipoles only to evaluate or stabilize its (K\rightarrow0) channels. The Gaussian form is periodic after the complete reciprocal-image sum and does not require deciding exactly where a dipole expansion ceases to be accurate.

## 5. Compute the SR kernel stably

Do not calculate

[
v^{\mathrm{SR}}=v-v^{\mathrm{LR}}
]

using ordinary subtraction near (K=0). Use

[
\boxed{
v^{\mathrm{SR}}(K)
==================

v_{\mathrm{dim}}(K)
\left[-\operatorname{expm1}
\left(
-\frac{K^2}{4\alpha^2}
\right)\right].
}
]

For 3D Ry units,

[
v_{\mathrm{dim}}(K)=\frac{8\pi}{K^2},
]

and

[
v^{\mathrm{SR}}(0)
==================

\frac{2\pi}{\alpha^2}.
]

This limit should be inserted analytically below a small-(K) threshold.

For the slab-truncated kernel, do **not** use the 3D limit. Expand the complete dimension-specific expression

[
v_{\mathrm{dim}}(K)
===================

\frac{8\pi}{K^2}T_{\mathrm{slab}}(K).
]

For (G_z=0) and (K_\parallel\rightarrow0), the unsplit slab interaction behaves as (1/|K_\parallel|), so multiplication by

[
1-e^{-K^2/(4\alpha^2)}
\sim K^2/(4\alpha^2)
]

makes (v^{\mathrm{SR}}) vanish linearly rather than approach the 3D constant.

Every dimensional kernel needs its own tested small-(K) series.

## 6. Reciprocal cutoff treatment matters

The identity

[
\sum_G f(Q+G)
]

is BZ periodic only when the reciprocal-image sum is treated consistently. A (Q)-dependent hard sphere can introduce small discontinuities as (G) vectors enter and leave.

For the LR piece:

* use a fixed global reciprocal-vector superset;
* include all vectors satisfying a worst-case Gaussian tolerance;
* select the cutoff so that
  [
  e^{-|Q+G|^2/(4\alpha^2)}<\epsilon_{\mathrm{LR}}
  ]
  for every omitted term and every (Q) in the BZ;
* preferably retain the Gaussian weight rather than applying an additional hard shell near significant weight.

For the SR coarse matrices, your existing (|q+G|) sphere can itself contaminate smoothness near the cutoff. Before judging transport interpolation, either:

1. use the complete common FFT-box (G) grid;
2. use a common subset present at every (q); or
3. introduce a smooth radial taper before the plane-wave cutoff.

For example,

[
v_{\mathrm{eff}}^{\mathrm{SR}}(K)
=================================

w_{\mathrm{cut}}^2(K),
v^{\mathrm{SR}}(K),
]

where (w_{\mathrm{cut}}=1) below (K_0) and smoothly falls to zero before (K_{\max}).

The squared window appears because there are two auxiliary-function legs. This should be converged against (K_0,K_{\max}).

A hard sphere changing membership across (q) could plausibly have contributed to the earlier apparent nonperiodicity. Your primer already identifies reciprocal-channel winding and fixed-(G)-slot interpolation as a serious issue. 

## 7. Split after defining one common transported frame

The frame must come exclusively from (C_q) or (X_q):

[
C_q=R_qS_q^2R_q^\dagger.
]

Do not define independent frames by diagonalizing:

[
\mathcal V_q^{\mathrm{SR}},
\qquad
\mathcal V_q^{\mathrm{LR}},
\qquad
\mathcal V_q.
]

Those eigenspaces can have crossings and arbitrary rotations unrelated to the physical pair subspace. Both SR and LR matrices must transform with the same link (T_{q\leftarrow q'}).

Similarly, use one fixed active rank over the full grid. If SR and LR use different spectral truncations, the algebraic identity between their sum and the total interaction will be lost.

## 8. Positivity and interpolation weights

For a nonnegative Coulomb kernel,

[
\mathcal V_q^{\mathrm{SR}}\succeq0,
\qquad
\mathcal V_q^{\mathrm{LR}}\succeq0.
]

This is a useful invariant.

If local multilinear interpolation is used,

[
\mathcal V_Q^{\mathrm{SR}}
==========================

\sum_i w_i
\widehat{\mathcal V}_{q_i}^{\mathrm{SR}},
\qquad
w_i\ge0,\quad \sum_iw_i=1,
]

then positive semidefiniteness is preserved automatically.

Higher-order polynomial or Fourier interpolation has weights of both signs and may produce small negative eigenvalues. Options are:

* initially use positive-weight multilinear interpolation;
* interpolate transported (\Phi_q) and reconstruct (\Phi v^{\rm SR}\Phi^\dagger);
* or project only tiny negative eigenvalues caused by interpolation noise.

I would not silently perform a large PSD projection; that could conceal genuine interpolation failure.

Always explicitly enforce numerical Hermiticity:

[
\mathcal V\leftarrow
\frac12(\mathcal V+\mathcal V^\dagger).
]

## 9. Treatment at (Q=0)

Never numerically form

[
v^{\mathrm{LR}}(0),F^*(0)F(0)
]

as infinity times zero.

For neutral interband transitions,

[
F_p(Q)
======

-iQ_i d_{p,i}
-\frac12Q_iQ_jQ_{p,ij}
+O(Q^3).
]

In 3D,

[
v(Q)\sim\frac{1}{Q^2},
]

so the dipole-dipole term has a finite but direction-dependent limit:

[
K_{pp'}^{\mathrm{LR}}
\sim
8\pi
\frac{
(Q\cdot d_p)^*
(Q\cdot d_{p'})
}{Q^2}.
]

In 2D,

[
v(Q)\sim\frac{1}{|Q|},
]

so the corresponding exchange behaves as (O(|Q|)), including the angular winding structure. This is the nonanalytic channel that should never be Fourier interpolated as part of the SR remainder. ([arXiv][1])

At exactly (Q=0), choose explicitly between:

* a directional (Q\to0) limit;
* an angular or mini-BZ cell average;
* the existing production head convention.

Those are different mathematical quantities. Do not use the isotropic mini-BZ value when calculating a directional exciton dispersion approaching (\Gamma).

For a finite-(\alpha) split, the mini-BZ head also splits:

[
\bar v_{\mathrm{head}}^{s}
==========================

\frac{1}{\Omega_{\rm mBZ}}
\int_{\rm mBZ}dQ,v^s(Q).
]

It is not generally correct to assign the entire existing `vhead` to the LR part. The invariant is

[
\bar v_{\mathrm{head}}^{\mathrm{SR}}
+
\bar v_{\mathrm{head}}^{\mathrm{LR}}
====================================

\bar v_{\mathrm{head}}.
]

## 10. Dipoles versus quadrupoles

If the finite-(K) LR form factors are evaluated directly, quadrupoles are needed primarily for:

* the exact (K\to0) limit;
* stabilization at extremely small (K);
* validation of angular behavior.

If a multipole model is used for the whole LR contribution, then in 3D:

* dipole–dipole contributes (O(1));
* dipole–quadrupole contributes (O(Q));
* quadrupole–quadrupole contributes (O(Q^2)).

A dipole-only subtraction can therefore leave an angularly nonanalytic (O(Q)) remainder that degrades higher-order Fourier interpolation. This is analogous to why dynamical quadrupoles are required beyond the Fröhlich dipole term in accurate electron–phonon interpolation. ([arXiv][2])

In 2D the corresponding orders are shifted by the (1/|Q|) kernel:

* dipole–dipole: (O(|Q|));
* dipole–quadrupole: (O(Q^2));
* quadrupole–quadrupole: (O(|Q|^3)).

Dipoles should capture the leading cusp, but quadrupoles may still matter for high-order smoothness.

## 11. Selecting (\alpha)

(\alpha) is not just a numerical cutoff. It determines how much physics is delegated to the exact LR path.

For the 3D Gaussian convention,

[
v^{\mathrm{SR}}(r)
\propto
\frac{\operatorname{erfc}(\alpha r)}{r}.
]

If (R_{\max}) is the maximum trustworthy real-space interpolation range, choose (\alpha) so that

[
\operatorname{erfc}(\alpha R_{\max})
\lesssim\epsilon_R.
]

The required reciprocal LR radius is roughly

[
K_{\mathrm{LR}}
\sim
2\alpha\sqrt{\log(1/\epsilon_G)}.
]

Thus:

* larger (\alpha): shorter SR range but more LR (G)-vectors;
* smaller (\alpha): fewer LR vectors but a longer-ranged SR object.

The best production value should be selected from a leave-one-out plateau, not solely from a heuristic proportional to grid spacing.

A useful error diagnostic is the residual (\alpha)-dependence of the off-grid result. The exact answer is independent of (\alpha); only interpolation and truncation errors introduce dependence.

## 12. Exact implementation sequence

### Coarse-grid preprocessing

For each coarse (q):

1. Build (C_q) and the fixed-rank frame
   [
   C_q=R_qS_q^2R_q^\dagger.
   ]

2. Recover
   [
   \Phi_q=S_qR_q^\dagger\zeta_q.
   ]

3. Evaluate stably
   [
   v_q^{\mathrm{SR}},\qquad v_q^{\mathrm{LR}}.
   ]

4. Construct
   [
   \mathcal V_q^{\mathrm{SR}}
   ==========================

   \Phi_qv_q^{\mathrm{SR}}\Phi_q^\dagger.
   ]

5. Optionally construct
   [
   \mathcal V_q^{\mathrm{LR}}
   ==========================

   \Phi_qv_q^{\mathrm{LR}}\Phi_q^\dagger
   ]
   for split-round-trip tests.

6. Verify
   [
   \mathcal V_q^{\mathrm{SR}}
   +
   \mathcal V_q^{\mathrm{LR}}
   ==========================

   S_qR_q^\dagger V_qR_qS_q.
   ]

7. Construct transport links from (C/X), not from either interaction piece.

### Arbitrary target (Q)

1. Generate wavefunctions at centroids only.
2. Form (X_Q,C_Q,R_Q,S_Q).
3. Compute target-to-stencil links (T_{Q\leftarrow q_i}).
4. Transport and interpolate:
   [
   \mathcal V_Q^{\mathrm{SR}}
   ==========================

   \sum_iw_i
   T_{Q\leftarrow q_i}
   \mathcal V_{q_i}^{\mathrm{SR}}
   T_{Q\leftarrow q_i}^\dagger.
   ]
5. Form
   [
   A_Q=X_QR_QS_Q^{-1}.
   ]
6. Apply
   [
   K_Q^{\mathrm{SR}}
   =================

   A_Q\mathcal V_Q^{\mathrm{SR}}A_Q^\dagger.
   ]
7. Add the separately evaluated low-rank
   [
   K_Q^{\mathrm{LR}}
   =================

   F_QD_Q^{\mathrm{LR}}F_Q^\dagger.
   ]

## 13. Tests the agent should regard as mandatory

The most important are:

[
v^{\mathrm{SR}}(K)+v^{\mathrm{LR}}(K)
=====================================

v(K)
]

to near machine precision, including the small-(K) series branches.

At every coarse point:

[
\mathcal V_q^{\mathrm{SR}}
+
\mathcal V_q^{\mathrm{LR}}
==========================

\mathcal V_q.
]

The sum must be independent of (\alpha).

Under any random auxiliary gauge (G_q),

[
\Phi_q\rightarrow G_q^\dagger\Phi_q,
\qquad
T_{Q\leftarrow q}\rightarrow
G_Q^\dagger T_{Q\leftarrow q}G_q,
]

the final physical kernel must remain unchanged.

For equivalent (Q) and (Q+G_0), verify equality of the **contracted total kernel**, not equality of fixed (G)-slots.

At a held-out coarse point, separately report errors in:

[
K^{\mathrm{SR}},
\qquad
K^{\mathrm{LR}},
\qquad
K^{\mathrm{total}}.
]

Finally, sweep (\alpha). If the total result changes substantially while the on-grid split identity is exact, the variation directly measures the off-grid interpolation error.

The most important architectural choice is therefore:

[
\boxed{
\text{parallel-transport and interpolate }
\mathcal V^{\mathrm{SR}}
\text{ in the orthonormal auxiliary frame;}
\quad
\text{add }
K^{\mathrm{LR}}
\text{ as a separate physical low-rank operator.}
}
]

That avoids both known failure mechanisms: the (C^{-1}) dual-coordinate amplification and the nonperiodic/singular small-(Q+G) channel.

[1]: https://arxiv.org/abs/2308.03012?utm_source=chatgpt.com "Maximally-Localized Exciton Wannier Functions for Solids"
[2]: https://arxiv.org/abs/2002.00628?utm_source=chatgpt.com "Electron-Phonon Beyond Fröhlich: Dynamical Quadrupoles in Polar and Covalent Solids"
