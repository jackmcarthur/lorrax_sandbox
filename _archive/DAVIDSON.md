Yes. Here is a plain **thick-restart block Davidson** for one fixed DFT (k)-point in an **NC pseudopotential plane-wave basis**, assuming the only expensive primitive you expose is a black-box block matvec
[
\mathrm{apply_H}_k:\Psi_k[G_Y,\alpha]\mapsto (H_k\Psi_k)[G_Y,\alpha],
]
where the plane-wave index (G) is sharded over a processor/GPU axis (Y), and the band/block index (\alpha) is **not** sharded.

I will write it in the exact distributed notation.

## 1. Assumptions and data layout

At fixed (k), the Hamiltonian is Hermitian in the orthonormal plane-wave basis,
[
H_k = T_k + V_{\rm loc} + V_{\rm NL},
\qquad
T_k(G)=\tfrac12 |k+G|^2.
]
Because this is **NC** plane waves, there is **no overlap matrix** (S); the eigenproblem is
[
H_k u_{n k} = \varepsilon_{n k} u_{n k}.
]

Let the GPU/process set be (Y={0,\dots,P_Y-1}).
GPU (y) owns the local subset (\mathcal G_y) of plane waves.

A distributed block
[
A[G_Y,\alpha]
]
means that on GPU (y) you store
[
A_y \in \mathbb C^{|\mathcal G_y|\times n_\alpha}.
]

So the wavefunction block for (n_\alpha) bands is stored as
[
A_y(G,\alpha),\qquad G\in\mathcal G_y.
]

Nothing is distributed over band index in this algorithm.

## 2. Small dense distributed primitives

All algorithmic communication is built from one primitive:
[
\mathrm{gram}(A,B)
==================

\sum_{y\in Y} A_y^\dagger B_y.
]
Locally each GPU forms
[
\Gamma^{(y)}*{AB}=A_y^\dagger B_y,
]
then an **all-reduce sum over (Y)** gives the global dense matrix
[
\Gamma*{AB}=\sum_y \Gamma^{(y)}_{AB}.
]

This dense result is then **replicated on all GPUs**.

This is used for:

[
A^\dagger B,\quad A^\dagger A,\quad |a_j|^2,\quad V^\dagger HV,\quad P^\dagger P,
]
and nothing ever requires an all-gather of the (G)-distributed wavefunctions.

## 3. Parameters to set at the top

I would define the algorithm by the following parameters.

### Spectral / convergence parameters

[
N_{\rm tgt}
]
number of lowest bands wanted at this (k).

[
\tau_{\rm res}
]
residual norm tolerance for convergence, typically test
[
|r_n| \le \tau_{\rm res}\max(1,|\varepsilon_n|).
]

[
N_{\rm maxit}
]
maximum Davidson outer iterations.

### Block / restart parameters

[
b
]
Davidson correction block size, i.e. how many unconverged roots you expand at one outer step.

[
m_{\max}
]
maximum Davidson subspace size before restart.

Usually set
[
m_{\max}=N_{\rm tgt}+p,b
]
with small restart depth (p), often (p=2,3,4).

### Linear dependence / orthogonalization parameters

[
\tau_{\rm dep}
]
threshold for dropping nearly dependent vectors from a candidate block.

[
n_{\rm ortho}
]
number of passes of subspace orthogonalization; in practice 1 or 2.
Two-pass DGKS-style reorthogonalization is the safe choice.

### Preconditioner parameters

For the standard reciprocal-space preconditioner, define

[
T_G = \tfrac12 |k+G|^2
]
for all local (G\in\mathcal G_y).

[
\delta_T > 0
]
small floor for mean kinetic energies.

If you use the Teter–Payne–Allan preconditioner, no further tuning is really required.

### Initialization / restart policy

You should decide at the top whether the initial guess (X_0) is taken from:

* previous nearby (k)-point,
* previous SCF/NSCF run,
* atomic/random low-(G) guesses.

For restart, choose either

* **thick restart to current Ritz vectors only**, or
* thick restart keeping current Ritz vectors plus a few extra unconverged directions.

For a first implementation I would do the first one:
[
V \leftarrow X,\qquad HV \leftarrow HX.
]

## 4. Arrays

### Distributed over (G_Y)

These live on every GPU only for its local (\mathcal G_y)-chunk.

[
V[G_Y,m]
]
current Davidson basis vectors.

[
W[G_Y,m]
]
their Hamiltonian images,
[
W = HV.
]

[
X[G_Y,N_{\rm tgt}]
]
current Ritz vectors.

[
HX[G_Y,N_{\rm tgt}]
]
their Hamiltonian images.

[
R[G_Y,b]
]
residual block for selected active roots.

[
P[G_Y,b]
]
preconditioned correction block.

[
HP[G_Y,b]
]
Hamiltonian image of the new correction block.

Also local reciprocal-space kinetic diagonal:
[
T[G_Y].
]

### Dense small replicated arrays

These are replicated on every GPU after all-reduce.

[
\Theta[m,m] = V^\dagger W
]
projected Hamiltonian.

[
C[m,m] \text{ or } C[m,N_{\rm tgt}]
]
eigenvectors of the projected problem.

[
\Lambda[N_{\rm tgt}]
]
current Ritz values.

[
B[m,b]=V^\dagger P
]
subspace orthogonalization coefficients.

[
S_P[b,b]=P^\dagger P
]
candidate block overlap.

Also masks / index lists for locked and active bands.

The only replicated arrays that can become nontrivial in size are the projected subspace matrices of dimension (m). This is the principal globally replicated memory cost in Davidson.

## 5. Preconditioner

Since (H_k) is only exposed as a black-box matvec, the standard robust choice is a **reciprocal-space diagonal kinetic preconditioner**.

The default choice I would recommend is the **Teter–Payne–Allan** form.

For a selected residual vector (r_n(G)), first compute a mean kinetic scale from the current Ritz vector (x_n):
[
\bar T_n
========

\frac{\sum_G T_G |x_n(G)|^2}
{\sum_G |x_n(G)|^2}.
]
Since (x_n) is normalized, this is just
[
\bar T_n = \sum_G T_G |x_n(G)|^2.
]
In distributed form:
[
\bar T_n
========

\sum_{y\in Y}\sum_{G\in\mathcal G_y}
T_G |x_{n,y}(G)|^2,
]
which is one all-reduce over a length-(b) vector if done for a block.

Define
[
x_{n,G}
=======

\frac{T_G}{\max(\bar T_n,\delta_T)}.
]

Then apply
[
p_n(G)=f(x_{n,G}),r_n(G),
]
with
[
f(x)=
\frac{27+18x+12x^2+8x^3}
{27+18x+12x^2+8x^3+16x^4}.
]

This has the right behavior:

* (f(0)=1), so low-(G) components are not over-amplified,
* (f(x)\sim \frac{1}{2x}) for large (x), so high-(G) components are damped roughly like an inverse kinetic operator.

The overall scale of (p_n) does not matter.

If you want an even simpler first implementation, you can use
[
p_n(G)=\frac{r_n(G)}{T_G+\alpha_n},
]
with
[
\alpha_n=\max(\bar T_n,\delta_T),
]
but the Teter form is the standard safer default in plane-wave codes.

## 6. The algorithm

Let (n_0\ge N_{\rm tgt}) be the size of the initial guess block; usually (n_0=N_{\rm tgt}).

### Step 0: initial guess and orthonormalization

Start from
[
X_0[G_Y,n_0].
]

Compute the overlap
[
S_0 = X_0^\dagger X_0
=====================

\mathrm{gram}(X_0,X_0).
]
This is a dense (n_0\times n_0) matrix replicated on all GPUs.

Diagonalize or Cholesky-factorize (S_0).
If using eigendecomposition,
[
S_0 = U s U^\dagger.
]
Drop directions with
[
s_i < \tau_{\rm dep}.
]
Then orthonormalize:
[
X_0 \leftarrow X_0 U s^{-1/2}.
]

Communication here: one all-reduce for (S_0).

### Step 1: initial (H)-application

Apply the black-box Hamiltonian:
[
HX_0 = \mathrm{apply_H}_k(X_0).
]

Set the Davidson basis
[
V \leftarrow X_0,\qquad
W \leftarrow HX_0,\qquad
m\leftarrow n_0.
]

### Step 2: Davidson outer iteration

For iteration (it=1,2,\dots,N_{\rm maxit}):

#### 2a. Projected Hamiltonian

Form
[
\Theta = V^\dagger W
====================

\mathrm{gram}(V,W).
]
Then symmetrize numerically:
[
\Theta \leftarrow \tfrac12(\Theta+\Theta^\dagger).
]

Communication: one all-reduce of an (m\times m) dense matrix.

#### 2b. Dense Rayleigh–Ritz solve

Solve the dense Hermitian problem
[
\Theta C = C E,
]
with eigenvalues in ascending order:
[
E=\operatorname{diag}(\varepsilon_1,\dots,\varepsilon_m),
\qquad
\varepsilon_1\le\varepsilon_2\le \cdots.
]

Take the first (N_{\rm tgt}) Ritz pairs:
[
C_N = C[:,1:N_{\rm tgt}],
\qquad
\Lambda = (\varepsilon_1,\dots,\varepsilon_{N_{\rm tgt}}).
]

These dense arrays are replicated on all GPUs.

#### 2c. Form Ritz vectors and their (H)-images

Compute
[
X = V C_N,
\qquad
HX = W C_N.
]

These are local dense GEMMs on each GPU:
[
X_y = V_y C_N,
\qquad
HX_y = W_y C_N.
]

No communication is needed here once (C_N) is replicated.

#### 2d. Residuals

For each target root,
[
r_n = Hx_n - \varepsilon_n x_n.
]
Block form:
[
R = HX - X,\operatorname{diag}(\Lambda).
]

Compute residual norms
[
\rho_n = |r_n|
==============

\sqrt{r_n^\dagger r_n}.
]
Distributed evaluation is
[
\rho_n^2
========

\sum_{y\in Y}\sum_{G\in\mathcal G_y}|r_{n,y}(G)|^2.
]

Communication: one all-reduce of a length-(N_{\rm tgt}) vector.

#### 2e. Locking

Define converged bands by
[
\rho_n \le \tau_{\rm res}\max(1,|\varepsilon_n|).
]

For the “lowest (N_{\rm tgt})” problem I would hard-lock only the lowest contiguous converged prefix.
That is, if bands (1,\dots,\ell) satisfy the criterion, lock those (\ell) and continue on the rest.

If all (N_{\rm tgt}) are locked, stop.

#### 2f. Choose active block

Let (A) be the list of the next unconverged target roots to improve, with
[
|A| = b_{\rm act} \le b.
]
Usually choose the lowest-index unconverged roots first.
Then take
[
R_A = R[:,A],\qquad
X_A = X[:,A],\qquad
\Lambda_A = \Lambda[A].
]

#### 2g. Precondition residuals

For each active root (n\in A), compute
[
\bar T_n
========

\sum_{y\in Y}\sum_{G\in\mathcal G_y}
T_G |x_{n,y}(G)|^2.
]

Then for each local (G),
[
\xi_{n,G}
=========

\frac{T_G}{\max(\bar T_n,\delta_T)},
]
and
[
p_n(G)=f(\xi_{n,G}),r_n(G),
]
with
[
f(\xi)=
\frac{27+18\xi+12\xi^2+8\xi^3}
{27+18\xi+12\xi^2+8\xi^3+16\xi^4}.
]

Collect these into
[
P = [p_n]_{n\in A}.
]

Communication: one all-reduce over a length-(b_{\rm act}) vector to get the (\bar T_n).

#### 2h. Orthogonalize the candidate block against the whole subspace

Form
[
B = V^\dagger P
===============

\mathrm{gram}(V,P).
]
Then update
[
P \leftarrow P - V B.
]

If using two-pass reorthogonalization, repeat this once:
[
B' = V^\dagger P,\qquad
P \leftarrow P - V B'.
]

Communication: one or two all-reduces of an (m\times b_{\rm act}) dense matrix.

#### 2i. Orthonormalize the candidate block internally

Form
[
S_P = P^\dagger P
=================

\mathrm{gram}(P,P).
]

Diagonalize
[
S_P = U s U^\dagger.
]
Drop directions with
[
s_i < \tau_{\rm dep}.
]

Then
[
P \leftarrow P,U,s^{-1/2}.
]

Let the remaining number of columns be (b'\le b_{\rm act}).

If (b'=0), then the new search directions were numerically dependent on the current subspace. In that case the safest action is to restart:
[
V\leftarrow X,\qquad W\leftarrow HX,\qquad m\leftarrow N_{\rm tgt},
]
and continue.

Communication: one all-reduce of a (b_{\rm act}\times b_{\rm act}) matrix.

#### 2j. Apply (H) to the new block

Compute
[
HP = \mathrm{apply_H}_k(P).
]

#### 2k. Augment the subspace

Set
[
V \leftarrow [V,P],\qquad
W \leftarrow [W,HP],\qquad
m \leftarrow m+b'.
]

No communication here besides what is internal to (\mathrm{apply_H}_k).

#### 2l. Restart if needed

If
[
m > m_{\max},
]
do a thick restart. The simplest choice is
[
V \leftarrow X,\qquad
W \leftarrow HX,\qquad
m \leftarrow N_{\rm tgt}.
]

That discards the accumulated correction directions and keeps only the current best Ritz vectors.

A slightly more aggressive variant is to keep
[
V \leftarrow [X, P_{\rm keep}],
]
for a small number of unconverged extra vectors, but the simplest robust implementation is the first one.

Then continue to the next iteration.

## 7. Communication summary

Outside of whatever communication is buried inside your black-box (H)-application, Davidson itself only needs **all-reduce sums over (Y)** for dense Gram-type matrices.

Specifically:

1. (X^\dagger X) for initial orthonormalization:
   one all-reduce of size (n_0\times n_0).

2. (V^\dagger HV) each outer step:
   one all-reduce of size (m\times m).

3. residual norms:
   one all-reduce of size (N_{\rm tgt}).

4. mean kinetic energies for the active block:
   one all-reduce of size (b_{\rm act}).

5. (V^\dagger P) for subspace orthogonalization:
   one or two all-reduces of size (m\times b_{\rm act}).

6. (P^\dagger P) for block orthonormalization:
   one all-reduce of size (b_{\rm act}\times b_{\rm act}).

There is **no algorithmic need to all-gather the distributed wavefunctions over (G)**.

## 8. What is replicated on all GPUs

Replicated everywhere:

* projected Hamiltonian (\Theta[m,m]),
* projected eigvectors (C[m,m]) or at least (C[:,1:N_{\rm tgt}]),
* Ritz values (\Lambda[N_{\rm tgt}]),
* small overlap / orthogonalization matrices (B[m,b]), (S_P[b,b]),
* active/locked masks and index lists.

Not replicated:

* the large PW-space arrays (V,W,X,HX,R,P,HP), which remain distributed over (G_Y).

So the algorithm is **G-sharded, dense-subspace-replicated**.

## 9. Notes specific to your use case

For **many conduction bands for GW**, the hard part is that (N_{\rm tgt}) is large, so the replicated dense subspace objects and the (V^\dagger HV) all-reduce become expensive. That is the exact reason PPCG becomes attractive later. But for a first robust eigensolver, the Davidson above is the right thing to write.

If you are using only one distribution axis (Y), then every GPU stores all (N_{\rm tgt}) current bands for its local (G)-chunk. That is usually the right design if your black-box (H_k\Psi) is already optimized for block application and GPU BLAS/FFT usage.

For a first implementation I would set:
[
b = 16,, 32,\text{ or }64,
\qquad
m_{\max}=N_{\rm tgt}+2b \text{ or } N_{\rm tgt}+3b,
]
use the Teter preconditioner, two-pass orthogonalization, and hard locking of the lowest contiguous converged roots.

If you want, I can rewrite this immediately in a more code-adjacent form with array names like

[
V[G_Y,\mu],\quad HV[G_Y,\mu],\quad X[G_Y,n],\quad R[G_Y,a],
]

and an explicit pseudocode loop that looks like JAX/CuPy/Python rather than math prose.
