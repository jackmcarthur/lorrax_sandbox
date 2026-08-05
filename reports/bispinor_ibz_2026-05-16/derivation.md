# Bispinor four-density vertex transformation under {S|τ} — derivation

**Scope.** LORRAX's bispinor pair-density carries a 4-vector "Lorentz" index
from the Pauli decomposition

$$
n_{i\sigma,\,j\sigma'}(\mathbf r)
=
\sum_{\mu_L=0}^{3} \zeta_{\mu_L,a}(\mathbf r)\,
\langle\sigma|\,\tilde\gamma^{\mu_L}\,|\sigma'\rangle\,
C^{a}_{ij},
\qquad
\tilde\gamma^{0}=I_{2},\;
\tilde\gamma^{1,2,3}=(\sigma_x,\sigma_y,\sigma_z),
$$

where the pair-density of two 2-component Pauli spinors $\psi_i,\psi_j$ is
$\rho^{\mu_L}_{ij}(\mathbf r) = \psi_i^\dagger(\mathbf r)\,\tilde\gamma^{\mu_L}\,\psi_j(\mathbf r)$.
We derive how $\rho^{\mu_L}$, and the corresponding bispinor Coulomb tensor
$V_q^{\mu_L,\nu_L}$, transform under a space-group operation
$g=\{U|\tau\}\in\mathcal G$ acting on direct fractional coordinates as
$x\mapsto Ux+\tau$. Throughout I use the math conventions handed in the prompt;
where I cite the LORRAX conventions document I mean
`reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`
(referred to as "**CONV**" below).

Reciprocal rotation: $R=U^{-T}$. Plane-wave convention:
$\psi_{n\mathbf q}(\mathbf x)=\sum_G c_{n\mathbf q}(G)\,e^{i 2\pi(\mathbf q+G)\cdot\mathbf x}$.
Unfolded q-point: $\mathbf q_1 = R\mathbf q + G_R$.

For an op $g$ with Cartesian rotation $\mathcal O = A\,U\,A^{-1}$
(where $A$ has the direct lattice vectors as columns):
$\det\mathcal O=\pm1$.
The *proper* part is

$$
R_{\text{proper}} \;=\;
\begin{cases}\mathcal O,&\det\mathcal O=+1,\\ -\mathcal O,&\det\mathcal O=-1,\end{cases}
$$

with $\det R_{\text{proper}}=+1$ by construction.
The SU(2) representative is built from this proper part:
$S_2(U) = e^{-i\theta\,\hat n\cdot\vec\sigma/2}$ with $(\theta,\hat n)$ extracted
from $R_{\text{proper}}$. **Inversion itself does not rotate spin**; the
improper sign is absorbed in the proper flip. This is exactly what
`SymMaps.get_spinor_rotations` does
(`common/symmetry_maps.py:548-632`, lines 570-575: `if det(R) < 0: R = -R`
before constructing the quaternion).

Nonsymmorphic $\tau$ affects only the plane-wave phase, never $S_2$.

---

## A1. τ-phase cancellation on the pair density at the same real-space point

Take two Bloch spinors $\psi_{i\mathbf q_i}$ and $\psi_{j\mathbf q_j}$ evaluated at the
*same* fractional $\mathbf x$. The unfold rule (CONV §"BGW convention", prompt
"Coefficient unfolding") gives, on the rotated coefficient,

$$
c_{n\mathbf q_1,A}(G_1)
= e^{-i 2\pi(\mathbf q_1+G_1)\cdot\tau}\,
\sum_{B} [S_2(U)]_{AB}\,c_{n\mathbf q,B}(G),
\qquad G = U^{T}(\mathbf q_1+G_1)-\mathbf q.
$$

Equivalently in position space (CONV (2)):
$\psi_{S\mathbf q}^A(\mathbf x) = [S_2(U)]_{AB}\,\psi_{\mathbf q}^B\!\bigl(U^{-1}(\mathbf x-\tau)\bigr)$.

In the pair-density $\rho^{\mu_L}_{ij}(\mathbf x) = \psi_i^\dagger(\mathbf x)\,\tilde\gamma^{\mu_L}\,\psi_j(\mathbf x)$
both factors are at the *same* $\mathbf x$. Apply the unfold to *both*:

$$
\psi_{S\mathbf q_i,i}^\dagger(\mathbf x)\,\tilde\gamma^{\mu_L}\,\psi_{S\mathbf q_j,j}(\mathbf x)
= \bigl[S_2(U)\psi_{\mathbf q_i,i}(U^{-1}(\mathbf x-\tau))\bigr]^\dagger
\,\tilde\gamma^{\mu_L}\,
\bigl[S_2(U)\psi_{\mathbf q_j,j}(U^{-1}(\mathbf x-\tau))\bigr].
$$

In *G-space* the τ-phase from the unfold of $\psi^\dagger$ is $e^{+i 2\pi(\mathbf q_{1,i}+G_{1,i})\cdot\tau}$
and from $\psi$ is $e^{-i 2\pi(\mathbf q_{1,j}+G_{1,j})\cdot\tau}$.
At equal $\mathbf x$ these phases are *the same factor with opposite signs*
applied as $\bar z \cdot z$ — they multiply to a residual phase that depends
only on the **difference** $(\mathbf q_{1,i}+G_{1,i}) - (\mathbf q_{1,j}+G_{1,j})$
of the wavevectors of the two factors, **not on the absolute** $\tau$ value
that sat inside each individual $\psi$.

In real space the same cancellation is even cleaner: $\psi^\dagger(\mathbf x)\psi(\mathbf x)$
sees $\bigl(\psi^\dagger\bigl(U^{-1}(\mathbf x-\tau)\bigr)\bigr)\bigl(\psi\bigl(U^{-1}(\mathbf x-\tau)\bigr)\bigr)$
with **no extra τ-phase** — the $e^{-i 2\pi(\cdot)\tau}$ on the spinor and its
conjugate cancel pointwise (each is a global complex unit and they appear as
$\overline{e^{-i\theta}}\cdot e^{-i\theta}=1$ when the *same* $\tau$ acts on
both legs at the same $\mathbf x$).

**Conclusion.** The pair density's transformation under $\{U|\tau\}$ inherits
**no τ-phase** at the same $\mathbf x$. This is exactly the property that lets
the scalar V_q unfold formula (CONV §"The ISDF V_q unfold formula")

$$
V_{\text{full}}[q_1,\mu',\nu']
= e^{2\pi i\,q_{\text{irr}}\cdot(L_{s,\mu'}-L_{s,\nu'})}\,
V_{\text{ibz}}[q_{\text{irr}},\alpha_s(\mu'),\alpha_s(\nu')]
$$

carry **no $\tau$-dependence** even on non-symmorphic ops (the only $\tau$
that appears is *inside* $L_{s,\mu}$ via $y_\mu = U^{-1}(x_\mu-\tau)=x_{\alpha(\mu)}+L_\mu$,
i.e. as an integer lattice wrap, not a phase). The bispinor V_q inherits
exactly the same property — the additional spinor sandwich $S_2^\dagger\,\tilde\gamma\,S_2$
in the next step is **purely real-orthogonal on the spin index** and never
brings $\tau$ back into the formula.

---

## A2. The spinor sandwich identity for $\mu_L\in\{1,2,3\}$

Let $S_2 = S_2(U)$ be the SU(2) representative built from the proper part
$R_{\text{proper}}$ of $\mathcal O = A U A^{-1}$ (so $S_2(U)\equiv S_2(R_{\text{proper}})$).
The standard SU(2)↔SO(3) homomorphism asserts that for any **proper rotation**
$\mathcal R\in\mathrm{SO}(3)$ with SU(2) representative $S=e^{-i\theta\hat n\cdot\vec\sigma/2}$,

$$
S^\dagger\,\sigma^{i}\,S \;=\; \sum_{j=1}^{3} \mathcal R^{\,j i}\,\sigma^{j}
\qquad\text{(equivalently)}\qquad
S\,\sigma^{i}\,S^\dagger = \sum_j \mathcal R^{\,i j}\sigma^{j}.
$$

(Sakurai *Modern QM* §3.3, eq. (3.3.16); Tinkham *Group Theory and QM* §3.5.
The transpose appears because we write $S^\dagger\sigma S$ vs $S\sigma S^\dagger$.)

For our case the relevant proper rotation is $R_{\text{proper}}$:

* if $\det\mathcal O=+1$: $R_{\text{proper}}=\mathcal O$, identity case;
* if $\det\mathcal O=-1$: $R_{\text{proper}}=-\mathcal O$. The improper sign
  ($\mathcal O = -R_{\text{proper}}$) corresponds to a **central inversion**
  composed with $R_{\text{proper}}$. Under inversion, axial vectors (the $\vec\sigma$
  components) are **invariant**: $i$ sends $\vec\sigma\to+\vec\sigma$, because
  $\vec\sigma$ is built from angular-momentum-like generators. This is the
  "inversion does not rotate spin" statement in the prompt's conventions and
  in CONV §"For improper rotations".

So the identity that holds for **any** $\mathcal O$ — proper or improper — is

$$
\boxed{\;
S_2(U)^\dagger\,\sigma^{\mu_L}\,S_2(U)
\;=\; \sum_{\nu=1}^{3} R_{\text{proper}}^{\,\nu\,\mu_L}\,\sigma^\nu,
\qquad \mu_L\in\{1,2,3\},
\;}
$$

with $R_{\text{proper}}=\mathrm{sign}(\det\mathcal O)\cdot\mathcal O$. This is the
exact identity LORRAX needs.

*Proof sketch.* For $\det\mathcal O=+1$, this is the textbook
SU(2) double cover. For $\det\mathcal O=-1$, write $\mathcal O = I_{\text{inv}}\,R_{\text{proper}}$
with $I_{\text{inv}}=-I_3$. On the spin operators,
$T_{\text{inv}}\sigma^i T_{\text{inv}}^\dagger = +\sigma^i$ (inversion is unitary
and *spin-trivial*). Therefore $S_2(U)^\dagger\sigma^i S_2(U) = S_2(R_{\text{proper}})^\dagger\,\sigma^i\,S_2(R_{\text{proper}})$
which by the proper case equals $\sum_j R_{\text{proper}}^{\,j i}\sigma^j$. $\square$

**Index convention.** The identity above uses the *first* index of
$R_{\text{proper}}$ as the *output* (new) spin-vector index and the *second*
as the *input* (old) — this is the row-major ($A^{ji}$ = "row $j$, column $i$")
reading of `R_proper[ν, μ_L]` in the fixture.

---

## A3. The $\mu_L=0$ (charge / CC) channel is invariant

$\tilde\gamma^0 = I_2$, so $S_2^\dagger\,I_2\,S_2 = I_2 = \tilde\gamma^0$
trivially. There is **no mixing** between $\mu_L=0$ and $\mu_L=1,2,3$
(an SU(2) conjugation cannot turn the identity into a Pauli matrix, since
$\mathrm{tr}(I_2\sigma^i)=0$ and traces are invariant under conjugation).
$R_{\text{proper}}$ acts as the trivial $1\times 1$ block on the $\mu_L=0$ channel.

This is consistent with `UNIQUE_TILES` in
`gw/v_q_bispinor.py:57-67`:

```python
UNIQUE_TILES: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (1, 2), (1, 3),
            (2, 2), (2, 3),
                    (3, 3),
)
```

Off-diagonal tiles with $\mu_L=0$ (i.e. (0,1), (0,2), (0,3) and their conjugates)
are absent: charge × magnetic-3-vector matrix elements vanish in the
unfold-symmetry block because, in the closed Coulomb tensor under the IBZ-fold,
charge and 3-vector channels live in disjoint irreps of the proper crystal
point group augmented with TRS. (Equivalently: $V_q^{0,j}$ for $j\neq 0$ is
killed by Hermiticity in the spin-trace; the bispinor tile catalog above is
the explicit consequence.)

**Therefore:**

* $V^{0,0}_{\text{full}}[q_1] = \text{unfold\_v\_q}(V^{0,0}_{\text{ibz}})[q_1]$
  (scalar formula, CONV).
* $V^{0,j}_{\text{full}}[q_1] = 0$ for $j\in\{1,2,3\}$
  (and likewise $V^{i,0}=0$ by Hermiticity).

---

## A4. Time-reversal sign on the 3-vector channels

Take $T = i\sigma_y K$ ($K$ = complex conjugation, antilinear). The action of
$T$ on a Pauli operator $O$ on the 2D spin space is

$$
T\,O\,T^{-1} = (i\sigma_y)\,\overline{O}\,(i\sigma_y)^{-1} = \sigma_y\,\overline{O}\,\sigma_y.
$$

Direct algebra:

* $\sigma_y\,\overline{\sigma_x}\,\sigma_y = \sigma_y\,\sigma_x\,\sigma_y = -\sigma_x$,
* $\sigma_y\,\overline{\sigma_y}\,\sigma_y = \sigma_y\,(-\sigma_y)\,\sigma_y = -\sigma_y$,
* $\sigma_y\,\overline{\sigma_z}\,\sigma_y = \sigma_y\,\sigma_z\,\sigma_y = -\sigma_z$,
* $\sigma_y\,\overline{I_2}\,\sigma_y = I_2$.

Hence $T\,\sigma^i\,T^{-1} = -\sigma^i$ for $i\in\{1,2,3\}$, and $T\,I_2\,T^{-1}=+I_2$.

So under TRS the 3-vector channel of the pair density acquires a **global
sign flip on every spin-vector index**, and the charge channel is invariant.
At the level of the $R$-matrix this is

$$
R^{\mu_L,\nu_L}_{\text{TRS, spatial-}s}
\;=\;
\begin{cases}+1, & \mu_L=\nu_L=0,\\
-R^{\,\mu_L,\nu_L}_{\text{proper, spatial-}s}, & \mu_L,\nu_L\in\{1,2,3\}.\end{cases}
$$

i.e. $R_{\text{proper}}^{\text{TRS}} = -R_{\text{proper}}^{\text{spatial}}$ on the
3×3 spin-vector block (with the charge block untouched).

**Is this absorbed into the existing `unfold_v_q` TRS conj-wrap?**

The existing scalar `unfold_v_q` applies, on the TRS half of the symmetry table
(CONV §"TRS-augmented case"):

```
V_full = V_spatial.conj()
```

i.e. a single global complex-conjugation of the unfolded tile. This conj-wrap
**already** handles the $\mu_L=0$ scalar V_q exactly, because the $V_q^{00}$
tile is Hermitian and the TRS image is its complex conjugate.

For the 3-vector tiles the picture is different: TRS contributes **two
independent operations** at the tensor level:

1. The same complex-conjugation that the scalar formula already applies.
   This comes from $q\to -q$ on the spinor *plane-wave indices*; it does
   *not* depend on the $(\mu_L,\nu_L)$ choice.
2. A **separate, multiplicative sign** $(-1)^{[\mu_L\in\{1,2,3\}]}\cdot(-1)^{[\nu_L\in\{1,2,3\}]}$
   from $T\sigma^i T^{-1}=-\sigma^i$. Each of the two $\psi$-legs of the
   pair-density contributes one factor of $-1$ when its $\sigma^{\mu_L}$ (or
   $\sigma^{\nu_L}$) factor is in the 3-vector block.

Step (1) is what the existing scalar `unfold_v_q` does on the TRS rows.
Step (2) is **new** for the bispinor extension — it has to be applied
explicitly on the $(\mu_L,\nu_L)\in\{1,2,3\}^2$ tiles **on top of** the
existing conj-wrap.

The product of two $-1$ factors on the same tile is $+1$, so:

* On the 6 "diagonal-in-3-vector" tiles $(1,1),(2,2),(3,3)$: the two minus
  signs cancel ⇒ **no additional TRS sign**. Conj-wrap alone is correct.
* On the 3 off-diagonal 3-vector tiles $(1,2),(1,3),(2,3)$: the two minus
  signs cancel again ⇒ **no additional TRS sign**. Conj-wrap alone is correct.
* On the cross tiles $(0, j)$ and $(i, 0)$ for $i,j\in\{1,2,3\}$: one minus
  one, one plus one, net $-1$ ⇒ **a separate sign is needed**. *But these
  tiles are absent from UNIQUE_TILES* (per A3), so this case never arises
  in practice.
* On the pure-charge tile $(0,0)$: no signs ⇒ conj-wrap alone is correct.

**Net answer to the prompt's question:** for the 7 tiles LORRAX actually stores
(the UNIQUE_TILES set), the TRS sigma-sign factorizes as
$(-1)^{[\mu_L\ne 0]}\cdot(-1)^{[\nu_L\ne 0]}$ which is $+1$ on every stored tile.
**The TRS $-1$ is absorbed for free into the existing conj-wrap; no separate
TRS sign needs to be applied to $V_q^{\mu_L,\nu_L}$ tiles for $(\mu_L,\nu_L)$
in UNIQUE_TILES.** The TRS $-1$ would surface only on the $(0,j)/(i,0)$ cross
tiles, which are gauge-zero by Hermiticity / point-group selection
(`v_q_bispinor.py:UNIQUE_TILES`).

However — and this is the load-bearing caveat — the **spatial** mixing
matrix $R_{\text{proper}}^{i,\alpha}R_{\text{proper}}^{j,\beta}$ in A5 still
needs to be applied with $R_{\text{proper}}^{\text{spatial}}$ (not the
TRS-flipped $-R_{\text{proper}}^{\text{spatial}}$), because the *spin-axis
rotation* on each spinor leg is the same on TRS and non-TRS rows of the
sym table (TRS doesn't reorient $\hat n$ of the SU(2) rotation; it flips the
overall sign of every $\sigma^i$, and that flip — as just shown — cancels
pairwise in the $(\mu_L,\nu_L)\in\{1,2,3\}^2$ block).

In symbols:

* spatial row $s$:   apply $R_{\text{proper}}^{\,s}\otimes R_{\text{proper}}^{\,s}$
  to mix the 3-vector indices, then no conj.
* TRS row $s$:       apply $R_{\text{proper}}^{\,s}\otimes R_{\text{proper}}^{\,s}$
  to mix the 3-vector indices (the **same** $R_{\text{proper}}$ as on the
  spatial row), then complex-conjugate the resulting tile (per the existing
  conj-wrap). The $-1\cdot-1=+1$ cancellation eats the TRS sigma-sign.

The fixture I emit therefore stores $R_{\text{proper}}^{\text{spatial}}$ as
the "spatial" half (rows $0..\text{ntran}-1$) and **also** stores the formal
$-R_{\text{proper}}^{\text{spatial}}$ as the "TRS" half (rows $\text{ntran}..2\,\text{ntran}-1$)
for transparency, but the consumer should *not* use the TRS-half entries to
mix 3-vector indices — on the stored UNIQUE_TILES the TRS sign is absorbed
into the conj-wrap and the spin-axis rotation reuses the spatial $R_{\text{proper}}$.
See `sym_index_layout` string in the npz for the per-row definitions.

---

## A5. Final rule for bispinor $V_q^{\mu_L,\nu_L}$ at full-BZ q

For each full-BZ q with parent IBZ q_irr and sym op index
$s=\text{sym\_idx\_q}[q]$, using
$U=\text{mtrx}^{-1}[s_{\text{spatial}}]$ as the forward direct-coord rotation,
$\mathcal O = A U A^{-1}$, $R_{\text{proper}}=\mathrm{sign}(\det\mathcal O)\,\mathcal O$:

**Charge × Charge:**

$$
V^{0,0}_{\text{full}}[q] = \text{unfold\_v\_q}\bigl(V^{0,0}_{\text{ibz}}\bigr)[q].
$$

**Charge × 3-vector:**

$$
V^{0,j}_{\text{full}}[q] = 0,\quad V^{i,0}_{\text{full}}[q] = 0,\qquad i,j\in\{1,2,3\}.
$$

(Gauge zero per `UNIQUE_TILES` in `v_q_bispinor.py`; see A3.)

**3-vector × 3-vector** ($i,j\in\{1,2,3\}$):

$$
V^{i,j}_{\text{full}}[q] = \sum_{\alpha=1}^{3}\sum_{\beta=1}^{3}
R_{\text{proper}}^{\,i,\alpha}(s_{\text{sp}})\,
R_{\text{proper}}^{\,j,\beta}(s_{\text{sp}})\;\cdot\;
\text{unfold\_v\_q}\bigl(V^{\alpha,\beta}_{\text{ibz}}\bigr)[q],
$$

where $s_{\text{sp}} = s$ if $s<\text{ntran}$, else $s-\text{ntran}$; the spatial
permutation $\alpha_s(\mu)$ and the umklapp phase $L_\mu$ inside the scalar
`unfold_v_q` are unchanged from the scalar case. If $s\ge\text{ntran}$ (TRS row),
follow with the **same** complex-conjugation that the scalar `unfold_v_q`
already applies on TRS rows; **no additional TRS sigma-sign** on $(\mu_L,\nu_L)\in\{1,2,3\}^2$
tiles (see A4).

Equivalently, in einsum form on the 3-vector block:

$$
V^{ij}_{\text{full}}[q]
= R^{i\alpha}\,R^{j\beta}\,U^{\alpha\beta}_{\alpha_s(\cdot),\alpha_s(\cdot)},
$$

where $U^{\alpha\beta}$ is the L-phase-applied IBZ tile per the scalar formula.

Implementation note: because the seven UNIQUE_TILES are stored as a
$3\times 3$ upper-triangle (plus charge), the consumer should materialise the
Hermitian-symmetrized $3\times 3$ matrix `Vibz3[α,β]` for each $(\mu,\nu)$
pair before contracting with $R\otimes R$. The contraction has identical
shape to the per-tile centroid arrays; cost is a per-q $3\times 3$ orthogonal
rotation on the 3-vector index, vanishing alongside the IBZ-unfold cost.

---

## References

- CONV: `reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`
  - "BGW convention" §
  - "ISDF V_q unfold formula" §  (scalar formula reused on each tile)
  - "TRS-augmented case" §        (the existing conj-wrap that we re-use)
- LORRAX bispinor V_q catalog: `sources/lorrax_D/src/gw/v_q_bispinor.py:57-67`
  (`UNIQUE_TILES`).
- LORRAX SU(2) extractor: `sources/lorrax_D/src/common/symmetry_maps.py:548-632`
  (`get_spinor_rotations`, including the `if det(R)<0: R = -R` proper-flip).
- Sakurai, *Modern QM* §3.3 for the SU(2)↔SO(3) sandwich identity.
- Tinkham, *Group Theory and QM* §3.5 for the inversion–spin commutation.
