# Steelman: the Q=0 BSE exchange kernel is DENSE in (k,k′), not k-block-diagonal

Role: argue the strongest case that ⟨cvk|K^x|c′v′k′⟩ couples **all** (k,k′) pairs.
Adjudication target: claim (b) — "the pair-coupling is δ_{kk′}". Claim (a) — "the
contracted object carries momentum shift 0 / only the q=0 Coulomb tile enters" — is
**true and not in dispute**; the whole point below is that (a) does not imply (b),
and the code owner's argument silently slides from one to the other.

**Verdict up front (stated even though I was asked to steelman one side): the DENSE
position is correct. The four live matvecs are wrong; they retain k as a spectator
axis where the physics requires a sum over k′.** Every independent line of evidence —
the real-space integral, the owner's own Hartree analogy, the Henneke reference the
code cites, BerkeleyGW's kernel builder, and the thermodynamic limit — converges on
the same conclusion, and none of them can be made to yield δ_{kk′}. This is not a
close call.

The disputed code:
`sources/lorrax_D/src/bse/bse_serial.py:62-64`
```python
S_V = jnp.einsum("kcvN,bcvk->bNk", M, X) / sqrt_nk   # k SURVIVES as output axis
U_V = jnp.einsum("MN,bNk->bMk", V_q0, S_V)           # k still spectator
V_term = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M), U_V) / sqrt_nk
```
`k` appears on both operands and the output of every einsum, so the exchange only
couples pairs at the **same** k. The correct action drops the k axis on the encode
(`'kcvN,bcvk->bN'`, summing k) and re-broadcasts the k-independent result on decode.

---

## 1. Real-space derivation: where δ_{kk′} would have to come from, and why it doesn't

Bare exchange kernel in real space is the *unscreened* Coulomb two-point function
`v(r,r′)=v(r−r′)` sandwiched between electron-hole pair densities (Henneke Eq. 2-16,
`context/Henneke-2020-*.md:173`; Rohlfing–Louie PRB 62 4927 Eq. 41):

```
⟨cvk | K^x | c′v′k′⟩ = ∫∫ dr dr′  [ψ*_{ck}(r) ψ_{vk}(r)]  v(r−r′)  [ψ*_{v′k′}(r′) ψ_{c′k′}(r′)]
                     =  ∫∫ dr dr′  Φ*_{cvk}(r)  v(r−r′)  Φ_{c′v′k′}(r′)                 (X.1)
```
with the pair density `Φ_{cvk}(r) ≡ ψ*_{ck}(r) ψ_{vk}(r)`.

**The Bloch phase cancels *within* a pair.** Write `ψ_{nk}(r)=N_k^{-1/2} e^{ik·r} u_{nk}(r)`
(cell-periodic `u`). Both factors of Φ carry the *same* k:
```
Φ_{cvk}(r) = (1/N_k) e^{-ik·r} u*_{ck}(r) · e^{+ik·r} u_{vk}(r) = (1/N_k) u*_{ck}(r) u_{vk}(r).   (X.2)
```
The `e^{±ik·r}` cancel exactly (Henneke states this verbatim,
`Henneke-2020-*.md:194`: "the factor e^{ik·r} cancels exactly due to the complex
conjugate operation"). **Therefore Φ_{cvk} is lattice-periodic — it carries crystal
momentum q=0 for every k.** Its Fourier support lives only on reciprocal-lattice
vectors G, with no k-dependent Bloch momentum.

**Momentum conservation from v(r−r′) fixes the *transfer*, not k.** Expand
`v(r−r′)=(1/Vol)Σ_{q,G} v(q+G) e^{i(q+G)·(r−r′)}`. The r-integral of (X.1) is
`∫dr Φ*_{cvk}(r) e^{i(q+G)·r}`. Because Φ is lattice-periodic (X.2), this integral
**vanishes unless q=0** — a lattice-periodic function has no matrix element with a
plane wave of nonzero crystal momentum q. The *same* argument applied to the
r′-integral again forces q=0. So the only Coulomb component that survives is the
**q=0 tile** `v(0+G)=v(G)`:
```
⟨cvk|K^x|c′v′k′⟩ = (1/N_k²) · |Ω^ℓ|/|Ω| · Σ_G v(G) [∫_Ω u*_{ck}u_{vk} e^{iG·r}] [∫_Ω u*_{v′k′}u_{c′k′} e^{−iG·r′}]
                 = (1/N_k) Σ_G v(G)  ρ_{cvk}(G)  ρ*_{c′v′k′}(G).                              (X.3)
```
(This is exactly Henneke Eq. 2-17 for `V_A` with the `1/N_k²` and the supercell
factor `N_k` combining to `1/N_k`; the ISDF form Eq. 4-3 replaces `Σ_G v(G) ρρ*` by
`Σ_{μν} M*(μ) Ṽ_{μν} M(ν)`, the exact structure of `V_q0` in the code.)

**Where δ_{kk′} would have to come from — and why it can't.** A δ_{kk′} would require
that the r-integral force q = (something involving k−k′), so that the two pair
densities could only "meet" when k=k′. That happens in the **direct** term W, where
the two densities are `ψ*_{ck}ψ_{c′k′}` (electron line) and `ψ*_{v′k′}ψ_{vk}` (hole
line) — **different** k on the two factors of each density, giving a residual phase
`e^{−i(k−k′)·r}`, hence transfer `q=k−k′` and the screened tile `W_{q=k−k′}`
(Henneke Eq. 2-17 W_A: the explicit `e^{−i(k−k′)·(r−r′)}` factor; code:
`bse_serial.py:69-78`, the FFT convolution over `q=k−k′`). **In the exchange term the
phase cancels within each pair (X.2), so nothing ties k to k′.** The two pair
densities each independently carry q=0; they couple through the *single* q=0 Coulomb
tile regardless of their k labels. k and k′ run **independently and fully** over the
BZ. δ_{kk′} never appears.

This is the precise disentangling of the owner's two claims:
- (a) "the contracted object has momentum shift 0" → **TRUE**: only `v(G)` at q=0
  enters (X.3). Both sides agree.
- (b) "the pair-coupling is δ_{kk′}" → **FALSE**: the q=0 tile is contracted between
  pair densities at **different** k and k′. q=0 is a statement about the *Coulomb
  line*, not about *equality of the two exciton k-labels*.

The code's `V_q0` **is** the correct object (the q=0 tile, ISDF-compressed). The bug
is purely in the contraction pattern: it feeds `V_q0` between `M_{cvk}` and `M_{c′v′k}`
at the *same* k, when (X.3) demands `M_{cvk}` and `M_{c′v′k′}` at *all* (k,k′).

---

## 2. The owner's Hartree analogy, term by term — it proves the k′-sum

Owner, verbatim: *"It's the same physics as the Hartree interaction, one does
V^H_{q=0}(G) = v_{q=0}(G) · FFT[Σ_k |ψ_{nk}(r)|²](G)."*

This analogy is exactly right, and it **refutes** the k-block-diagonal conclusion,
because the `Σ_k` the owner wrote *inside* the FFT is the very sum the code drops.
Map it term by term:

| Hartree | BSE exchange | 
|---|---|
| density `ρ(r)=Σ_{n,k} |ψ_{nk}(r)|² = Σ_{n,k} ψ*_{nk}(r)ψ_{nk}(r)` | encode `S(r_ν)=Σ_{c′,v′,k′} ψ*_{c′k′}(r_ν)ψ_{v′k′}(r_ν) X_{c′v′k′}` = Σ_{k′} M_{c′v′k′}(ν)X |
| the sum `Σ_k` runs over **all** k → one k-independent ρ(r) | the encode sum `Σ_{k′}` runs over **all** k′ → one k-independent S(ν) |
| `V^H(G)=v_{q=0}(G)·ρ(G)` (q=0 Coulomb tile) | `U(μ)=Σ_ν V_q0(μ,ν) S(ν)` (q=0 Coulomb tile) |
| `V^H(r)` is the **same** potential felt at every k: `⟨ψ_{ik}|V^H|ψ_{ik}⟩` | decode `[K^xX]_{cvk}=Σ_μ M*_{cvk}(μ) U(μ)` — same k-independent U re-read at every k |

The Hartree potential is built by summing the density over **all** occupied (n,k),
producing **one** potential `V^H(r)` that is then felt identically at every k. It does
**not** depend on "which k′ contributed" — a state at k feels the field of the charge
at every k′. Structurally that is `encode = Σ_{k′}(…)`, `decode at k` — i.e. **dense
in (k,k′)**. The owner's own formula has `Σ_k` *inside* the object that gets FFT'd and
multiplied by the q=0 Coulomb; the LORRAX encode `'kcvN,bcvk->bNk'` **keeps k as an
output axis instead of summing it**, which is the Hartree analogy with the `Σ_k`
deleted. If you actually did Hartree the way the code does exchange, you would build a
*separate* potential from each k's density alone and let it act only back on that same
k — that is not the Hartree interaction, it is a self-interaction restricted to one
k-point, and it would vanish as the grid refines (see §5).

So the analogy is correct and establishes claim (a) — the q=0 Coulomb tile — but the
`Σ_k` it contains is exactly claim (not-b). The owner proved the dense position and
read it as the diagonal position.

---

## 3. Henneke Eq. 4-5: the reference the code cites has an explicit outer Σ_{k′}

The matvec's docstring lineage traces to Henneke-2020 (`context/`). Its exchange
matvec, Eq. 4-5 (`Henneke-2020-*.md:450-456`), regrouped for efficiency, is:
```
[V_A X](i_v i_c k) = (1/N_k) Σ_μ  ū_{i_c k}(r̂_μ) u_{i_v k}(r̂_μ)          ← DECODE at k (bra pair @ k)
                     · { Σ_ν Ṽ_{A,μν}                                     ← q=0 Coulomb tile
                         · ( Σ_{k′} ( Σ_{j_c} u_{j_c k′}(r̂_ν)             ← ENCODE, explicit Σ_{k′}
                               ( Σ_{j_v} ū_{j_v k′}(r̂_ν) X(j_v j_c k′) ) ) ) }
```
Read the parenthesization: the innermost object `Σ_{k′} Σ_{j_c,j_v} u_{j_c k′}(ν)
ū_{j_v k′}(ν) X(j_v j_c k′)` **sums over k′ and has no free k index** — it is a pure
function of the centroid `r̂_ν`. Call it `S(ν)`. Henneke then applies `Ṽ_{A,μν}`
(→`U(μ)`, still k-free), and only in the outermost factor does a k reappear, and it
enters **solely through the decode orbitals** `ū_{i_c k}(r̂_μ) u_{i_v k}(r̂_μ)` — the
bra pair density at the output k. Text confirms (`Henneke-2020-*.md:458`): "one can
first perform contractions over j_v, j_c, **and k′** to obtain a quantity that only
depends on r̂_ν." The matvec cost analysis (`:443`, Eq. 4-4) is even more explicit:
`[V_A X](i_v i_c k) = Σ_{j_v,j_c,k′} V_A(i_v i_c k, j_v j_c k′) X(j_v j_c k′)` — a
sum over k′.

Compare index-for-index to the code:
| Henneke Eq. 4-5 | `bse_serial.py:62-64` | match? |
|---|---|---|
| `S(ν)=Σ_{k′,c′,v′} M_{c′v′k′}(ν) X` — **no k** | `S_V = 'kcvN,bcvk->bNk'` — **k kept** | **NO** |
| `U(μ)=Σ_ν Ṽ_{μν} S(ν)` — no k | `U_V='MN,bNk->bMk'` — k spectator | mismatch inherited |
| `[V_AX]_{cvk}=(1/N_k)Σ_μ M*_{cvk}(μ) U(μ)` | `'kcvM,bMk->bcvk'` | decode OK, but fed wrong U |
| prefactor `1/N_k` (single) | `1/√N_k · 1/√N_k = 1/N_k` | **prefactor agrees** |

The prefactor is fine (both sides give `1/N_k` total — so this is *not* a
normalization dispute; the owner is right that the overall `1/N_k` is correct). The
sole discrepancy is the encode's spectator `k`: Henneke sums it (`->bN`), the code
keeps it (`->bNk`). The code implements Henneke Eq. 4-5 with the outer `Σ_{k′}`
deleted.

---

## 4. BerkeleyGW: the kernel builder loops every (ik,ikp) for the exchange, q=0 tile

BGW assembles the full dense kernel matrix, both direct and **exchange**, over all
pair-of-k-points blocks.

- **The transition space is the full square.** `distrib_kernel.f90:80`:
  `peinf%nck = (xct%nkpt_co*xct%ncb_co*xct%nvb_co)**2`. The number of kernel blocks is
  `(N_k N_c N_v)²`, i.e. the dense `(cvk)×(c′v′k′)` matrix, not `N_k N_c N_v`
  (which is what a k-diagonal kernel would need).
- **Nested ik/ikp enumeration.** `distrib_kernel.f90:228` `do ikp=1,xct%nkpt_co`
  nested under the ik loop; the block owner is indexed
  `((ik−1)*nkpt_co+(ikp−1))*…` (`:236,:247,:252`) and stored as
  `peinf%ik(ipe,·)=ik`, `peinf%ikp(ipe,·)=ikp` (`:287-288`). Every (ik,ikp) pair is a
  block.
- **The kernel builder fills exchange per (ik,ikp).** `kernel_main.f90:401` `do
  ii=1,peinf%nckpe` pulls `ik=peinf%ik(...,ii)`, `ikp=peinf%ikp(...,ii)`
  (`:408-409`) and calls `mtxel_kernel(...,bsex,ii,ik,ikp,...)` (`:471-474`) — `bsex`
  (the exchange block) is an output for **every** (ik,ikp).
- **The exchange itself: q=0 Coulomb tile between different-k pair densities.**
  `mtxel_kernel.f90:635-700`:
  - `mvc = <vk|e^{iG·r}|ck>` — pair density at **ik** (`:652-657`).
  - `mvpcp = <vkp|e^{iG·r}|ckp>` — pair density at **ikp** (`:659-664`).
  - `get_vcoul(.true.,.false.)` at `qflag==1` → the **q=0** bare Coulomb with head
    zeroed (`:691`; the `.true.` argument zeroes `vcoul(G=0)`).
  - `gx_sum_TDA(xct,…,vcoul,mvc,mvpcp,bsex,ivp,icp,ikp,iv,ic,ik)` (`:695-696`) sums
    `Σ_G mvc*(G) vcoul(G) mvpcp(G)` into the (ik,ikp) block of `bsex`.

  This is exactly (X.3): a **single q=0 Coulomb tile** (owner's claim (a): TRUE)
  contracted between pair densities at **different** k=ik and k′=ikp (claim (b):
  FALSE — dense). BGW's exchange is q=0 in the Coulomb line and dense in (k,k′)
  simultaneously; the two are not in tension. LORRAX's `--ring-check`
  (`bse_ring_comm.py:853-996`) cannot catch the discrepancy because it validates the
  ring matvec against the serial matvec, which implements the *same* k-diagonal
  contraction — there is no dense reference in the tree.

---

## 5. Thermodynamic limit: k-diagonal exchange vanishes as 1/N_k — unphysical

Take a normalized, k-delocalized (Wannier-like) exciton, the case where exchange
matters most: `X_{cvk} = a_{cv}/√N_k` with `Σ_{cv}|a_{cv}|²=1` (so `Σ_{cvk}|X|²=1`).
Assume the pair amplitude `M_{cvk}(ν)` is smooth in k (true away from band
crossings), with k-average `m̄(ν)=(1/N_k)Σ_k Σ_{cv} M_{cvk}(ν) a_{cv} = O(1)`.

**Dense (correct) form.** Let `A(ν)=Σ_{c′v′k′} M_{c′v′k′}(ν) X_{c′v′k′}
= N_k^{−1/2} Σ_{k′}Σ_{c′v′} M_{c′v′k′}(ν) a_{c′v′} = N_k^{−1/2}·N_k·m̄(ν)=√N_k · m̄(ν)`.
Then
```
⟨X|K^x|X⟩_dense = (1/N_k) Σ_{μν} A*(μ) V_q0(μ,ν) A(ν)
                = (1/N_k) · N_k · Σ_{μν} m̄*(μ) V_q0(μ,ν) m̄(ν)
                = Σ_{μν} m̄*(μ) V_q0(μ,ν) m̄(ν)  =  O(1).                          (X.4)
```
**Intensive — converges to a finite constant as N_k→∞.** This is the physical
singlet-triplet / exchange splitting: it is the exchange self-energy of the exciton,
which must approach a well-defined thermodynamic-limit value as the mesh refines.

**k-diagonal (code) form.** Per-k inner object `b_k(ν)=Σ_{cv} M_{cvk}(ν) a_{cv}=O(1)`;
`X_{cvk}=a_{cv}/√N_k` gives the code's `S_V[·,k]=b_k(ν)/√N_k`. Then
```
⟨X|K^x|X⟩_diag = (1/N_k) Σ_k Σ_{μν} (b_k*(μ)/√N_k) V_q0(μ,ν) (b_k(ν)/√N_k)
               = (1/N_k²) Σ_k [ Σ_{μν} b_k*(μ) V_q0(μ,ν) b_k(ν) ]
               = (1/N_k²) · N_k · O(1)  =  O(1/N_k)  →  0.                        (X.5)
```
**The k-diagonal exchange scales as 1/N_k and vanishes as the grid refines.** The
singlet-triplet splitting — which in TDA is *entirely* the exchange term (the
D and W terms are spin-blind for the singlet/triplet pair) — would shrink toward zero
as you converge the k-mesh. That is unphysical and grid-diagnostic: a correctly
implemented BSE converges the LT/exchange splitting to a nonzero constant; this bug
makes it a decreasing function of N_k that a convergence study would expose as "the
singlet and triplet merge as I add k-points." The ratio (X.4)/(X.5) = O(N_k): on a
12×12 MoS₂ mesh the code captures ~1/144 of the physical exchange splitting. This is
the quantitative statement behind DEAD_CODE.md §1.1's "silently wrong singlet
spectrum."

(Why Si 4×4×4 SOC passed at ~3 meV, `bse_serial`/STATUS: Si's low excitons are
triplet-dominated and the G=0 exchange is weak by orthogonality, so the missing
factor-of-N_k multiplies a near-zero number. Exchange-insensitive fixture.)

---

## 6. Reconciling (a) vs (b): exactly what the owner has right and wrong

- **Right (a):** the contracted Coulomb object is the q=0 tile only. Confirmed by
  (X.3), Henneke Eq. 4-3/4-5 `Ṽ_A`, and BGW `get_vcoul(.true.,.false.)` at qflag=1.
  The `V_q0` array in the code is the correct object. The overall `1/N_k` prefactor
  is correct. The Hartree analogy is apt.
- **Wrong (b):** "q=0 tile" ≠ "δ_{kk′}". q=0 constrains the *Coulomb momentum
  transfer* (both pair densities are lattice-periodic, so only reciprocal-lattice G's
  enter), not the *equality of the two exciton k-labels*. The pair densities at k and
  k′ are independent; the q=0 tile couples every k to every k′. The owner's own
  Hartree formula carries `Σ_k` inside the density — that `Σ_k` **is** the k′-sum the
  code omits. Claim (a), correctly followed through, *is* the dense position.

The confusion is natural because in the **direct** term the phrase "q" genuinely
means "k−k′" and the k,k′ dependence lives in `W_{k−k′}`; the code's W path
(`bse_serial.py:69-78`) correctly convolves over q=k−k′ and *is* dense in (k,k′).
The owner (correctly) noticed exchange uses only q=0 while direct uses all q — and
then mistakenly read "only q=0" as "only k=k′." Exchange is dense too; it is just
dense with a *constant* (k,k′-independent) coupling `V_q0`, whereas direct is dense
with a `(k−k′)`-dependent coupling `W_{k−k′}`. "k-block-diagonal exchange" is the
error; "constant-in-(k,k′) exchange coupling" is the truth, and a constant coupling
is maximally dense, not diagonal.

---

## 7. Finite-Q generalization (H^{Q≠0}) — REQUIRED SECTION

### (i) The finite-Q exchange action in the pair basis, and the convention reconciliation

LORRAX pair basis (`context/parallel_bse_algos.md:9`, `designs/finite_q_bse.md`):
`|v k, c k+Q⟩` — hole at `(v,k)`, electron at `(c,k+Q)`, exciton COM momentum **+Q**.
The amplitude is labelled by the **hole** k: `X^Q(v,c,k)`. BGW uses the mirror
convention (shift on valence, COM `−Q`, `bgw_fine_grid_reference.md:466-475`); it is a
relabeling, not different physics.

Finite-Q pair density (electron shifted): `M^Q_{cvk}(G)=⟨c,k+Q| e^{i(Q+G)·r} |v,k⟩`.
Now the residual Bloch phase inside the pair is `e^{i((k+Q)−k)·r} · e^{iG·r}·(cell)
= e^{i(Q+G)·r}·(cell-periodic)` — the electron and hole carry **different** crystal
momenta differing by Q, so the pair density carries net crystal momentum **Q**, not
0. Repeating the §1 momentum-conservation argument: the r-integral now survives only
when the Coulomb momentum transfer equals **Q**, selecting the **q=Q tile** `v(Q+G)`
including **G=0** (finite, `4π/|Q|²`, no head divergence). Exchange action:
```
[K^x_Q X^Q](v i_c k) = (1/N_k) Σ_μ  ū_{i_c,k+Q}(r̂_μ) u_{i_v,k}(r̂_μ)            ← decode: bra pair @ (k, k+Q)
                       · { Σ_ν Ṽ^Q_{μν}                                          ← q=Q Coulomb tile, G=0 incl.
                           · ( Σ_{k′}  Σ_{j_c,j_v} u_{j_c,k′+Q}(r̂_ν) ū_{j_v,k′}(r̂_ν) X^Q(j_v j_c k′) ) }
Ṽ^Q_{μν} = |Ω| Σ_G (4π/|Q+G|²) conj(ζ^V_μ(G)) ζ^V_ν(G)      (parallel_bse_algos.md:22)
```
(matches `designs/finite_q_bse.md`'s `V^Q_A`). The encode still sums **k′**; the
single Q-tile `Ṽ^Q` is constant across (k,k′); the decode reintroduces k through the
bra pair `(k,k+Q)`.

**Is "X_cvk → Y_{cv,k+Q}" the same statement as Q-block-diagonality of H^BSE in a
fixed-Q pair basis? Yes — reconcile as follows.** The owner's phrasing
"K^X_Q X_cvk → Y_{cv k+Q}" describes the action in the **unshifted** labeling where X
is indexed by a single k and the operator *moves* momentum by Q. In the **fixed-Q
pair basis** `|v k, c k+Q⟩`, one instead labels the amplitude by the hole k and holds
Q **fixed**: every basis vector already carries COM momentum Q, and H^{BSE} acts
**within** that fixed-Q block (`k→k` in the label, because the "+Q" is absorbed into
the basis definition). These are the same operator written in two labelings:
- Unshifted labeling: input carries momentum p (the hole-k picture with X_cvk), output
  carries p+Q; H^X_Q is the block of H that shifts momentum by exactly Q — i.e. H is
  **block-diagonal in Q** (a given Q maps momentum-p sector to momentum-(p+Q) sector,
  and different Q's don't mix). "X_cvk→Y_{cv,k+Q}" is precisely the statement that the
  operator lives in the single-Q block.
- Fixed-Q pair basis: fold the +Q into the ket, index by hole k; H^{BSE}(Q) is a
  matrix in (v c k)×(v′ c′ k′) at that fixed Q. Its eigenproblem `H^{BSE}(Q) A^S(Q) =
  E_S(Q) A^S(Q)` is the finite-Q exciton dispersion.

So the owner's "→ Y_{cv,k+Q}" and "H is Q-block-diagonal in a fixed-Q pair basis" are
**the same statement**, viewed with the +Q shift either explicit (on the output label)
or absorbed (into the basis). The ± ambiguity the owner flagged ("k+Q or maybe −Q on
convention") is exactly the LORRAX(+Q on electron) vs BGW(−Q on hole) mirror: whether
you shift the electron by +Q or the hole by −Q, and whether you label X by the hole-k
or electron-k. Both give the same spectrum E_S(Q)=E_S(−Q) for a TR-symmetric crystal;
only the index bookkeeping differs. **This is not a dispute with the dense position —
Q-block-diagonality (across different Q) is orthogonal to (k,k′)-density within a
fixed Q.**

### (ii) Within one fixed-Q block, is (k,k′) coupling dense or diagonal? — Dense.

The identical §1 argument at Q≠0: the pair densities `M^Q_{cvk}` and `M^Q_{c′v′k′}`
each carry net momentum Q (fixed), and are otherwise **independent in k and k′**.
Momentum conservation fixes the Coulomb transfer to q=Q (the tile `v(Q+G)`, G=0
included), which is a **single constant tile** across the whole block. Nothing ties k
to k′. Encode sums k′, decode reintroduces k → **dense in (k,k′)** exactly as at Q=0.
The only differences from Q=0 are (1) the tile is `Ṽ^Q` instead of `Ṽ^{Q=0}`, and (2)
`Ṽ^Q` keeps its G=0 element (finite `4π/|Q|²`), whereas `Ṽ^{Q=0}` excludes/rank-1-
injects the divergent head. BGW confirms: `get_vcoul(.not.energy_loss,.true.,finiteq,
qpg0_ind)` at `qflag≠1` (`mtxel_kernel.f90:689`) builds the `v(Q+G)` column and does
**not** zero the head for finite Q ("we should never zero head of exchange when using
finite Q," `:688`) — still called inside the same per-(ik,ikp) block loop, so still
dense in (k,k′). **Q≠0 does not rescue the k-diagonal code; it is wrong there too.**

### (iii) Architecture: what must change under the DENSE contraction structure

Current Q=0 dataflow (`kernel_dataflow_trace.md`, `bse_serial.py:59-64`): S-encode
(`M`,`X`→`S`) → `V_q0` contract → decode (`M*`→`V_term`); W via 3-D FFT convolution
over q=k−k′. Under the **correct dense** exchange the changes are:

1. **Exchange encode becomes k-summed (k-global), not k-local.** Change the three
   einsums at `bse_serial.py:62-64` (and the parallel forms in `bse_simple.py:101-131`,
   `bse_ring_comm.py:262-300 apply_V_ring`, dead `bse_jax.py:108-121`) from
   ```
   'kcvN,bcvk->bNk' ;  'MN,bNk->bMk' ;  'kcvM,bMk->bcvk'
   ```
   to
   ```
   S(ν)   = einsum('kcvN,bcvk->bN', M, X) / √N_k      # SUM over k — one k-free vector
   U(μ)   = einsum('MN,bN->bM', V_q0, S)              # k-free
   V_term = einsum('kcvM,bM->bcvk', conj(M), U) / √N_k  # broadcast same U to every k
   ```
   In the **ring**/sharded path this means the encode `S` must be **all-reduced across
   the k-shard** (S is a global sum over k′; today each rank keeps its own k-slice).
   That is one extra reduction over the ζ-index vector `S(ν)` (length `n_rmu`, tiny) —
   cheap, and it is the *only* new communication. The decode then broadcasts the
   reduced `U(μ)` back to every local k. Note: `bse_ring_comm.py`'s history already had
   this — the dropped `v_couples_k`/`couple_k` parameter at ancestor `81ca040`
   (DEAD_CODE.md §1.2) summed `S_total` over k; the fix is to **restore that reduction
   as the only path**, not add a parallel one.

2. **Q index — one tile swap, no new density index.** For H^{Q≠0} the exchange needs
   `Ṽ^Q = V_qmunu[Q_flat]` in place of `V_q0`, and the **conduction** tensors read at
   the shifted grid index `k+Q` with the umklapp phase (`designs/finite_q_bse.md`:
   `psi_c_Q[k,c,s,μ]=e^{−2πi G_umk·s_μ}·psi_full[kpQ_index[k],…]`, `eps_c_Q` similarly).
   The **exchange contraction pattern is byte-identical** to the Q=0 dense form above —
   it just receives `Ṽ^Q` in the tile slot and `psi_c_Q` in the conduction slots. No
   tensor gains a new *density* axis; the amplitude `X` stays rank `(b,c,v,k)` at fixed
   Q. The full `V_qmunu(nq,μ,ν)` is already on disk (`bse_io.py:389-411`); only the
   q=0 slice is read today (`_read_vq0_sharded`, `bse_io.py:221-265` — parametrize with
   `q_index=Q_flat`, single reader).

3. **Q=0 falls out as the Q→0 special case — single path, no fork.** Set Q=(0,0,0):
   `kpQ_index[k]=k`, `G_umk=0` (no phase), `Ṽ^Q → Ṽ^{Q=0}` = `V_q0` **with the head
   rank-1-injected** (the head injection is gated on `Q==(0,0,0)`, since only at Q=0 is
   `4π/|Q+G|²` divergent at G=0; `designs/finite_q_bse.md`). The dense k-summed encode
   is identical for all Q. **Therefore the correct architecture has exactly ONE
   exchange matvec** — k-summed encode, tile contract, broadcast decode — with Q
   entering only as (a) which conduction grid-index/phase the loader supplies and (b)
   which `V_qmunu` slice fills the tile slot. Any design that keeps a separate "k-local
   exchange for Q=0" and "k-summed exchange for Q≠0" is a **defect** (two parallel
   paths, violates the no-redundancy rule): the Q=0 path is just wrong, and the
   Q-general k-summed path *is* the Q=0 path at Q=0.

4. **W (direct) is unchanged by all of this.** It already convolves over q=k−k′
   (dense, correct); at finite Q only the conduction ψ read shifts to k+Q and the
   `W̃_{k−k′,μν}` tile is Q-independent (`designs/finite_q_bse.md`,
   `parallel_bse_algos.md:38-43`). No W change is implied by the exchange fix.

Net: the fix is **remove one spectator axis** (the encode's `k`) — turning k-local
into a k-global reduction — and the Q generalization then rides on the existing
loader-side k+Q remap with a single tile swap, with Q=0 recovered as the special case.
One exchange code path, not two.

---

## 8. Final position

**DENSE. The k-block-diagonal exchange in `bse_serial.py:62-64` (and the three
sibling matvecs) is a correctness bug, not a convention.** Five independent
derivations — the real-space Coulomb integral (§1), the owner's own Hartree analogy
carried to its conclusion (§2), the Henneke Eq. 4-5 the code descends from (§3), the
BerkeleyGW kernel builder (§4), and the thermodynamic-limit scaling (§5) — all yield
`⟨cvk|K^x|c′v′k′⟩ = (1/N_k) Σ_{μν} M*_{cvk}(μ) V_q0(μ,ν) M_{c′v′k′}(ν)` with **no
δ_{kk′}**, and none can be made to produce one. The owner is correct that only the
q=0 Coulomb tile enters (claim a) and that the `1/N_k` prefactor is right; the error
is the inference from "q=0 tile" to "k-block-diagonal" (claim b) — the q=0 tile is a
constant coupling contracted between pair densities at **all** independent (k,k′),
which is maximally dense, and the `Σ_k` in the owner's Hartree formula *is* the k′-sum
the code omits. At finite Q the same holds within each fixed-Q block (dense in k,k′,
with the q=Q tile including G=0), and Q-block-diagonality across different Q is a
separate, compatible fact. The fix is a one-axis change (k-summed encode + broadcast
decode, i.e. `'kcvN,bcvk->bN'`), it restores the reduction that ancestor revision
`81ca040` already had, and it makes Q=0 the Q→0 limit of a single finite-Q exchange
path.
