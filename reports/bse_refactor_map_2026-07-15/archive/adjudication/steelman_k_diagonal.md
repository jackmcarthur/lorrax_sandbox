# Steelman: is the k-block-diagonal Q=0 BSE exchange correct?

**Role:** construct the strongest possible case that the as-coded k-block-diagonal
exchange contraction (`bse_serial.py:62-64` and the three siblings) is the *correct*
Q=0 BSE exchange. Adjudicate claim **(b)** — that the pair coupling is δ_{kk′} —
precisely.

**Verdict up front (stated even though it is against the side I was asked to argue):**
The k-block-diagonal contraction is **not defensible** as the Q=0 BSE exchange for
N_k > 1. Every steelman I can build either (i) reduces to the trivially-true
single-k (molecular / Γ-only) special case, or (ii) silently changes the operator
into something that is no longer the BSE exchange kernel. Three independent
references — Henneke Eq (2-16)/(4-5), BerkeleyGW `gx_sum_TDA` driven by the
`(ik,ikp)` loop in `kernel_main.f90`, and a from-scratch index derivation — all give
a kernel that is **dense in (k,k′)**. The owner's intuition is correct about a
*different* true statement (only the q=0 Coulomb tile enters; and H^BSE is
block-diagonal in the exciton COM momentum Q). Neither of those implies δ_{kk′}. The
strongest honest form of the owner's position collapses onto claim **(a)**, which was
never in dispute.

The failing steelman is itself the finding.

---

## 0. The precise object in dispute

As coded (`src/bse/bse_serial.py:62-64`, verbatim einsums):

```
S_V[b,ν,k] = Σ_{c,v}   M[k,c,v,ν]  X[b,c,v,k]        / √N_k     # k KEPT
U_V[b,μ,k] = Σ_ν       V_q0[μ,ν]   S_V[b,ν,k]                    # k KEPT
V_term[b,c,v,k] = Σ_μ  conj(M[k,c,v,μ]) U_V[b,μ,k]  / √N_k       # k KEPT
```

Collapsing the three steps (the two 1/√N_k multiply to 1/N_k):

```
(K^x X)[b,c,v,k] = (1/N_k) Σ_{μν} M*_cvk(μ) V_q0(μ,ν) [ Σ_{c′v′} M_c′v′k(ν) X[b,c′,v′,k] ]     (CODE)
```

The **only** k that appears in the source bracket is the *same* k as the output.
That is claim **(b)**: pairs at k couple only to pairs at the same k.

The disputed alternative (audit DEAD_CODE §1.1, `bse_serial.py` "fix direction"
line 89):

```
(K^x X)[b,c,v,k] = (1/N_k) Σ_{μν} M*_cvk(μ) V_q0(μ,ν) [ Σ_{k′,c′v′} M_c′v′k′(ν) X[b,c′,v′,k′] ]   (DENSE)
```

The source bracket is now **k-independent** — one vector S(ν) = Σ_{k′,c′v′} …, broadcast
to every output k on decode. Equivalently, the matrix element is

```
⟨cvk|K^x|c′v′k′⟩ = (1/N_k) Σ_{μν} M*_cvk(μ) V_q0(μ,ν) M_c′v′k′(ν)     — no δ_kk′.     (DENSE elt)
```

Note the structure of (DENSE elt): it is a **separable / rank-≤N_μ** operator in the
full (c,v,k) index space. The k-diagonal CODE keeps exactly the block-diagonal
(k′=k) sub-blocks of this operator and discards every off-diagonal (k′≠k) block. So
the two are not related by any scalar normalization: CODE is a strict submatrix of
DENSE. They have different rank, different spectrum, different eigenvectors. No
convention converts one into the other (this already kills the "1/N_k absorbs a
k-sum" idea; see S2).

Both sides agree on claim **(a)**: only the single q=0 Coulomb tile `V_q0(μ,ν)`
enters (momentum transfer of the interaction is 0, G=0 excluded by valence–conduction
orthogonality, Henneke Eq 2-32). The dispute is exclusively about δ_{kk′}.

---

## 1. The steelman attempts (each strongest form, and why it fails)

### S1 — "It's a local (contact / TDDFT ALDA-type) exchange kernel, which is k-diagonal."

**Strongest form.** In TDDFT the xc kernel f_xc(r,r′) = f(r) δ(r−r′) is local. A local
kernel yields a matrix element ∫ ψ*_ck(r)ψ_vk(r) f(r) ψ*_c′k′(r)ψ_v′k′(r) dr with a
single spatial integral — maybe that forces k=k′.

**Why it fails.** It does not force k=k′. Even for a strictly local kernel,
```
K_loc(cvk,c′v′k′) = ∫ Ω^ℓ  ρ*_cvk(r) f(r) ρ_c′v′k′(r) dr,   ρ_cvk(r)=ψ*_ck(r)ψ_vk(r)
```
is a full dense matrix in (k,k′): ρ_cvk and ρ_c′v′k′ are two different lattice-periodic
densities, and the integral of their product against f(r) is generically nonzero for
k≠k′. Locality in **r** (δ(r−r′)) is not locality in **k**. A δ(r−r′) kernel makes the
kernel *G-diagonal in reciprocal space* — which is precisely the bare-exchange case
already (v(G)δ_{GG′}) — and that case is manifestly dense in (k,k′) below (S5, §2).
Locality removes the G,G′ double sum, not the k,k′ double sum. Fails.

### S2 — "A normalization convention: the 1/N_k out front already *is* the averaged k-sum."

**Strongest form.** The dense form carries 1/N_k and an inner Σ_{k′} (N_k terms); maybe
the code's 1/N_k with no Σ_{k′} is the same operator under a convention where X is a
k-averaged amplitude, or where 1/N_k denotes "average over k′" rather than "sum then
divide".

**Why it fails, concretely.** Track the factors. Henneke's V_A carries an explicit
1/N_k (Eq 4-3), and the matvec (Eq 4-5) is `(1/N_k) Σ_μ ū_ic,k u_iv,k {Σ_ν Ṽ_A,μν (Σ_{k′} …)}`
— the 1/N_k and the Σ_{k′} **coexist**; the 1/N_k does not stand in for the sum. The
code independently reproduces that same 1/N_k as √N_k · √N_k in encode·decode
(`bse_serial.py:60,62,64`) — so the prefactor is *already fully accounted for* and has
no spare factor of N_k to represent an averaged k-sum. If the code intended
S(ν)=⟨M,X⟩_k averaged over k, the decode would have to *not* re-divide, and the
per-k output would have to be constant in k — neither is true (`U_V[b,μ,k]` is k-dependent;
decode divides again). Moreover, as noted in §0, CODE is a strict submatrix of DENSE —
submatrix vs full matrix cannot be related by any scalar prefactor, because they have
different rank. A convention can move where a 1/N_k sits; it cannot restore N_k−1
dropped off-diagonal blocks. Fails.

### S3 — "Supercell / Born–von-Kármán Γ-only folding makes the exchange one block."

**Strongest form.** Henneke Eq (2-11): the whole periodic problem lives on a supercell
Ω^ℓ whose reciprocal cell contains the k-grid as its Γ-folded points. In a Γ-only
supercell there is a single k = Γ; a single k-point is trivially "k-diagonal". LORRAX's
molecular ISDF-BSE ancestor (Hu–Shao–Cepellotti–da Jornada–Lin–Thicke–Yang–Louie 2018,
Henneke ref [13]) is a **molecule** — one k-point — where the code's einsum
`'kcvN,bcvk->bNk'` with N_k=1 is exact.

**Why it fails.** The supercell/Γ-only picture, applied correctly, is the *fully
k-summed* (DENSE) picture, not the per-primitive-k-diagonal (CODE) picture. Folding the
N_k primitive-cell k-points to a single supercell Γ means the exchange is **one** block
that couples **all** primitive-cell pair labels — i.e. the Σ_{k′} is exactly the
intra-supercell sum. The code, by contrast, keeps N_k *separate* blocks indexed by the
primitive-cell k and never couples them. So the supercell argument, taken to its
conclusion, is an argument *for* DENSE.

What S3 *does* establish — and this is the single strongest true kernel of the entire
steelman — is the **N_k = 1 special case**:

> **k-diagonal exchange is exact if and only if N_k = 1** (molecule / Γ-only / one
> k-point). For N_k = 1 there is no off-diagonal block to drop, CODE ≡ DENSE, and the
> code is correct.

This is almost certainly the historical origin of the code: a molecular ISDF-BSE matvec
(`'cvN,bcv->bN'`) had a batch `k` axis grafted on (`'kcvN,bcvk->bNk'`) for the periodic
port, without adding the cross-k coupling that periodicity introduces. The single-k
correctness then survived as a latent bug for every N_k > 1 run. That is the precise,
defensible residue of the steelman — and it is not a defense of the periodic code.

### S4 — "Momentum conservation: a q=0 interaction can only couple k to itself."

**Strongest form (closest to the owner's words).** The exchange uses the q=0 Coulomb
tile. "q=0 interaction" sounds like "zero momentum transfer," which sounds like it
cannot move a pair from k′ to k.

**Why it fails.** This conflates two different momenta. The interaction line carries
momentum q. Each e–h **pair** carries a *net* crystal momentum. In the Q=0 (optical)
BSE, every pair is built as (electron at k, hole at k) → **net pair momentum = 0**, for
*every* k. Two objects that each individually carry net momentum 0 can be connected by
a q=0 interaction *regardless* of their internal k, because momentum is already
conserved pairwise (0 = 0). The q=0 selection rule is `k_e − k_h = Q = 0` *within each
pair* — it says nothing about k vs k′ *between* the bra pair and the ket pair. Formally:
the exchange vertex is Σ_G M*_cvk(G) v(G) M_c′v′k′(G) with M_cvk(G)=⟨ck|e^{iGr}|vk⟩; the
G-sum is unrestricted and nonzero for k≠k′ (the two pair densities are different
periodic functions with overlapping Fourier content). There is **no** Kronecker δ_{kk′}
anywhere in the momentum bookkeeping. Fails. (This is the exact point of confusion the
adjudication is meant to resolve; see §3.)

### S5 — "Complete the Hartree analogy and keep k unsummed."

**Strongest form (the owner's explicit argument).** Verbatim: *"It's the same physics
as the Hartree interaction, one does V^H_{q=0}(G) = v_{q=0}(G) · FFT[Σ_k |ψ_nk(r)|²](G)."*
Perhaps the Hartree structure keeps k local.

**Why it fails — the analogy, completed, is the decisive argument *against* the code.**
Read the owner's own formula: the Hartree potential is sourced by a density
**Σ_k |ψ_nk|²** — the density is *summed over all k*. Map the analogy term-by-term onto
BSE exchange:

| Hartree | BSE exchange (Q=0) |
|---|---|
| source density ρ(r) = **Σ_k** \|ψ_nk(r)\|² | transition density n_X(r′) = **Σ_{k′c′v′}** X_{c′v′k′} ψ*_c′k′(r′) ψ_v′k′(r′) |
| potential V^H(r) = ∫ v(r,r′) ρ(r′) dr′ | Φ(r) = ∫ V(r,r′) n_X(r′) dr′ |
| matrix element ⟨nk\|V^H\|nk⟩ (each k) | (K^x X)_cvk = ∫ ψ*_ck(r)ψ_vk(r) Φ(r) dr (each k) |

The **source** of the exchange potential is the transition density
n_X(r′) = Σ_{k′} …, summed over **all** k′ — exactly the analogue of the Hartree
Σ_k |ψ_nk|². Then that *one* potential Φ(r) is projected onto each output pair (c,v,k).
That is precisely the DENSE contraction: k-summed encode → single potential → k-local
decode. The Hartree analogy therefore *requires* the Σ_{k′} the code omits. The code's
error, in the analogy's language, is building a *separate* Hartree density from each k
in isolation — `ρ_k(r)=|ψ_nk|²` per k, no Σ_k — which is not the Hartree potential of
anything. The owner's own equation contains the k-sum that the owner's code drops.

Formally the exchange matvec is `n_X → Φ → project`:
```
n_X(r′) = Σ_{k′,c′,v′} ψ*_c′k′(r′) ψ_v′k′(r′) X_{c′v′k′}          # Σ over k′  ← the disputed sum
Φ(r)    = ∫ V(r,r′) n_X(r′) dr′                                    # single potential, no k label
(K^x X)_cvk = ∫ ψ*_ck(r) ψ_vk(r) Φ(r) dr                          # k-local decode
```
In the ISDF basis this is exactly S(ν)=Σ_{k′c′v′}M_c′v′k′(ν)X, U(μ)=Σ_ν V_0(μν)S(ν),
out_cvk = Σ_μ M*_cvk(μ)U(μ) — the DENSE form. Fails (for the code); the analogy is sound
and points the other way.

---

## 2. The three independent references (all say DENSE)

**(R1) Henneke 2020 — the package's own cited reference.**
- Kernel definition Eq (2-16) (`context/Henneke-2020…md:173`):
  `V_A(i_v i_c k, j_v j_c k′) = ∫∫ ψ̄_ick(r)ψ_ivk(r) V(r,r′) ψ̄_jvk′(r′)ψ_jck′(r′) dr dr′`
  — a matrix indexed by **both** k and k′, with **no** δ_{kk′}. Contrast the diagonal
  term D (Eq 2-15, `:169`), which *does* carry `δ_{k,k′}` explicitly — the paper writes
  the Kronecker delta when it means one, and V_A has none.
- ISDF matvec Eq (4-4)/(4-5) (`:443,454`): `[V_A X](i_v i_c k) = Σ_{j_v,j_c,k′} V_A(…) X(…)`
  with an **explicit outer Σ_{k′}**, regrouped (Eq 4-5) as
  `(1/N_k) Σ_μ ū_ick(r_μ)u_ivk(r_μ) {Σ_ν Ṽ_A,μν (Σ_{k′}(Σ_jc u_jck′(r_ν)(Σ_jv ū_jvk′(r_ν) X(j_v j_c k′))))}`.
  The braced factor is **k-independent** (only k′ is summed inside; the sole k-dependence
  is the ū_ick(r_μ)u_ivk(r_μ) prefactor *outside* the brace). That is DENSE verbatim.

**(R2) BerkeleyGW — the production reference.**
- The kernel main loop iterates over **pairs** (ik, ikp):
  `kernel_main.f90:401` `do ii=1,peinf%nckpe` with `ik=peinf%ik(...)`, `ikp=peinf%ikp(...)`
  (`:408-409`), calling `genwf_kernel` for `ik` and for `ikp` (`:440-447`) and then
  computing the kernel block for that (ik,ikp). Every (ik,ikp) pair is visited — a dense
  N_k × N_k kernel.
- The exchange itself, `gx_sum_TDA` (`gx_sum.f90:33-87`), for a given (ik,ikp):
  `mvpcp(:,ivp,icp) = vcoul(:) * mvpcp(:,ivp,icp)` (`:53`) then
  `gemm('c','n', …, mvc(:,:,:,isc), mvpcp(:,:,:,iscp), …, outtemp)` (`:59-61`) and
  `bsex(iit,…) += outtemp(iv,ic,ivp,icp)` with
  `iit = peinf%wown(ivp_in,icp_in,ikp,iv_in,ic_in,ik)` (`:73-75`). Per element:
  `bsex[cvk, c′v′k′] = Σ_G conj(M_vc(G;ik)) · vcoul(G) · M_v′c′(G;ikp)` — the two pair
  densities are read at **ik** and **ikp** independently, stored at an index that carries
  **both** ik and ikp. No δ. Dense.
- BGW zeros only the head: `get_vcoul(vbar=.true.…)` sets `vcoul(1)=0` (`mtxel_kernel.f90:673-674`,
  `gx_sum.f90:49` "modified Coulomb potential where vbar(G=0)=0") — matching claim (a),
  the q=0/G=0 exclusion; orthogonal to the (k,k′) question.

**(R3) Self-derivation (from Rohlfing–Louie exchange, no external source needed).**
The bare-exchange (electron–hole exchange) kernel is the direct-Coulomb-of-the-transition-
density term:
```
K^x_{(cvk),(c′v′k′)} = ⟨ck,v k | v | c′k′, v′k′⟩_exchange
                     = (1/N_k) Σ_{G≠0} [⟨ck|e^{iGr}|vk⟩]* v(G) [⟨c′k′|e^{iGr}|v′k′⟩]
```
Derivation: exchange = the pair at (c,v,k) annihilates into the density fluctuation
ρ_cvk(r)=ψ*_ck(r)ψ_vk(r) at r; the bare Coulomb v(r−r′) propagates it; it re-creates the
pair (c′,v′,k′) from ρ*_c′v′k′(r′). Fourier: ρ_cvk(r)=Σ_G M_cvk(G)e^{iGr}/√Ω with
M_cvk(G)=⟨ck|e^{iGr}|vk⟩ (G runs over the reciprocal lattice; the *interaction* momentum
is exactly q=0 because both densities are lattice-periodic — claim (a)). v is
G-diagonal (Eq 2-28), so K^x = (1/N_k) Σ_G M*_cvk(G) v(G) M_c′v′k′(G). k and k′ enter
through two independent pair densities; there is no mechanism producing δ_{kk′}. G=0
drops by ⟨ck|vk⟩=0 (Eq 2-32 / orthogonality) — again claim (a). QED: dense.

All three agree, by three different routes (analytic paper, production Fortran, first
principles). No independent reference anywhere in the tree produces δ_{kk′}
(`kernel_dataflow_trace.md:234` "no independent dense reference anywhere"; the
`--ring-check` gate compares two implementations of the *same* wrong formula,
DEAD_CODE §1.1 `:76-79`).

---

## 3. What the owner's intuition *is* correctly capturing

The owner's statement bundles two claims; separating them dissolves the dispute.

**(a) "the object that gets contracted has a momentum shift of 0" — TRUE, undisputed.**
The interaction that enters Q=0 exchange is the single q=0 Coulomb tile `V_q0(μ,ν)`
(with G=0 excluded). Both the code and the audit agree; neither proposes summing over
q-tiles for exchange. This is the whole of the "same physics as Hartree, V^H_{q=0}"
observation, and it is right.

**(b) "the exchange kernel should be k-block-diagonal" — FALSE for N_k > 1.**
"q=0 interaction" (a statement about the *interaction line*) does not imply "δ_{kk′}"
(a statement about the *pair labels*). §1-S4 shows why: every Q=0 pair is individually
momentum-neutral, so a q=0 line couples all k to all k′. The Hartree analogy the owner
invokes (§1-S5) *requires* the Σ_{k′} the k-diagonal code drops — the source density is
Σ_k, not per-k.

**The finite-Q remark is the tell.** The owner writes (verbatim):
*"K^X_{Q=0} X_cvk → Y_cvk and K^X_Q X_cvk → Y_{cv k+Q} (or −Q on convention)."*
This is a statement about the **exciton COM momentum Q**, and it is **correct** — but it
is *inter-block* structure (different Q do not mix), not *intra-block* (k,k′) structure.
Reconciling the conventions (§4-i): "X_cvk → Y_{cv,k+Q}" is the *same statement* as
"H^BSE is block-diagonal in Q, in a fixed-Q pair basis" — once you fix whether the
amplitude label k anchors on the hole (valence) or the electron (conduction). It is
**not** a statement that within a fixed Q the (k,k′) coupling is diagonal. The owner's
mental model is (correctly) organized around Q-block-diagonality and is (incorrectly)
being read as k-block-diagonality. Q-block-diagonal is true; k-block-diagonal is the
bug. The code's δ_{kk′} is an *intra-Q-block* diagonality that no Q-block argument
licenses.

**Precise true statement the owner's intuition corresponds to:**
> The Q=0 exchange uses only the q=0 Coulomb tile (a); and H^BSE is block-diagonal in the
> exciton COM momentum Q (the finite-Q remark). Both true. Neither implies the pair
> coupling within a Q-block is diagonal in k — and it is not (b is false). The only regime
> in which (b) holds is N_k = 1.

---

## 4. Finite-Q section (H^{Q≠0}) — required

### (i) The finite-Q exchange action in the pair basis, and the ±Q convention

Fixed-Q TDA pair basis: `|v k, c k+Q⟩` — hole in valence band v at k, electron in
conduction band c at k+Q; exciton COM momentum +Q (LORRAX convention,
`parallel_bse_algos.md:9`; BGW uses the mirror, shift-on-valence / COM −Q,
`bgw_fine_grid_reference.md:466-475`). The amplitude is labeled `X^Q(c,v,k)`, **anchored
on the valence/hole k**. Pair density with momentum transfer Q:
```
M_cvk(Q,G) = ⟨c,k+Q| e^{i(Q+G)·r} |v,k⟩.
```
Finite-Q exchange action (dense in k,k′ within the fixed-Q block):
```
(K^x_Q X^Q)[c,v,k] = (1/N_k) Σ_{μν} M*_cvk(Q,μ) Ṽ^Q(μ,ν) [ Σ_{k′,c′,v′} M_c′v′k′(Q,ν) X^Q[c′,v′,k′] ]
Ṽ^Q(μ,ν) = |Ω| Σ_G (4π/|Q+G|²) conj(ζ^V_μ(G)) ζ^V_ν(G)        (parallel_bse_algos.md:22)
```
Same k-summed encode / broadcast decode as DENSE; the only changes vs Q=0 are
(1) the pair density carries the Q momentum shift, (2) the Coulomb tile is Ṽ^Q with
**G=0 included** (see ii). Output amplitude lives in the same fixed-Q pair space.

**Where ±Q enters.** The convention ambiguity is entirely the choice of **which member
of the pair the amplitude label k anchors on**:
- anchor on hole (valence) k: `X^Q(c,v,k)` ↔ pair `|v k, c k+Q⟩`, electron shifted by +Q;
- anchor on electron (conduction) k: `X̃^Q(c,v,k) = X^Q(c,v,k−Q)` ↔ pair `|v k−Q, c k⟩`,
  hole shifted by −Q.

These are the *same* physical states relabeled by k→k∓Q. The owner's "X_cvk → Y_{cv,k+Q}"
describes the operator in a convention where the *input* is written in the Q=0
(unshifted) pair labels and the *output* pair has its electron pushed to k+Q — i.e. it
tracks the momentum label of the output pair space. **Is "X_cvk → Y_{cv,k+Q}" the same
statement as Q-block-diagonality?** Yes: reconciling the two, once you identify the pair
`|v k, c k+Q⟩` with the label (c,v,k) [valence-anchored] versus (c,v,k+Q)
[conduction-anchored], the "shift the output by +Q" mapping is exactly the statement that
the operator maps the fixed-Q pair space to itself and never mixes different Q. The +Q
vs −Q is the valence-anchor vs electron-anchor choice, nothing more. It is a statement
about the **block (Q) structure**, and — crucially — it is **silent on the intra-block
(k,k′) structure**. It does not assert (and must not be read as asserting) k-diagonality
within the block; §4-ii shows the intra-block coupling is dense, exactly as at Q=0.

### (ii) Within one fixed-Q block: dense or diagonal in (k,k′)?

**Dense — same adjudication as (b), now at Q≠0.** The pair densities are
`M_cvk(Q,G)=⟨c,k+Q|e^{i(Q+G)r}|v,k⟩` and `M_c′v′k′(Q,G)`; the exchange couples them
through the *single* tile v(Q+G):
```
⟨cvk|K^x_Q|c′v′k′⟩ = (1/N_k) Σ_G M*_cvk(Q,G) v(Q+G) M_c′v′k′(Q,G)     — no δ_kk′.
```
Two pairs both carrying COM momentum Q are connected by the Q-momentum interaction line
for *any* k,k′ (the pairwise momentum balance is Q=Q, independent of k). Dense in
(k,k′), for exactly the same reason as Q=0. BGW confirms: `gx_sum_TDA` is still called
per (ik,ikp) block at finite Q (same `kernel_main.f90` loop), just with the Q-shifted
`mvc`/`mvpcp` and the finite-Q `vcoul`.

**v(Q+G) now includes G=0.** At Q≠0 the head term 4π/|Q|² is **finite** and must be
kept — there is no valence–conduction orthogonality protection because
`⟨c,k+Q|v,k⟩=0` no longer removes the G=0 Fourier component of a *momentum-transferring*
density. BGW is explicit: `get_vcoul(.not.energy_loss, .true., finiteq, qpg0_ind)`
(`mtxel_kernel.f90:689`) with the code comment *"we should never zero head of exchange
when using finite Q"* (`:688`), and `vcoul(iqpg0)` is only zeroed in the `energy_loss`
special case (`gx_sum`/`get_vcoul`, `mtxel_kernel.f90:1111-1113`). So the default
finite-Q exchange keeps the full v(Q+G) head; head-zeroing is the optical-limit (Q→0)
special case. This matches Henneke Eq (2-28) (`|k+G|²`, no exclusion) reducing to
(2-32) only at k=0.

### (iii) Architecture — what must change under the DENSE (my-position) contraction

Current dataflow (`kernel_dataflow_trace.md:29-52`, `bse_serial.py:59-78`): exchange =
per-k **S-encode** → `V_q0` contraction → per-k **decode**; direct W = 3-D FFT
convolution over q=k−k′. The finite-Q design doc (`designs/finite_q_bse.md`) proposes
finite-Q as "conduction k-remap + Q-tile swap, matvec byte-identical to Q=0."

Under the correct (DENSE) contraction, the required changes are:

1. **Fix Q=0 exchange to be k-summed FIRST (prerequisite).** Change the encode from
   `'kcvN,bcvk->bNk'` to `'kcvN,bcvk->bN'` (drop the k output axis — Σ over k′), and the
   decode from `'kcvM,bMk->bcvk'` to `'kcvM,bM->bcvk'` (broadcast the k-independent U(μ)
   to every k). `V_q0` contraction (`'MN,bN->bM'`) loses its k axis. This is a
   one-einsum-per-step change in `bse_serial.py:62-64` and the siblings
   (`bse_simple.py`, `apply_V_ring` in `bse_ring_comm.py:262-300`). The exchange
   encode/decode **become k-summed / k-broadcast**, not k-local. (D and W are untouched:
   D is genuinely δ_kk′; W's q=k−k′ convolution is already k-nonlocal and correct.)

2. **Add Q to the exchange only as: (Q-shifted pair density) + (Q Coulomb tile).**
   - Pair density gains a Q index / shifted-k read: `M_cvk(Q,ν)` uses the conduction
     orbital at the wrapped grid index k+Q. Because `psi_full` stores u_{n,k}(r_μ) at
     **every** k and at k-independent centroids (`kernel_dataflow_trace.md:169`),
     `u_{c,k+Q}(r_μ)` is already on disk — a gather `psi_c_Q[k]=e^{-i2πG_umk·s}·psi_full[kpQ_index[k]]`
     (`finite_q_bse.md` dataflow), no new ζ fit, no shifted NSCF. Valence stays at k.
   - Coulomb tile: read `V_Q = V_qmunu[Q_flat]` instead of `V_q0`, and **do not** inject
     the divergent q→0 head (the head rank-1 injection at `bse_io.py:468-513` is a Q=0-only
     branch); include G=0 directly since 4π/|Q|² is finite.
   The exchange encode is still `Σ_{k′}` (k-summed) — Q does not change that; it only
   changes *which* orbital the pair density reads and *which* tile multiplies S(ν). The
   output pair lands at the +Q-shifted conduction label (i)'s convention.

3. **W (direct):** structurally unchanged — the convolution kernel W̃_{k−k′,μν} is
   **Q-independent** (`finite_q_bse.md`); only the conduction ψ read is Q-shifted
   (valence at k, conduction at k+Q). The existing 3-D FFT over q=k−k′ is preserved; the
   loader supplies the Q-shifted conduction cache. (This part of the finite_q design is
   already correct.)

4. **Does Q=0 fall out as Q→0?** **Yes, and it must be built that way.** With (1) done,
   set Q=0: `M_cvk(0,ν)=M_cvk(ν)`, `V_Q → V_q0`, and G=0 drops back out by orthogonality
   (Henneke 2-32). One code path, Q as a parameter that defaults to 0. **Flag:** the
   current `finite_q_bse.md` "matvec byte-identical to Q=0" promise is only safe *after*
   step (1). If finite-Q is built as "conduction-remap on top of the *existing*
   k-diagonal Q=0 matvec," it **inherits the δ_kk′ bug at every Q** and produces a second
   parallel wrong path. The correct sequencing is: fix Q=0 to DENSE (k-summed encode),
   then finite-Q is genuinely the same path with a Q-shifted read and a tile swap — a
   single unified matvec, no parallel Q=0 special case. This is exactly the "no parallel
   old/new paths / single source of truth" discipline the design must honor.

---

## 5. Final position

**I cannot defend the k-block-diagonal exchange as the Q=0 BSE exchange for N_k > 1.**
The strongest steelman reduces to the true-but-narrow statement that k-diagonal exchange
is exact **iff N_k = 1** (molecule / Γ-only), which is the likely historical origin of
the code (a molecular ISDF matvec with a batch k axis grafted on) and not a defense of
its use on k-grids. Every attempt to make it correct for N_k > 1 either (S1, S4) confuses
locality-in-r or interaction-momentum-q with locality-in-k, (S2) tries to launder a
dropped N_k−1 blocks through a scalar prefactor that provably cannot do so, or (S3, S5)
invokes an analogy (supercell, Hartree) that, completed correctly, *requires* the Σ_{k′}
the code omits. The owner's Hartree equation `V^H = v·FFT[Σ_k|ψ_k|²]` contains the very
k-sum the code drops.

The owner's intuition is right about two separate, true things — only the q=0 tile enters
(a), and H^BSE is block-diagonal in the exciton COM momentum Q (the finite-Q remark).
Neither is claim (b). Claim (b) — δ_kk′ within a Q-block — is a physics-correctness bug in
every live matvec (`bse_serial.py:62-64`, `bse_simple.py:101-131`,
`apply_V_ring` `bse_ring_comm.py:262-300`, dead `bse_jax.py:108-121`), confirmed by
Henneke (2-16)/(4-5), by BerkeleyGW (`gx_sum_TDA` + the dense `(ik,ikp)` loop), and by
first-principles derivation. **Fix: k-summed encode / k-broadcast decode (the DENSE form),
which also makes finite-Q fall out as the Q→0 limit of one unified matvec.**

---

### Evidence index (file:line)
- Code under dispute: `sources/lorrax_D/src/bse/bse_serial.py:62-64` (and 60,64 prefactors);
  siblings `bse_simple.py:101-131`, `bse_ring_comm.py:262-300` (`apply_V_ring`),
  `bse_jax.py:108-121`.
- Henneke: `src/bse/context/Henneke-2020…md:169` (D has δ_{kk′}), `:173` (V_A, no δ),
  `:426` (Eq 4-3 ISDF V_A), `:443` (Eq 4-4 Σ_{k′}), `:454` (Eq 4-5 k-independent brace),
  `:271`/`:308` (Eqs 2-28/2-32, G=0 exclusion = claim (a)).
- BerkeleyGW dense loop: `BSE/kernel_main.f90:401,408-409,440-447` (per-(ik,ikp) blocks);
  exchange `BSE/gx_sum.f90:33-87` (esp. `:53,59-61,73-75`); head handling
  `BSE/mtxel_kernel.f90:673-674,688-689,1111-1113`.
- Audit: `archive/DEAD_CODE.md` §1.1 `:35-91`; `archive/files/kernel_dataflow_trace.md:29-52,234,368-388`.
- Finite-Q: `archive/designs/finite_q_bse.md` (reference kernel block, dataflow),
  `context/parallel_bse_algos.md:9,11-12,14-17,22,38-43` (conventions, W Q-independent,
  V^Q tile).
