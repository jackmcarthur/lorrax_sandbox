# Exchange-kernel adjudication — VERDICT

_2026-07-15. Three independent agents (one assigned to steelman each position +
one empirical), all landed **dense in (k,k′)**, high confidence — including the
k-diagonal steelman, which concluded its own position is defensible only for
Nk=1. Full writeups: `steelman_k_diagonal.md`, `steelman_dense.md`,
`empirical_bsemat.md` (dump commands included)._

## Verdict

The Q=0 BSE exchange kernel is **dense in (k,k′)**:

```
⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν} M*_cvk(μ) V_q0(μ,ν) M_c'v'k'(ν)    — no δ_kk'
```

The as-coded k-block-diagonal contraction (all four matvecs,
`bse_serial.py:62-64` reference) computes only the (k,k) diagonal of this matrix
with the 1/Nk prefactor retained. DEAD_CODE.md §1.1 (B1) stands.

## Reconciling the owner's argument — the two claims are separable

- **(a) "the object that gets contracted has a momentum shift of 0" — TRUE and
  undisputed.** Only the q=0 Coulomb tile `V_q0` enters; BGW does exactly this
  (`mtxel_kernel.f90` contracts the q=0 vcoul between pair densities). The q=0
  refers to the **interaction line**: every Q=0 pair (ck, vk) is individually
  momentum-neutral, so a q=0 line couples all k to all k′ — it imposes no
  k-selection rule.
- **(b) "therefore δ_kk′" — FALSE for Nk>1.** The δ_kk′ structure belongs to the
  **direct** term, where the two density factors carry phase e^{−i(k−k′)r} and
  select the tile W_{k−k′}. In the exchange, the Bloch phase cancels *within*
  each pair density (both states at the same k), so k and k′ run independently.
- **The Hartree analogy completes on the dense side.** V^H_q=0(G) =
  v(G)·FFT[Σ_k |ψ_nk|²](G) has the k-sum *inside* the FFT'd density. The
  exchange action is exactly a transition-Hartree: build
  n_X(r) = Σ_{k'c'v'} X_{c'v'k'} ψ_c'k' ψ*_v'k' (**summed over k′**), one q=0
  Hartree-like solve, then project at each k. The code is this analogy with the
  Σ_k′ deleted.

## Evidence highlights

1. **BGW's own kernel matrix on disk** (the decisive, paper-free check):
   `/mats/exchange` in `runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5` (Q=0,
   qflag=1) has off-diagonal k-blocks the same order as the diagonal — diag
   (ik=ikp=0) max 0.387 / mean 0.091; off-diag (0,1) max 0.269; (0,5) max 0.292;
   (0,63) max 0.151. 8×8 run: diag(0,0)=0.349, off(0,7)=|0.0403−0.117i|=0.124.
   Structurally varying and decaying with k-separation — not noise. Axis
   identity cross-checked against the head diagonal (=1.0 exactly) to rule out
   a transpose error. Writer `bsewrite.f90:476,516-517` allocates the full
   dense (…,ik,ikp) hyperslab by construction; `storage=0` = no symmetry folding.
2. **BGW source**: `kernel_main.f90:401-474` computes bsex for every (ik,ikp)
   pair; `distrib_kernel.f90:80` sizes the kernel as (Nk·Nc·Nv)².
3. **Henneke 2020** (the implementation's own reference): Eq. 4-5 has the
   explicit outer Σ_k′ and a k-free contraction bracket; Eq. 2-16 carries both
   k and k′ with no delta, while the D term explicitly writes δ_kk′ where one
   is meant.
4. **Thermodynamic limit**: for a normalized k-delocalized exciton the dense
   exchange expectation is O(1) (intensive); the coded form scales O(1/Nk) → 0.
   The TDA singlet-triplet splitting is pure exchange, so the bug makes it
   vanish under mesh refinement (~1/144 captured on a 12×12 mesh). This is also
   why the Si 4×4×4 SOC validation (exchange-insensitive manifold) passed at 3 meV.
5. **No normalization rescue**: the coded operator is a strict k-block-diagonal
   submatrix; no scalar prefactor restores Nk−1 dropped off-diagonal blocks.

## Finite-Q (owner requirement)

- The owner's statement "K^X_Q X_cvk → Y_cv,k+Q (or −Q on convention)" is the
  statement that **H^BSE is block-diagonal in the exciton COM momentum Q** —
  TRUE, and compatible with (not in tension with) intra-block density. The ±Q
  ambiguity is the pair-labeling anchor: +Q-on-electron (|vk, c k+Q⟩, the
  LORRAX finite_q design's choice) vs −Q-on-hole (BGW's mirror). Within one
  fixed-Q block, the (k,k′) exchange coupling is **dense**, through the single
  tile v(Q+G) — **including G=0** at Q≠0 (finite 4π/|Q|²; BGW
  `mtxel_kernel.f90:688`: "never zero head of exchange when using finite Q").
- **Architecture**: the correct fix makes the exchange *cheaper* and finite-Q
  *simpler*. Encode drops its spectator k — `'kcvN,bcvk->bN'` (S becomes a
  k-free ζ-space vector; one small all-reduce in the ring path) — decode
  broadcasts back at every k. Finite-Q then = read conduction ψ at wrapped
  k+Q (already on disk at centroids) + swap the tile V_q0 → V_Q (G=0 kept) +
  umklapp phase; Q=0 falls out as the Q→0 special case of ONE unified exchange
  path — no parallel Q=0/finite-Q code. NB: the prior `finite_q_bse.md` promise
  that the matvec is "byte-identical to Q=0" would have inherited the δ_kk′ bug
  at every Q; the B1 fix must land first, then that design's remap applies.

## Consequence for the program

B1 fix is **greenlit**: its own gated commit, validated against (i) a dense
2-k/N=144 reference and (ii) BGW `bsemat.h5` exchange blocks directly (now that
the layout mapping is documented in `empirical_bsemat.md`). Expect eigenvalue
shifts on singlet/bright states in any exchange-sensitive comparison after the
fix; the Si triplet-dominated manifold should move little (re-anchor gates
accordingly).
