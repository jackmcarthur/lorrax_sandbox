# Empirical adjudication: is BGW's BSE exchange kernel k-block-diagonal?

Role: the **empirical check** — no papers, only BGW's own on-disk kernel matrix
for the validated Si 4×4×4 run. Question (b) from the dispute: within the Q=0
kernel, is the exchange pair-coupling `δ_{kk'}` (LORRAX code) or dense in
`(k,k')` (audit DEAD_CODE §1.1)?

**Verdict: DENSE.** BGW allocates, computes, and writes a full `k × k'` exchange
block; the off-diagonal `ik≠ikp` exchange elements are nonzero and the same
order of magnitude as the diagonal. LORRAX's k-block-diagonal exchange
(`bse_serial.py:62-64`) drops physics BGW keeps. The audit is correct; the code
owner's claim (b) is wrong. Claim (a) — only the q=0 Coulomb tile enters — is
correct and not in dispute.

---

## 1. Dataset layout — pinned from spec + writer before touching data

Files: `runs/Si/04_si_4x4x4_bse/{00_bgw_bse, 00_bgw_bse_8x8, 01_bgw_bse_vcoul}/bsemat.h5`
(`00_bgw_bse_8x8_haydock` is a symlink to `00_bgw_bse_8x8`). Primary analysis on
`00_bgw_bse`.

Header scalars (`h5dump -d /bse_header/...`):

| field | value | meaning |
|---|---|---|
| flavor | 2 | complex |
| nvb, ncb | 4, 4 | restricted TDA (n1b=n2b=4) |
| ns / nspinor | 1 / 2 | spinor (SOC) run, so `nk·ns = nk` |
| nk | 64 | full-BZ k-points |
| qflag | 1 | **Q=0 calculation** |
| exciton_Q_shift | (0,0,0) | Q=0 confirmed |
| nblocks / storage | 1 / 0 | restricted; **no symmetry folding, full block stored** |

`storage=0` matters: the `k×k'` block is stored in full, not reconstructed from
symmetry — so off-diagonal elements read from disk are genuinely computed values.

**Spec** (`docs/docs_bgw/bsemat.h5.spec:177-199`): `/mats/exchange` is rank-7,
Fortran dim order `(flavor, n1b, n1b, n2b, n2b, nk·ns, nk·ns)`, and

    K^x(v,v',c,c',k,k') = ∫∫ ψ_ck(r)* ψ_vk(r) v(r−r') ψ_c'k'(r') ψ_v'k'(r')*

carries **both** k and k' with no `δ_{kk'}` — dense by definition of the stored
object.

**Writer** (`sources/BerkeleyGW/BSE/bsewrite.f90`) confirms the exact memory
order and that the two k-axes are independent:
- `bsewrite.f90:476` — `data(1,iv,ivp,ic,icp,ik,ikp) = dble(bsemat(...))`
  → Fortran layout `(flavor, iv, ivp, ic, icp, ik, ikp)`.
- `bsewrite.f90:516-517` — `offset(6)=(ik-1)+(is-1)*nkpt_co`,
  `offset(7)=(ikp-1)+(isp-1)*nkpt_co` → ik and ikp are separately-looped,
  independent hyperslab axes. BGW writes the whole dense `ik × ikp` matrix.

**Per-axis mapping** (written down before dumping — this is where pattern-match
fails). `h5ls` reports C order = reverse of Fortran. C-order shape
`{64, 64, 4, 4, 4, 4, 2}` maps to:

| C axis (h5dump `-s`/`-c` slot) | size | meaning | Fortran dim |
|---|---|---|---|
| 0 | 64 | **ikp** (k') | 7 |
| 1 | 64 | **ik** (k) | 6 |
| 2 | 4 | icp (c') | 5 |
| 3 | 4 | ic (c) | 4 |
| 4 | 4 | ivp (v') | 3 |
| 5 | 4 | iv (v) | 2 |
| 6 | 2 | flavor (re,im) | 1 |

So a fixed-k-pair band block is `-s "ikp,ik,0,0,0,0,0" -c "1,1,4,4,4,4,2"`
(256 complex pairs = 4⁴ band quads). The size-64 axes being the two outermost C
axes is unambiguous: the four size-4 axes are bands, the size-2 is flavor.

## 2. Dump commands and data

Login-node rule respected: **no bare python**; `module load lorrax_agent && lxstatus`
returned exit 1 (no active pool), so numerics done with `h5dump` subsets + `awk`
(a shell tool). Modulus reducer over a band block:

```bash
F=runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5
dumpblock () {  # args: dataset ikp ik   (C-order outer axes)
  h5dump -d /mats/$1 -s "$2,$3,0,0,0,0,0" -c "1,1,4,4,4,4,2" $F \
   | sed -n '/DATA {/,/}/p' | sed 's/^.*)://' | tr ',' '\n' \
   | grep -oE '[-+]?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?' \
   | awk 'NR%2==1{re=$1} NR%2==0{im=$1; m=sqrt(re*re+im*im);
          if(m>mx)mx=m; s+=m; c++} END{printf "max=%.4e mean=%.4e n=%d\n",mx,s/c,c}'
}
```

(Pitfall recorded: the `(i,j,...):` index prefix on each h5dump line contains
digits — the `sed 's/^.*)://'` strip is mandatory, else all datasets alias to the
index tuples and read identically. First pass without the strip gave bogus
`4.242641=3√2` "values"; corrected below.)

`00_bgw_bse` — |element| over each `k`-pair band block (256 pairs):

| block | exchange | head | body |
|---|---|---|---|
| **diag** ik=ikp=0 | max 3.87e-1, mean 9.1e-2 | 1.00e0, 6.3e-2 | 1.92e-1, 2.5e-2 |
| **diag** ik=ikp=5 | max 4.42e-1, mean 7.4e-2 | 1.00e0, 6.3e-2 | 2.36e-1, 1.9e-2 |
| off ik=0 ikp=1 | max 2.69e-1, mean 6.4e-2 | 4.36e-1, 7.7e-2 | 1.75e-1, 3.8e-2 |
| off ik=0 ikp=5 | max 2.92e-1, mean 6.5e-2 | 3.87e-1, 8.5e-2 | 1.54e-1, 4.1e-2 |
| off ik=3 ikp=17 | max 9.65e-2, mean 3.0e-2 | 1.26e-1, 3.5e-2 | 1.72e-1, 6.3e-2 |
| off ik=0 ikp=63 | max 1.51e-1, mean 5.5e-2 | 9.68e-2, 2.9e-2 | 1.45e-1, 5.2e-2 |

`00_bgw_bse_8x8` single-element cross-check (iv=ivp=ic=icp=0),
`-c "1,1,1,1,1,1,2"`:
- diag (0,0): `0.349148, −3.97e-22` → real, as a k-diagonal band-diagonal
  element should be.
- off (0,7): `0.0403, −0.117` → |·| = 0.124, plainly nonzero.
- off (0,40): `−0.00252, −0.000728` → |·| = 0.0026, small but nonzero;
  exchange decays with k-separation but does not vanish.

## 3. Sanity checks (did I transpose an axis?)

1. **Band-block count = 256 = 4⁴** for every dump → the four size-4 axes are
   fully spanned as bands; the two selected outer axes are the size-64 k-axes,
   not bands. If I had put a k-axis in a band slot the count would not be 256.
2. **head diagonal max = exactly 1.000000, mean = 0.0625 = 1/16** independent of
   which diagonal k → the head (macroscopic q→0 direct term) has the expected
   identity-like structure on the k-diagonal, confirming axis 0==axis 1 really is
   "same k" there. Off-diagonal head is a different, smaller number → the two
   outer axes are genuinely distinct k indices.
3. **Direct (head/body) is also dense in k** (off-diagonal nonzero) — expected,
   since `W(r,r')` couples k,k' too. Exchange and direct share the same k-axis
   structure, so no exchange-specific transpose slipped in.
4. Off-diagonal magnitudes **vary with the k-pair** (0.27 at ikp=1 → 0.097 at
   (3,17) → 0.15 at ikp=63), i.e. structured k-dependence, not a constant
   broadcast artifact and not floating-point noise (noise would be ~1e-16, not
   1e-1).

## 4. Source-level confirmation (kernel main loop)

`sources/BerkeleyGW/BSE/kernel_main.f90`: outer `do ik=1,qg%nf` over the **full**
BZ (`:300`) × distributed `ikp=peinf%ikp(...)` (`:409`); `bsex(:,:,:)` is
accumulated per `(ik,ikp,ic,icp,iv,ivp)` (`:461-478`) and handed to `bsewrite`
(`:557-560`). Exchange is computed for every `(ik,ikp)` pair, not only
`ik==ikp`. Line 12 notes exchange uses the "proper" part of the Coulomb (G=0
excluded at Q=0) — that is claim (a) (the q=0 tile, head-excluded), and it is
orthogonal to the dense-k structure.

---

## 5. Why the owner's Hartree analogy proves the *opposite* of claim (b)

The owner: exchange is "the same physics as the Hartree interaction,
`V^H_{q=0}(G) = v_{q=0}(G)·FFT[Σ_k |ψ_nk(r)|²](G)`." Correct — and note the
`Σ_k`. In one-body Hartree you build the density `ρ(G) = Σ_{k'} ⟨k'|e^{-iGr}|k'⟩`
(**a sum over all k'**), multiply by `v(G)`, then contract with a *single*
orbital's density to get its potential. The two-body BSE exchange kernel is the
same object promoted to a matrix:

    (K^x X)_{cvk} = Σ_{c'v'k'} [ M*_{cvk}(G) · v(G) · M_{c'v'k'}(G) ] X_{c'v'k'}
                  = M*_{cvk}(G) · v(G) · [ Σ_{c'v'k'} M_{c'v'k'}(G) X_{c'v'k'} ]

The bracket `Σ_{c'v'k'} M_{c'v'k'}(G) X_{c'v'k'}` **is** the Hartree-style
generalized density summed over all k' (the exact analog of `Σ_k |ψ_k|²(G)`).
The single index k survives only on the *output projection* `M*_{cvk}` — the
"which orbital feels the potential" index — never as a constraint `k' = k` on the
accumulation. Claim (a) (only `v_{q=0}(G)`) is exactly right; but the q=0 line
still couples every `(cvk)` to every `(c'v'k')`. The analogy demands the k'-sum
the code omits. Momentum-**shift**-zero (one Coulomb tile) ≠ k-**block**-diagonal
(`δ_{kk'}`).

The LORRAX code (`bse_serial.py:62-64`, and the same pattern in the other three
matvecs):

```python
S_V    = einsum("kcvN,bcvk->bNk", M, X) / sqrt_nk   # keeps k → per-k density
U_V    = einsum("MN,bNk->bMk", V_q0, S_V)
V_term = einsum("kcvM,bMk->bcvk", conj(M), U_V) / sqrt_nk
```

keeps `k` on `S_V` (`->bNk`), so the accumulated "density" is built from a single
k and fed back to that same k. That is precisely the `(k,k)` diagonal of the
dense kernel, with the `Σ_{k'}` dropped — while the `1/Nk` prefactor is retained
unchanged. It is not a normalization convention; it discards the off-diagonal
blocks that Section 2 shows are the same magnitude as the diagonal.

---

## 6. Required section — H^{Q≠0}

### (i) Finite-Q exchange in the pair basis, and reconciling the +Q/−Q convention

Fixed-Q pair basis `|v k, c k+Q⟩`: hole at `(v,k)`, electron at `(c,k+Q)`,
amplitude `X^Q(v,c,k)` **labeled by the hole momentum k** (LORRAX convention,
`parallel_bse_algos.md:9`; BGW mirrors it — shift on the valence state, COM
`−Q`, `bgw_fine_grid_reference.md:466-475` — a pure relabeling). Kernel
(`designs/finite_q_bse.md`, Rohlfing–Louie):

    K^x_Q(vck, v'c'k') = Σ_G M*_{vc}(k,Q,G) v(Q+G) M_{v'c'}(k',Q,G)
    M_{nn'}(k,Q,G)     = ⟨n,k+Q| e^{i(Q+G)·r} |n',k⟩

**Where the ±Q ambiguity enters:** the sign of the COM label flips with which
particle carries the shift (electron at k+Q ⇒ COM +Q; hole shifted ⇒ COM −Q).
It is a labeling choice, not different physics; fix "label by hole k, electron at
k+Q" and it is settled.

**Is the owner's "X_cvk → Y_cvk+Q" the same as Q-block-diagonality?** Partly, and
the reconciliation is the crux. There are two *independent* axes:

- **Q selects the block.** H^BSE is block-diagonal in Q: amplitudes of different
  COM momentum never mix. Within one block, both input `X^Q` and output `Y^Q`
  live in the same space `{(v,c,k) : all k}` at fixed Q. This is the true,
  correct "Q-block-diagonal" statement.
- **k,k' index densely within the block** (Section (ii)).

The owner's `X_cvk → Y_cvk+Q` is bookkeeping for *where the electron sits*
(k+Q), i.e. it names the output pair by the electron momentum while naming the
input by the hole momentum. Under a consistent hole-k labeling the operator does
**not** translate the amplitude — it maps `{X^Q(vck)}_{all k}` to
`{Y^Q(vck)}_{all k}`. So "X_cvk→Y_cvk+Q" **is** the Q-block-diagonality
statement (Q in ⇒ Q out, no Q-mixing) *once the electron/hole labels are made
consistent* — and it is **not** a statement about k,k' coupling inside the block.
At Q=0 the two axes look conflatable because `k+Q=k`, which is exactly the trap:
fixing the COM (Q=0) is misread as fixing the pair momentum (`δ_{kk'}`). They are
different constraints. The owner is right that H is Q-block-diagonal; that does
not make it k-block-diagonal.

### (ii) Within one fixed-Q block: dense or diagonal in (k,k')?

**Dense** — identical structure to Q=0. Pair densities `⟨c,k+Q|e^{i(Q+G)r}|v,k⟩`;
Coulomb `v(Q+G)` now **includes G=0** (finite `4π/|Q|²`, no head exclusion, no
rank-1 injection — except in the `energy_loss`/optical special case,
`kernel_main.f90:287-289`). The element `M*_{vck}(Q,G) v(Q+G) M_{v'c'k'}(Q,G)` is
nonzero for all k,k'. Section 2's Q=0 evidence is the `Q→0` limit of exactly this
object, so the empirical dense-in-k result carries to Q≠0.

### (iii) Architecture — what changes under the dense (correct) contraction

Current exchange dataflow (`kernel_dataflow_trace.md`): per-k S-encode →
`V_q0` contract → per-k decode; W via 3-D FFT convolution over `q=k−k'`.

**Q=0 fix (the same edit that enables finite-Q).** Make the exchange encode
k-**summed** and the decode a broadcast:

```python
S_V    = einsum("kcvN,bcvk->bN",  M, X) / sqrt_nk   # drop k: global Σ_{k'} density
U_V    = einsum("MN,bN->bM",      V_q0, S_V)
V_term = einsum("kcvM,bM->bcvk",  conj(M), U_V) / sqrt_nk
```

The intermediate `S_V` loses its k-axis (a reduction over all `(k',c',v')`); the
decode broadcasts the shared length-`n_rmu` vector back to every k. This is
strictly **cheaper** (one `[b,N]` vector, not `[b,N,k]`) — dropping the
off-diagonal was never a performance win, just wrong physics. Applies identically
to all four matvecs.

**Finite-Q (H^{Q≠0}) under this structure:**
- **Pair amplitude gains a shifted-k read, not a new contraction shape.**
  `M^Q[k,c,v,μ]` is built from `ψ_c[k+Q]` (conduction gathered at the Q-rolled
  k-index) and `ψ_v[k]`. Only a k-axis gather/roll is added; einsum shapes are
  unchanged.
- **`V_q0` → `V_Q` tile.** Read the `q=Q` slice of the on-disk
  `V_qmunu(nq,μ,ν)` tensor (all q tiles already on disk, `bse_io.py:389-411`;
  today only the q=0 slice is read, `:460`). `Ṽ^Q_μν` includes G=0; the q→0 head
  injection (`bse_io.py:468-513`) becomes the `Q=0` special case only.
- **Exchange encode/decode stay k-summed** — the *same* dense structure as the
  Q=0 fix. `S^Q[b,ν]=Σ_{k',c',v'} M^Q_{c'v'k'}(ν) X^Q[...]`; `U=Ṽ^Q S^Q`;
  `Y^Q[b,cvk]=M^Q*_{cvk}(μ) U[b,μ]`. No k-locality returns at finite Q.
- **W (direct):** conduction read at `k+Q`, convolution kernel `W̃` is
  Q-independent (`designs/finite_q_bse.md`), `q=k−k'` loop unchanged.
- **Which tensors gain a Q index:** `M` (via shifted-k read of `ψ_c`), the
  Coulomb tile (`V_q0`→`V_Q`); the FFT `W̃` does **not**.

**Q=0 falls out as Q→0, single path.** `Q=0 ⇒ k+Q=k` ⇒ `M^Q→M`, `V_Q→V_q0+head`,
`W` unchanged. The k-summed exchange encode is *required* for Q=0 correctness
anyway (Sections 2, 5), so the Q=0 bug-fix and the finite-Q generalization are
the **same** code change. Flag: any design that keeps the current per-k Q=0
encode and bolts on a separate dense finite-Q path would be two parallel paths
for one operator — reject it. One k-summed exchange encode/decode, parameterized
by (Q-shift on `M`, Q-tile of V), covers both.

---

## Appendix — reproduction

```bash
F=runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5
h5ls -r $F | grep -E 'mats|exchange'          # shape {64,64,4,4,4,4,2}
h5dump -d /bse_header/kpoints/qflag $F         # 1  (Q=0)
h5dump -d /bse_header/kpoints/exciton_Q_shift $F   # 0 0 0
h5dump -d /bse_header/params/storage $F        # 0  (full, unfolded)
# diagonal band block:  -s "ik,ik,..."   off-diagonal: -s "ikp,ik,..."
h5dump -d /mats/exchange -s "0,0,0,0,0,0,0"  -c "1,1,4,4,4,4,2" $F   # ik=ikp=0
h5dump -d /mats/exchange -s "63,0,0,0,0,0,0" -c "1,1,4,4,4,4,2" $F   # ik=0 ikp=63
h5dump -d /mats/exchange -s "7,0,0,0,0,0,0"  -c "1,1,1,1,1,1,2" \
   runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/bsemat.h5                  # 0.0403,-0.117
```

Pool check `module load lorrax_agent && lxstatus` → exit 1 (none active); all
numerics via `h5dump` + `awk`, no bare python on the login node.
