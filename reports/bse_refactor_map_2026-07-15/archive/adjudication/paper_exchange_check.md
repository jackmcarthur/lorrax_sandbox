# BSE exchange kernel — verification against the published ISDF-BSE paper

_2026-07-16. Verifies the B1 "dense (k,k′) exchange" fix (landed on
`agent/bse-phase2`, HEAD `5d3819f`) against the equations of the paper the
implementation descends from._

## Source used

**Paper:** F. Henneke, L. Lin, C. Vorwerk, C. Draxl, R. Klein, C. Yang,
_"Fast optical absorption spectra calculations for periodic solid state
systems,"_ Communications in Applied Mathematics and Computational Science
**15**(1), 89–113 (2020); preprint **arXiv:1907.02827**. Authors/venue/title
confirmed by web search; reference Julia implementation `fhenneke/BSE_k_ISDF.jl`.

The published PDF (arXiv:1907.02827) is a binary stream that the fetch model
could not OCR. **I worked from the local digest**
`sources/lorrax_A/src/bse/context/Henneke-2020-Fast optical absorption spectra c.md`,
which is a Mathpix OCR of the _published_ CAMCoS article — it carries the
journal header, the verbatim abstract, and the paper's own `\tag{}` equation
numbers (2-14…2-20, 4-1…4-7). All equation numbers below are the paper's.

---

## 1. The paper's exchange expressions (verbatim, with indices explicit)

### 1.1 Hamiltonian block structure — Eq. (2-15)

```
                ⎡  D + 2 V_A − W_A        2 V_B − W_B      ⎤
H_BSE =         ⎢                                          ⎥
                ⎣ −2 V̄_B + W̄_B      −D − 2 V̄_A + W̄_A ⎦
```

Note the **explicit factor of 2 on the exchange** blocks (`2 V_A`, `2 V_B`).
TDA (paper §2.2, digest line 180): "_Within the … Tamm-Dancoff approximation …
both V_B and W_B are neglected_", so the TDA Hamiltonian is
`H_BSE = D + 2 V_A − W_A` (digest line 439; the "W_B" printed there is an OCR
slip for W_A — the A-block screened term).

### 1.2 Bare-exchange integral — Eq. (2-16), and periodic form Eq. (2-17)

```
V_A(i_v i_c k , j_v j_c k′) = ∫∫  ψ̄_{i_c k}(r) ψ_{i_v k}(r)  V(r,r′)
                                    ψ̄_{j_v k′}(r′) ψ_{j_c k′}(r′)  dr dr′        (2-16)

                            = (1/N_k²) ∫∫  ū_{i_c k}(r) u_{i_v k}(r)  V(r,r′)
                                    ū_{j_v k′}(r′) u_{j_c k′}(r′)  dr dr′         (2-17)
```

Conjugation, per element:
- **Row / output (i, at r):** `ū_{i_c} u_{i_v}` = **conj(conduction)·valence**.
- **Col / input (j, at r′):** `ū_{j_v} u_{j_c}` = **conj(valence)·conduction**.
- Digest line 194: the Bloch phase `e^{ik·r}` cancels within each pair density,
  so **V_A/V_B carry no `e^{-i(k-k′)}` phase** (only W_A/W_B do). ⇒ the exchange
  couples every `k` to every `k′` through the single `q=0` interaction — **no
  `δ_{kk′}`**.

`V_B` (Eq. 2-20) is `V_A` with the **input** pair density conjugated
(`ū_{j_c k′} u_{j_v k′}` instead of `ū_{j_v k′} u_{j_c k′}` — c↔v swapped in the
ket).

### 1.3 ISDF form of V_A — Eq. (4-3), tile Eq. (4-1)

```
V_A(i_v i_c k, j_v j_c k′) ≈ (1/N_k) Σ_{μν}^{N_μ^V}
        ū_{i_c k}(r̂_μ) u_{i_v k}(r̂_μ)   Ṽ_{A,μν}   ū_{j_v k′}(r̂_ν) u_{j_c k′}(r̂_ν)   (4-3)

Ṽ_{A,μν} = 𝒱(ζ_μ^V, ζ_ν^V) = (1/N_k) ∫∫ ζ̄_μ^V(r) V(r,r′) ζ_ν^V(r′) dr dr′        (4-1),(2-18)
```

Prefactor bookkeeping: **one explicit `1/N_k`** in (4-3), and the tile
`Ṽ_{A}` carries a **second `1/N_k`** internally (Eq. 2-18) → net `1/N_k²`,
consistent with (2-17). The tile conjugates its **first (μ)** index (`ζ̄_μ`).

### 1.4 Fast matvec application of V_A — Eq. (4-4)/(4-5)

```
[V_A X](i_v i_c k) = Σ_{j_v j_c k′} V_A(i_v i_c k, j_v j_c k′) X(j_v j_c k′)       (4-4)

= (1/N_k) Σ_μ ū_{i_c k}(r̂_μ) u_{i_v k}(r̂_μ)                       ┐ DECODE (output k)
     · { Σ_ν Ṽ_{A,μν}                                             ├ COULOMB (μ←ν)
         · ( Σ_{k′} ( Σ_{j_c} u_{j_c k′}(r̂_ν)                     ┐
                      ( Σ_{j_v} ū_{j_v k′}(r̂_ν) X(j_v j_c k′) )))} ┘ ENCODE (k′-SUMMED)  (4-5)
```

Per-element structure of (4-5):

| Stage | Paper expression | Which orbital conj. | k′ handling |
|---|---|---|---|
| **Encode** `S(ν)` | `Σ_{k′,j_c,j_v} u_{j_c k′}(r̂_ν) ū_{j_v k′}(r̂_ν) X(j_v j_c k′)` | **valence** `ū_{j_v}`; conduction NOT | **explicit `Σ_{k′}`** → S is k-free |
| **Coulomb** | `Σ_ν Ṽ_{A,μν} S(ν)` | tile conj. on μ | — |
| **Decode** `[V_AX](i k)` | `(1/N_k) Σ_μ ū_{i_c k}(r̂_μ) u_{i_v k}(r̂_μ) · (…)_μ` | **conduction** `ū_{i_c}`; valence NOT | broadcast to **every** output k |

Digest line 458 states the encode "_perform[s] contractions over j_v, j_c, and
**k′**_" first, producing "_a quantity that only depends on r̂_ν_" — i.e. the
paper's own text confirms the encode is **k′-summed into a k-free ζ-space
vector**, exactly the B1 fix. (Contrast W_A Eq. (4-6): there the `k′`-sum is a
**convolution** `W̃_{k−k′}` — that is where `δ`-like `k`-locality lives, not in
the exchange.)

Define, in the code's notation, the **canonical pair density**
`M_{cvk}(μ) ≡ conj(u_{ck}(r̂_μ)) u_{vk}(r̂_μ)` = conj(cond)·val. Then the paper's
row/output density = `M`, and its col/input density = `conj(M)`. So the paper's
operator is:

```
PAPER:   [V_A X](out) = (1/N_k) Σ_μ  M_out(μ)  Σ_ν Ṽ_{A,μν}  Σ_in conj(M_in(ν)) X(in)
         → M on the OUTPUT (decode), conj(M) on the INPUT (encode).
```

---

## 2. The landed code — composed exchange expressions

Every live exchange path builds the **same** canonical density
`M[k,c,v,μ] = Σ_s conj(ψ_c[k,c,s,μ]) ψ_v[k,v,s,μ]`
(`bse_serial.compute_pair_amplitude`, `bse_serial.py:12-14`) — the spinor index
`s` is **traced here** (see §4). All five compose:

```
CODE:    S(ν)   = (1/√N_k) Σ_{k,c,v}  M_{cvk}(ν)  X(b,c,v,k)          ENCODE — k SUMMED
         U(μ)   =           Σ_ν  V_q0(μ,ν)  S(ν)                       COULOMB
         VX(cvk)= (1/√N_k) Σ_μ conj(M_{cvk}(μ)) U(μ)   [∀ k]           DECODE — broadcast
         → conj(M) on the OUTPUT (decode), M on the INPUT (encode).
```

The two `1/√N_k` compose to the single `1/N_k` the dense formula requires; the
second `1/N_k` of (2-17) lives inside `V_q0` (the tile analogue of `Ṽ_A`).

| # | Implementation | file:line | Encode `S` | Coulomb `U` | Decode `VX` |
|---|---|---|---|---|---|
| 1 | `bse_serial.apply_bse_hamiltonian_single_device` | `bse_serial.py:52-54` | `einsum("kcvN,bcvk->bN", M, X)/√nk` | `einsum("MN,bN->bM", V_q0, S)` | `einsum("kcvM,bM->bcvk", conj(M), U)/√nk` |
| 2 | `bse_simple.build_bse_simple_matvec` | `bse_simple.py:89-133` | `M_Y=conj(ψc)ψv`; `einsum("kcvN,bcvk->bN")/√nk` | `einsum("MN,bN->bM", V_q0, S)` | `M_X=conj(ψc)ψv`; `einsum("kcvM,bM->bcvk", conj(M_X), U)/√nk` |
| 3 | `bse_ring_comm.apply_V_ring` (A-block) | `bse_ring_comm.py:233-268` | ring: `A=Σ conj(ψc)·(ψv·X)`; `S=Σ_k A /√nk` (l.255) | `einsum("MN,bN->bM",V_q0,S)`+`psum` (l.257-8) | `M_X=conj(ψc)ψv`; `einsum("kcvM,bM->bcvk",conj(M_X),U)`; `/√nk` (l.264-8) |
| 4 | `bse_stack_matvec` V term | `bse_stack_matvec.py:138-145` | `einsum("kcvN,bcvk->bN", M_Y, X)/√nk` | `einsum("MN,bN->bM", V_q0, S)` | `einsum("kcvM,bM->bcvk", conj(M_X), U)/√nk` |
| 5 | dense gate `_build_dense_H` → `Kx` | `test_bse_dense_reference.py:66,72-73` | `Kx(cvk,CVK)=Σ conj(M_cvk) V_q0 M_CVK / nk` (built as full (k,k′) matrix) | — | — |

**B-block exchange** (`build_bse_ring_matvec_full`): `_apply_B`
(`bse_ring_comm.py:582-588`) calls `apply_V_ring_B`
(`bse_ring_comm.py:503-522`), which invokes `apply_V_ring` with **conjugated**
encode orbitals `conj(psi_c_Y), conj(psi_v_Y)` → encode density becomes
`conj(M)` on the input while the decode still uses `conj(M_X)`, i.e.
`V_B(out,in) = Σ conj(M_out) V_q0 conj(M_in)`. That is `V_A` with the **input**
pair conjugated — structurally the paper's `V_B` (Eq. 2-20). The full matvec
assembles `S=[[A,B],[−B†,−A†]]` (`bse_ring_comm.py:590-601`), matching the sign
pattern of Eq. (2-15).

**Dense-gate check on disk:** `test_full_H_matches_dense` /
`test_DV_matches_dense` (`test_bse_dense_reference.py:183-211`) assert
`matvec(X) == H @ X` for serial + simple + ring, where `H = diag(D)+Kx−Kd` uses
the row-conjugated `Kx` above. Passing ⇒ all four matvecs are self-consistent
with one dense `(k,k′)` exchange matrix.

---

## 3. Term-by-term agreement (paper Eq. 4-3/4-5 vs. code)

| Property | Paper (4-3)/(4-5) | Code (all 5 paths) | Agree? |
|---|---|---|---|
| **k′-sum in encode** | explicit `Σ_{k′}` (l.454,458) → S k-free | `einsum("kcvN,bcvk->bN")` / ring `Σ_k A` — **k summed out** | ✅ |
| **Dense in (k,k′), no δ** | phase cancels (l.194); couples all k↔k′ | decode broadcasts U to every k; no `δ_{kk′}` | ✅ |
| **Single V_q0 (q=0) tile** | one `Ṽ_{A,μν}` (4-1) | one `V_q0[μ,ν]` | ✅ |
| **Explicit prefactor** | one `1/N_k` out front (tile carries 2nd) | `1/√N_k · 1/√N_k = 1/N_k` (tile = V_q0) | ✅ |
| **Coulomb contraction** | `Σ_ν Ṽ_{A,μν} S(ν)` (μ←ν) | `einsum("MN,bN->bM", V_q0, S)` | ✅ |
| **Conjugation SIDE** | conj on **INPUT** (encode); M on OUTPUT | conj on **OUTPUT** (decode); M on INPUT | ⚠ transpose — see §3a |
| **Factor of 2 on V_A** | `2 V_A` in H (2-15) | **no ×2** in any matvec | ⚠ intentional — see §4 |
| **B-block (non-TDA)** | `2 V_B − W_B`; V_B = V_A w/ ket c↔v conj | `V_B − W_B`; `apply_V_ring_B` conj's encode ψ | ✅ struct. (×2 per §4) |
| **TDA** | drop V_B, W_B; A = D+2V_A−W_A | TDA matvecs = D + V − W | ✅ struct. (×2 per §4) |

### 3a. Conjugation side — transpose convention, not an error

Paper: `V_A(out,in) = Σ M_out V_q0 conj(M_in)`. Code:
`Kx(out,in) = Σ conj(M_out) V_q0 M_in`. These place the complex conjugate on
**opposite** sides, so `Kx_code = V_A_paper^T` (`= conj(V_A_paper)`, since V_A is
Hermitian — paper l.180 "V_A and W_A are Hermitian"). A Hermitian matrix and its
transpose have **identical (real) eigenvalues**; the eigenvectors are complex-
conjugated, and `ε₂(ω)` (Eq. 4-7) is invariant under that global conjugation.

Why the code's side is the right one: the physical exchange element is
`⟨c v k| K^x |c′v′k′⟩` with the **bra (output) pair density conjugated** — the
BGW / Rohlfing–Louie convention. The code conjugates the output (`conj(M_out)`),
matching BGW; the paper's Eq. (4-3) typesets the conjugate on the input. This is
confirmed empirically: `STATUS.md:53,67` reports the pre-B1 code (same
conjugation convention) reproduced BGW eigenvalues to **~3 meV** and total
`‖d‖²` to **machine precision**. **Verdict: convention, not a discrepancy — and
the code's convention is the BGW/physical one.**

---

## 4. The missing factor of 2 — intentional, correct for nspinor=2

The paper's `2 V_A` (Eq. 2-15) is the **spin multiplicity of the singlet
channel** in a spin-restricted (`nspinor=1`, spin-degenerate) formulation: the
paper's `V_A` (Eq. 2-16) integrates spinless orbitals, and the ×2 restores the
spin sum for the singlet exchange (the triplet has 0). The code carries **no
×2** (`bse_serial.py:70`, `bse_simple.py:177`, `bse_stack_matvec.py:151`, and
the dense reference `H = diag(D)+Kx−Kd`, `test_bse_dense_reference.py:91`).

This is **intentional and correct** for LORRAX's regime, cross-checked against:

- `kernel_dataflow_trace.md:311-319`: "_Exchange M is spinor-traced at same k
  (`kcsm,kvsm->kcvm`) … **No singlet factor 2 on K^x and no n_spin switch in any
  matvec: correct for nspinor=2 spinor runs** (the validated Si-SOC case)_".
- `kernel_dataflow_trace.md:307-310`: nspinor is inferred from the WFN;
  **there is no collinear `nspin=2` path in `bse/`** — LORRAX BSE runs are
  spinor (SOC) only.
- The spin sum is done **microscopically**: `compute_pair_amplitude`
  (`bse_serial.py:12-14`) traces the 2-component spinor index `s`
  (`Σ_s conj(ψ_c) ψ_v`). With an explicit spinor trace there is no spin
  degeneracy to multiply by — the ×2 would **double-count** and be wrong.
- `n_spin / n_spinor` enter only as the `ε₂` prefactor at the absorption stage
  (`16π²/(V·N_k·n_spin·n_spinor)`, `STATUS.md:122`,
  `kernel_dataflow_trace.md:126,252-253`), user-supplied, defaults
  `n_spin=1, n_spinor=2`.

**Known caveat (not triggered here):** a scalar `nspinor=1` WFN run would get a
triplet-like (unscaled) exchange with no guard — logged as **C4** in
`kernel_dataflow_trace.md:315-318,483-486` and `DEAD_CODE.md`. That is a
documented limitation of the spinor-only codebase, not a defect in the landed
nspinor=2 path.

**Spinor-trace placement:** the paper has no spinor index; the code's `Σ_s` sits
inside `M` (both encode and decode), so the exchange sees a single spin-traced
scalar density per (c,v,k,μ). Placement is consistent across all five paths.
(The direct `W` term traces the conduction line `t` and valence line `s`
separately against a scalar `W_{μν}` — charge-channel screening only,
`kernel_dataflow_trace.md:311-314`; out of scope for the exchange check.)

---

## 5. Verdicts

| Implementation | Verdict |
|---|---|
| `bse_serial.apply_bse_hamiltonian_single_device` (`bse_serial.py:52-54`) | **AGREES** — dense k′-summed encode, single V_q0, `1/N_k`. Conjugation on output = BGW/physical convention (paper's transpose); no ×2 correct for nspinor=2. |
| `bse_simple.build_bse_simple_matvec` (`bse_simple.py:89-133`) | **AGREES** — identical composed exchange to serial. |
| `bse_ring_comm.apply_V_ring` (A-block, `bse_ring_comm.py:233-268`) | **AGREES** — ring encode k-sums (l.255), single V_q0 (l.257), broadcast decode (l.265); same convention/prefactor. |
| `bse_ring_comm` B-block exchange `apply_V_ring_B` (`bse_ring_comm.py:503-522`, via `_apply_B` l.583) | **AGREES (structural)** — `conj(M)` on both encode/decode = paper's V_B (ket c↔v conj, Eq. 2-20); assembled as `[[A,B],[−B†,−A†]]` per Eq. (2-15). Same ×2-absent (nspinor=2) rationale. |
| `bse_stack_matvec` V term (`bse_stack_matvec.py:138-145`) | **AGREES** — identical composed exchange to serial. |
| dense gate `Kx` (`test_bse_dense_reference.py:66,72-73`) | **AGREES** — dense `(k,k′)` matrix `Σ conj(M) V_q0 M / N_k`, no `δ_{kk′}`; self-consistent with the matvecs (gate passes) and with the paper's structure (row-conjugated = physical convention). |

**Overall: the landed B1 exchange fix agrees with Henneke et al. (2020) on every
load-bearing structural point** — the k′-sum in the encode (Eq. 4-5), density in
(k,k′) with no `δ_{kk′}` (Eq. 2-17 phase cancellation), the single q=0 tile
(Eq. 4-1), the Coulomb contraction, and the single explicit `1/N_k` prefactor.
Two deviations from the paper's _literal_ formulas are both deliberate and
correct: (i) the complex conjugate sits on the output/bra density (the BGW /
physical convention; the paper writes the Hermitian transpose — identical
spectrum and ε₂), and (ii) the paper's `×2` singlet factor is absent because
LORRAX BSE is nspinor=2-only and traces the spin inside the pair density, where
a `×2` would double-count. Both are documented (`STATUS.md`,
`kernel_dataflow_trace.md`) and the convention is validated against BGW to 3 meV.
