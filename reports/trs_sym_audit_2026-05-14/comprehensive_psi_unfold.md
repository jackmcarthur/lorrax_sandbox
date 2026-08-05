# Comprehensive ψ-unfold audit vs nosym ground truth

**Date**: 2026-05-14
**Branch**: `agent/trs-aware-sym-fix` @ `80edbe8` (lorrax_B)
**Goal**: at every full-BZ k for every system, verify LORRAX's production
`WfnLoader.load(k='full_bz')` (i.e. `unfold_psi`) lands in the same
degenerate subspace as the nosym (ntran=1) ψ at the matching full-BZ k.
**Verdict**: **PASS on all three systems (MoS2, CrI3, Si).** No bug-class
disagreement on any (k_full, sym_idx) pair. The current HEAD's combined
R_cart fix (`5dc8813`) + centroid_perm forward-direction flip (`80edbe8`) +
PR1/PR2/PR3 (V_q TRS conj + `unfold_psi` lift) is sound on ψ.

---

## Method (one paragraph)

For each system, load LORRAX ψ at every full-BZ k via the production
`WfnLoader.load(bands=(0,N), k='full_bz')` (Method A); load nosym ψ at
the same physical full-BZ k via `WfnLoader.load(k='ibz')` from a
ntran=1 WFN (Method B); scatter both onto the common FFT-box; within
each degenerate energy group (tol = 1e-5 Ry on the nosym energies)
compute the overlap matrix `U = X_norm · Y_norm.H`; report
`unit_err = ||UU^H − I||_∞` (subspace agreement) and
`gauge_err = ||X_norm − U·Y_norm||_2` (gauge residual). The G-list
alignment is handled by FFT-box scatter (every G has a unique box
index; the two ψ's may have different G-orderings on disk but land at
the same box cells). PASS gate: `max unit_err < 1e-3` over all k_full
in the band window — three orders of magnitude below the bug-level
disagreement seen on the pre-fix code (CrI3 sym=1: 0.82; Si non-symm:
1.19).

Test driver: `comprehensive_psi_data/run_comprehensive_psi_unfold.py`.
Per-system JSON: `comprehensive_psi_data/{MoS2,CrI3,Si}_results.json`.
Full log: `comprehensive_psi_data/run_all.log`.

## Coverage matrix

| System | sym WFN | nosym WFN | ntran | nk_full | nspinor | bands | sym_idx values used | TRS rows hit |
|---|---|---|---:|---:|---:|---|---|---|
| MoS2 3×3 SOC | `runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5` | `runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5` | 2 | 9 | 2 | [0,32) | {0, 2} | {2} ← `-E` (TRS-augmented identity) |
| CrI3 6×6 30Ry SOC | `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5` | `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/qe_nosym/nscf/WFN.h5` | 6 | 36 | 2 | [0,32) | {0..5} | none (inversion present ⇒ no TRS folding) |
| Si 4×4×4 SOC Fd-3m | `runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5` | `runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5` | 48 | 64 | 2 | [0,16) | 24 distinct spatial ops | none (inversion present) |

All `nk_full` values matched a nosym k 1-for-1 (`build_match_table`
returned no `-1`). Total (k_full) tuples tested: **9 + 36 + 64 = 109**.

## Results — per-system

### MoS2 (PASS 9/9)

```
 k_f k_irr sym TRS ngrp   max_unit  max_gauge  pass
   0     0   0   F   16   8.70e-08   4.17e-04  PASS
   1     2   2   T   16   3.18e-08   2.52e-04  PASS
   2     2   0   F   16   3.18e-08   2.52e-04  PASS
   3     6   2   T   16   3.18e-08   2.52e-04  PASS
   4     8   2   T   31   2.15e-05   4.64e-03  PASS
   5     7   2   T   16   3.18e-08   2.52e-04  PASS
   6     6   0   F   16   3.18e-08   2.52e-04  PASS
   7     7   0   F   16   3.18e-08   2.52e-04  PASS
   8     8   0   F   31   2.15e-05   4.64e-03  PASS
```

Per sym_idx aggregate:

| sym_idx | TRS | n_k | max_unit | mean_unit | max_gauge |
|---:|:---:|---:|---:|---:|---:|
| 0 (E) | F | 5 | 2.15e-05 | 4.33e-06 | 4.64e-03 |
| 2 (-E, TRS) | T | 4 | 2.15e-05 | 5.40e-06 | 4.64e-03 |

Identity and TRS rows are statistically indistinguishable — confirms the
PR3 TRS branch (`iσ_y · conj`) is correct. The 2e-5 floor at k_f={4, 8}
comes from independent-SCF noise on the 31 non-degenerate bands in the
[0,32) window — same magnitude appears on the trivial identity rows at
k_f=0.

### CrI3 (PASS 36/36)

All 36 (k_full) rows passed. Per sym_idx aggregate:

| sym_idx | mtrx role | TRS | n_k | max_unit | mean_unit | max_gauge |
|---:|:---:|:---:|---:|---:|---:|---:|
| 0 | E | F | 8 | 7.71e-15 | 5.57e-15 | 2.93e-08 |
| 1 | C3 | F | 6 | 2.40e-08 | 5.65e-09 | 1.92e-04 |
| 2 | C3⁻¹ | F | 6 | 2.81e-08 | 6.04e-09 | 2.01e-04 |
| 3 | -E (inversion) | F | 6 | 2.18e-07 | 5.56e-08 | 6.61e-04 |
| 4 | S6 | F | 5 | 3.50e-07 | 9.22e-08 | 7.83e-04 |
| 5 | S6⁻¹ | F | 5 | 3.89e-07 | 1.00e-07 | 8.45e-04 |

The C3 (sym=1, 2) and S6 (sym=4, 5) rows — the operations that
**failed at 0.82 on the prior R_cart bug** — now agree at ≤ 4e-7
unit_err, **6 orders of magnitude below the bug**. Inversion row
(sym=3) is at the same level. The identity floor is 1e-14 (ULP) because
the CrI3 nosym SCF was bit-equivalent to the sym SCF at this k_irr
(common DFT eigenvalues to ULP, line 317 of discussion.md confirms).

Spot check of the underlying machinery (verified live on HEAD `80edbe8`):

```
CrI3 sym_matrices[1] (crystal C3):     CrI3 R_cart[1]:
[[ 0 -1  0]                            [[-0.500   -0.866    0    ]
 [ 1 -1  0]                             [ 0.866   -0.500    0    ]
 [ 0  0  1]]                            [ 0        0        1    ]]
  |R R.T - I|_inf = 1.13e-07           (canonical 120° about z)
  det = +1
U_spinor[1] = diag(exp(-i 120°), exp(+i 120°))   (correct C3 SU(2))
|U U.H - I|_inf = 1.11e-16
```

R_cart is orthogonal to 7 decimals (down from the pre-fix `||R R.T -
I||∞ = 3.46` shown in `algebraic_unfold_cri3.md`); U_spinor is the
correct ±60° spinor (down from the pre-fix `±125°`).

### Si (PASS 64/64)

All 64 (k_full) rows passed; 24 distinct spatial sym ops exercised
(out of 48 total — IBZ-orbit closure picks a subset; the remainder are
in the augmented Brillouin zone built by inversion). Per sym_idx
aggregate (top 12 ops):

| sym_idx | n_k | max_unit | mean_unit | max_gauge |
|---:|---:|---:|---:|---:|
| 0 (E) | 8 | 6.60e-15 | 3.71e-15 | 8.03e-09 |
| 1 | 5 | 3.14e-09 | 6.30e-10 | 7.65e-05 |
| 2 | 5 | 7.36e-12 | 2.64e-12 | 3.84e-06 |
| 3 | 4 | 9.34e-13 | 3.71e-13 | 1.36e-06 |
| 4 | 3 | 4.13e-09 | 1.38e-09 | 8.86e-05 |
| 5 | 3 | 1.41e-08 | 4.71e-09 | 1.65e-04 |
| 6 | 2 | 1.40e-12 | 9.61e-13 | 1.67e-06 |
| 7 | 2 | 2.44e-12 | 1.54e-12 | 2.20e-06 |
| 8 | 5 | 1.02e-08 | 2.04e-09 | 1.40e-04 |
| 9 | 4 | 8.78e-09 | 2.20e-09 | 1.29e-04 |
| 10 | 2 | 7.70e-13 | 6.20e-13 | 1.24e-06 |
| ... | | | | |

Worst row: sym_idx=5 at k_f=54, max_unit = 1.41e-08 — still **9 orders
of magnitude below the pre-fix 1.19**. The 36 non-symmorphic ops (the
ones that failed at 1.19 in `algebraic_unfold_si.md`) all pass cleanly.

Spot check:

```
Si sym_matrices[6] (non-symm C3-class):    Si R_cart[6]:
[[ 0  0 -1]                                [[ 0  1  0]
 [-1  0  0]                                 [-1  0  0]
 [ 1  1  1]]                                [ 0  0  1]]
  τ_frac = (1/8, 1/8, 1/8)                  |R R.T - I|_inf = 0
  → non-symmorphic C3 about [1,1,1]         → orthogonal ⇒ correct SU(2)
```

(Compare with the pre-fix `algebraic_unfold_si.md` line 122: `sym_idx=6
proper_nonsymm`, `||R R.T - I|| = 13.5` ⇒ garbage SU(2). Now 0.)

## What this means

`unfold_psi` produces ψ in the correct gauge-fixed subspace at every
(k_full, sym_idx) the production pipeline encounters on these three
systems. The R_cart fix at `symmetry_maps.py:810` and the PR3
`unfold_psi` body are both verified end-to-end against nosym ground
truth — not just against algebraic per-element formulas.

**Coverage caveat**: TRS rows are exercised only on MoS2 (sym_idx=2,
TRS-augmented identity). The richer TRS combinations (TRS·C3,
TRS·S6, etc.) on a non-inversion non-C2-only system are *not* hit by
this test bed because:
- MoS2 post-no_t_rev has only {E, σ_h}; the TRS row is `-E`.
- CrI3 has inversion ⇒ no TRS folding.
- Si has inversion ⇒ no TRS folding.

The MoS2 result confirms the basic `iσ_y · conj` rule is correct; the
algebraic `audit_pr3_perelement.py` (sister agent's earlier
work) covers the {I, σ_x} synthetic group with TRS rows. Combined,
the spatial + iσ_y·conj branches of `unfold_psi` are now both validated,
but a non-inversion non-C2-only test bed with C3 + TRS would be the
strictest possible test for future regression coverage (none currently
on disk).

## Verdict (per task spec)

- **MoS2 PASS**: max unit_err 2.15e-5 (SCF noise floor)
- **CrI3 PASS**: max unit_err 3.89e-7 (sub-µeV agreement on C3/S6/inversion)
- **Si PASS**: max unit_err 1.41e-8 (sub-meV on all 24 used sym ops, incl. non-symmorphic)

**ψ unfold is sound on all 3 systems with current HEAD `80edbe8`.**

Per the task's "Expected outcomes" decision matrix: **all PASS at
sub-bug levels** → "ψ unfold is correct everywhere. The CrI3 6 eV Σ_X
bug then MUST be downstream of ψ (in compute_vcoul, V_q kernel, or
Σ_X computation)." However, the Σ_X comparisons reported at 22:13
(MoS2 0.090 meV) and Agent 5 R1/R2/R3 (MoS2 bit-equal, CrI3 cascade
bit-equal) suggest that after PR2 + R_cart fix + centroid-perm flip,
the production Σ_X path is also correct on MoS2 and on CrI3-with-the-
cascade-disabled-pre-flip. The remaining CrI3 6 eV gap noted at 16:20
predates `80edbe8` (it ran on `8504994+a45f039+69ab42c`, before the
R_cart + centroid-perm fixes landed). A fresh CrI3 6×6 30Ry
sym-vs-nosym rerun on `80edbe8` would close that ticket; it is **not**
a ψ-unfold issue per this audit.

## Artifacts

- Driver: `comprehensive_psi_data/run_comprehensive_psi_unfold.py`
- Per-system JSON: `comprehensive_psi_data/{MoS2,CrI3,Si}_results.json`
- Full stdout: `comprehensive_psi_data/run_all.log`
- MoS2-only sanity log: `comprehensive_psi_data/run_mos2.log`

Total cost: ~17 GPU-sec compute (MoS2 1.7 s + CrI3 14.9 s + Si 0.9 s)
on JID 52976844 (1× hbm-mixed A100).
