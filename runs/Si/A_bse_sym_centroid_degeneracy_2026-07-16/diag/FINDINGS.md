# Where the Si BSE symmetry actually breaks — verdict + numbers trail

**Date:** 2026-07-16 · sym-arm restart `work_sym/tmp/isdf_tensors_792.h5`, old-arm
`work_old/tmp/isdf_tensors_960.h5` · lorrax_A @ `agent/bse-phase2` · 1 GPU (salloc
JID 56001468, released). Scripts: `diag/rung*.py`, runner `diag/run_probe.sh`.

## Verdict

The "broken symmetry" is NOT in LORRAX's GW/ISDF ingredients. DFT energies and
ψ-at-centroids are symmetry-covariant to machine precision, including the full
degenerate multiplets at Γ. The suspected causes — centroid placement, the ψ
IBZ→full-BZ unfold, the ζ-fit — are refuted by direct measurement.

**The dominant symmetry-breaker is band-window truncation of degenerate
multiplets at high-symmetry k-points.** The dense-BSE analysis uses a fixed
nv/nc transition window (here 4v4c) that keeps only part of a degenerate
multiplet at high-symmetry k. At Γ the valence top (Γ₂₅′) and conduction bottom
(Γ₁₅) are each 6-fold degenerate (nspinor=2 ⇒ 3 spatial × 2 spin); nv=4/nc=4
keeps 4 of 6. This makes the transition density — and hence the exchange/direct
kernels — non-covariant specifically at those k, and that is where the
splitting comes from.

A secondary, genuine defect exists: the ISDF Coulomb/screening tiles V0/W0 are
~3% non-covariant under the centroid permutation (raw), ~8% after q=0 head
injection. But it contracts to ~1e-4 in the kernel and does not drive the
observed splittings.

## Rung-0 — the two arms took different code paths, split identically

`gw_old.out`: centroid orbit closure FAILED (41218/46080) → V_q/W built directly
on all 64 full-BZ q (unfold=full-BZ, no tile sym-unfold). `gw_sym.out`: IBZ
cascade ACTIVE → V_q/W on 8 IBZ q then unfolded via centroid π_R (unfold=IBZ→full).
Kx covariance 9.38% (old) vs 9.37% (sym); Kd 5.24% vs 5.22%; 4v4c doublet 520.7
vs 518.5 μeV — identical to 3 sig figs. Rules out tile-construction path,
centroid closure, ISDF rank (960 vs 792) in one shot.

## Rung 1 — energies exactly covariant

Max spread of e_nk within any k-star (group by `sym.irr_idx_k`), per band:
0.0000 μeV (both arms). D is not the ingredient.

## Rung 5/8 — ψ at centroids exactly covariant (refutes the ψ-unfold hypothesis)

Valence-band overlap Gram on the centroid grid, eigenvalue R-invariance within
k-stars:

| band set | worst within-star eig rel-spread |
|---|---|
| val [4,8) (BSE window) | 1.7e-15 |
| cond [8,12) (BSE window) | 1.6e-15 |
| occ [0,8) | 2.5e-15 |
| deepval [0,4) | 2.0e-15 |

k-diagonal transition-density Gram (identity metric): 7.5e-15. With V0/W0
metric: only 1.06e-4 / 4.6e-4. The `unfold_psi` output is symmetry-correct.
(Old arm's non-closed centroids break the Σ_μ Gram at 6.9% — that is centroid
non-closure, already ruled out by rung-0.)

Direct Γ-point covariance (Gram-corrected projector D=(Ψ†Ψ)⁻¹Ψ†Ψ_R, symmorphic
ops): every closed Γ multiplet is covariant — [0,1]→9e-11, [2..7] (Γ₂₅′ 6-fold)
→9e-10, [0..8)→8e-10, [8..13] (Γ₁₅ 6-fold)→1e-9. The cut BSE windows leak:
[4,8)→0.65, [8,12)→0.66.

## Rung 6 — the ISDF tiles ARE ~3% non-covariant (real but subdominant)

Direct band-free tile test on the sym arm (q=0 ⇒ V0(μ,ν) must equal
V0(α(μ),α(ν)) for all 48 spatial ops):

| tile | max_rel violation | worst op |
|---|---|---|
| V0 raw | 3.16e-2 | 38 (nonsymmorphic) |
| V0 head-injected | 7.78e-2 | 40 (symmorphic) |
| W0 raw | 3.02e-2 | 43 (nonsymmorphic) |

Genuine defect; head injection roughly doubles it. But k-diagonal kernel blocks
are covariant to 1e-4 — the tile non-covariance contracts away.

## Rung 7 — clincher: tiles/head are NOT the cause

Little-group-symmetrized tiles (V0 fully α-invariant, ΔV0=1.6%, ΔW=1.5%) and
head on/off, 4v4c doublet split:

| variant | mfd0 split |
|---|---|
| A baseline (head, raw W) | 518.45 μeV |
| B no head (raw) | 519.16 |
| C symmetrized tiles + head | 518.32 |
| D symmetrized tiles, no head | 519.03 |

Enforcing tile covariance and removing the head change nothing.

## Rung 10 — Γ on-site block: cut window destroys degeneracy, closed window restores it

Isolated Γ exciton block:
- 4×4 CUT [4,8)×[8,12): all 16 eigenvalues non-degenerate (every multiplet
  fully lifted).
- 6×6 CLOSED [2,8)×[8,14): eigenvalue multiplets clean — splits 0, with one
  3-fold at 1.18 μeV and one at 36 μeV (~100× smaller than the cut-window
  splitting). D_Γ = (e_c−e_v) = 2.5404 eV × Identity, so the diagonal cannot
  split anything.

## Rung 11 — the split lives on the high-symmetry (cut-multiplet) k-stars

Exact eigenvector attribution of the 4v4c doublet split (reconstructs to
518.45 μeV), on-site contribution by k-star:

| star | size | on-site split contribution |
|---|---|---|
| 0 = Γ (cut 6-fold) | 1 | −3252 μeV |
| 1 (high-sym) | 8 | +4303 μeV |
| 2 | 4 | +538 |
| 3 | 6 | −266 |
| 4 (generic) | 24 | −208 |
| 5 | 12 | −241 |

The 518 μeV net is a near-cancellation of large (±3000–4300 μeV) contributions
concentrated on the small-orbit / high-symmetry stars where the window cuts
multiplets. Generic (size-24) k contribute little. This is why larger windows
don't help (6v6c→2409, 8v8c→2030 μeV) — no fixed (nv,nc) count is
symmetry-closed at every high-symmetry k, so each window cuts a different
multiplet.

## Root cause + implication for the BGW comparison

Cutting a degenerate multiplet in the transition space breaks the exciton's
point-group symmetry in any BSE implementation — a property of the finite
(nv,nc) window, not of LORRAX. The "BGW ~2 μeV" reference (its bsemat.h5 is
absent per PHASE2_LOG — the comparison was never actually run on this fixture)
must correspond to a symmetry-closed / full-multiplet transition window. The
right LORRAX-vs-BGW gate is a window degenerate-closed at the contributing
high-symmetry k (or a degeneracy-averaged manifold comparison), NOT a raw
fixed 4v4c cut.

Remaining genuine LORRAX to-do (independent of the window issue): the ~3%
non-covariance of V0/W0 under the centroid permutation (worst on nonsymmorphic
ops; head injection worsens it) — a real symmetry defect in the ISDF
Coulomb/screening tile assembly, subdominant here (~tens of μeV floor) but
worth a fix in the tile / head-injection path.

## Methodology note

An early version of these probes hardcoded the wrong band window (n_occ=4
instead of 8 — the WFN is nspinor=2, so 8 electrons = 8 occupied spinor bands);
caught via an eigenvalue sanity check (baseline must reproduce 518.45 μeV) and
re-ran with the correct window [4,8)×[8,12). All numbers above are post-fix.
The rung6-(A) Kx block-Frobenius numbers (~10%) are partly confounded by
conduction-window truncation and are superseded by the truncation-free tile
test rung6-(B) and the eigenvalue-based rung10/rung11.
