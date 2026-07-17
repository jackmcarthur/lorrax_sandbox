# Round 2 — closed-window residual attribution + tile-defect root cause

**Date:** 2026-07-16 · sym-arm restart `work_sym/tmp/isdf_tensors_792.h5`
(orbit-closed 792 centroids ⇒ centroid perm α exists) · lorrax_A @
`agent/bse-phase2` · 1 GPU (salloc 56013945, released). Scripts `diag2/*.py`,
runner `diag/run_probe.sh`. Read `diag/FINDINGS.md` (Round 1) first.

## Verdict (three decisive results)

1. The closed-window residual IS the tile defect. Γ-on-site closed block
   [2,8)×[8,14): little-group-symmetrizing the q=0 V0/W0 tiles collapses the
   36.39 μeV residual to 0.000 μeV (head off). In the full coupled BSE, genuine
   exciton multiplets split up to 15.4 μeV (raw) → <1 μeV (covariant tiles).
2. The tile non-covariance is BORN in the ζ-fit, is TAU-BLIND (not a
   fractional-translation-phase bug), and W just inherits V.
3. Root cause = the SAME degenerate-multiplet-cut mechanism as the BSE window,
   now in the SCREENING ISDF fit window, amplified by CCT conditioning (3.6e9).

## Task 2 — bisection (diag2/cov_ladder.py, q=0 identity T[α(μ),α(ν)]==T[μ,ν])

| rung | object | max rel viol | τ-sensitive |
|---|---|---:|---|
| L0 | ψ@centroids (closed sets) | 1e-15 (R1 rung5) | — |
| L1 | C0 CCT Gram (fit INPUT) | 4.26e-3 | no |
| L2 | G0=ζ̃_{q=0}(G=0) (fit OUTPUT) | 8.64e-2 | NO (G=0⇒phase 1) |
| L3 | V0 = Σ_G conj(ζ̃)vζ̃ | 3.16e-2 | — (τ cancels leg-to-leg) |
| L4 | W0 screened | 3.02e-2 | — |
| L5 | ΔW = W0−V0 | 6.9e-2 | — |

corr(V0 per-op viol, W0 per-op viol)=0.997 ⇒ W inherits V; the W solve adds no
independent defect. The 8.6% at G=0 is τ-BLIND yet worst on a nonsymmorphic op
⇒ "nonsymmorphic-worst" is base rate (36/48 ops), NOT a phase signature —
REFUTES the Round-1 phase hypothesis.

Per-element proof V_q assembly is faithful: for q=0 (IBZ parent, no unfold),
V0[μ,ν]=Σ_G conj(ζ̃_μ(G))v(G)ζ̃_ν(G). A covariant ζ obeys ζ̃_{α(μ)}(G)=
e^{-iG·τ}ζ̃_μ(RᵀG); substituting both legs the τ phases cancel and (v(RᵀG)=v(G),
R-closed sphere) V0[α(μ),α(ν)]=V0[μ,ν] exactly. So V0 non-covariance ⟺ ζ̃
non-covariance; the bilinear contract preserves whatever covariance ζ has.

## Task 2 — depth (diag2/zeta_probe.py)

SEED — screening band-window cut (decisive):

| screening window | closed? | C0 cov viol |
|---|---|---:|
| [8,16) | yes | 5.84e-10 |
| [8,20) | yes | 5.79e-10 |
| [8,28) | yes | 6.54e-10 |
| [8,36) | yes | 7.18e-10 |
| [8,40) | yes | 7.04e-10 |
| [8,60)=production (nband=60) | NO | 4.26e-3 |

Every degeneracy-closed conduction top ⇒ machine-precision (6e-10); only the
production window (top band 59 = cut multiplet) breaks. The 0.4% seed is 100%
the top-band cut.

AMPLIFIER — cond(C0)=3.586e9 (λmax 4.4e-3, λmin 1.2e-12, eff-rank@1e-8=738/792);
ζ=C⁻¹Z amplifies the 0.4% seed 20.3× (G0 8.64e-2 / C0 4.26e-3). With a closed
window the seed→6e-10, amplified→~1e-8 (negligible) — the amplifier only bites
because of the seed.

## Task 1 — closed-window residual (diag2/closed_window.py, closed_window_full.py,
full_multiplet.py)

(A) Γ-on-site block [2,8)×[8,14) — 2×2 grid {raw,sym q=0 tiles}×{head off,on}:

| variant | max intra-multiplet split |
|---|---:|
| raw, no head (=R1 rung10 baseline) | 36.39 μeV |
| raw, head on | 34.70 μeV |
| sym tiles, no head | 0.000 μeV |
| sym tiles, head on | 3.06 μeV |

Symmetrizing the q=0 tiles ANNIHILATES the 36 μeV residual. The 3.06 μeV
head-on residual is G0's own 8.6% non-covariance (raw G0 injected).

(B) Full-H closed window [0,8)×[8,16). The per-q stabilizer projection is a
no-op for generic q; correct full-BZ covariance needs IBZ-symmetrize + unfold.
diag2/closed_window_full.py: extract IBZ reps → SPATIAL little-group symmetrize
→ re-expand with production common.symmetry_maps.unfold_v_q. Gates:
ROUNDTRIP unfold(V_ibz_raw) vs production V_full = 0.000e+00 (wiring exact);
DIRECT q=0 spatial covariance of V0_sym = 6.4e-16 (raw was 3.16e-2).

The size-8 "mfd0" spanning 2031 μeV is NOT a broken multiplet — covariant tiles
leave it at 2031.41 μeV (Δ0.09), i.e. it is physically-distinct excitons (a
1 meV energy-clustering artifact, same as R1 rung11's "8v8c→2030 μeV").

GENUINE full-BSE multiplets (diag2/full_multiplet.py: covariant-tile degenerate
groups, tol 5μeV; raw-tile split of the SAME states):
- max |λ_raw − λ_cov| over all states = 8.78 μeV
- worst genuine-multiplet raw-tile split = 15.41 μeV (size 5)
- clean size-2 multiplets: cov_split 0.4–0.75 μeV → raw_split 4–15 μeV

So in the full coupled BSE the tile defect splits exciton multiplets by ~4–15
μeV (raw) → <1 μeV (covariant) — smaller than the isolated Γ-block's 36 μeV
because the exciton is delocalised over k, but still the dominant residual and
approaching the BGW ~2 μeV scale as tiles→covariant. (Covariant residuals up to
~9 μeV on some groups are my spatial-only symmetrization leaving TRS-folded W_q
tiles imperfect; a fully covariant set from the fix would close them.)

## Task 3 — proposed fix (design-level; NOT implemented)

Degeneracy-round the SCREENING ISDF fit band window so its top boundary does
not split a degenerate multiplet at any k. Location: common/meta.py:99
`b_id_4_user = int(nband)` → gw_init.fit_zeta `band_range_right=(b1,b4)`; today
b_id_4=_round_up(nband, world_size) rounds UP only for device divisibility,
never degeneracy. b_id_3=nelec+ncond (σ top) has the same exposure.

Design:
- add `round_band_window_to_closed_shell(energies_kn_ry, b_hi, tol_ry,
  direction='down')` IN gw/degen_average.py (it already owns the
  contiguous-degenerate-group detection; BGW TOL_Degeneracy=1e-6 Ry). Route the
  screening-window top through it — no new sym helper, no parallel machinery
  (respects unified-sym-action / no-new-API-layer).
- round DOWN to the largest boundary b≤nband with min_k(e[k,b]−e[k,b-1])>tol;
  warn with the dropped band count. Does NOT change the σ OUTPUT band set;
  closes only the ζ/χ0 fit window.
- No CCT-conditioning change (closed window ⇒ seed 6e-10).

Not implemented: it changes screening physics (drops bands) and needs a policy
call (down/warn/error) — fails the "small, unambiguous, gate-protected" bar; this
is the write-plan-and-stop case. It is a GW-init file the concurrent W(ω) agent
does not touch, so it can land independently.

## Task 4 — degeneracy gate design

Γ-on-site closed-window BSE degeneracy assertion on a session restart fixture:
- build H_Γ = D+Kx−Kd at Γ over an auto-detected degeneracy-closed (nv,nc)
  window; pure numpy eigvalsh on a ≤(nc·nv)² matrix — ms, 0 GPUs, no 2nd GW run
  (mirrors test_bse_dense_reference fixture reuse; honours no-16-GPU-gating).
- two-tier threshold tied to the fix: raw production tiles → assert <50 μeV
  (guards the 36 μeV floor vs regression); with the screening-window fix →
  tighten to <5 μeV. Optional symmetrized-tile variant asserts <1 μeV as a
  window-mechanism invariant independent of the tile fix.
- File: NEW tests/test_bse_degeneracy.py (does NOT touch the concurrent agent's
  bse_w_exact.py/bse_feast.py/solvers/). Piggyback the gnppm/bse_dense_state
  session fixture.

Not implemented: the μeV threshold must be characterised once against the chosen
committed fixture (MoS2 gnppm) — the Si numbers here are from a run-dir restart,
not a repo fixture; committing an unvalidated-threshold gate would be a false
green. Skeleton + one-time calibration is the hand-off.

## Files
- diag2/cov_ladder.py → bisect_sym.json (Task-2 bisection)
- diag2/zeta_probe.py → zeta_probe_sym.json (seed + amplifier)
- diag2/closed_window.py → closed_window_sym.json (Task-1 A + naive B)
- diag2/closed_window_full.py → closed_window_full_sym.json (Task-1 B, correct)
- diag2/full_multiplet.py → full_multiplet_sym.json (full-BSE genuine multiplets)
