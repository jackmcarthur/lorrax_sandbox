# LORRAX user manual — kickoff, first four chapters, review round 1

**Branch:** lorrax_B `agent/manual` (off main @ `adc2197`).
Commits: `54279e3` (outline + chapters 1/4/5/6, 20 files under `manual/`),
`0d5822d` (3-lens review fixes).

## What exists

`manual/` — one .md per section, LaTeX math, no LaTeX formatting yet.
`manual/00_outline.md` is the contract: full chapter plan (~52 pp), editorial
threads T1–T5, pre-writing blockers for Ch. 7, and the frequency-doc source map.
Drafted: Ch 1 (intro, capability matrix, LORRAX-vs-BGW pros/cons, units), Ch 4
(objects on (r,r′), two-sums rule, N_k log N_k, time-integration heuristics, QP
solvers), Ch 5 (ISDF ansatz, σ⁰ spinor trace, the fit via P_k, orbit closure,
rank), Ch 6 (truncation, mini-BZ MC average, cutoffs, q→0 pointer).

## Review round 1 (three Explore agents, per-lens)

| Lens | Headline findings |
|---|---|
| Code fidelity | 11 verified-false/misleading claims; top: §6.3 stated default `bare_coulomb_cutoff` = 4×Ecut — **actual default is ecutwfc, deliberately matching BGW** (gw_init `_resolve_cutoff`); §4.5 qp_solver strings `on_shell`/`sc_eigenvalues` don't exist (`fixed_point`/`self_consistent`); `mc_average_vcoul_body` is ON by default, 3D-only; Z-factor IS a central difference (dE=0.5 eV = BGW default); ridge only on 1×1 mesh; 0D box not wired through V_q driver at HEAD |
| Physics/pedagogy | χ⁰'s −2 prefactor is the collinear factor, contradicts spinor-first (fix: trace in pair density, prefactor −1 + transpose term); COHSEX split as written double-counted bare X; §4.4 crossing-regime floor is imposed broadening, NOT the gap; crossing node counts are O(A)~tens-to-100, not "short"; hedge "no other code" claims |
| Genre/structure | Fair and genre-fit overall; bibliography promised in App F which is symbols/units (→ new App G); §12.2/§12.3/§13.5 refs were unpinned (outline now enumerates Ch 12/13 subsections); Ch 4/6 ~30% over page budget; T3 currently 0% delivered (all figures placeholders) |

**Adjudication of conflicts (verified in code):** (1) cutoff default — reviewer
right, sandbox memory was stale (default changed upstream to ecutwfc; memory file
updated); (2) "bispinor GN-PPM gate can't exist" — reviewer wrong,
`tests/regression/bispinor_debug/bispinor_test.in` has `bispinor=true` +
`compute_mode=gn_ppm`; instead §1.2 wording clarified (charge channel screened,
transverse bare).

All accepted fixes applied in `0d5822d`. Deferred (flagged in files/outline):
per-q ζ-fit residual logging verification (TODO in 5.5), Ch 4/6 length trim (do
after content settles), T3 figures (need runs).

## Next

1. Ch 2 (installation) + Ch 3 (tutorial) — mechanical, source material exists.
2. Ch 7 after pre-writing blockers: GN B-normalization from
   `fit_gn_ppm_from_wc_pair`, window-scheme verification, fresh periodic Σ(ω) gate.
3. Ch 11 input reference regenerated from `gw_config.py` groups.
4. T3 figures: n_μ sweep + cutoff sweep on the Si/MoS₂ fixtures (1-GPU jobs).
