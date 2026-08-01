# ASSERTIONS.md — rebuild checklist, 2026-07-31

Signed by the rebuild agent (Claude Fable 5, branch `frontera-rebuild`).
Every statement below was checked in this session; deferrals are stated
plainly rather than claimed fixed.

## Every present file earns its place

| Files | Verdict |
|---|---|
| `AGENTS.md` (+`CLAUDE.md` symlink), `CLAIMS.md`, `GATES.md`, `INVARIANTS.md` | (i) current — written this session against the verified 2026-07-31 state |
| `docs/HLO_HOWTO.md`, `docs/PORTABILITY.md` | (i) current — pointers verified against the repo at `ecf461e` |
| `docs/BGW_LORRAX_MATCHING.md`, `PARSE_OUTPUTS.md`, `skills/compare/SKILL.md`, `skills/build_inputs/SKILL.md` | (i) — machine-independent physics/parsing content; stale paths repointed (receipts -> `git show 6324f27a:`, archives -> `_archive/`) |
| `skills/execute_workflow/SKILL.md`, `skills/checkpoint/SKILL.md` | (i) — rewritten for Frontera this session |
| `docs/docs_bgw/`, `docs/docs_qe/`, `docs/docs_gwjax/` | (i) — reference specs for BGW/QE/LORRAX inputs; COHSEX_INPUT.md carries a currency note deferring to the repo `manual/` |
| `tools/hlo/analyze_hlo_dump.py`, `tools/compare_bgw_gwjax.py`, `tools/README.md` | (i) — the two load-bearing survivors of `scripts/`, moved/restored and (analyzer) extended |
| `fastloop/PLAN.md`, `fastloop/run_minideck.py` | (i) — deliberate scaffold; both state they are not yet runnable end-to-end |
| `templates/`, `assets/pseudopotentials/` | (i) — deck-building inputs, machine-independent |
| `runs/`, `reports/`, `scripts/` READMEs | (i) — one-line contracts for purged directories |
| `_archive/` (7 entries) | (ii) — each has a dated reason in `_archive/README.md`; non-normative by declaration |
| Everything else that was here on clone (155k run files, 71 reports, 40+ scripts, lxrun modulefiles, uv scaffolding, dead gitlinks) | (iii) — deleted; recoverable at git commit `6324f27a` |

## The six findings

| Finding | Status |
|---|---|
| (a) Provenance archaeology | ADDRESSED: `CLAIMS.md`, 16 seeded rows with jobids, append-only directive. Caveat: the gloo-failure reps are cited via scorecard §AY rather than individual jobids. |
| (b) No fast execution loop | SCAFFOLDED, NOT SOLVED at rebuild time. *(Superseded 2026-07-31, same day: `fastloop/` was built and certified — jobs 7884926/7884936, CLAIMS rows 17-19; the loop is now ~3 min for both legs. See `fastloop/PLAN.md`.)* |
| (c) Sharding invisible at call sites | ADDRESSED: existing analyzer checked against current needs — collectives it already had; layout-boundary (transpose/copy/bitcast) table and `--forbid` gate added; `docs/HLO_HOWTO.md` written. Caveat: the new code is syntax-checked (login python 3.7) but has not yet run against a real dump — first user should treat its output with rule-2 skepticism. |
| (d) Gated-path combinatorics | ADDRESSED: `GATES.md`, defaults read from `src/ffi/{fft,gemm,gate}.py` at `ecf461e`. |
| (e) Physics-machine seam bugs | ADDRESSED: `INVARIANTS.md`, 9 rows, enforcing refusal named where one exists (`rank_criterion.py`, `collectives.py::warm_mesh_cliques`). |
| (f) Which-tree-is-truth | ADDRESSED: "Which tree is truth" paragraph in `AGENTS.md`, incl. bundles-pin-src and the same-day landing rule. |

Also requested: Frontera-vs-Perlmutter and MKL-vs-Cray guidance has one
home (`docs/PORTABILITY.md`), pointing at repo docs; the CUDA wheel pin is
stated as the repo states it (`jax[cuda12]` — pyproject switched from
`cuda13` to `cuda12`; flip back only for a newer-driver machine;
`pyproject.toml` authoritative). *(Corrected 2026-07-31: this sentence had
the pin backwards.)*

## Fresh-agent orientation

A fresh agent can orient from `AGENTS.md` alone (90 lines): it carries the
tree-of-truth paragraph, the 5-line current state, the truth table, the 8
rules, and the distrust list. Full orientation (AGENTS + the three
ledgers) is ~165 lines. Task extras: a skill (~60 lines) or
`docs/HLO_HOWTO.md` (~75 lines). Nothing requires reading `_archive/`
(8,044 lines) or the scorecard (9,400+ lines); both are grep targets only.

## Known unresolved items, stated plainly

1. The lorrax_D `head-wing-fix` stash is NOT in this repository: its
   `.patch` files were blocked by the old `.gitignore` and exist only on
   the (outage-bound) Perlmutter checkout. See
   `_archive/rescue_2026-07-22/README.md`. Owner action required.
2. The QE -> pw2bgw deck-building leg is not certified on Frontera
   (stated in `skills/execute_workflow/SKILL.md`).
3. `templates/*.in` semantic currency against the current input parser
   was not re-verified by a run in this session.
4. `tools/compare_bgw_gwjax.py` was restored, not re-run (needs an
   in-container job).
