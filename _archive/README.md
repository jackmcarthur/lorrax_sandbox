# _archive/

Deliberately retained historical material. Nothing here is normative: where an entry contradicts `AGENTS.md`, `CLAIMS.md`, or the main-repo docs, the archive is wrong. Grep it; do not read it linearly.

| Entry | Archived | Why kept |
|---|---|---|
| `CHANGELOG.md` | 2026-07-31 | Session memory of the Perlmutter/local-GPU era (2026-04 to 07), 352 KB. Provenance for the BGW-matching conventions still cited in `docs/BGW_LORRAX_MATCHING.md`. |
| `KNOWN_SANDBOX_ERRORS.md` | 2026-07-31 | Error log of the old environment. Almost all entries concern paths/tools that no longer exist; kept because a few describe BGW/QE input pitfalls that recur. |
| `rescue_2026-07-22/README.md` | 2026-07-31 | Records that the lorrax_D `head-wing-fix` stash was saved only as local `.patch` files on Perlmutter which the whitelist `.gitignore` blocked from commit. **Those patches are NOT in this repo**; recovery requires the Perlmutter checkout or branch `rescue/lorrax-D-worktree-2026-07-22` (worktree part only). `.gitignore` now whitelists `*.patch` so this cannot recur. |
| `DAVIDSON.md` | 2026-07-31 | Unreviewed thick-restart block-Davidson derivation (GPU-era, conversational provenance). Possibly useful for future distributed-eigensolver work; not current practice. |
| `gn_bug_plan.md` | 2026-07-31 | GN-PPM head-correction plan, resolved during the Perlmutter campaigns. Physics conclusions were folded into `docs/BGW_LORRAX_MATCHING.md`. |
| `profiling_stack/` | 2026-07-31 | GPU/xprof profiling skill built around Perlmutter `lxrun`/Shifter. On Frontera use `docs/HLO_HOWTO.md` + `tools/hlo/`. The `cpu_addendum.md` here still has valid generic notes on CPU trace interpretation. |

Deleted outright in the 2026-07-31 rebuild (recoverable from git history at `6324f27a`): `runs/` contents, `reports/` contents (71 dirs), `scripts/` contents, `tmp/`, `uneven_test_out/`, `modulefiles/` (lxrun pool overlay), `sources/` gitlinks, `pyproject.toml`/`src/` (uv package scaffolding), `agents_xprof.md`, `docs/JAX_PROFILING.md` (folded into `docs/HLO_HOWTO.md`).
