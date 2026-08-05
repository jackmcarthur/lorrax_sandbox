# Agent 4 — cohsex.in reference + cross-doc canonicalization + sandbox-vs-repo split

**Slice:** `COHSEX_INPUT.md` (sandbox), `templates/cohsex.in` (sandbox),
`src/gw/gw_config.py` `_DEFAULTS` (lorrax_D), sandbox/repo boundary.
**Out of scope:** `docs/PHYSICS_COMPREHENSIVE.md`, `MEMORY_MODEL.md`,
`ENVIRONMENT_COMPREHENSIVE.md`, `AGENT_TODO.md`, plans/, advanced/ — those
are covered by agents 1-3.

---

## 1. Scope

The files I read directly:

- `/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/COHSEX_INPUT.md` (411 lines)
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/gw_config.py` (1017 lines; `_DEFAULTS` dict is the ground truth)
- `/pscratch/sd/j/jackm/lorrax_sandbox/templates/cohsex.in` (56 lines)
- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_install_maintain_blitz_2026-05-13/consensus.md`
- `lorrax_D/docs/` directory listing (structure, not full content of each file — that's agents 1-3)

The `_DEFAULTS` dict is on branch `agent/install-blitz-integration` (HEAD
`3079a1f`), which is the canonical integration-blitz state. Commit `5cadd4b`
deleted `psig_k_chunk_size` (it no longer appears in `_DEFAULTS`).

---

## 2. Per-doc verdict

| Doc | Action | Justification |
|-----|--------|---------------|
| `docs/docs_gwjax/COHSEX_INPUT.md` | **REGENERATE** (then move to repo) | 45 undocumented parser keys; 4 stale "docs" keys; wrong default for `output_file`; wrong section structure vs current `LorraxConfig` groups. Hand-maintenance is already losing. |
| `templates/cohsex.in` (sandbox) | **UPDATE now; eventual MOVE to repo** | Uses 4 deprecated/unknown keys (`x_only`, `use_chunked_isdf`, `write_no_head_vw`, `output_file`, `sigma_debug_split_contrib`); doesn't match any `_DEFAULTS` entry for those. Trivially fixable today. |

---

## 3. Section A — cohsex.in reference drift (the core audit)

### 3.1 Parser keys NOT documented in COHSEX_INPUT.md (undocumented — 45 keys)

The parser (`_DEFAULTS`) has **86 keys** total. `COHSEX_INPUT.md` documents
**45 keys** (by `### \`key\`` headers). That leaves **45 parser keys with
zero documentation**.

Grouped by `LorraxConfig` sub-dataclass:

**System / geometry (all documented):** `nval`, `ncond`, `nband`, `sys_dim` — OK.

**FilePaths — undocumented:**
- `centroids_file_current` — bispinor Gordon-current centroids; new in bispinor phase-1.
- `kin_ion_file` — T + V_ion + V_NL matrix elements file. Appears in the
  `sigma_freq_debug.dat` column table but not as a settable key.
- `sigma_diag_file` — the replacement for deprecated `output_file`.
  **Critical gap**: users who read only the doc will try to set `output_file`
  and get a deprecation warning they weren't warned was coming.
- `eqp0_file` — BGW-format QP energies (auto-written; user-settable filename).
- `eqp1_file` — Z-linearized QP energies (new key, not mentioned).
- `sigma_kij_h5_file` — streaming accumulation output; mentioned in the
  `sigma_omega_accumulation` description but not as its own key.

**Mode flags — undocumented:**
- `compute_mode` — the new canonical mode axis (`"auto"|"x_only"|"cohsex"|"gn_ppm"|"hl_ppm"`). This is the most important missing key: the doc still describes `x_only` and `do_screened` as the primary control surface.
- `do_G0` — G=0 head inclusion flag.
- `no_degen_averaging` — BGW-mirror band-averaging toggle.
- `degen_avg_tol_ry` — degenerate-band tolerance.

**Backend — all undocumented:**
- `use_ffi_io` — parallel-HDF5 FFI toggle (replaces implied "always uses phdf5").
- `gspace_mode` — `host_cache` vs `file_reread` ψ(G) lifecycle.
- `isdf_memory_mode` — legacy string mapping to `ScreeningSolver` enum.
- `cusolvermp_charge` — `auto|on|off` override for ζ-fit Cholesky solver.
- `cusolvermp_lu` — `auto|on|off` override for ζ-fit LU solver.
- `gamma_contract_mode` — `take|einsum|scan` variant for γ-double-contract kernel.

**Memory / chunking — partially documented (`memory_per_device_gb` only):**
- `chunk_size` — legacy band-chunk knob (-1 = no chunking).
- `band_chunk_size` — band-chunk size (default 16).
- `r_chunk_size` — r-chunk override (0 = auto).
- `use_aot_chunk_chooser` — AOT chunk-chooser gate.
- `chunk_chooser_mode` — `heuristic|analytic` AOT variant.
- `gflat_chunk_size` — gflat-axis chunk; 0 = one-shot or planner-picked.
- `vq_g_chunk_size` — V_q inner G-axis GEMM chunk; 0 = auto.

**Head / Coulomb — partially documented (`wcoul0_source`, `vhead`, `whead_0freq`, `whead_imfreq`):**
- `wcoul0_eta` — Coulomb head eta parameter (undocumented).
- `mc_average_vcoul_body` — mini-BZ Coulomb body averaging (default True; undocumented).
- `bare_coulomb_cutoff` — Coulomb cutoff in Ry (default None = ecutwfc). Sandbox memory
  notes flag this as critical for BGW comparisons. Completely absent from the doc.
- `zeta_cutoff` — ζ-sphere cutoff (Ry); must be ≥ `bare_coulomb_cutoff`.
- `use_bgw_vcoul` — diagnostic BGW vcoul override.
- `bgw_vcoul_file` — path to BGW vcoul file.
- `bgw_vcoul_sym_wfn` — aux WFN for 48-op symmetry group extraction.

**Screening — mostly documented; missing:**
- `regenerate_minimax_tables` — gate to force minimax table regeneration (undocumented).

**PPM — partially documented; missing:**
- `ppm_model` — `"gn"|"hl"` explicit model selector (doc mentions GN-PPM but
  not the key).
- `ppm_head_omega_h_ry` — override Ω_h directly for BGW comparison (undocumented).
- `ppm_sigma_target_error` — σ-quadrature minimax tolerance.
- `ppm_sigma_max_nodes` — σ-quadrature max nodes.
- `sigma_omega_batch_size` — ω-batch size for accumulation.
- `sigma_window_edge_factor` — window edge factor (default 1.5).
- `sigma_omega_accumulation` — `auto|kij|kij_stream` (documented in §8 but not as a `### \`key\`` header entry).
- `ppm_sigma_scale`, `ppm_sigma_flip_neg` — documented but under wrong section.

**Debug — partially documented; missing:**
- `debug_omega` — float override for debug ω probe.
- `sigma_debug_quadrature_samples` — number of samples for quadrature debug.
- `w_copies_debug_file` — output filename for W copies debug.
- `sigma_freq_debug_file` — output filename for freq debug.
- `write_wfn_h5` — gate for end-of-run WFN_qp.h5 write.

**BSE — entirely undocumented:**
- `get_centroids_fi` — fine-k BSE centroid gate.
- `wfn_fi_min`, `wfn_fi_max` — fine-k band window.
- `kgrid_fi` — fine-k grid spec.

### 3.2 Keys the doc claims but the parser does NOT know (stale / lies — 4 keys)

| Doc key | Status |
|---------|--------|
| `x_only` | **DELETED from parser.** Functionality subsumed by `compute_mode = "x_only"`. Doc §3 leads with this key. |
| `use_chunked_isdf` | **Never in `_DEFAULTS`.** Was a structural flag before chunked was made the default; doc and template both show it. |
| `write_no_head_vw` | **Not in `_DEFAULTS`.** Appears in template and doc §10. Possibly a debug key handled outside the `_DEFAULTS` path, or deleted. |
| `sigma_debug_split_contrib` | **Not in `_DEFAULTS`.** Appears in template. The consensus Q4 called this out explicitly: "deprecated/auto-derived or typo?" Author's call needed. |

### 3.3 Default-value mismatches

| Key | Doc says | Parser default |
|-----|----------|---------------|
| `output_file` | default `"eqp0_noqsym.dat"` | **Key deleted.** Replacement is `sigma_diag_file` (default `"sigma_diag.dat"`). The doc §9 also shows `output_file` as a user key. |
| `use_chunked_isdf` | default `true` | **Not in parser.** |
| `sigma_omega_h5_file` | default `""` | Parser: `"sigma_mnk.h5"` — non-empty default, the doc says empty. |

### 3.4 Structural mismatch

`COHSEX_INPUT.md` uses a 11-section narrative structure (Restart & ISDF Basis,
Band Window, Calculation Mode, …). The parser's `LorraxConfig` now has 9
sub-dataclasses (`FilePaths`, `HeadConfig`, `ScreeningConfig`, `PPMConfig`,
`MemoryConfig`, `BackendConfig`, `DebugConfig`, `BSEConfig`, `K_POINTS`). The
two structures are not aligned. An auto-generated doc would naturally follow the
dataclass grouping; regeneration should adopt that as the new section structure.

---

## 4. Section B — Should COHSEX_INPUT.md live in the repo or the sandbox?

**Re-affirm: move to `lorrax_C/docs/COHSEX_INPUT.md`.**

The argument is unchanged from the install-blitz consensus: a second user doing
`git clone lorrax; pip install -e .` has no path to discovering the cohsex.in
reference. The sandbox is personal infrastructure. The doc is the user-facing
reference for the primary input format.

**Minimum-viable move (before Blitz #2 auto-generation):**

Copy `COHSEX_INPUT.md` as-is into `lorrax_C/docs/`, with a one-line header
noting it is hand-maintained pending auto-generation from `_cohsex_schema.py`.
This takes 5 minutes and immediately fixes the "second user has no reference"
problem. The doc being stale (per §3 above) is a pre-existing problem; it is
no worse in the repo than in the sandbox, and now at least it's reachable.

**Until Blitz #2 lands:** The author maintains it, but the gap analysis in §3
means it should not be patched by hand — the 45 missing keys make a hand-patch
a multi-hour job that will drift again immediately. The right answer is: copy
as-is now, auto-generate via Blitz #2, then hand-maintain only the conceptual
prose (intro paragraphs, physics-level descriptions) that cannot be
auto-generated from the schema.

---

## 5. Section C — templates/cohsex.in

The sandbox template at `/pscratch/sd/j/jackm/lorrax_sandbox/templates/cohsex.in`
has **five problems**:

1. `output_file = eqp0.dat` — deprecated key, parser emits a `DeprecationWarning`
   and ignores it. Should be `sigma_diag_file = sigma_diag.dat` (if custom path
   wanted) or removed (the default is `sigma_diag.dat`).
2. `x_only = false` — key does not exist in `_DEFAULTS`. Parser silently ignores it.
3. `use_chunked_isdf = true` — same; not in `_DEFAULTS`.
4. `write_no_head_vw = true` — not in `_DEFAULTS`; effect unknown.
5. `sigma_debug_split_contrib = true` — not in `_DEFAULTS`; consensus Q4 flags this.

**Recommendation: update the template inline now (5-minute fix).** The template
is copied verbatim into every new run directory by the Build-Inputs skill. A
run started today would get a `DeprecationWarning` from `output_file` and silently
lose three debug flags. The template should be:

- Remove `x_only`, `use_chunked_isdf`, `write_no_head_vw`, `sigma_debug_split_contrib`.
- Replace `output_file = eqp0.dat` with `sigma_diag_file = sigma_diag.dat`.
- Add `compute_mode = cohsex` or `compute_mode = gn_ppm` as the explicit mode
  flag (replacing the `do_screened = true` + `use_ppm_sigma = true` pair with
  the canonical single knob).

**Longer term:** Move the template into `lorrax_C/templates/cohsex.in` so the
repo is self-contained. The sandbox `templates/` can then symlink or just import
the canonical version. This is a direct analogue of Blitz #5's
"move universal slice upstream" recommendation.

**Do not auto-generate the template from the parser schema** (as suggested in
the blitz plan). The template should be a *minimal example* — showing only the
keys a new user needs to touch. An auto-generated template with all 86 keys and
their defaults would be 200+ lines of noise for a first-time user. The right
split is:
- Auto-generate the *reference* (`COHSEX_INPUT.md`) from the schema.
- Hand-curate the *template* (`cohsex.in`) to show the 10-15 keys that vary
  between runs.

---

## 6. Section D — sandbox-vs-repo split (meta-question)

### What belongs in the repo (`lorrax_C/docs/`)

| Content | Current home | Action |
|---------|-------------|--------|
| `COHSEX_INPUT.md` | sandbox `docs/docs_gwjax/` | Copy to repo immediately; auto-generate with Blitz #2 |
| `templates/cohsex.in` | sandbox `templates/` | Move to repo `templates/`; sandbox Build-Inputs skill references repo copy |
| `docs/PHYSICS_COMPREHENSIVE.md` | repo `docs/` | Already in repo — keep |
| `docs/MEMORY_MODEL.md` | repo `docs/` | Already in repo — keep |
| `docs/ENVIRONMENT_COMPREHENSIVE.md` | repo `docs/` | Already in repo — update for blitz changes |
| `docs/MINIMAX_QUADRATURE.md` | repo `docs/` | Already in repo |
| `docs/ZETA_V_Q_ALGORITHMS.md` | repo `docs/` | Already in repo |
| `docs/SYMMETRY_COMPREHENSIVE.md` | repo `docs/` | Already in repo |

### What is genuinely sandbox-only

| Content | Why sandbox-only |
|---------|-----------------|
| `skills/` (Build-Inputs, Execute-Workflow, Compare, Checkpoint) | Multi-agent orchestration scaffolding; Perlmutter-specific; not user-facing |
| `reports/` | Session history; valuable internally but not for second users |
| `runs/` | Calculation outputs |
| `AGENTS.md` (sandbox top-level) | Sandbox conventions; references sandbox-specific tools (`lxrun`, `lxattach`) |
| `modulefiles/lorrax_agent/` | Pool-coordination overlay; explicitly split off from upstream by Blitz #5 |
| `audit_pr*.py`, per-session scripts | Session artifacts |

### Where the boundary needs a callout in PORTING.md

One paragraph, approximately:

> **Sandbox vs. repo.** The canonical user-facing documentation lives in
> `lorrax_C/docs/` and is shipped with the package. The sandbox at
> `lorrax_sandbox/` is a single-site research environment containing
> multi-agent workflow scripts (`skills/`), run history (`runs/`), and the
> `lorrax_agent` overlay module for shared-allocation coordination. A second
> user cloning the repo does not need the sandbox; everything needed to run
> LORRAX is in `lorrax_C/` itself.

This paragraph does not currently exist anywhere. It should be in `PORTING.md`
(which is already being rewritten per the blitz consensus) and referenced from
`docs/index.md`.

---

## 7. Section E — cross-doc canonicalization: three single-source-of-truth rules

Ranked by leverage (which one closes the most doc-vs-code drift per hour of
work):

### Rule 1 (highest leverage): Cohsex.in reference auto-generated from `_cohsex_schema.py`

**Status:** Blitz #2, deferred. The gap is 45 undocumented keys + 4 stale
keys. No hand-patch is sustainable. This is the install-blitz consensus
#2 and remains the #1 highest-leverage canonicalization.

**Mechanism:** Introduce `src/gw/_cohsex_schema.py` with a `Field(name, type,
default, doc_str)` table. `tools/gen_cohsex_input_md.py` walks it and emits
`docs/COHSEX_INPUT.md`. CI test `test_cohsex_doc_freshness.py` regenerates
and fails on diff. The parser's `_DEFAULTS` becomes a view of the schema
(or the schema IS `_DEFAULTS` with annotations). This closes the drift
permanently.

**Comparable exemplars:** Quantum ESPRESSO's `INPUT_PW.txt` is auto-generated
from a `INPUT_PW.def` Python table (see `PW/Doc/gen_inputs.py` in the QE
source). ABINIT's variables reference is generated from
`~4000-line abipy/abio/abivar_database.py`. At LORRAX's scale (one author,
~86 keys), the QE pattern is the right analogue: a single Python dict
generates the Markdown, CI fails on diff, no hand-sync needed.

**Estimated cost:** 1 day (schema file + generator + CI test). Closes 45
undocumented entries permanently.

### Rule 2 (medium leverage): API reference auto-generated via `pdoc`

**Status:** `docs/gen_api_docs.sh` is already scaffolded. `docs/index.md`
references it. Per CONTEXT.md, the `src/isdf/` → `src/gw/isdf_fitting.py`
rename broke the `index.md` links months ago.

**Mechanism:** Run `pdoc` in CI (or document how to run it locally); keep
`docs/api/` as a generated artifact. The docstrings in `gw_config.py` (the
`LorraxConfig` module docstring is already 20 lines of good content; the
sub-dataclass docstrings are solid) are the source of truth. The only
maintenance cost is keeping docstrings current — which authors do anyway.

**Estimated cost:** 0.5 day to fix `gen_api_docs.sh` and add it to CI. Closes
the "no API reference" gap without writing prose.

### Rule 3 (lower leverage but quick): Single ENV_VARS table in `ENVIRONMENT_COMPREHENSIVE.md`

**Status:** Proposed in install-blitz consensus, never landed. The CONTEXT.md
flags `LORRAX_SC_*`, `LORRAX_V_Q_*`, `ISDF_*` as still read from `os.environ`
in code; docs may reference them inconsistently as env vars or cohsex.in keys.

**Mechanism:** A single `ENV_VARS` table listing every `os.environ.get(...)` key
in the codebase, with name, type, default, and which cohsex.in key (if any)
supersedes it. This table replaces the scattered mentions in `ENVIRONMENT_COMPREHENSIVE.md`
and prevents future confusion between "user sets this in cohsex.in" vs "user
exports this before running."

**Estimated cost:** 2-3 hours to grep all `os.environ.get` calls and compile
the table. Not auto-generated, but small enough to maintain by hand once written.

---

## 8. What I'd write fresh

Nothing new to write. The gap is not missing prose — it's 45 keys that need
auto-generated documentation. Writing them by hand today would create 200 more
lines that drift again next month.

The one thing that doesn't exist and should: a **10-line QUICKSTART block** at
the top of `COHSEX_INPUT.md` showing a minimal working `cohsex.in` for a
COHSEX run and a GN-PPM run. This is the first thing a new user needs. It
should use only `_DEFAULTS`-present keys and no deprecated flags.

Estimate: 20 lines. Write it once; it won't drift because it only uses keys
whose defaults are intentional.

---

## 9. Open questions

**Q4 (blocks template update and schema allow-list):** Are `sigma_debug_split_contrib`,
`write_no_head_vw`, and `use_chunked_isdf` deprecated/auto-derived (so: parser
should warn and ignore), or were they accidentally dropped from `_DEFAULTS`
(so: parser should grow them back)? The install-blitz consensus flagged this as
author's call. Until answered, the template cannot be fully cleaned without
risk of silently disabling a flag that was previously load-bearing.

**Q-bse:** The BSE sub-block (`get_centroids_fi`, `wfn_fi_*`, `kgrid_fi`) is
entirely undocumented. Is this because it's not yet production-ready and
shouldn't be exposed, or just an oversight? If the former, `_DEFAULTS` should
mark those keys as `_INTERNAL` so the auto-generator can skip them.

**Q-schema-vs-defaults:** The Blitz #2 plan says "introduce `_cohsex_schema.py`
with a Field table." But `_DEFAULTS` is already a near-complete schema (has
name, default, and inline comments for every key). The simplest implementation
is to add a parallel `_DOCS` dict (name → doc string) and a `_TYPES` dict
(name → type string) rather than a new file. Worth deciding before writing
Blitz #2 so the schema doesn't end up duplicating `_DEFAULTS` a third time.

---

## Summary table (CONTEXT output format)

| # | Recommendation | Priority | Cost |
|---|---------------|----------|------|
| 1 | Update `templates/cohsex.in`: remove 5 deprecated/unknown keys | Now | 10 min |
| 2 | Copy `COHSEX_INPUT.md` to `lorrax_C/docs/` as-is | Now | 5 min |
| 3 | Implement Blitz #2: `_cohsex_schema.py` + auto-gen + CI test | Next | 1 day |
| 4 | Add 20-line QUICKSTART block to top of `COHSEX_INPUT.md` | Next | 30 min |
| 5 | Move `templates/cohsex.in` to `lorrax_C/templates/`; sandbox references repo copy | Soon | 15 min |
| 6 | Add sandbox-vs-repo boundary paragraph to `PORTING.md` | With blitz #3/#5 | 15 min |
| 7 | Implement Rule 2: fix `gen_api_docs.sh` + add to CI | Week 2 | 0.5 day |
| 8 | Implement Rule 3: single ENV_VARS table in ENVIRONMENT_COMPREHENSIVE | Week 2 | 2-3 hr |
