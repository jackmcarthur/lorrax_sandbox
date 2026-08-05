You are **Agent 4 of 4** on the LORRAX docs paring team.

**Read first:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/CONTEXT.md`

## Your slice: **Cohsex.in reference + cross-doc canonicalization + sandbox-vs-repo split**

| Doc / artifact | Lines | Where |
|---|---:|---|
| `COHSEX_INPUT.md` | 411 | **Sandbox** (`/pscratch/sd/j/jackm/lorrax_sandbox/docs/docs_gwjax/`) — NOT in the lorrax_C repo. |
| `templates/cohsex.in` | ~40 | **Sandbox** template; copied into every new run dir. |
| `src/gw/gw_config.py` | (parser) | The `_DEFAULTS` dict ground truth. Last touched by main (commit `5cadd4b` deleted `psig_k_chunk_size`). |
| Sandbox `AGENTS.md` | (read-only) | Sandbox conventions; references docs/docs_gwjax/. |
| Sandbox `PARSE_OUTPUTS.md`, `runs/.../cohsex.in` examples | — | Downstream consumers of the schema. |

Plus the **cross-doc canonicalization** question: what content
belongs in the lorrax_C repo (`docs/`) vs the sandbox? Multiple docs
across the other three agents' slices touch this same boundary.

## What to look for

### A. Cohsex.in reference drift

The sandbox's `COHSEX_INPUT.md` is hand-maintained. The parser's
`_DEFAULTS` dict (at `src/gw/gw_config.py` on integration HEAD —
remember main DELETED `psig_k_chunk_size` per `5cadd4b`) is ground
truth. Concretely:

1. Run (mentally, by reading the parser): `grep '"[a-z_]*":' src/gw/gw_config.py | grep -v '^#' | sort -u` — that's the set of keys the parser recognizes.
2. Compare to what `COHSEX_INPUT.md` documents (grep for `### \`key\``
   or similar).
3. Identify:
   - Keys the parser knows but the doc doesn't (undocumented).
   - Keys the doc claims but the parser doesn't (lies / stale).
   - Doc text descriptions that disagree with the parser default.

This is the table that motivated Blitz #2 (deferred). Document the
drift; recommend the path forward.

### B. Should COHSEX_INPUT.md live in the repo or the sandbox?

The install-blitz consensus said it should live in `lorrax_C/docs/`
because a second user `git clone`-ing the repo doesn't get the
sandbox. Blitz #2 was supposed to move it; deferred. **Re-affirm or
revise this recommendation given current state.**

If you re-affirm:
- What's the minimum-viable move? Just copy it as-is, or wait for
  Blitz #2's auto-generation?
- Until Blitz #2 lands, who maintains it?

### C. Templates/ in sandbox

`/pscratch/sd/j/jackm/lorrax_sandbox/templates/cohsex.in` is what the
Build-Inputs skill copies into every new run dir. Per round-1 of the
install-blitz, this template is itself stale (uses `output_file`
which is deprecated, etc.). Recommend:
- Update the template inline now.
- Auto-generate the template from the parser schema (Blitz #2-style).
- Move the template into the lorrax_C repo at `templates/cohsex.in`
  so it's not sandbox-only.

### D. Sandbox-vs-repo doc split — the meta-question

Look across what the other agents are reviewing (the lorrax_C `docs/`
tree). Identify everything that's *also* in the sandbox or *should
be* (cohsex.in reference, agent-team templates, skills, recent reports
mentioning install-blitz consensus).

Recommend:
- Which content belongs in the repo (for second-user reach).
- Which is genuinely sandbox-only (multi-agent A/B/C/D coordination
  scaffolding, reports/, the sandbox `templates/` that drive
  Build-Inputs).
- Where the boundary needs a one-paragraph callout in PORTING.md.

This question intersects Agent 2's "install/porting" and Agent 3's
"plans/WIP" slices — you're the natural arbiter for "this lives in
the wrong repo."

### E. Cross-doc canonicalization patterns

The most leverage in docs paring usually comes from **regeneration
from a single source**:
- Cohsex.in reference → from parser schema.
- Module-map / API reference → from docstrings via `pdoc` (already
  scaffolded per `docs/gen_api_docs.sh`).
- Modulefile env-var reference → from a single ENV_VARS table
  (proposed in install-blitz consensus, never landed).

Recommend up to three such "single source of truth" rules,
prioritized by leverage (which one closes the most doc-vs-code
drift).

### F. Synthesis tip

You are explicitly the synthesis-leaning agent here. The other three
each own ~3000-4500 lines of doc; you own the structural questions
above plus ~500 lines of cohsex-related content. Your draft can be
shorter (target ~250-400 lines) but should commit to the
boundary-defining recommendations the other three will then defer to.

### G. Web search latitude

Look at how comparable scientific-software packages handle their
"input file" reference (BerkeleyGW's input docs, Quantum ESPRESSO's
INPUT_PW.txt, ABINIT's variables reference, JAX's API reference).
Cite. Pick the closest exemplar to LORRAX's scale (one author, FFI +
configparser + cohsex.in pattern).

## Output

Write to:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/agent_4.md`

Cover the six sections from CONTEXT §"Output structure", with
emphasis on **boundary recommendations** (sandbox-vs-repo split,
auto-generation rules).

## Constraints (from CONTEXT)

- Read-only on both `sources/lorrax_D/` and the sandbox-level docs.
- No compute.
- Stay in your own report file. Do NOT read sibling drafts.
- Stop when written. Print: `Agent 4 done — see agent_4.md`.

Start by comparing the parser's `_DEFAULTS` keys against
`COHSEX_INPUT.md`'s documented set.
