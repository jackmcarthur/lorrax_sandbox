You are **Agent 1 of 4** on the LORRAX docs paring team.

**Read first:**
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/CONTEXT.md`

It has the framing, the integration-branch state, the verdict
vocabulary (KEEP / MERGE / ARCHIVE / DELETE / REGENERATE), and the
output structure.

## Your slice: **Core physics / algorithm reference docs**

These are the big, math-heavy comprehensive references:

| Doc | Lines | Notes |
|---|---:|---|
| `docs/PHYSICS_COMPREHENSIVE.md` | 977 | §11 explicitly supersedes §3-5; how much of §3-5 should remain? |
| `docs/ZETA_V_Q_ALGORITHMS.md` | 1146 | New (May 17-19 memory-model refit). May subsume parts of PHYSICS §11 or vice versa. |
| `docs/SYMMETRY_COMPREHENSIVE.md` | 665 | New. Overlap with PHYSICS §on-symmetry? With the TRS-aware unfold commits in code? |
| `docs/MEMORY_MODEL.md` | 1004 | +1030 lines added recently (per-stage refit). May overlap with the "Memory" sections inside PHYSICS or ZETA_V_Q. |
| `docs/MINIMAX_QUADRATURE.md` | 340 | GW frequency integral discretization. Standalone but check if it overlaps with the minimax/PPM material in PHYSICS / `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` (the GN_PPM doc is owned by Agent 3). |

Total: ~4100 lines in your slice. The opportunity here is the
biggest single-doc reduction. The risk is shredding load-bearing
prose someone needs.

## What to look for

### A. Supersession patterns

- `docs/PHYSICS_COMPREHENSIVE.md`: round-1 of the install-blitz noted
  "§11 supersedes §3-5" with a top-pointer. Read it. Decide
  per-section: keep, archive-in-place (with a clear deprecation
  pointer), or move to `docs/archive/`. Don't preserve "historical
  prose for completeness" — if the section is wrong vs current code,
  it actively misleads.
- `docs/ZETA_V_Q_ALGORITHMS.md` is new and likely overlaps with PHYSICS
  §11. Which one is the canonical home for ζ-fit / V_q algorithm
  prose? Pick one and recommend the other be folded in or pointed at.
- `docs/MEMORY_MODEL.md` may now duplicate the "memory model" portions
  of ZETA_V_Q_ALGORITHMS. Same question.

### B. Math notation consistency

These four docs have ~4100 lines of equations. Skim for:
- Equation numbering reuse across docs (Eq. 5 in PHYSICS vs Eq. 5 in
  ZETA_V_Q_ALGORITHMS — confusing if a section moves).
- Symbol conventions (Σ_X vs Σ_x, ζ vs zeta, μ_L = i indefinite case
  from the bispinor work).

Don't propose a rewrite of the math. Just call out where consistency
work would pay off if a section moves.

### C. Stale code references inside the math docs

These docs cite specific code paths (e.g. "implemented in
`src/common/isdf_fitting.py:1458-2260`"). Spot-check at least 5 of
these citations against current code — line ranges drift, function
names change. Document the drift; recommend either de-citing or
shifting to a `:line` permalink at a specific commit.

### D. Internal "see X" references

If §11 of PHYSICS says "see MEMORY_MODEL.md §3.2" and MEMORY_MODEL.md
has been refit, the §-numbers may not exist anymore. Check the
top-of-doc table-of-contents for each of your four docs and verify
cross-references still resolve.

### E. Web search latitude

If you want to compare against an exemplar (e.g., JAX's docs, scipy's
math docs), web search is authorized. Cite. But the goal is paring
LORRAX's docs, not adopting an external structure wholesale.

## Output

Write to:
`/pscratch/sd/j/jackm/lorrax_sandbox/reports/lorrax_docs_pare_2026-05-22/agent_1.md`

Cover the six sections from CONTEXT §"Output structure". Commit to a
verdict per doc (and per section where finer granularity matters).
The recommendation table is the most important deliverable; the prose
around it is supporting evidence.

## Constraints (from CONTEXT §"Constraints")

- Read-only on `sources/lorrax_D/`.
- No compute.
- Stay in your own report file. Do NOT read `agent_2.md`, `agent_3.md`,
  `agent_4.md`.
- Stop when written. Print: `Agent 1 done — see agent_1.md`.

Start by reading the four docs in your slice (use the integration
branch HEAD on `sources/lorrax_D/agent/install-blitz-integration`).
