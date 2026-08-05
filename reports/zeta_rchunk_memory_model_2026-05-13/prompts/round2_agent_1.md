# Round 2 — Agent 1: critique + revision

You finished round 1 and wrote `reports/zeta_rchunk_memory_model_2026-05-13/agent_1.md`. The other three agents wrote `agent_2.md`, `agent_3.md`, `agent_4.md` in the same directory, independently.

**Now read those three files.** Then write `reports/zeta_rchunk_memory_model_2026-05-13/agent_1_v2.md` containing the five sections below. Be terse where you can, exhaustive where you must.

## 1. Cross-reading notes

For each of the other three drafts (agent_2, agent_3, agent_4), 5–15 lines:
- What's **stronger** than my v1 (specific section, claim, formula)
- What's **weaker or wrong** (specific section, claim, formula)
- What's **new** — tensors, coefficients, sharding observations, code references, edge cases — that I did not have

Cite by `agent_M.md §N` and your own `agent_1.md §N`.

## 2. Revisions to my v1

Specific places where I now believe my v1 was wrong, incomplete, or imprecise. For each: cite my own section + the new view + the evidence (code line, other-agent reference, or "on reflection").

## 3. Where I still disagree

After reading the others, the **specific mathematical or factual claims** where I still hold my position against one or more agents. Not preferences. For each disagreement:
- The other agent's claim (file + section)
- My counter-claim with reasoning
- What evidence would actually resolve it — be concrete (an HLO dump of which kernel, a specific code line to inspect, a one-GPU smoke test with which inputs)

## 4. Consolidated open questions

The set of items **all four of us should treat as unresolved**. Cross-reference: if Agent 3's §X open question is the same thing as my §Y, say so explicitly. Promote any single-agent flagged uncertainty to a team-level open question if it's load-bearing.

## 5. Recommended next steps

3–6 concrete actions that would close the remaining gaps. Examples of "concrete":
- "Run `fit_zeta_to_h5` once with `JAX_DEBUG_HLO_DUMP=1`, grep for `pair_density` shape, count distinct lifetime offsets"
- "Read `psi_G_store.fetch_psi_rchunk` at lines XYZ and confirm where the FFT box materializes"
- "Add a planner unit test that asserts `r_chunk_picker(CrI3 config) ≈ 12500 within ±20%`"

Avoid "more analysis needed" — every item must say *what* analysis, *on what*, *expecting what*.

---

**Constraints:**
- Read-only on `sources/lorrax_A/`.
- No compute. Desk research only.
- Read `agent_2.md`, `agent_3.md`, `agent_4.md` — those are the inputs to this round.
- Stop when `agent_1_v2.md` is written. Print "Agent 1 round 2 done".
