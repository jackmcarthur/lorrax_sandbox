# TRS-blind sym handling audit — STATUS

**Started**: 2026-05-14
**Working tree**: `sources/lorrax_B` (branch `agent/zeta-bc-scan-shardmap`, HEAD `c796420`)
**Mission**: enumerate and fix every code path that conflates spatial-only sym (`sym_matrices`, length `ntran`) with TRS-augmented sym (`sym_mats_k`, length `2·ntran`).

## The bug (one paragraph)

`symmetry_maps.py:129-130` augments `sym_mats_k` to `[mtrx, -mtrx]` (length `2·ntran`) so it can fold k/q grids using time-reversal symmetry. `find_irreducible_qpoints` (and `find_symmetry_ops_simple`) then resolve a sym index into this augmented table — values `≥ ntran` indicate TRS-augmented operations. Downstream consumers (`_unfold_v_q_ibz_to_full`, the centroid-permutation table `sym_perm` built by `compute_centroid_sym_perm`, the ψ k-unfold) use **only the spatial half** of the sym table. Mismatched indexing → silent OOB clips in JAX gather → wrong physics, with no exception raised.

The k-side path has a documented warning ("Non-symmorphic phases are NOT applied for these k-points. Use `noinv=.true.` in QE to avoid this.") that admits the issue exists. The q-side path has no warning at all and silently produces wrong V_q.

## Evidence

- MoS2 3×3, same orbit-closed centroid file, two runs (IBZ cascade vs. forced full-BZ):
  - ζ at IBZ q's: **bit-identical** — fit_zeta IBZ-only path is correct
  - Σ_X eigenvalues: **max |Δ| = 6.89 eV at k=0 band 44** — wrong by orders of magnitude
- Trace at `/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_rchunk_memory_model_2026-05-13/mos2_same_basis_ibz_vs_fullbz.md` (subagent partial output)
- The MoS2 NSCF emits `SymMaps: 4/9 full-BZ k-points require time-reversal symmetry for unfolding` at runtime.

## Existing nosym runs (Agent 4 references; do NOT need to rerun BGW)

| Path | What it pairs with | Use |
|---|---|---|
| `runs/MoS2/02_mos2_3x3_nosym` | `runs/MoS2/00_mos2_3x3_cohsex` | MoS2 sym-vs-nosym Σ comparison |
| `runs/MoS2/B_03_mos2_3x3_nosym_60Ry` | same family @ higher cutoff | |
| `runs/Si/02_si_4x4x4_nosym` | `runs/Si/01_si_4x4x4_nosymmorphic` | Si has inversion — sym/nosym MUST agree to ULP regardless of the TRS bug. Use as **null hypothesis** check. |
| `runs/Si/03_si_10x10x10_nosym_timing` | (perf reference) | |
| `runs/Si_pseudobands/00_si_2x2x2_60Ry/21_lorrax_cohsex_nosym_parity` | the rest of `00_si_2x2x2_60Ry/` | named "parity" — was set up as a sym/nosym correctness check |
| `runs/Si_pseudobands/00_si_2x2x2_60Ry/{26,27,28,29}_cohsex_pb_*_nosym*` | corresponding `_sym` variants | pseudobands + sym/nosym 4-way grid |
| `runs/sym_fix_test/{bi,mos2}_{sym,nosym}` | each other | DFT-only scaffolds (no GW yet); could be used for fresh post-fix runs if budget allows |

## What each agent owns

1. **Agent 1 — Scope**: enumerate every TRS-blind site → output `agent_1_scope_report.md`
2. **Agent 2 — Convention design**: design canonical TRS abstraction + write the patch → output `agent_2_design.md` + the patch on a sub-branch
3. **Agent 3 — Synthetic + e2e regression**: extend the synthetic round-trip test to expose the bug + prove fix → output `agent_3_test_report.md`
4. **Agent 4 — Existing nosym reference audit**: use the table above to identify pre-fix vs post-fix correctness signals. NO new BGW runs; only LORRAX reruns where existing nosym BGW (or LORRAX-nosym) results are already on disk → output `agent_4_reference_audit.md`

## File discipline

- All four agents write to `reports/trs_sym_audit_2026-05-14/`.
- Cross-agent messages → `discussion.md` (timestamped, agent-tagged).
- Each agent's primary deliverable → `agent_N_<topic>.md`.
- Do NOT touch the prior `reports/zeta_rchunk_memory_model_2026-05-13/` folder (closed).
