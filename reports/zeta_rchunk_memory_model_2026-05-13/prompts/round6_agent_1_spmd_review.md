# Round 6 — Agent 1: SPMD reviewer

Read `round6_discussion.md` first (mission, coordination, validation gates). Read `round5_unified_plan.md` (you contributed §2 to it; full plan is the source of truth now).

## Your role

You don't write code. You **review Agent 2's implementation as it lands** for SPMD-safety regressions. Specifically check:

1. **`in_specs`/`out_specs` correctness** on the rewritten `c_q_from_psi_sm._local` and `z_q_from_psi_sm._local`. The `psi_l_X`/`psi_r_X` specs are `P(None, 'x', None, None)`; outputs `P(None, 'x', 'y')`. Confirm Agent 2 didn't accidentally change these.
2. **Scan carry shapes**. Per round 5 final: `(P_l_acc, P_r_acc)` rank-5 each, `c128[nk, ns, r_loc, mu_loc, ns]` per-rank, where `mu_loc = n_rmu/p_x` and `r_loc = r_chunk/p_y`. Confirm initialization is `jnp.zeros(...)` inside the shard_map body (per-rank-local).
3. **The all_gather invariant** (the missing-piece finding from Round 5): per-iter `jax.lax.all_gather(psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)` — IFFT happens BEFORE the gather, not after. Confirm Agent 2's body has this ordering.
4. **No sharding constraints inside the scan body** — `with_sharding_constraint` calls inside a scan inside a shard_map can trigger SPMD WhileOp inflation (see `solve_zeta` comment at `isdf_fitting.py:1119-1141`). Confirm there are none.
5. **`psi_l_X` band-axis slicing**: `lax.dynamic_slice_in_dim` on the band axis is safe (replicated within the shard_map body), but verify Agent 2 doesn't accidentally use a *non-static* slice length.
6. **Mask approach over slice approach** for L/R per-bc band windows — confirm.

## Workflow

1. Re-read `round5_unified_plan.md` §2 (your contribution).
2. Watch for Agent 2's progress in `round6_discussion.md` "Agent 2 → others" section.
3. As Agent 2 lands sub-pieces (smoke test, kernel rewrite, tests), read the diffs on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap` and check the 6 invariants above.
4. Post review notes under "Agent 1 → others". Use `[BLESS] item N` or `[CONCERN] item N — <reason>` markers so Agent 2 can quickly see your status.

## Output

Append your running review to the discussion file. When Agent 2 publishes "Agent 2 round 6 done" AND you've blessed all 6 invariants, print "Agent 1 round 6 review passed". If you find a SPMD-safety regression, write a `BLOCKER:` line and stop.

Read-only on `sources/`. No commits. ~30–60 min of active reviewing once Agent 2 starts producing code.
