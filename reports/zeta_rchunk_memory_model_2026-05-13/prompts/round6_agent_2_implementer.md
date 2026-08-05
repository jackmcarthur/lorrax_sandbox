# Round 6 — Agent 2: Implementation lead

Read `round6_discussion.md` first. Then **read `round5_unified_plan.md` end-to-end** — it's 10.5k words and IS the source of truth. You wrote part of it, so you know the structure, but the synthesis Agent 4 produced incorporates Agents 1, 3, 4's findings on top of yours. Re-read it carefully; specifically §§2, 3, 6.

## Workflow (sequential, do NOT skip gates)

### Phase 1: G0 smoke test (BEFORE the kernel rewrite)

**Critical.** The plan flags io_callback-inside-scan-inside-shard_map as "novel composition" with no prior art. Verify it works on a synth WFN first.

1. Create a new test file `sources/lorrax_B/tests/test_io_callback_nested.py`. Skeleton in `round5_unified_plan.md` §3 (the code block starting `# tests/test_io_callback_scan_in_shard_map.py`).
2. The test: a minimal `shard_map` body that does `lax.scan` over a small index axis; inside the scan body, `io_callback` pulls a per-iter slice from a host-resident numpy array; the scan accumulates. Compare against a Python loop equivalent.
3. Run on **single-device CPU** (`JAX_PLATFORMS=cpu`) for the unit test. The composition either works or fails — environment doesn't matter for the smoke test.
4. If G0 passes: commit the test (`tests/test_io_callback_nested.py`), post "Agent 2 G0 passed" in `round6_discussion.md`, proceed to Phase 2.
5. **If G0 fails**: write a `BLOCKER:` line in the discussion file with the error, stop, and wait for the orchestrator. Do NOT attempt the kernel rewrite on shaky ground.

### Phase 2: Restore `_slice_local_tile_bc` in PsiGStore

The helper was added in `cdd0fba` and removed in `5cadd4b`. Restore it. Reference: `round5_unified_plan.md` §3 + Agent 2's original `agent_2_structural_fix.md` §4a sketch.

- Method signature: `_slice_local_tile_bc(x_idx, y_idx, bc_idx) -> np.ndarray` returning `(nk_tot, bpd_max, ns, ngkmax)`.
- Pad short bcs with zeros (math-neutral when L/R masks are applied).
- Re-add the `_bpd_max` field to `__init__`.
- DO NOT bring back any of the other removed bits (`_slice_local_tile_bc` is the only resurrection).

### Phase 3: Rewrite `c_q_from_psi_sm` and `z_q_from_psi_sm`

**The load-bearing change.** Reference: `round5_unified_plan.md` §6 implementation sketch.

- Inside each function's `_local` shard_map body, replace the all-at-once einsum with `lax.scan` over `n_bc` iterations.
- Per iter: io_callback for the bc tile → IFFT (local) → r-slice → **`jax.lax.all_gather(psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)`** → einsum into `(P_l_acc, P_r_acc)` carries.
- L/R per-bc band windows: mask approach (`jnp.where` on rank-local band axis).
- IFFT BEFORE all_gather (per Round 5 Agent 3's correction; reverse order blows the FFT box to ~80 GB).
- Post-scan: existing IFFT → γ̃-contract → FFT → transpose tail (unchanged).

### Phase 4: Simplify `_make_fit_one_rchunk_kernel._kernel`

- Delete the `psi_G_device_full` lazy-property dependency. Agent 2 will need to revisit `PsiGStore` and remove the buggy property (it's still in the class but no longer called).
- Delete the `psi_Y_parts.append(...)` / `jnp.concatenate` chain (already deleted in `5cadd4b` — confirm).
- The new `_kernel` body just calls the new `z_q_from_psi_sm` / `c_q_from_psi_sm` directly with `psi_l_X`, `psi_r_X` + a closure-captured `psi_G_store` reference (the helpers will pull bcs via io_callback inside their shard_map).

### Phase 5: G1 bit-identity gate

Mirror Agent 4's gate spec from `round5_unified_plan.md` §5:
- MoS2 3×3 synth WFN, single-channel charge.
- Run today's `_kernel` body vs the new body. `rtol=1e-10, atol=1e-12` (NOT bit-equal; sum order differs).
- If G1 fails: stop, debug. Write `BLOCKER:` if you can't resolve quickly.

### Phase 6: Commit

Single commit on `agent/zeta-bc-scan-shardmap`. Commit message enumerates the changes + quotes G0 + G1 results.

Then post "Agent 2 round 6 done" in the discussion file. **At that point** Agents 1, 3, 4 take over for review and validation gates G2 + G3.

## Constraints

- **Branch**: `sources/lorrax_B` `agent/zeta-bc-scan-shardmap`. Don't create a new branch.
- **Bit-identity is the gate, not feel.** If G1 fails by even one rounding unit, debug — don't waive.
- **Smoke test G0 is non-negotiable.** Don't skip it because "io_callback should work."
- Communicate via `round6_discussion.md` (poll before each work cycle). Reviewers may flag concerns in real time; respond before continuing.
- If the SLURM allocation isn't up yet for the bit-identity test, MoS2 3×3 should run on a single GPU via `lxrun` once it lands. Get the smoke test (CPU) done first, then wait if needed.

Print "Agent 2 round 6 done" when committed + G1 passed.
