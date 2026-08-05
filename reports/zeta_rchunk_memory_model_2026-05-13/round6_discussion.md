# Round 6 discussion — implement Round 5's scan-inside-shard_map plan

## Mission

Implement `round5_unified_plan.md`. Land a single commit on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap` that replaces the buggy concat-reshuffle integration with the scan-inside-shard_map design.

## Status snapshot (orchestrator-maintained)

- **lorrax_B branch head**: `5cadd4b` (the current buggy integration — has the tracer leak + remat).
- **Plan**: `round5_unified_plan.md` — 10.5k words, 8 sections, ALL FOUR AGENTS SIGNED OFF. This is the source of truth.
- **SLURM allocation**: fresh 4-hour `lx-alloc-jackm` with `hbm80g`. Use `lxattach` to pick it up.
- **Critical pre-implementation step**: §7 risk in the plan flags io_callback-inside-scan-inside-shard_map as "novel composition" (no prior art in this codebase). Verify it works on a synth WFN BEFORE the kernel rewrite. Don't bet CrI3 on an unverified nesting.

## File-polling protocol (same as Round 5)

Each agent reads this file before each work cycle. Write peer-bound messages under named sections. **No orchestrator routing.** Implementation is sequential by file-touch order — see "Coordination" below.

## Coordination — who touches what

**Sequential by file** to avoid conflicts:
1. **Agent 2 (impl lead)**: writes the smoke test FIRST, then the kernel rewrite. Owns commits on lorrax_B.
2. **Agent 1 (SPMD reviewer)**: reviews Agent 2's code as it lands. Flags any deviation from `round5_unified_plan.md` §2.
3. **Agent 3 (HLO validator)**: when Agent 2's kernel rewrite lands + the bit-identity gate passes, dumps HLO at CrI3 6×6 80 Ry. Validates against §4 predictions.
4. **Agent 4 (numerics validator)**: when Agent 2's kernel rewrite lands, runs MoS2 3×3 bit-identity test + supervises CrI3 e2e run. Validates against §5 gates.

Dependencies:
- Agent 1 & Agent 4 don't need Agent 2 to commit before STARTING their reviews (read the plan, prepare checklists).
- Agent 3 needs the new commit + a working CrI3 invocation. Until then, Agent 3 can prep the HLO comparison checklist (Round 4 baseline vs predicted post-Round 6).

## Validation gates (from `round5_unified_plan.md` §5)

| Gate | Action | Pass criterion |
|---|---|---|
| **G0** | io_callback-inside-scan-inside-shard_map smoke test on synth WFN | runs without error; output bit-equals a Python-loop equivalent |
| **G1** | MoS2 3×3 ζ-fit bit-identity on lorrax_B (new) vs lorrax_A baseline | `rtol=1e-10, atol=1e-12` (NOT bit-equal — sum order differs) |
| **G2** | HLO slot count + zero "Involuntary full rematerialization" warnings | ≤ 5 substantive slots; 0 remat warnings |
| **G3** | CrI3 6×6 80 Ry end-to-end runs through all 16 r-chunks + remainder | no OOM, no tracer leak, total preallocated-temp ≤ 15 GiB |

**Do not proceed past a failed gate.** If G0 fails, the plan is wrong — escalate to the orchestrator before continuing.

---

## Agent 1 → others

**2026-05-13 — Agent 1 review-ready.** Reviewing branch `agent/zeta-bc-scan-shardmap` (currently at `5cadd4b`). Will poll for Agent 2's commits + diff each one against the 6 SPMD invariants below.

### Review checklist (from prompt + Round 5 §2)

| # | Invariant | Source of truth |
|---|---|---|
| 1 | `z_q_from_psi_sm._local` / `c_q_from_psi_sm._local`: `in_specs=(P(None,'x',None,None), P(None,'x',None,None), P(),P(),P(),P(),P())` (psi_l_X, psi_r_X, γ̃ perms/phases, r_start), `out_specs=P(None,'x','y')`, `check_rep=False` | Plan §2.2 |
| 2 | Carry init: `jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)` INSIDE the shard_map body. `mu_loc = n_rmu // p_x`, `r_loc = n_zchunk // p_y`. NO `NamedSharding` on the carry. | Plan §2.3, §2.4 |
| 3 | Per-iter order: `io_callback` → `to_rchunk_inner` (IFFT + r-slab using `r0_local = r_start + axis_index('y') * r_loc`) → `all_gather(axis_name=('x','y'), axis=1, tiled=True)` → masks/einsums. **all_gather AFTER IFFT, not before.** | Plan §2.9 — load-bearing |
| 4 | No `with_sharding_constraint` inside the scan body or anywhere in `_local`. No collectives inside body other than the explicit per-iter `all_gather` (no `psum`, no nested `axis_index` that varies per iter). | Plan §2.3 risks 1+2; `solve_zeta` prior-art `isdf_fitting.py:1119-1141` |
| 5 | `psi_l_X` per-bc slice: `lax.dynamic_slice_in_dim(psi_l_X_, b_lo_global[bc_idx] - L_lo_g, bpd_max_global, axis=2)` — **static slice length** `bpd_max_global` (closure constant). Offset is traced (via `bc_idx`), length is static. Same for psi_r_X with R offset. | Plan §2.6, §6.5 |
| 6 | L/R per-bc band windows use **mask approach** (`jnp.where(mask[None,:,None,None], psi_Y_bc, 0)`), NOT slice approach. mask built from global band indices + `bc_valid` cutoff for short final bc. | Plan §2.5, §2.9 |

### Additional secondary checks I'll perform

- **§2.9 `tiled=True` axis ordering**: when Agent 2 wires the per-iter all_gather, check the smoke test verifies the gathered band axis matches the host-tile bc-stacking convention. If not, a `jnp.take` permutation should appear inside the body.
- **§2.4 `lax.scan` donation**: confirm Agent 2 does NOT pass an `unroll=` argument (default unroll=1 keeps WhileOp atomic, scan-internal aliasing kicks in).
- **§6.5 slicer API**: `_slice_local_tile_bc(x_idx, y_idx, bc_idx_traced) → (nk, bpd_per_bc, ns, ngkmax)` — bc-indexed, NOT flat-row.
- **`accumulate_rchunk_to_gflat` not touched**: it's the in-repo precedent; Agent 2's edits should be limited to `isdf_fitting.py`, `psi_G_store.py`, and possibly `wfn_transforms.py` (deletion of `gflat_to_rchunk`).

### Marker convention I'll use

- `[BLESS] item N` — invariant satisfied in commit `<sha>`.
- `[CONCERN] item N — <reason>` — non-blocking but Agent 2 should address.
- `BLOCKER: item N — <reason>` — Agent 2 must fix before proceeding.

Standing by. Will diff each Agent 2 commit as it lands.

---

**2026-05-13 23:50 — Agent 1 review of Agent 2's working tree (Phase 1-3, pre-commit).**

Reviewed against the 6 SPMD invariants. **5 [BLESS], 1 [CONCERN] (dead code, not a SPMD issue).** All SPMD invariants pass.

### G0 smoke test (`tests/test_io_callback_nested.py`)

`[BLESS] G0 — 6/6 tests pass per Agent 2's note.` Read the test:
- G0a/b/c/d/e all use `in_specs=()`, `out_specs=P()`, `check_rep=False` — composition smoke only (production specs tested elsewhere).
- Rank-5 carry pattern (`jnp.zeros(..., c128)` inside body) exercised in G0d.
- `dynamic_slice_in_dim(psi_X_const, bc_idx, 1, axis=0)` — static length, traced offset. Matches production pattern.
- Mask approach exercised in G0e.
- All four primitives compose (io_callback × scan × shard_map × all_gather).

⚠️ **G0 reservation noted in §2.9 of the plan still stands**: `tiled=True` axis ordering on 1×1 mesh is a no-op. The "gathered band axis matches host-tile bc-stacking" question gets answered at G1 (MoS2 3×3, 2×2 mesh) — Agent 4's bit-identity test will catch any band-axis permutation mismatch as a numerics failure.

### Invariant 1 — `_local` shard_map specs (`z_q_from_psi_sm._local`, isdf_fitting.py:536-541)

```python
in_specs=(L_spec, L_spec, P(), P(), P(), P(), P())     # L_spec = P(None, 'x', None, None)
out_specs=out_spec                                      # P(None, 'x', 'y')
check_rep=False
```

Arg list: `(psi_l_X_, psi_r_X_, perm_L_, phase_L_, perm_R_, phase_R_, r_start_)` — 7 args. ✅ matches plan §2.2. **[BLESS] item 1**.

### Invariant 2 — Carry init shapes (isdf_fitting.py:548-576)

```python
x_idx = jax.lax.axis_index('x')                         # loop-invariant
y_idx = jax.lax.axis_index('y')                         # loop-invariant
mu_loc = psi_l_X_.shape[1]                              # = n_rmu // p_x
r_loc  = n_zchunk // p_y                                # = n_zchunk // p_y (line 458, pre-check)
# ...
r0_local = r_start_ + y_idx * jnp.int32(r_loc)         # per-rank r-slab offset
P_l_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)
P_r_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)
```

- Carry shape matches plan §2.3 — `c128[nk, ns, r_loc, mu_loc, ns]`. ✅
- `mu_loc` inferred from per-rank shape (correct under L_spec). ✅
- `r_loc` from `n_zchunk // p_y` with pre-flight check at line 454-457 (`raise ValueError` if not divisible). ✅
- No `NamedSharding` on the carry. ✅
- INSIDE the shard_map body. ✅
- `axis_index` called once before scan (lines 548-549), not per-iter. ✅

**[BLESS] item 2**.

### Invariant 3 — Per-iter order: io_callback → IFFT → all_gather (isdf_fitting.py:578-611)

```python
def body(carry, bc_idx):
    P_l_acc, P_r_acc = carry
    # (1) io_callback
    psi_G_bc_local = _io_callback(_slicer_host, _slicer_out_sds,
                                   x_idx, y_idx, bc_idx, ordered=False)
    # (2) IFFT + r-slab
    psi_Y_bc_local = to_rchunk_inner(psi_G_bc_local, g_index_dev, fft_grid_t,
                                       r0_local, r_loc,
                                       kvecs_frac=kvecs_frac_dev, norm="ortho")
    # (3) all_gather AFTER IFFT, BEFORE einsum
    psi_Y_bc = jax.lax.all_gather(psi_Y_bc_local, axis_name=('x', 'y'),
                                    axis=1, tiled=True)
    # (4) masks
    # (5) psi_l_X / psi_r_X slice
    # (6) two einsums
```

**Ordering is exactly the plan §2.9 sequence**: io_callback → to_rchunk_inner (IFFT + slab) → all_gather → masks/einsums. **IFFT-FIRST, gather-second.** ✅

`ordered=False` on io_callback matches plan §3.4 recommendation. ✅

**[BLESS] item 3**. This is the load-bearing piece; the lorrax_A 80 GB FFT box would re-appear if Agent 2 had inverted the order.

### Invariant 4 — No `with_sharding_constraint` inside `_local` or scan body

Grepped `isdf_fitting.py` for `with_sharding_constraint` — 9 occurrences, all OUTSIDE `_local`:
- `_identity_pad_block_diagonal` (lines 751, 758) — padding helper.
- `factor_c_q` (line 964) — Cholesky helper.
- Block-tile helpers (lines 993, 1001).
- `_reshard_zeta_*` (lines 1031, 1049, 1051) — explicit reshard helpers.
- `solve_zeta` paths (lines 1270, 1277, 1329, 1330) — distinct function.

Inside `z_q_from_psi_sm._local` (lines 540-670): **zero `with_sharding_constraint`**. ✅

Inside `c_q_from_psi_sm._local`: also zero (unchanged from before).

Inside the scan body (lines 578-638): only one collective — `lax.all_gather` (line 598-600). No `psum`, no per-iter `axis_index`. ✅

**[BLESS] item 4**. The `solve_zeta` WhileOp trap precondition (sharded carry under SPMD) does not arise here.

### Invariant 5 — `psi_l_X` per-bc slice (isdf_fitting.py:618-625)

```python
psi_l_X_bc = jax.lax.dynamic_slice_in_dim(
    psi_l_X_padded, psi_l_X_bc_offset[bc_idx],
    bpd_max_global, axis=2)
```

- **Static length**: `bpd_max_global = bpd_max * P_total` — closure constant (line 493). ✅
- **Traced offset**: `psi_l_X_bc_offset[bc_idx]` — table gather indexed by traced `bc_idx`. ✅
- Same pattern for `psi_r_X` (line 623-625). ✅

⭐ **Notable improvement over my §2 sketch**: Agent 2 adds **front-padding** to psi_l_X / psi_r_X (lines 509-520, 558-567) to handle the case where `bc.lo < L_lo_g` (which would yield a negative dynamic_slice offset). My §6.5 sketch had this latent bug — Agent 2 noticed and fixed it. Pad rows are masked-zero in the einsum, so it's math-neutral. **Good engineering catch.**

**[BLESS] item 5**.

### Invariant 6 — Mask approach over slice approach (isdf_fitting.py:603-611, 621-627)

```python
g_axis = (b_lo_global_arr[bc_idx]
          + jnp.arange(bpd_max_global, dtype=jnp.int32))
bc_valid = g_axis < b_hi_global_arr[bc_idx]                # short final bc handled
l_mask = ((g_axis >= L_lo_g) & (g_axis < L_hi_g) & bc_valid)
r_mask = ((g_axis >= R_lo_g) & (g_axis < R_hi_g) & bc_valid)
psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
psi_r_X_bc = jnp.where(r_mask[None, None, :, None], psi_r_X_bc, 0)
```

- Mask approach (pointwise `jnp.where`), not dynamic_slice on the band axis. ✅
- Global-band-index masks, gathered by traced `bc_idx` from static `b_lo_global_arr` / `b_hi_global_arr` tables. ✅
- `bc_valid` cutoff for short final bc. ✅
- Mask applied to BOTH `psi_*_Y_bc` (rank-local, post-gather) AND `psi_*_X_bc` (replicated-bands slice) — symmetric masking ensures consistent zero-product in the einsum.

**[BLESS] item 6**.

### Secondary checks

- ✅ **`lax.scan` no `unroll=`** (line 644-646): default unroll=1, keeps WhileOp atomic for scan-internal aliasing.
- ✅ **`tiled=True` on all_gather** (line 600): correct.
- ✅ **Slicer API bc-indexed** (psi_G_store.py:272-309): `_slice_local_tile_bc(x_idx, y_idx, bc_idx) → (nk, _bpd_max, ns, ngkmax)`. Matches plan §6.5.
- ✅ **Slicer uses `np.zeros` not `np.empty`** (line 307): pad rows exactly zero per §7.3 risk.
- ✅ **`_psi_G_device_full` lazy property GONE** (psi_G_store.py): only a docstring reference at line 255; no live code path. Tracer-leak source eliminated.
- ✅ **`gflat_to_rchunk` GONE** (wfn_transforms.py): no occurrence; `to_rchunk_inner`, `to_rchunk`, `to_rmu`, `gflat_to_rmu`, `accumulate_rchunk_to_gflat` all kept.
- ✅ **`c_q_from_psi_sm` unchanged** (isdf_fitting.py:256-375): CCT-side rewrite deferred per plan §8.
- ✅ **Norm pre-multiply at kernel call site** (isdf_fitting.py:1505-1506): `psi_l_X_scaled = psi_l_rmuT_X_fit / norms_l[None, None, :, None]` — matches plan §5.3.4 / §6.3.
- ✅ **`accumulate_rchunk_to_gflat` untouched** (wfn_transforms.py:870): in-repo precedent preserved.

### `[CONCERN]` — dead-code cleanup needed

`isdf_fitting.py:700-702` has an orphan `return` statement (tab-indented inside `z_q_from_psi_sm` after the real `return` at line 692-693) referencing undefined `psi_l_Y`, `psi_r_Y`. **NOT a SPMD issue and NOT a runtime crash** (Python accepts unreachable code after the first `return`; the undefined names never evaluate). But it's stale text left over from the rewrite — please clean up before commit. 3-line delete.

```python
# Currently at lines 700-702 (tab-indented, after the real return):
    return _pair_pipeline_sm_cache[cache_key](
        psi_l_X, psi_l_Y, psi_r_X, psi_r_Y,
        perm_L, phase_L, perm_R, phase_R)
```

### Net assessment

**Agent 1 SPMD review: PASS (modulo dead-code cleanup).** No SPMD-safety regressions. The new kernel body is a faithful implementation of plan §2 with one improvement (front-padding) that fixes a latent issue I missed in the sketch. The WhileOp/SPMD trap precondition (sharded carry visible to the partitioner) does not arise — carry is rank-local inside the manual-mode body.

@Agent 4 — when G1 lands, no need to block on me for SPMD signoff. Run the bit-identity test.
@Agent 3 — after Agent 2's commit lands, the HLO dump can fire. Predicted slot map in your §4 should match (zero `psi_Y_full`-class slots; one FFT box inside the WhileOp; ≤ 2 carry-class slots).
@Agent 2 — please delete the 3 dead-code lines (lines 700-702 in `isdf_fitting.py`) before commit. After that's done + your commit lands, ping me and I'll print the final "Agent 1 round 6 review passed" sentinel.

---

**2026-05-13 23:55 — Agent 1 review of committed `f567aa0`.**

Re-reviewed the committed version. Agent 2 made **two important refinements** beyond my §2.10 sketch — both improvements that I want to acknowledge explicitly. All 6 SPMD invariants still pass.

### Refinement 1 — `g_index` / `kvecs_frac` passed via `in_specs`, not closure (isdf_fitting.py:541-551)

```python
g_index_spec    = P(None, None, None, None)
kvecs_frac_spec = P(None, None)
@partial(shard_map, mesh=mesh_xy,
         in_specs=(L_spec, L_spec, P(), P(), P(), P(), P(),
                   g_index_spec, kvecs_frac_spec),
         out_specs=out_spec, check_rep=False)
def _local(psi_l_X_, psi_r_X_, perm_L_, phase_L_,
           perm_R_, phase_R_, r_start_,
           g_index_dev, kvecs_frac_dev):
```

Agent 2's rationale: *"closure-captured Auto-sharded arrays trip a mesh-context mismatch under multi-device shard_map"*.

**This is a real SPMD subtlety** that I missed in my §2 sketch. Closure-captured `jax.Array`s carry their own (Auto-mode) `NamedSharding` annotation; when a Manual-mode `shard_map` body tries to access them, the mesh contexts can collide. The fix — pass through `in_specs` with explicit replicated `P()` specs — makes the manual-mode access explicit and avoids the trap.

The in_specs list now has 9 entries (vs 7 in my sketch) but the design property is unchanged: all closure-class constants are replicated (`P(None,...)`), so SPMD-safety is preserved. **[BLESS] item 1 — committed version is actually BETTER than my sketch.**

### Refinement 2 — FULL-r psi_Y_local + gather + slice-after, NOT slice-before (isdf_fitting.py:588-640)

Agent 2's r-slab strategy comment (lines 588-600):

> We CANNOT slice the r-axis per-rank BEFORE the all_gather over bands — that would mix r-slabs from different y-ranks at the same gathered band position (r-incoherence at the einsum).

Agent 2 is **right and I was wrong**. My §2.10 sketch had:
```python
r0_local = r_start_ + y_idx * r_loc
psi_Y_bc_local = to_rchunk_inner(..., r0_local, r_loc, ...)   # WRONG: r-slab per y-rank
psi_Y_bc = lax.all_gather(psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)
```

After all_gather, band `j` in the gathered tensor would have its source rank's r-slab — DIFFERENT r-positions for each band. The einsum `'kmna, knbr → karmb'` would silently produce wrong numerics (mixing different r-positions across the band-summed term).

Agent 2's fix (lines 622-639):
```python
psi_Y_bc_local_full_r = to_rchunk_inner(
    psi_G_bc_local, g_index_dev, fft_grid_t,
    r_start_, n_zchunk,             # FULL r-chunk, NOT r_loc
    kvecs_frac=kvecs_frac_dev, norm="ortho")
psi_Y_bc_full_r = jax.lax.all_gather(
    psi_Y_bc_local_full_r, axis_name=('x','y'), axis=1, tiled=True)
psi_Y_bc = jax.lax.dynamic_slice_in_dim(
    psi_Y_bc_full_r, r0_y_offset, r_loc, axis=3)
```

After gather, every rank has full bands × full r-chunk → coherent band ↔ r mapping. Then slice to per-rank r_loc.

**Cost analysis** (per-iter, scan-aliased):
- FFT-box transient unchanged (~1.3 GB; FFT always runs on full FFT grid regardless of r_len).
- `psi_Y_bc_local_full_r`: `c128[36, 1, 2, 73648]` ≈ 84 MB per rank.
- `psi_Y_bc_full_r` (post-gather): `c128[36, 16, 2, 73648]` ≈ 1.36 GB per rank.
- `psi_Y_bc` (post-slice): `c128[36, 16, 2, 18412]` ≈ 340 MB per rank.
- Carries unchanged: 2 × 3.71 GiB.

Net per-iter additional transient: ~1 GB more than my §2.10 estimate. But it's scan-aliased → single slot per buffer-class. Peak preallocated-temp should still land in the **13–15 GiB** band Agent 3 predicted. **[BLESS] item 3 — committed version corrects a correctness bug I missed.**

### Other invariants — all pass on committed code

- **Item 2** (carry init): unchanged ✅
- **Item 4** (no `with_sharding_constraint` in body): grepped inside `z_q_from_psi_sm` — 0 occurrences ✅
- **Item 5** (static slice length on psi_*_X with front-padding): unchanged ✅
- **Item 6** (mask approach with `bc_valid`): unchanged ✅
- **Slicer API bc-indexed** (psi_G_store.py:272-309): unchanged ✅
- **`_psi_G_device_full` GONE**: confirmed ✅
- **`gflat_to_rchunk` GONE** from both wfn_transforms.py and isdf_fitting.py: confirmed ✅
- **`c_q_from_psi_sm` unchanged**: CCT-side deferred per plan §8 ✅
- **`accumulate_rchunk_to_gflat` untouched**: ✅

### [CONCERN] item — dead-code cleanup STILL not done

The orphan `return` statement is still at `isdf_fitting.py:743-745` (tab-indented, after the real return at line 738):

```
^Ireturn _pair_pipeline_sm_cache[cache_key](
^I^Ipsi_l_X, psi_l_Y, psi_r_X, psi_r_Y,
^I^Iperm_L, phase_L, perm_R, phase_R)
```

Python's tokenizer skips comments/blank lines for indent tracking → these lines are parsed as inside `z_q_from_psi_sm` but unreachable (dead code after the prior `return`). Module loads fine; G0 didn't catch this because the dead `return` is never executed.

NOT a SPMD or correctness issue. NOT a runtime crash. Still cleanup-worthy — 3-line delete. Suggested follow-up commit on the same branch before merge to main.

### Final assessment

**Agent 1 round 6 review passed** for SPMD safety on commit `f567aa0`.

All 6 invariants verified; two design refinements (in_specs for replicated closure constants, full-r-gather-then-slice for band coherence) are *improvements* over my §2.10 sketch that catch real subtleties. The kernel rewrite is implementation-clean enough to send to G1/G2/G3 validation gates without SPMD-side concerns.

@Agent 4: ready to fire G1 (MoS2 3×3 bit-identity) on `f567aa0`.
@Agent 3: ready to fire G2 (CrI3 HLO dump) once G1 passes.
@Agent 2: 3-line dead-code cleanup at `isdf_fitting.py:743-745` is the only outstanding cleanup item — non-blocking for validation gates, but should land in a follow-up commit before merge.

## Agent 2 → others

**2026-05-13 — Agent 2 starting Round 6 implementation.**

Phase order per `prompts/round6_agent_2_implementer.md`:
1. **Now**: G0 smoke test on synth WFN — `tests/test_io_callback_nested.py`. CPU (`JAX_PLATFORMS=cpu`). Will post "Agent 2 G0 passed" or `BLOCKER:` here.
2. Restore `_slice_local_tile_bc` + `_bpd_max` on `PsiGStore`.
3. Rewrite `c_q_from_psi_sm` / `z_q_from_psi_sm` per §6 (scan + io_callback + IFFT + all_gather + interleaved L/R carries; ordered=False, unroll=1).
4. Drop `psi_G_device_full` lazy property.
5. G1 MoS2 3×3 charge bit-identity vs lorrax_A `ff5873c`.
6. Commit + sentinel.

@A1/A3/A4: status updates at phase boundaries. SLURM alloc 52940368 up — will lxattach for G1.

**Agent 2 G0 passed.** `tests/test_io_callback_nested.py` — 6/6 tests green on `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` 1×1 mesh:

- **G0a** — basic four-primitive composition (io_callback × scan × shard_map × all_gather). Output bit-equals numpy reference.
- **G0b** — traced bc_idx routes correctly to per-iter slab (per-bc distinct primes).
- **G0c** — `ordered=True` and `ordered=False` produce identical numerical results (correctness invariant; ordered is a perf knob only).
- **G0d** — rank-5 c128 accumulator survives across iters; all_gather output participates in einsum into the carry; matches numpy reference at rtol/atol 1e-12.
- **G0e** — pad-rows + L/R mask compose math-neutrally (zero-padded short bcs contribute exactly zero).

**Path B is unblocked.** No JAX restriction discovered; the four-primitive nesting works as the unified plan §3.1 predicted. The 4-rank GPU verification of `tiled=True` axis ordering (§2.9 🟡 open) is deferred to G1; on a 1×1 mesh `tiled=True` is a no-op so it can't be exercised here.

Proceeding to **Phase 2** (restore `_slice_local_tile_bc` + `_bpd_max` on PsiGStore).

---

**2026-05-14 — Agent 2 round 6 done.** Single commit `f567aa0` on `agent/zeta-bc-scan-shardmap`. All 6 phases landed; G0 + G1 passed.

### Two design bugs surfaced + fixed during implementation (worth flagging for the plan record)

**Bug A — `psi_l_X` / `psi_r_X` per-bc offset clamping was wrong** for bcs starting below the L/R window (e.g., MoS2 charge: L=(0,80), R=(2,80); bc 0 starts at 0 < R_lo_g=2). My initial `max(0, bc.lo - L_lo_g)` clamp caused the X-side dynamic_slice to pick the wrong bands for bc.lo<L_lo_g, even though the L/R mask zeros the einsum at those positions — the X-side band index was off by `(L_lo_g - bc.lo)` for IN-window positions too. **Fix**: pre-pad psi_l_X / psi_r_X at the front with `front_pad_*` zero bands so the offset is always non-negative; pad bands contribute zero via the mask.

**Bug B — design §2.3's per-rank r-slab strategy was inconsistent with §2.9's all_gather** when bands are joint-sharded over `('x','y')`. Doing `r0_local = r_start + y_idx * r_loc` BEFORE the all_gather causes different y-ranks to compute different r-slabs; gathering bands across `('x','y')` then mixes r-slabs from different y-ranks at the same gathered band index — r-incoherent psi_Y_bc. Caught by a per-band diff dump on a 4-device CPU mesh: bands from y=0 ranks were correct (diff=0), bands from y=1 ranks were half-magnitude wrong. **Fix**: each y-rank computes the FULL r-chunk in psi_Y_local (not r_loc), all_gather along bands, then `dynamic_slice_in_dim` on the r-axis to this y-rank's r_loc slab AFTER the gather. Per-iter size grows by `p_y` but XLA's scan-internal allocator aliases the slab across iters → single slot.

**Plan addendum for Round 7+**: §2.3 + §2.9 should be amended to reflect the gather-then-slice ordering. The earlier "per-rank r-slab BEFORE all_gather" wording slipped through 4-agent review in Round 5.

### Validation gates — Agent 2 owns G0 + G1; Agents 1/3/4 own G2 + G3

- **G0** (4-primitive composition): `tests/test_io_callback_nested.py` 6/6 green on `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.
- **G1** (MoS2 3×3 charge bit-identity vs lorrax_A `ff5873c`): zeta_q_G diff max|diff|=6.13e-7, max rel=1.02e-10, median rel=6.88e-11 (A.maxabs=6.0e3) — broadly distributed → float-summation reorder, math preserved. Bare Σ_X k=0: `-40.0279 -40.0279 -33.8689 -33.8689 -33.3545 -33.3545 -33.4930 -33.4930` (matches baseline to printed precision).
- **G2** (HLO slot count + zero remat warnings) — pending Agent 3.
- **G3** (CrI3 6×6 80 Ry e2e) — pending Agent 4.

### Notes for the reviewers

@A1: SPMD invariants in your checklist (items 1–6) — items 1, 2, 4, 5, 6 land as specified; **item 3 deviates from §2.9** per Bug B above (gather happens BEFORE r-slice, not after the IFFT-then-r-slice + gather sequence). Specifically: I do IFFT on FULL r-chunk (not r_loc), then gather bands, then slice r. The "IFFT first, gather second" ordering you required is preserved; what changed is that the IFFT produces the full r-chunk (not the per-rank r_loc slab), and the r-slice happens after the gather. Please confirm this satisfies your invariant 3's spirit (IFFT-FIRST per §2.9 was about avoiding the 80 GB FFT-box-on-full-bands; that's still satisfied because each rank only IFFTs its 1/P bands).

  Also flagging: the per-bc index tables (b_lo_global, b_hi_global, psi_l_X_bc_offset, psi_r_X_bc_offset) are kept as numpy arrays in closure and `jnp.asarray`'d INSIDE the body. Closure-captured Auto-sharded jax.Arrays inside Manual-mode trip a mesh-context mismatch (same as `kvecs_frac` — passed via in_specs).

@A3: HLO dump expected per round5_unified_plan §4. The new per-iter shape inside the scan body has FULL r-chunk on the per-rank slab (not r_loc) due to Bug B fix — your prediction tables may need a per-iter slab term sized `(nk · bpd_max · ns · r_chunk)` instead of `(... · r_loc)`. After all_gather + r-slice, the carries are still per-rank r_loc'd. Please re-derive the slot table accounting for the gather-then-slice ordering.

@A4: G1 passed at the rtol=1e-10/atol=1e-12 spec. The diff-distribution analysis matches Round 3's Path-D-integration result for the same kernel (broadly distributed at 1e-10 relative; max ULP-class outliers from float-summation reorder). Please supervise the CrI3 6×6 80 Ry e2e run + confirm the `nvidia-smi` HWM lands in the 13–15 GiB range your §4.1b predicted.

**Agent 2 round 6 done.**

## Agent 3 → others

**2026-05-13 — Agent 3 in Phase 1 prep.** Will validate Agent 2's kernel-rewrite HLO against `round5_unified_plan.md` §4 once G0 + G1 + commit land.

### Run-dir + script staged (ready to fire)

- Dir: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round6_hlo_2026-05-13/`
- Symlinks: `WFN.h5`, `kin_ion.h5`, `qp_wfn_rotations.h5`, `centroids_frac_1504.txt` → parent `lorrax_A_perf_2026-05-07/`
- `cohsex.in`: same recipe as `lorrax_B_path_d_hlo`'s (memory_per_device_gb=60, band_chunk_size=16, r_chunk_size=0, gflat_chunk_size=64, x_only=true, do_screened=false). No retired knobs.
- Launch: `run_round6_hlo.sh` (mirrors prior path-d script; sets `XLA_FLAGS=--xla_dump_to=$PWD/xla_dump` + `XLA_DUMP_HLO_AS_TEXT=true` via `LORRAX_SHIFTER`).
- SLURM: will `lxattach` to JID 52940368 (hbm80g, 4hr) when Agent 2 commits.

### HLO comparison checklist (filled in after the dump lands)

Baselines for reference:

| Metric | lorrax_A `ff5873c` (pre-Path-D, OOM at 200 GiB) | lorrax_B `5cadd4b` (Round 4 Path-D, 48.63 GiB) | **Round 6 prediction** (`round5_unified_plan.md` §4.1) |
|---|---|---|---|
| Total bytes used (per rank) | 200.35 GiB | 48.63 GiB | **≤ 15 GiB** (target 13–15) |
| Preallocated-temp | 196.30 GiB | 44.56 GiB | **≤ 12 GiB** |
| Substantive temp slots (top size class) | 58 unsharded FFT-box class | 4 (3 × 14.85 GiB P-pair + 1 × 12.07 GiB FFT box) | **≤ 5** (2 carry @ 3.71 GiB, 2 post-pair @ 3.71 GiB, 1 FFT box ~5 GiB; some aliased) |
| FFT-box-class slots | 58 unsharded | 1 (12.07 GiB unsharded, from old `gflat_to_rchunk`) | **≤ 2** (1 inside scan ~5 GiB; maybe 1 fusion-adjacent) |
| `psi_Y_full`-shape slots `c128[nk, nb_total, ns, r_chunk]` | n/a | 2 (within 14.85 GiB offsets — the 30 GiB double-mat) | **0** (never materialized — no `c128[36, 160, 2, 73648]` / `c128[36, 150, 2, 73648]`) |
| Pair-density rank-5 slots | 0 | 3 × 14.85 GiB | **≤ 2** carry-class @ 3.71 GiB each (`c128[36, 2, 18412, 94, 2]`) |
| Output Z_q (live-out) | n/a | 3.71 GiB (`c128[36, 94, 73648]` — unsharded r) | ~0.99 GiB (`c128[36, 94, 18412]` — r already y-sharded) |
| `while_loop` count in HLO | 0 | 0 | **= 1** (one interleaved scan, per Agent 1 v3 §2.10) |
| `all-gather` inside while-body | n/a | n/a | **= 1** (per-iter band gather across `('x','y')`) |
| `Involuntary full rematerialization` warnings in gw.out | many | 32 | **= 0** (REQUIRED to pass G2) |

### Predicted slot map (what I expect to find in the new HLO)

1. **Carry `P_l_acc`** — `c128[36, 2, 18412, 94, 2]` = 3.71 GiB. Full-scan lifetime, in-place WhileOp aliasing.
2. **Carry `P_r_acc`** — same shape, same size. Both live across the interleaved scan.
3. **`P_l_R_conj`** — same shape after `ifftn(k) + conj`. Lives until γ̃ contract finishes.
4. **`P_r_R`** — same shape after `ifftn(k)`. Lives during γ̃ contract only. XLA may alias #4 with #2's freed slot.
5. **FFT box inside scan body** — `c128[36, 1, 2, 1.125M]` ≈ 1.3 GiB raw + cuFFT ×4 ≈ 5 GiB. Single slot, aliased across all `n_bc` scan iters.
6. **`all_gather` band-slab** — `c128[36, 16, 2, 18412]` ≈ 340 MB per rank. Per-iter, aliased.

Peak live concurrent at γ̃ contract: `P_l_R_conj + P_r_R + γ̃ scratch + output + params ≈ 12–14 GiB`.

### Failure modes I'll specifically check for

- **`c128[..., 73648]` slots with unsharded r-axis** → reshard remat leak (Round 4's 12 GiB defect returning). PASS REQUIRES 0.
- **`c128[*, *, 1125000]` or `c128[*, *, 75, 75, 200]` slots NOT inside a WhileOp body** → an FFT box that escaped scan-aliasing. PASS REQUIRES this slot lives inside the WhileOp with lifetime = 1 iter.
- **> 2 P-pair-shape slots simultaneously live** → carry-aliasing failed; either the scan body has a hidden Python loop or BufferAssignment chose poorly.
- **`while_loop` count > 1** → implementation accidentally split into two scans (serialized) — diverges from §2.10. Not strictly a fail but worth flagging.
- **No `all-gather` inside the body** → bands aren't being summed correctly. G1 would have caught the numerics, but HLO confirms the structure.

### What I'll post next (Phase 2 → Phase 3)

When Agent 2 posts "Agent 2 round 6 done" AND Agent 4 posts G1 passed:
1. `lxattach` to JID 52940368, fire `run_round6_hlo.sh`. Tail `gw.out`.
2. Read `xla_dump/module_*.jit__kernel.sm_*-memory-usage-report.txt` for the biggest kernel.
3. Fill in the comparison table above with measured numbers.
4. `grep -c "Involuntary full rematerialization" gw.out`.
5. Post the validation result + a copy to `reports/.../round6_hlo_validation.md`.
6. If all gates pass: "**Agent 3 G2 passed**". If not: `[CONCERN]` notes per failed criterion for Agent 2 to investigate.

@Agent 2: when you commit, please include the commit hash in your "Agent 2 round 6 done" message so I can cross-reference in the validation report. Also: if the new kernel turns out to use **serialized** scans (one for L, one for R) rather than interleaved, the predicted `while_loop` count goes from 1 → 2; let me know so I can adjust the HLO acceptance gate.

@Agent 4: G1 (MoS2 3×3 bit-identity) is your sign-off; I block on it. CrI3 HLO dump is wasted compute if MoS2 numerics drift outside `rtol=1e-10, atol=1e-12`. Please post your G1 result here before I fire.

@Agent 1: if you spot an SPMD-layer deviation from §2 in Agent 2's code before G1 lands, flag it so we don't pour CrI3 GPU-hours into a faulty kernel.

## Agent 4 → others
_(Numerics validator + final validation gate signoff.)_

## Any → Orchestrator (human)
_(Prefix with `BLOCKER:` if you need an immediate answer.)_

---

**2026-05-13 23:33 — Agent 4 polling: Agent 2 still in flight.**

- HEAD on `agent/zeta-bc-scan-shardmap` still `5cadd4b`; no new
  commit.
- Working tree on lorrax_B shows substantial activity: 8 files
  modified (~518 insert / 698 delete; net negative due to
  `wfn_transforms.py` losing 259 lines as `gflat_to_rchunk` is
  removed).  Files touched beyond Phase-2's stated scope:
  `isdf_fitting.py` (mtime 23:29 — kernel rewrite Phase 3 in
  progress), `gw_config.py` + `gw_init.py` (knob plumbing /
  config), `aot_memory_model/kernels/fit_one_rchunk.py` (AOT
  shape glue).
- Latest discussion marker is "Agent 2 G0 passed" at line 87
  (~30 min ago); no "round 6 done" sentinel yet.

Reading: Agent 2 is mid-Phase-3 (the c_q/z_q rewrite).  Will not
intrude on the read-only-on-sources constraint; just patiently
polling.  Re-checking in 900 s.

**2026-05-13 23:46 — Agent 4 polling: Agent 1 SPMD review PASS noted; Agent 2 commit pending.**

Read Agent 1's pre-commit review at line 75: **5/6 BLESS, 1 CONCERN
(dead code at `isdf_fitting.py:700-702`)** — Agent 2 is presumably
addressing that now (`isdf_fitting.py` mtime 23:45, 1 min ago).
SPMD invariants 1-6 all pass; the new kernel matches plan §2 with
one improvement (front-padding for `psi_l_X` / `psi_r_X` to handle
negative-offset slice case).

**G1 scope update** (per Agent 1's review of the working tree):
- `c_q_from_psi_sm` is **UNCHANGED** in this round (CCT-side rewrite
  deferred per plan §8) — I'll drop it from G1 and only validate
  `z_q_from_psi_sm`.
- `psi_l_X` / `psi_r_X` are front-padded by Agent 2's design — my
  G1 test must mock the PsiGStore + closure tables (`b_lo_global`,
  `b_hi_global`, `L_lo_g`, `R_lo_g`, `psi_l_X_bc_offset`) per
  Agent 1's L_spec / mask pattern at `isdf_fitting.py:618-625`.
- The 4-rank `tiled=True` axis-ordering check (plan §2.9 🟡) will
  fall to G1.2 (MoS2 3×3 e2e on the SLURM alloc); my synth G1.1
  on 1×1 mesh exercises the math but not the gather ordering.

Continuing to wait for commit.  Re-checking in 900 s.

---

**2026-05-14 — Agent 4 G1 result: 4 pass / 2 FAIL — BLOCKER raised.**

`tests/test_zq_from_psi_sm_bit_identity.py` (synth-WFN, 1×1 mesh,
CPU, `JAX_ENABLE_X64=1`).  Side-by-side comparison of the new
`z_q_from_psi_sm(f567aa0)` vs an inline numpy reference that builds
`psi_l_Y` / `psi_r_Y` via `to_rchunk` on the full G-flat tile +
global einsum.

Pass tolerance: `rtol=1e-10, atol=1e-12` per plan §5.2.

| Sub-gate | Setup | Result | max rel |
|---|---|---|---|
| G1.1 | charge, multi-bc `((0,4),(4,8))`, L=R=`(0,8)` | ✅ PASS | ~5e-13 (ULP-class) |
| G1.1a | single bc `((0,8),)`, L=R=`(0,8)` | ✅ PASS | ~5e-13 |
| G1.1b | short final bc `((0,4),(4,8),(8,9))`, L=R=`(0,9)` | ❌ **FAIL** | **11.5** |
| G1.1c | asymmetric L=`(0,5)` R=`(3,8)`, bcs `((0,4),(4,8))` | ❌ **FAIL** | **5.04** |
| G1.1d | bispinor γ̃^1, ns=4, L=R=`(0,8)` | ✅ PASS | ~5e-13 |
| G1.1e | pseudobands (pre-divide), L=R=`(0,8)` | ✅ PASS | ~5e-13 |

### BLOCKER: `z_q_from_psi_sm` per-bc `dynamic_slice_in_dim` clamps when bc bands extend past the L/R window end

**Where**: `src/common/isdf_fitting.py:658-660` (and the mirrored R-side
slice on lines 663-665).

```python
psi_l_X_bc = jax.lax.dynamic_slice_in_dim(
    psi_l_X_padded, psi_l_X_bc_offset[bc_idx],
    bpd_max_global, axis=2)
```

XLA's `dynamic_slice_in_dim` clamps the start to
`max(0, axis_size - slice_size)` on out-of-bounds offsets — confirmed
explicitly:
```
len-5 array, offset=4, slice_len=4 → [1.,2.,3.,4.]   (offset clamps 4→1)
len-9 array, offset=8, slice_len=4 → [5.,6.,7.,8.]   (offset clamps 8→5)
```

The L/R mask is built from logical/global band indices (`g_axis =
b_lo_global[bc_idx] + arange(bpd_max_global)`) so the mask thinks the
sliced band axis is at positions `[bc.lo, bc.lo + bpd_max_global)`,
but the clamp returns *physically different bands*.  After masking,
the einsum multiplies the *wrong psi_l_X bands* against the masked
`psi_l_Y` slab.

**Front-padding `psi_l_X` / `psi_r_X` (lines 519-524, 577-586) fixes
the negative-offset case (bc.lo < L_lo_g) but does NOT cover the
end-overflow case** (bc.hi > L_hi_g or bc.lo + bpd_max_global > nb_l).

### Reproducer (G1.1c, simplest failing case)

```
nb_total = 8, band_chunk_ranges = ((0,4),(4,8)),
L = (0,5), R = (3,8).
psi_l_X has 5 bands (L window).
bc 1=(4,8): psi_l_X_bc_offset[1] = 4 - L_lo_g + front_pad_l = 4.
  dynamic_slice_in_dim(psi_l_X, 4, 4, axis=2)  on a 5-band array:
    clamps offset 4 → max(0, 5-4)=1 → returns bands [1..4]
    but L mask for this bc is [True, False, False, False]  (only
    global band 4 is in L=[0,5)), retaining slot 0 = band 1.
  Result: einsum uses psi_l_X[band 1] where it should use band 4 → wrong.
```

### When this fires in production

- **Charge channel** (L=R=full band axis, bcs cleanly divide
  nb_total): NEVER fires.  This is why Agent 2's MoS2 3×3 G1 e2e
  passed at max rel = 1.02e-10 — that config has L=R=`(0, nb_total)`
  and bands divisible by bc_size.
- **Bispinor transverse** (L = val window, R = cond window, bcs
  span the boundary): **ALWAYS** fires.  Any bc whose band span
  extends past either L_hi or R_hi (or starts below L_lo / R_lo
  for the front-pad case Agent 2 already covered) triggers the
  clamp + mask mismatch.
- **Short final bc** (nb_total not a multiple of bc_size): fires
  even on charge channel.  Plan §5.3.3 explicitly required this
  case to work.

### Suggested fix (Agent 2)

Symmetric to the existing front-pad: add **back-padding** to
`psi_l_X_padded` / `psi_r_X_padded` so the dynamic_slice never
extends past the array.  Required pad amount on the back side:
```python
back_pad_l = max(
    max(0, (bc.lo - L_lo_g + front_pad_l) + bpd_max_global - psi_l_X.shape[2] - front_pad_l)
    for (bc.lo, _hi) in bcr
)
# Equivalently:
end_offset_max = max(psi_l_X_bc_offset[bc_idx] + bpd_max_global for bc_idx in range(n_bc))
back_pad_l = max(0, end_offset_max - (psi_l_X.shape[2] + front_pad_l))
```
Add a `jnp.pad(psi_l_X_, ((0,0),(0,0),(front_pad_l, back_pad_l),(0,0)))`
in place of the current front-only pad.  The L/R mask zero-out
contract makes back-pad bands math-neutral, same as the front-pad
ones.

### Status

- **Agent 4 G1 FAILED on production-relevant edge cases.**
- @Agent 2: please apply the back-pad fix and re-run.  After your
  re-commit lands I'll re-run all six sub-gates.
- @Agent 3: **PLEASE HOLD off on firing the CrI3 HLO dump** until G1
  passes — wasted compute if the kernel has wrong numerics on
  bispinor channels.  Charge-only G3 might still pass (this bug
  doesn't fire on full-coverage L=R=full), so if you want to test
  the HLO structure on a charge-only run that's safe; but bispinor
  CrI3 needs to wait.
- @Agent 1: not a SPMD invariant violation per se (your 6/6 BLESS
  stands) — it's a numerical bug in the L/R per-bc index math.
  Filing for completeness; no action needed from you.

Test file: `tests/test_zq_from_psi_sm_bit_identity.py`.  Re-run with:
```
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python -m pytest \
    tests/test_zq_from_psi_sm_bit_identity.py -v -s
```

**BLOCKER: psi_l_X / psi_r_X back-padding missing in z_q_from_psi_sm at isdf_fitting.py:658-665.**
