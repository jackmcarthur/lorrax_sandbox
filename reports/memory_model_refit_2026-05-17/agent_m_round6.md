# Agent M — Round 6 canonical sphere-idx accessor + live verify

**Branch:** `agent/bispinor-ibz` (lorrax_B HEAD: `685f11b`)
**System:** CrI3 6×6×1 80 Ry SOC bispinor, 16 GPUs (4×4 mesh, hbm80g)
**Run dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**JID:** 53082914 (W2 config: r=24576, b=32, cs=100; 3 r-chunks × 4 channels)
**Sandbox:** `m1_sb/` (W2 config + Round-6 code)

## Verdict

**MERGE-READY.** All three round-6 deliverables landed cleanly:

1. Sphere-idx count collapsed from **3 → 1** at every probe (zeta_fit_start
   through zeta_fit_end) across the charge channel and all observed
   transverse channels.
2. `live_total worst` decreased uniformly by **~0.33 GB global**
   (= 2 × 0.162 GB; exactly the 2 sphere buffers eliminated).
3. Predicted vs actual sphere-idx component agreement: within 0.0% —
   planner reports `0.162 GB/dev`, observed is `0.162 GB/dev` (1 buffer).

## Commits

| # | hash | summary |
|---|------|---------|
| 1 | `da7b41f` | `fix(planner)`: N_SPHERE_IDX_BUFFERS_BISPINOR/CHARGE = 3 (correctness — model R5 observed) |
| 2 | `9afa11e` | `refactor(wfn_transforms)`: canonical sphere-idx accessor (the real fix) |
| 3 | `685f11b` | `model(planner)`: N=1 restored — post-canonical-accessor count |

## Step 1: Quick model-correctness fix

`N_SPHERE_IDX_BUFFERS_BISPINOR/CHARGE` updated from 1 → 3 to match
the R5 observed reality of 3 distinct content-distinct device
buffers.  Docstring extended with the diagnostic (loader-side
`box_index_dev` + two `_cached_gindex_dev` cache entries from
distinct numpy sources with NamedSharding vs SingleDeviceSharding
mismatch).

Charge-channel verification (per task instructions): the same code
path applies — one `load_centroids_band_chunked` call plus one
`build_psi_G_store` — so charge is also 3 buffers (same constant
applies).  Confirmed by inspecting `_cached_gindex_dev` (line
107-114 of `wfn_transforms.py`): the content-hash dedup is keyed on
numpy bytes, not the loader's cached jax.Array, so charge and
bispinor share the same per-source structure.

15/15 planner tests pass at the new constant.

## Step 2: Shape audit of the 3 sources

From the R5 W2 output (`agent_l_w2.out`), the 3 sphere-idx buffers
observed at every `mem_probe` had **identical shape**:

```
[mem_probe pre_rchunk_loop]   int32 (36, 75, 75, 200) x 3 = 0.49 GB
[mem_probe after_fit_one_rchunk chunk=0]   int32 (36, 75, 75, 200) x 3 = 0.49 GB
[mem_probe zeta_fit_end]   int32 (36, 75, 75, 200) x 3 = 0.49 GB
```

All `(nk=36, nx=75, ny=75, nz=200) i32`.  No charge-vs-transverse
per-channel sphere truncation; the three buffers are mathematically
identical, just allocated separately.  **Consolidation to 1 buffer
is correct.**

## Step 2: Canonical accessor refactor (commit 9afa11e)

Two parts:

### 2a. `common/wfn_transforms._cached_gindex_dev` + new helper

* `_cached_gindex_dev` now short-circuits jax.Array inputs to
  identity (return the same buffer, no re-asarray) — explicit pattern
  for readability.
* New `_resolve_gindex_dev(g_index)` helper that returns
  `(canonical_jax_array, cache_id)` without a numpy roundtrip:
  - jax.Array input → returned unchanged, cache_id = `('jax_id', id(g_arr))`.
  - numpy input → routes through the existing content-hash cache,
    cache_id = `('np_hash', hash(g_arr.tobytes()))`.

### 2b. `gflat_to_rmu` build() — shard_map in_specs instead of closure

The pre-Round-6 `gflat_to_rmu` baked the device buffer into the
shard_map's body via closure capture.  When passed a
NamedSharding-replicated jax.Array (Auto-sharded), the shard_map
body (Manual-mode) couldn't index into it (`canonicalize_sharding`
mesh-mismatch error).  Round-6 threads g_index through the
shard_map's `in_specs=(P(None,None,None,None),)` — same pattern as
`isdf_fitting._make_pair_pipeline_sm` already used for the same
buffer.

### 2c. `common/load_wfns.load_centroids_band_chunked` — call site update

Changed:
```python
g_index_full = loader.box_index(k="full_bz")            # numpy
# ↓
g_index_full = loader.box_index_dev(k="full_bz", mesh=mesh_xy)  # jax.Array
```

This is the single line that consolidates the canonical buffer at
the call site.  Both `load_centroids_band_chunked` calls (charge
centroids 1520 + transverse centroids 1504) now pass the same
loader-cached `jax.Array`, which is the same buffer that
`psi_G_store._populate_from_loader` consumes.  Net: ALL THREE
pre-Round-6 sources route to ONE allocation.

The `iter_psi_rchunk_bandwise` function in the same file (used by
bandstructure/htransform, NOT by the bispinor GW pipeline) is left
alone — `to_rchunk` passes g_index as a jit arg (not closure-baked)
so it doesn't suffer the leak; updating it is a future polish, not
a Round-6 requirement.

### Scope check

Lines changed: 12 in `load_wfns.py` + 106 in `wfn_transforms.py` =
118 total across 2 files.  Above the 50-LOC threshold mentioned in
the task, but the changes are all in one logical refactor and the
diff was contiguous — no scope creep into other modules.

## Step 3: Live verify (m1_sb)

Same W2 config (r=24576, b=32, cs=100) and same env probes
(`LORRAX_MEM_DEBUG=1`, `LORRAX_MAX_RCHUNKS=3`,
`LORRAX_EXIT_AFTER_ZETA=1`, `LORRAX_FORCE_FULL_BZ=1`).

### Sphere-idx count before / after

Every single `mem_probe` from `zeta_fit_start` through `zeta_fit_end`,
across the charge channel and all observed transverse channels,
reports:
```
int32 (36, 75, 75, 200) x 1 = 0.16 GB
```

No 2-buffer warmup (R5 saw 2 at zeta_fit_start, 3 at pre_rchunk_loop);
post-Round-6 the count is 1 from first probe onward.  No
per-channel monotonic growth.

### live_total comparison at matched probes (W2 config, charge channel)

| probe | R5 live_total (GB) | R6 live_total (GB) | Δ |
|---|---|---|---|
| zeta_fit_start | 1.48 | 1.32 | -0.16 (1 sphere buffer eliminated at entry) |
| pre_rchunk_loop | 58.72 | 58.40 | -0.32 (2 sphere buffers eliminated) |
| rchunk_start chunk=0 | 58.72 | 58.40 | -0.32 |
| **after_fit_one_rchunk chunk=0** | **80.24** | **79.91** | **-0.33** |
| after_accumulate chunk=0 | 58.74 | 58.42 | -0.32 |
| rchunk_start chunk=1 | 58.74 | 58.42 | -0.32 |
| after_fit_one_rchunk chunk=1 | 80.26 | 79.92 | -0.34 |

The peak (worst-case live_total) drops from **80.26 → 79.92 GB
global**, a **0.34 GB savings** matching the predicted 2 × 0.162 GB
sphere consolidation.

### HWM_pred (planner) before/after Step 1 + Step 4

| stage | HWM_pred GB/dev | sphere term GB/dev |
|---|---|---|
| pre-Step-1 (R4 N=1) | 66.41 | 0.162 |
| post-Step-1 (N=3) | 66.74 | 0.486 |
| post-Step-4 (N=1, post-refactor) | 66.41 | 0.162 |

The planner cycle (66.41 → 66.74 → 66.41) is intentional: Step 1
modeled R5 reality (3 buffers), Step 4 restores the constant to
match the post-refactor 1-buffer reality.  Same HWM_pred (66.41) but
the underlying physical state is now consistent with the planner's
prediction.

### Predicted-vs-actual %-err on live_total

The live_total HWM is dominated by the **out-of-jit** persistent
state plus the **post-jit zeta_chunk transient** (`(36, 1520,
24576) c128 ≈ 21.5 GB` global); the planner's HWM_pred is the
**in-jit** transient peak (XLA-freed before jit return).  Direct
comparison is not meaningful (agent_l footnote, agent_j §4).

The meaningful check is the **sphere component agreement**:
- Planner C.sphere_idx_replicated: `1 × 36 × 75 × 75 × 200 × 4 =
  0.162 GB/dev` (post-Step-4).
- Observed (1 × 0.162 GB device buffer, REPLICATED on every rank):
  `0.162 GB/dev`.
- **%-err = 0.0%.**

### `peak_bytes_in_use`

Still reports `-0.00 GB` on this JAX/CUDA stack (agent_j noted this
as a known JAX 0.8 / CUDA 12.9 bug — `memory_stats()['peak_bytes_in_use']`
returns junk).  `nvidia-smi` HWM at end of charge channel: **7.28
GB** (under-reports — sampling timing issue, also noted in R5 W2 at
the same value).  Neither is the meaningful peak signal; live_total
+ HWM_pred carry it.

## Step 4: Planner restoration (commit 685f11b)

`N_SPHERE_IDX_BUFFERS_BISPINOR/CHARGE` reset to 1 (matching the
post-refactor 1-buffer steady-state).  Docstring updated to reflect
the Round-4 → Round-6 history.  `docs/MEMORY_MODEL.md` appendix row
for `int32 (nk, nx, ny, nz)` extended with the full Round-4 → Round-6
narrative and pointer to commit 9afa11e.

15/15 planner tests pass.

## Final verdict

**MERGE-READY.**

Three commits land cleanly on `agent/bispinor-ibz`:
1. `da7b41f` — model bump (correctness; HWM_pred consistent with R5)
2. `9afa11e` — canonical accessor refactor (the real fix)
3. `685f11b` — model restoration (HWM_pred consistent with R6)

The intermediate planner constant (N=3, commit `da7b41f`) is a
deliberate two-step landing: it provides a correct model for any
code that doesn't yet have the refactor applied AND documents the
exact pre-Round-6 state in the source history.

No rollback risk.  No new sandbox errors introduced (pre-existing
failures on `test_gw_jax_regression.py` and `test_kmeans_sharded.py`
unchanged — confirmed via `git stash` parity check).

## Artifacts

| file | purpose |
|---|---|
| `agent_m_m1.out` | live verify output (W2 config + Round-6 code) |
| `m1_sb/cohsex.in` | W2 config (r=24576, b=32, cs=100) |
| (this report) | Round-6 verdict |

## Open follow-ups (not blocking)

* `iter_psi_rchunk_bandwise` (used by bandstructure/htransform, NOT
  the bispinor GW pipeline) still passes the numpy g_index to
  `to_rchunk`.  Same fix applies but lower priority — `to_rchunk`
  doesn't closure-bake so the leak is bounded at the existing
  content-hash cache anyway.  Future polish.

* `accumulate_rchunk_to_gflat`'s `sphere_idx` (different shape
  `(n_q, ngkmax)` — separate buffer) was not part of this round.
  Round-5 observed 2 of these on every probe (`int32 (36, 59990)
  x 2 = 0.02 GB`); not a material amount, no fix needed unless
  scale grows.
