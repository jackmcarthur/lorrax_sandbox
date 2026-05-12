# Bispinor pipeline rebase + chunker findings — handoff

Session 2026-05-04. Goal was to make `memory_per_device_gb` work as a
single user-facing knob for production CrI3 60Ry bispinor on 30 GB
A100s. Got partway; documenting state so the next session can finish.

## What's committed and where

| Branch | Status | Contains |
|---|---|---|
| `agent-B/refactor-compute-vcoul` | pushed to origin | 5 commits ahead of `main`: V_q split (f6a3400, 02b333d), bispinor pipeline + LU + print-gating (9397e35), tee_stdout helper (914a225), chunker recalibration (c3c1c26) |
| `agent-B/wip-budget-split` | local only | One WIP commit (17bfb5e) on top of the above: budget-split chunker (`WFN_WORKSPACE_FRAC = 0.30`) + lowered defaults (`target_utilization 0.97→0.80`, runtime overhead `max(0.8 GB, 5%×budget)`) |
| `origin/main` | 21 commits ahead of branch base | Major cleanup: deleted my V_q split files, replaced with unified `compute_all_V_q` in `compute_vcoul.py` + per-dimension `coulomb/` package + LorraxConfig sub-dataclasses |

Stash list is empty (the budget-split stash was promoted to
`agent-B/wip-budget-split`).

## What we learned about the chunker (validated)

1. **Budget-split policy is correct and safe.** Capping
   `fft_inloop ≤ 30% × budget` makes `band_chunk` shrink to the
   mesh-divisibility floor before `chunk_r` is forced too small. MoS2
   60Ry passes unchanged (cap is non-binding for small systems);
   CrI3 60Ry 8-GPU's plan goes from `chunk_r=4584` (246 chunks, 16 h)
   to `chunk_r=43216` (27 chunks).

2. **`query_fft_peak_bytes` is roughly accurate.** Direct probe on
   CrI3 60Ry 8-GPU: AOT-predicted 10.37 GB at bc=8, runtime FFT call
   adds ~2.6 GB per call (just the per-rank input/output at sharded
   /8). cuFFT plan workspace is sub-GB. *The chunker's coefficients
   are not the source of the gap.*

3. **The actual gap (predicted 24 GB → runtime 37+ GB OOM) is the
   bc-loop unrolling.** `fit_one_rchunk` iterates
   `for bc_idx in band_chunk_ranges:` as Python — XLA sees 10 separate
   `io_callback + FFT + reshard + accumulate` traces in one big jit.
   Cumulative live ψ buffers across the unrolled trace match the host
   ψ(G) cache size (~26 GB for CrI3 8-GPU at band_chunk=8), and the
   25.4 GiB single-buffer alloc failures we saw line up with that.

4. **The fix is structural: scan-ify the bc-loop.** Per-iteration
   donation only works at jit boundaries, so the only way to free
   each `psi_bc_Y` before the next is to make every iteration use the
   *same compiled body* under `jax.lax.scan`. Then runtime memory
   stays at one bc's working set instead of 10×.

## Why the rebase wasn't done in this session

Origin/main deleted my entire V_q split (`v_q_tile.py`,
`v_q_driver.py`, `v_q_lorentz.py` — 2217 lines) and replaced it with
a unified `compute_all_V_q(zeta_io, ...)` plus a per-dimension
`coulomb/` package (`Bulk3D`, `Slab2D`, `Box0D` under one `SysDim`
dispatcher). The bispinor functionality on `agent-B/...` was built
*against the old structure* and needs to be re-expressed against the
new one — that's a real port, not a 3-way merge.

Concrete delta vs main:

```
src/gw/v_q_driver.py    — 997 lines deleted
src/gw/v_q_lorentz.py   — 379 lines deleted
src/gw/v_q_tile.py      — 841 lines deleted
src/gw/coulomb_kernel.py— 605 lines deleted
src/gw/compute_vcoul.py — +2097 lines (new compute_all_V_q etc.)
src/gw/coulomb/         — NEW package (base.py, bulk_3d.py, slab_2d.py, box_0d.py)
src/gw/gw_config.py     — 775 lines changed (sub-dataclasses)
```

## Plan for the next session

### Phase 1: rebase mechanically

```bash
git checkout agent-B/refactor-compute-vcoul
git rebase --onto origin/main main agent-B/refactor-compute-vcoul
```

Expect commits f6a3400 + 02b333d to drop entirely (their files no
longer exist). Commit 9397e35 will conflict heavily — most of its V_q
content needs to be re-implemented; only the LU-branch piece in
`common/isdf_fitting.py` survives mechanically. 914a225 (runtime tee)
and c3c1c26 (chunker recalibration) should mostly apply with manual
fixups for the new `cfg.memory.*` sub-dataclass.

If the rebase is too messy, fall back to:

```bash
git checkout origin/main
git checkout -b agent-B/refactor-compute-vcoul-v2
# cherry-pick salvageable commits one at a time
git cherry-pick 914a225  # runtime tee — should mostly apply
git cherry-pick c3c1c26  # chunker recalib — adapt to cfg.memory.*
# bispinor work re-implemented as new commits (Phase 2)
```

### Phase 2: bispinor V_q on the new interface

Origin/main's `compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, ...)`
([gw_init.py:590](sources/lorrax_B/src/gw/gw_init.py#L590)) handles
**one** ζ file → **one** V_qmunu. For bispinor we need **four**
calls (one per channel: charge `zeta_q.h5` plus three transverse
`zeta_q_mu{1,2,3}.h5`) plus a Lorentz tile combiner.

Sketch of the new `compute_V_q_bispinor` wrapper:

```python
def compute_V_q_bispinor(
    zeta_paths,  # dict {0: 'zeta_q.h5', 1: 'zeta_q_mu1.h5', 2: ..., 3: ...}
    wfn, meta, mesh_xy, cfg, ...
):
    # 1. Compute V_q^{μ_L,μ_L} (the 4 diagonal channels) with the
    #    appropriate Coulomb dressing.  Each call uses compute_all_V_q
    #    against its channel's ζ.
    V_diag = {}
    for mu_L in (0, 1, 2, 3):
        V_diag[mu_L] = compute_V_q_one_channel(
            zeta_paths[mu_L], wfn, meta, mesh_xy, cfg,
            vertex_mu_L=mu_L,  # threads through the Coulomb dressing
        )

    # 2. Off-diagonal transverse tiles (μ_L ≠ ν_L, both > 0): need
    #    cross-channel ζ products.  For each unique (i,j) with i<j ∈
    #    {1,2,3}, compute V^{i,j} = ⟨ζ^i | D^{i,j}(q+G) | ζ^j⟩.
    #    Three unique tiles: (1,2), (1,3), (2,3).
    V_offdiag = {}
    for i, j in [(1,2), (1,3), (2,3)]:
        V_offdiag[(i,j)] = compute_V_q_offdiag_tile(
            zeta_paths[i], zeta_paths[j], wfn, meta, mesh_xy, cfg,
            vertex_pair=(i, j),
        )

    # 3. Hermitian-transpose: V^{j,i} = (V^{i,j})†
    for (i, j), V_ij in V_offdiag.items():
        V_offdiag[(j, i)] = V_ij.conj().swapaxes(-1, -2)

    # 4. Coulomb-gauge zeros: V^{0,i} = V^{i,0} = 0 for i ∈ {1,2,3}.
    # Total: 1 (charge) + 3 (transverse diagonal) + 6 (off-diagonal,
    # 3 computed + 3 hermitian) = 10 nonzero, 6 zero.
    return assemble_lorentz_tiles(V_diag, V_offdiag, ...)
```

The transverse-channel Coulomb dressing
`D^{i,j}(q+G) = (4π/|K|²)(δ_{ij} − K̂_iK̂_j)` lives outside the
existing `coulomb/` package (it's a (3,3) matrix in spatial Cartesian
indices for each (q+G), not a scalar). The cleanest place to put it
is a new `coulomb/transverse.py` that wraps the dimension-aware
`v_qG` from `Bulk3D`/`Slab2D`/`Box0D` and applies the projector.

### Phase 3: bispinor zeta fit (3 transverse channels)

Already worked on the branch's `gw_init.py` — sequential calls to
`fit_zeta_chunked_to_h5` with `vertex_mu_L=1,2,3` after the charge
fit. The `vertex_mu_L` thread already exists in `isdf_fitting.py` on
the branch. Need to verify the new `fit_zeta` signature on main
accepts the parameter (or thread it through if not).

The `compute_L_q_from_CCT` LU branch (Schur-form CCT is indefinite for
transverse vertices, so `jnp.linalg.cholesky` returns NaN — must use
LU) is the smallest and most portable bispinor change. ~30-50 lines.
This should land first as a standalone commit.

### Phase 4: re-apply budget-split chunker

The WIP commit on `agent-B/wip-budget-split` adds `WFN_WORKSPACE_FRAC = 0.30`
gate before `_find_r_chunk`'s retry loop. After rebase, this needs to
be re-applied against main's `compute_optimal_chunks` which uses
`cfg.memory.per_device_gb` and `cfg.memory.chunk_target_utilization`
in place of the old flat fields. Should be ~20 lines of context-shift.

### Phase 5: structural fix for bc-loop

Independent of the rebase: scan-ify the bc-loop in
`_make_fit_one_rchunk_kernel` ([isdf_fitting.py:981-990](sources/lorrax_B/src/common/isdf_fitting.py#L981-L990))
to give `psi_bc_Y` proper inter-iteration donation. Required changes:

1. Encode `bc_classify` results as fixed-shape int tensors (each bc →
   `(tag_code, payload[0..3])`) so the scan body can dispatch with
   `jax.lax.cond` or masked slices.
2. Change `psi_G_store.fetch_psi_G` to accept dynamic `(bc_lo, bc_hi)`
   ints and return a fixed-shape array (the bc_size is uniform within
   the scan, padding the last bc if needed).
3. Replace the Python `for bc_idx, ... in enumerate(bc_classify):`
   with `jax.lax.scan` over `n_bcs`.

This is the *real* fix for the CrI3 OOM. Estimated ~2-3 hours.
Without it, the chunker's per-bc accounting is structurally honest
but the kernel runs at 10× the model's predicted live-set on
unrolled traces.

## Current status of CrI3 60Ry bispinor

Doesn't end-to-end on 8 or 16 GPU within a 30 GB budget today, even
with the budget-split chunker. The blocker is Phase 5 (bc-loop
scan-ification), not coefficient calibration.

MoS2 60Ry bispinor *does* end-to-end cleanly with `memory_per_device_gb = 30.0`
as the only knob. Current run dir:
`runs/MoS2/D_60Ry_bispinor/` ([cohsex.in](runs/MoS2/D_60Ry_bispinor/cohsex.in)
is the single-knob form).

## File pointers

- **Budget-split chunker**: `agent-B/wip-budget-split` branch, commit 17bfb5e
- **Bispinor pipeline (old structure)**: `agent-B/refactor-compute-vcoul` branch, commit 9397e35
- **AOT memory model with budget-split chooser**: [aot_memory_model/chooser.py:468-592](sources/lorrax_B/src/gw/aot_memory_model/chooser.py#L468-L592) (the reference implementation we ported)
- **bc-loop to scan-ify**: [isdf_fitting.py:981-990](sources/lorrax_B/src/common/isdf_fitting.py#L981-L990)
- **LU branch for transverse channels**: in `compute_L_q_from_CCT` on the branch (commit 9397e35 diff in `common/isdf_fitting.py`)
- **MoS2 reference run dir**: `runs/MoS2/D_60Ry_bispinor/`
- **CrI3 reference run dir**: `runs/CrI3/B_nonbisp_baseline/`
- **FFT memory probe (built this session)**: `runs/CrI3/B_nonbisp_baseline/probe_fft_memory.py`
