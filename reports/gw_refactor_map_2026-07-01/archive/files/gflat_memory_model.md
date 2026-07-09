# src/gw/gflat_memory_model.py (1007 LOC)

Deep-read notes for the GW refactor map, 2026-07-01. All line numbers refer to
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src/gw/gflat_memory_model.py`.

## Purpose

Analytic per-rank HBM memory model + deterministic chunk-size picker for the
G-flat ζ-fit + V_q pipeline. Models five HBM peaks (A: centroid load, B:
CCT/Cholesky, C: fit_one_rchunk fused jit, D: accumulate_rchunk_to_gflat,
E: V_q per-tile) and picks `(band_chunk, r_chunk, gflat_chunk_size)` to land
near `target_utilization × budget`. Pure arithmetic — no physics equations, no
device allocation, no file I/O; every coefficient is pinned to an HLO/live_arrays
calibration report cited in comments (memory_model_refit_2026-05-17 round series).

Category: **resource mgmt: memory planner**.

## Entry points (grep evidence across src/, tests/, tools/, scripts/)

| Symbol | Callers |
|---|---|
| `plan_gflat_chunks` | `src/gw/gw_init.py:587,606` (the only production caller — runs on EVERY rank so all ranks agree on chunk sizes); `tests/test_planner_refit_2026-05-17.py` (6 call sites); `tests/test_band_chunk_size_floor.py` (6 call sites) |
| `GFlatChunkPlan` (return type) | consumed in `gw_init.py`: `.format()` printed rank-0; `.band_chunk → chunks['band_chunk']`, `.r_chunk → chunks['chunk_r']`, `.hwm_bytes → chunks['gflat_hwm_gb']`, `.gflat_chunk_size → chunks['gflat_chunk_size']` |
| `GFLAT_CHUNK_SIZE_CAP` | `tests/test_planner_refit_2026-05-17.py:30,174,176,186` only |
| `N_SPHERE_IDX_BUFFERS_BISPINOR` / `_CHARGE` | `tests/test_planner_refit_2026-05-17.py:31,221,280` only |
| doc cross-refs (comments, not imports) | `src/common/load_wfns.py:374`, `src/common/gamma_matrices.py:153` cite this module for slot accounting |

Kwargs of `plan_gflat_chunks` that NO caller (production or test) ever passes,
per grep: `use_ibz_T`, `use_query_fft_peak_bytes`, `n_q_ibz`,
`pair_density_slots_charge`, `pair_density_slots_transverse`,
`fft_box_factor_A`, `fft_box_factor_D`, `r_chunk_override` is passed by gw_init;
gw_init passes the **legacy alias** `fft_box_factor=4.0` (not `fft_box_factor_A`).

## Function-by-function

### `_bytes_c128(*dims, shard=1)` — L91-97
Per-rank c128 byte count: `16 · Π dims / shard`. Used by every peak function.

### `_bytes_i32(*dims, shard=1)` — L100-106
Same for int32 (4 bytes). Only used by `_sphere_idx_replicated_bytes`.

### `_default_pair_density_slots()` — L109-135
Returns XLA-BufferAssignment-calibrated count of concurrent rank-5 pair-density
buffers in `fit_one_rchunk`: **4 on CPU XLA, 3 on GPU XLA** (falls back to 3 if
jax unimportable). Imports jax at call time and reads `jax.default_backend()`.
Calibration provenance: `agent_d_hlo_calibration.md` (GPU),
`CPU_PLANNER_LANDED_2026-05-20.md` (CPU). Called only inside
`plan_gflat_chunks` when the two `pair_density_slots_*` kwargs are None (always,
in practice).

### `_round_pow2_down(n)` — L138-142
Largest power of 2 ≤ n, ≥ 1. Used once (band_chunk pick, L884).

### `_largest_divisor_le(n, cap)` — L145-153
Largest divisor of n ≤ cap. **ZERO callers anywhere** (grepped
`_largest_divisor_le` across src/tests/tools/scripts — only the definition
hits). Dead code, likely a leftover from a pre-refit band_chunk picker.

### Module constants — L170, L207-208
- `GFLAT_CHUNK_SIZE_CAP = 100` (L170): empirical cuFFT plan-algorithm-crossover
  cap; cs=1414 OOM'd at production CrI3 80Ry (agent_f probe); cs≤100 stays in
  the "factor-2" regime; perf-neutral (cs=1 within 15% of cs=360 wall).
- `N_SPHERE_IDX_BUFFERS_BISPINOR = 1`, `N_SPHERE_IDX_BUFFERS_CHARGE = 1`
  (L207-208): replicated `g_index[k,nx,ny,nz] int32` buffer count after the
  Round-6 canonical-accessor dedup (was 8 bispinor / 3 charge pre-fix; 40-line
  comment L172-206 documents the leak history: source-of-truth allocation at
  `common/gvec_fft_box.py:55`). Both now equal 1, so the `is_bispinor`
  selection between them at L726-727 is vestigial.

### `GFlatChunkPlan` dataclass — L211-258
Fields: `band_chunk, r_chunk, n_r_chunks, gflat_chunk_size (Optional[int] but
"always int after the 2026-05-17 cap"), hwm_bytes, peak_breakdown (dict
name→bytes), peak_components (per-term dict), bottleneck (binding peak name),
budget_bytes`. `.format()` (L224-258) renders the human log: chunk sizes,
budget, HWM %, sorted peak totals, then per-peak component groups keyed by
letter prefix (`"A."`, `"B."`, ...) split at the first `.` — non-prefixed keys
bucket under `_misc`.

### `_sphere_idx_replicated_bytes(*, nq, fft_grid, n_buffers)` — L281-287
`n_buffers × 4 · nq·nx·ny·nz` bytes, **NOT divided by p_xy** (replicated
per-rank). Docstring block L263-280 documents the "cross_jit_leaked"
lifetime class: (nk,nx,ny,nz) int32 ≈ 0.16 GB/rank per buffer on CrI3 80Ry
((36,75,75,200)). Called from all five peak functions.

### `_peak_A_centroid_load(...)` — L297-327
Pre-loop ψ(G)→IFFT→sample-at-r_μ per channel. Terms (returned as `"A.*"` dict):
- `centroid_out_filling` = c128(nk, ns, mu, nb_per_load)/p
- `phase_table` = c128(nk, n_rtot) replicated
- `fft_box` = c128(band_chunk, ns, n_rtot) × fft_box_factor_A (default 4.0);
  audited 2026-05-17: the real kernel (`wfn_transforms.py:611-851`,
  `gflat_to_rmu._kernel`) batches on flat (nk·nb_local) inside shard_map, box
  is per-rank already, no nk factor; `band_chunk` proxies `cs`.
- `sphere_idx_replicated`
Note the `nq` arg here is used only for the sphere-idx term.

### `_peak_B_cct_chol(...)` — L330-361
CCT + Cholesky pre-loop on full (μ,ν). Terms (`"B.*"`):
- `centroids_persistent` = **4×** c128(nk, ns, mu, nb_total)/p_xy — 4 physical
  buffers per channel: ψ_l/ψ_r each in rmuT_X + rmu_Y transposed form
  (agent_a finding #1 + agent_g census §6, live_arrays-confirmed)
- `P_l_plus_P_r_open_spin` = 2 × c128(nk, ns, ns, mu, mu)/p_xy
- `C_q` = c128(nq, mu, mu)/p_xy; `L_q` same shape (Cholesky factor)
- `sphere_idx_replicated`

### `_peak_C_fit_one_rchunk(...)` — L364-427
The per-r-chunk fused jit peak. Persistent base (`"C.*"`, filtered `v > 0`):
- `centroids_persist` = 4 × c128(nk, ns, mu, nb_total)/p_xy
- `L_q` = c128(nq, mu, mu)/p_xy
- `gflat_acc` = c128(nq_disk, mu, ngkmax)/p_xy — charged in BOTH Peak C and
  Peak D persistent bases (Round-10/agent_q: the two jits have isolated
  transient slots, so not double-counting; live_arrays sig
  `c128(36, 1520, 59990)` cited)
- `sphere_idx_replicated`
Transient:
- `P_pair_concurrent_slots` = pair_density_slots × c128(nk, ns, ns, mu,
  r_chunk)/p_xy — the dominant term; XLA aliases psi_bc_Y, FFT box, Z_q into
  the SAME lifetime slots (HLO-verified agent_d M1)
- `zeta_out` = c128(nq, mu, r_chunk)/p
Round-6 note: bc-loop is `lax.scan(unroll=1)` (commit f567aa0) so n_bc no
longer multiplies the FFT-box term — which explains the **unused parameters**
`band_chunk, n_bc, n_rtot, p_x, p_y, fft_box_factor, is_charge_channel`
still in the signature but never referenced in the body (pre-scan-era cruft).
Source refs baked in comments: `common/isdf_fitting.py:625-627, 713-720`.

### `_peak_D_accumulate(...)` — L430-511
`accumulate_rchunk_to_gflat` peak (runs after fit_one_rchunk returns; P_l/P_r
freed, ζ_chunk still live). Persistent (`"D.*"`): same 4-buffer centroids +
L_q + gflat_acc + sphere-idx as Peak C. Transient:
- `zeta_chunk` = c128(nq_disk, mu, r_chunk)/p_xy (donated via
  donate_argnums=(1,))
- `accumulate_fft_box` = c128(gflat_chunk_size, n_rtot) × fft_box_factor_D
  (default 2.0 — no ns axis, ζ is spin-traced upstream; kernel at
  `wfn_transforms.py:1057-1107`; HLO-verified agent_d M2 modules 0474/0363,
  2 box-sized slots)
Optional branch (L483-499): when `use_query_fft_peak_bytes=True` and mesh
given, imports `common.fft_helpers.query_fft_peak_bytes` for an AOT-measured
cuFFT peak (`NamedSharding(mesh_xy, P(None,None,None,None))`, fft_axes=(1,2,3));
silently `except Exception: pass` back to the constant formula. **No caller
ever sets this flag** — dead-in-practice branch, kept "for calibrating the
planner".

### `_peak_E_v_q_per_tile_transient(...)` — L518-589
Per-tile peak inside `_compute_V_q_g_flat_one_tile`'s per-q kernel
(`gw/v_q_g_flat.py:271-465` per agent_i §2). Terms (`"E.*"`, filtered `v > 0`):
- `V_acc` = c128(n_q_ibz, mu_L, mu_R)/p_xy (donated in-place; post-unfold output
  piggybacks the same slot)
- `v_q_table_replicated` = c128(n_q_ibz, ngkmax) replicated
- `zeta_L_all` = c128(n_q_ibz, mu_L, ngkmax)/p_xy (pre-loop slab pre-read,
  v_q_g_flat.py:372-384)
- `zeta_R_all` = 0 if `same_zeta` else c128(n_q_ibz, mu_R, ngkmax)/p_xy —
  TT-off-diagonal (same_zeta=False) is the dominant tile
- `zeta_L_on_x_axis` = c128(mu_L, ngkmax)/**p_x** (resharded to P('x',None),
  replicated on y); `zeta_R_on_y_axis` = c128(mu_R, ngkmax)/**p_y**
- `V_q_block` = c128(mu_L, mu_R)/p_xy
- `g0_acc` = c128(n_q_ibz, mu_L)/p_x if write_g0
- `psi_centroids_persistent` = 2 × c128(nk, ns, mu, nb_total)/p_xy — caller-
  scope ψ_r centroids retained via `psi_rmu_Y`/`transverse_wfn_data` closures
  during all 7 tiles (agent_h §4, agent_i §5); docstring admits nb_total
  over-counts (real slab is nb_r ≈ 160) — "conservative bias preferred"
- `sphere_idx_replicated` — note called with `nq=nk` here (vs `nq=nq`
  elsewhere; same value in practice since nq=nk_tot in the planner)

### `_peak_E_v_q_unfold(*, n_q_full, mu_L, mu_R, p_xy)` — L592-607
Returns `{"E.V_acc_full_BZ": c128(n_q_full, mu_L, mu_R)/p_xy}` — the post-loop
`unfold_v_q` output (`common/symmetry_maps.py:392-470`). Aliased into the V_acc
slot, so it is surfaced in `peak_components` for display but **excluded from
E_total** (see L977-982).

### `_peak_E_v_q_bispinor_buffer(*, n_q_full, mu_T, p_xy, use_ibz_T)` — L610-631
Lorentz-mix transient when the transverse IBZ cascade is active
(`gw/v_q_bispinor.py:587-728`): `tt_full_in_9_tiles` = 9 × c128(n_q_full, mu_T,
mu_T)/p_xy + `tt_mixed_6_tiles` = 6 × same. Returns 0-dict when
`use_ibz_T=False` — which is **always**, since no caller passes `use_ibz_T`.

### `plan_gflat_chunks(...)` — L638-1007
The public entry. Deterministic r-first picker, no iterative search:

1. **Setup** (L704-736): legacy `fft_box_factor` alias overwrites
   `fft_box_factor_A`; mesh sizes `p_x·p_y = p_xy = p`; `mu =
   meta.n_rmu_padded` falling back to `meta.n_rmu`; `nq = meta.nk_tot`
   (q-grid ≡ k-grid assumption); `fft_grid` from meta or cube-root fallback
   for SimpleNamespace test metas; `n_sphere_buffers` via the (now-degenerate)
   bispinor/charge branch; `n_q_ibz` defaults to full BZ (conservative);
   `target = budget_gb·1e9 · target_utilization`.
2. **r_chunk** (L753-810): slope `α_C = pair_density_slots ·
   c128(nk,ns,ns,mu)/p_xy + c128(nq,mu)/p`; constant `c_C_const` = 4×centroids
   + L_q + gflat_acc + sphere-idx (re-derived inline, duplicating
   `_peak_C_fit_one_rchunk`'s persistent block); `r_chunk =
   clamp(headroom_C/α_C, [min(mu, n_rtot), n_rtot])`, also floored at
   `ceil(n_rtot/max_chunks)` and rounded down to a multiple of p_xy (min p_xy).
   Override path recomputes the natural cap and prints a WARNING with the
   Peak-C estimate at the override.
3. **gflat_chunk_size** (L812-875): `GFLAT_CHUNK_FLOOR=4`, `bc_floor_factor=4`;
   base_D re-derived inline (duplicating `_peak_D_accumulate` persistent +
   zeta_chunk); `cs = min(headroom_D/fft_per_row, CAP=100)` rounded down to a
   multiple of 4, floored at 4. Override wins over the cap with a printed
   WARNING quoting the runtime Peak-D estimate (`peak_D_at_override = base_D +
   fft_per_row·cs`, formula "replicated here as a quick estimate").
4. **band_chunk** (L877-900): `bc_cap = 0.5·target / (c128(ns,n_rtot)·
   factor_A)` clipped to nb_total, rounded down to a power of 2, then bumped
   up to a multiple of p_xy ≥ p_xy via `_bump_band_chunk_to_mesh_floor`
   (nested closure, L744-751; cap `max(nb_total, p_xy)`). Override path prints
   the mesh-floor bump message (tested by test_band_chunk_size_floor.py).
   L885-888: `band_chunk_pre` computed, then `if ... and band_chunk_pre <
   p_xy: pass` — a **no-op dead branch**.
5. **Peaks + HWM** (L902-1007): evaluates all five peak functions;
   `peak_E = peak_E_off` (TT off-diag, same_zeta=False) when bispinor else
   `peak_E_cc` (same_zeta=True, write_g0=True); `E_total = E_per_tile +
   E_lorentz` (unfold excluded — aliased slot); `hwm = max` over
   {A_centroid, B_CCT_chol, C_fit_one_rchunk, D_accumulate, E_v_q};
   `bottleneck` = argmax. Both `mu_L=mu_R=mu` for every Peak-E call — the
   planner does not model distinct charge/transverse centroid counts
   (production uses ~640:200 per the bispinor-centroid-ratio memory note).

## Flags / config consumed

Pure function — no direct config reads. Via the gw_init call site the
cohsex.in ↔ kwarg wiring is:
- `memory_per_device_gb` → `budget_gb` (`cfg.memory.per_device_gb`)
- `r_chunk_size` → `r_chunk_override` (`cfg.memory.r_chunk_override`, >0 wins)
- `band_chunk_size` → `band_chunk_override` (`cfg.memory.band_chunk_size`)
- `gflat_chunk_size` → `gflat_chunk_size_override` (`cfg.memory.gflat_chunk_size`;
  Round-4 fix: threaded as kwarg so the planner HWM reflects the runtime cs)
- `cfg.bispinor` → `is_bispinor`
- **NOT wired**: cohsex.in `chunk_target_utilization` feeds only the legacy
  `compute_optimal_chunks`; the gflat planner gets a hand-tuned
  `target_utilization=0.80` at the call site (docstring at L687-688 claims the
  wiring exists — stale).

`jax.default_backend()` is read at call time (pair-density slot count).

## I/O

None. No files read or written; output is the returned dataclass + rank-0
`print` of `plan.format()` and override WARNING lines.

## Cross-module deps

- `jax.sharding.{Mesh, NamedSharding, PartitionSpec}` (Mesh for the mesh_xy
  arg; NamedSharding/P only inside the never-taken `use_query_fft_peak_bytes`
  branch)
- `jax` (lazy, in `_default_pair_density_slots`)
- `common.fft_helpers.query_fft_peak_bytes` (lazy, never-taken branch)
- `jax.numpy as jnp` imported at L83 and **never used**

## Dead suspects

1. `_largest_divisor_le` (L145-153) — grep for the name across
   src/tests/tools/scripts hits only the definition. Dead.
2. `import jax.numpy as jnp` (L83) — `jnp` never referenced in the file. Dead
   import (and forces jax import at module load).
3. `use_query_fft_peak_bytes` branch in `_peak_D_accumulate` (L483-499) +
   kwarg (L658) — no caller ever passes True (grepped
   `use_query_fft_peak_bytes` repo-wide: only gw_init comments about the
   *other* AOT model and fft_helpers itself). Dead-in-practice, kept for
   calibration.
4. `use_ibz_T` kwarg (L657) → `_peak_E_v_q_bispinor_buffer` 9+6-tile branch
   (L624-631) — no caller passes it; always returns the 0.0 dict. The
   function's non-trivial branch is unreachable in the current tree.
5. `n_q_ibz`, `pair_density_slots_charge/transverse`, `fft_box_factor_A/D`
   kwargs — never passed by any caller; only defaults used (gw_init passes the
   *legacy alias* `fft_box_factor=4.0`, which equals the default anyway).
6. Unused params of `_peak_C_fit_one_rchunk`: `band_chunk, n_bc, n_rtot, p_x,
   p_y, fft_box_factor, is_charge_channel` accepted but unreferenced in the
   body (obsolete since the Round-6 scan change removed the n_bc×FFT-box term).
7. L885-888 no-op: `band_chunk_pre` + `if band_chunk != band_chunk_pre and
   band_chunk_pre < p_xy: pass` — vestigial (a removed warning?).
8. `_bump_band_chunk_to_mesh_floor` is defined before step 1 (L744-751) but
   used only in step 3 — not dead, just oddly placed.

## Redundancy suspects

1. Picker constants duplicate the peak functions: `c_C_const` (L775-781)
   re-derives `_peak_C_fit_one_rchunk`'s persistent block term-by-term;
   `base_D` (L821-833) re-derives `_peak_D_accumulate`'s persistent +
   zeta_chunk. Any formula change must be made in two places.
2. `peak_D_at_override` (L850-851) is a third copy of the Peak-D FFT-box
   formula, self-described as "replicated here as a quick estimate".
3. `fft_box_factor` legacy alias next to `fft_box_factor_A` (L647-648,
   L704-706) — classic old/new parallel path; the only production caller still
   uses the legacy name.
4. `N_SPHERE_IDX_BUFFERS_BISPINOR` vs `_CHARGE` both = 1 post-Round-6; the
   `is_bispinor` selection (L726-727) and the L723-725 comment ("we measured
   8 buffers; non-bispinor charge-only is 3") are stale relative to the
   constants.
5. `p` vs `p_xy` are the same value (`p = p_xy`, L710) yet both are threaded
   through every peak-function signature and used interchangeably
   (`shard=p` vs `shard=p_xy`) — a latent-refactor seam, harmless today.

## Weird code

1. L723-725 comment claims 8 bispinor / 3 charge sphere-idx buffers; the
   constants above it are both 1 (post-Round-6). Stale narrative.
2. `GFlatChunkPlan.gflat_chunk_size: Optional[int]` with inline comment
   "always int after the 2026-05-17 cap" — type lies, kept for API stability?
3. Magic constants: `GFLAT_CHUNK_SIZE_CAP=100` (empirical cuFFT crossover),
   `GFLAT_CHUNK_FLOOR=4` + `bc_floor_factor=4` ("cuFFT plan amortisation"),
   `0.5 * target` in the band_chunk cap (L882 — undocumented ½ factor),
   `target_utilization=0.94` default vs 0.80 at the only production call site
   vs 0.97 cohsex.in default for the *other* chunker. Three different
   utilization numbers for one concept.
4. Docstring L687-689 says `target_utilization ← cohsex.in
   chunk_target_utilization` but gw_init explicitly documents that this knob
   is NOT wired to this planner (feeds legacy `compute_optimal_chunks` only).
5. `nq = meta.nk_tot` (L714) — q-grid silently assumed identical to k-grid;
   `nk` and `nq` then coexist as distinct-looking variables with equal values,
   and `_peak_E_v_q_per_tile_transient` is called with `nq=nk` for its
   sphere-idx term while others use `nq=nq`.
6. Peak E models `mu_L = mu_R = mu` everywhere (L936-966), i.e. no distinct
   transverse vs charge centroid counts, and always evaluates worst-case
   TT-off-diagonal for bispinor — conservative by design but diverges from the
   production 640:200 μ_C:μ_T split.
7. E_total commentary (L977-979): "V_acc_full_BZ is aliased ... so we subtract
   it" — code never adds it in the first place (it lives only in
   `peak_E_all`/`peak_components` for display). Comment describes a subtraction
   that is really an omission.
8. `except Exception: pass` (L498-499) silently swallows AOT-measurement
   failure — intentional fallback but invisible when it triggers.
9. Fallback `fft_grid = (round(n_rtot**(1/3)),)*3` (L721-722) exists solely so
   SimpleNamespace test metas work — planner accuracy silently degrades if
   production meta ever lacks `fft_grid`.
10. gw_init caller estimates `ngkmax ≈ 0.06 · n_rtot` when meta lacks it
    (gw_init.py:597) — magic 0.06 lives outside this file but sizes gflat_acc,
    a first-order Peak C/D term.
