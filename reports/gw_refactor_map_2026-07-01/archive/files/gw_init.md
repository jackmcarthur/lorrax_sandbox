# src/gw/gw_init.py — deep-read notes (2026-07-01)

**LOC:** 1328. **Module docstring role:** "ISDF fitting orchestration and memory-aware chunk sizing for LORRAX GW."

## Purpose

Top-level orchestrator for the ISDF preparation stage of the GW pipeline: sizes device-memory chunks (three overlapping planners), loads centroid-sampled ψ, drives the ζ fit (`fit_zeta_to_h5`), drives the bare-Coulomb `V_qmunu` build (scalar and bispinor 7-tile paths), writes restart state, and builds the 4-copy `Wavefunctions` bundle. Entry from `gw.gw_jax.main` via `prepare_isdf_and_wavefunctions`.

Category: **orchestration + resource mgmt (memory planner) — pipeline stage glue between I/O, ISDF fit, and V_q physics**.

## Entry points (grep evidence: src/, tests/, tools/, scripts/)

| symbol | callers |
|---|---|
| `prepare_isdf_and_wavefunctions` | `src/gw/gw_jax.py:169` (only caller) |
| `get_effective_chunk_size` | `src/gw/gw_jax.py:152` |
| `compute_optimal_chunks` | internal (`prepare_isdf_and_wavefunctions:1197`); `tests/archive/test_chunked_wfn_loading.py:49,784` (archived test) |
| `fit_zeta` | internal only (`prepare_isdf_and_wavefunctions:1227`) — no external callers found (grepped `\bfit_zeta\(` across src/tests/tools/scripts) |
| `compute_V_q` | internal only (`prepare_isdf_and_wavefunctions:1246`) — grepped `\bcompute_V_q\(`; note `gw/v_q_g_flat.py` has an unrelated same-named function |
| `build_wavefunction_bundle` | internal only (called twice in `prepare_isdf_and_wavefunctions`, both non-restart and restart paths) |
| `get_bandranges` | re-exported by `src/gw/__init__.py:8`; **no actual call sites** — `psp/get_DFT_mtxels.py:170` defines its OWN copy and calls that at :836 |
| `read_lorrax_input` / `read_cohsex_input` | re-exports from `.gw_config` (line 516, "backward-compatible"); imported *via gw_init* by `bandstructure/htransform.py:879`, `scripts/checks/sigma_direct_check.py:68`, `scripts/checks/w_from_eps0_0d_check.py:58` |
| `_apply_aot_chunk_model` | internal only (`fit_zeta:574`) |

## Function-by-function

### Memory-model helpers (lines 22–151)

- Constants (33–36): `_BYTES_C128 = 16.0`; `_FFT_COPIES = 3` ("cuFFT input + output + scratch (lower bound)"); `_ZCT_ADDITIONAL_COEF = 3` (AOT-measured, ref `scripts/validate_fft_workspaces.py`).
- `_bytes_c128(*dims, shard)` (39–44): bytes of complex128 array per device after sharding.
- `_ChunkAlphas` dataclass (47–62): per-cr byte coefficients for the five r-chunk moments. Fields: `α_pair, α_psi_Y_bc, α_zcol, α_z_slice, α_gather, c_solve, m_psi_G_bc`. (Greek-letter identifiers in source.)
- `_build_chunk_alphas(...)` (65–77): builds `_ChunkAlphas` from dims. Note comment: α_pair scales by ns² because the unified open-spin pair density is rank-5 `(nk, ns, ns, μ, cr)` for ALL channels (charge γ̃⁰=I and transverse γ̃ⁱ=αⁱ).
  - `α_pair = bytes(nk,ns,ns,mu)/(p_x·p_y)`, `α_psi_Y_bc = bytes(nk,band_chunk,ns)/p_y`, `α_zcol = bytes(nq,mu)/p`, `α_z_slice = bytes(mu)/p`, `α_gather = bytes(mu)/p + 2·bytes(mu)`, `c_solve = bytes(mu,mu)/p_x² + 3·bytes(mu,mu)`, `m_psi_G_bc = bytes(nk,band_chunk,ns,nr)/p`.
- `_fft_moment(cr, base, fft_inloop_bytes, a, n_bc=1)` (80–100): `base + 2·α_pair·cr + fft_inloop + n_bc·α_psi_Y_bc·cr`. Long comment: `_make_fit_one_rchunk_kernel` Python-unrolls the bc-loop inside one jit so n_bc post-reshard ψ_bc_Y buffers can be concurrently live (was missing 17 GB on CrI3 16-GPU at chunk_r=112016, band_chunk=16, n_bc=5).
- `_zct_moment` (103–105): `base + (2+3)·α_pair·cr`.
- `_reshard_moment` (108–110): `base + 3·α_zcol·cr` (input + output + NCCL scratch).
- `_solve_moment` (113–115): `base + 2·α_zcol·cr + q_batch·(2·α_z_slice·cr + c_solve)`. **Zero grep hits outside its definition — dead** (the driver re-derives the same expression inline in `_eval_stages` line 276 and `_max_cr_per_stage` line 143).
- `_gather_moment` (118–120): `base + α_zcol·cr + q_gather·α_gather·cr`. **Also only referenced via `_eval_stages`? No — `_eval_stages` line 277 re-derives inline; `_gather_moment` itself has zero call sites → dead.** (Grepped `_solve_moment\|_gather_moment\|_fft_moment\|_zct_moment\|_reshard_moment` in src: `_fft_moment`, `_zct_moment`, `_reshard_moment` ARE called in `_eval_stages` 273–275; `_solve_moment` and `_gather_moment` are NOT.)
- `_max_cr_per_stage(headroom, fft_cost_in_loop, a, *, nr_max, m_budget, m_zct_cap, n_bc)` (123–150): closed-form linear inversion `cr ≤ (headroom − c)/α` per moment; returns dict `{'fft','zct','reshard','solve','gather'}`; optional zct soft-cap from `zct_stage_cap_gb`.

### `compute_optimal_chunks(meta, mesh_xy, memory_budget_gb, target_utilization=0.97, verbose, n_b_left, n_b_right, r_chunk_override, zct_stage_cap_gb)` (154–404)

Legacy heuristic chunk planner for the 6-stage ISDF r-chunk loop (FFT / Pair / ZCT / Reshard / Solve / Gather; docstring gives per-stage buffer multipliers, "XLA HLO-calibrated, CrI3 16 GPUs"). Steps:
1. Persistent centroid storage check (X+Y sharded copies, full-load `nb` vs post-slice `nb_l+nb_r`); raises if over budget.
2. Pre-loop band-chunked FFT sizing: `bpd_max = (headroom − m_phase)/(3·bytes(nk,ns,nr))`; `band_chunk = min(nb, bpd_max·p)`.
3. C_q build stage check: `m_centroids + 2·bytes(nk,μ,μ)/p + bytes(nq,μ,μ)/p`.
4. In-loop FFT workspace via `common.fft_helpers.query_fft_peak_bytes` on the UNSHARDED shape `(nk, band_chunk, ns, nx, ny, nz)`, `fft_axes=(-3,-2,-1)`, sharding `P(None, ('x','y'), None, None, None, None)`, complex128 — queried exactly per band_chunk value (`_fft_inloop_bytes`, 345–358). Big comment (329–342) documents the historical bug: old `fft_per_k` dropped the nk factor → Si 10×10×10 4-GPU under-predicted by ~19 GiB (module_0147.jit__kernel memory-usage-report; top preallocated-temp = 18.54 GiB = 3 × nk·bpd·ns·nr·16).
5. `_find_r_chunk` (291–326): base = m_centroids + m_L_q; `n_bc = ceil(nb/band_chunk)`; picks cr = min over stage limits, rounds cr down to a multiple of p_total (solve sharding); returns None if infeasible → outer loop halves `bpd` and retries (361–369).
6. Sizes `q_chunk` (solve) and `q_gather` (H5 write) from leftover headroom (373–378).
7. Returns dict: `band_chunk, chunk_r, q_chunk, q_gather, k_chunk, memory_estimate{peak_estimate_gb, budget_gb, bottleneck, available_vcoul_gb, limit_info}`.

Inner `_eval_stages` (258–289) forward-evaluates all five moments at chosen cr and sizes a `k_batch` (`fft_head·0.5/fft_per_k` — magic 0.5 safety factor at line 288).

**Key fact for refactor:** its `chunk_r` and `band_chunk` picks are unconditionally OVERWRITTEN later by the gflat planner in `fit_zeta` (lines 636–637). Only `q_chunk`, `memory_estimate.available_vcoul_gb` (V_q budget) and `band_chunk` (used once for the centroid load at 1221–1225, before fit_zeta re-picks) survive as real consumers. `q_gather` and `k_chunk` are returned but never read anywhere (grepped `q_gather`/`k_chunk` across src — only gw_init hits).

### `_apply_aot_chunk_model(chunks, cfg, meta, mesh_xy, *, band_range_left, band_range_right, print_fn, rank0)` (407–501)

Runs the AOT (ahead-of-time XLA memory-analysis) chunk chooser/predictor from `gw.aot_memory_model` (`predict_kernel_peak, SysDims, MeshSpec, Knobs, choose_chunks_analytic, choose_chunks_heuristic, describe_chunks`). Two modes via `cfg.memory.use_aot_chunk_chooser`:
- True: chooser (heuristic 20/80 default, or `chunk_chooser_mode="analytic"` regressed fit for kernel `"fit_one_rchunk"`, tag `"current"`) overrides `chunks['chunk_r']`/`chunks['band_chunk']` in place; keeps q_chunk/q_gather/k_chunk from the heuristic.
- False: predict-only; logs AOT peak for γ = runtime/AOT-pred calibration.
Returns AOT-predicted peak GB or None if import fails (silent `except Exception` → heuristic fallback).
**Note:** any chunk_r/band_chunk it sets is then clobbered by the gflat planner in `fit_zeta` (636–637), so chooser mode only affects the logged prediction in practice — redundancy/dead-path suspect.

### `get_effective_chunk_size(chunk_size)` (504–512)

Flag decoder: −1→None (all bands), 0→64 (auto), 1–2048→explicit, else ValueError. Caller: `gw_jax.py:152` (`meta.chunk_size = ...` from `config.memory.chunk_size`).

### Re-exports (515–516)

`from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401` — backward-compat shim; three external files still import via gw_init.

### `get_bandranges(nv, nc, nband, nelec)` (519–529)

Legacy: returns `nvrange=[nelec−nv,nelec], ncrange=[nelec,nelec+nc], nsigmarange=[nelec−nv,nelec+nc], n_fullrange=[0,nband], n_valrange=[0,nelec]`. Docstring says "used by psp/get_DFT_mtxels.py" — but that file defines its own identical copy (`psp/get_DFT_mtxels.py:170`) and calls the local one. gw_init's copy has zero call sites (only the `gw/__init__.py` re-export). **Dead + duplicate.**

### `fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir, psi_rmu_Y, psi_rmuT_X, chunks, print_fn)` (532–869)

Drives `common.isdf_fitting.fit_zeta_to_h5`. Returns `(zeta_h5_path, mem_est, transverse_wfn_data)` — docstring stale: says "Returns (zeta_h5_path, mem_est)".

Sequence:
1. `set_gamma_contract_mode(cfg.backend.gamma_contract_mode)` — module-level global for the γ̃·γ̃ kernel (comment: threading a kwarg through shard_map bodies "would be churn").
2. Band windows: `band_range_left = (b0, b3)` (all val + sigma cond), `band_range_right = (b1, b4)` (sigma val + all cond).
3. Prints legacy planner memory estimate; runs `_apply_aot_chunk_model` on EVERY rank (comment 570–573: historical rank-0-only guard caused mismatched band_chunk across ranks → mismatched NCCL buffers in `_fft_gather_reshard` → hang on 2nd band chunk).
4. **G-flat planner (581–642):** `gw.gflat_memory_model.plan_gflat_chunks(...)` with `target_utilization=0.80` HARD-CODED (long comment 613–619: hand-tuned for bispinor 4-channel slack; the cohsex.in `chunk_target_utilization` knob feeds ONLY the legacy planner), `fft_box_factor=4.0`, `max_chunks=64`, `_ngkmax_est = meta.ngkmax or 0.06·n_rtot` (magic 0.06), `_nq_disk_est = nk_tot` (conservative full-BZ). Overrides `chunks['band_chunk']`, `chunks['chunk_r']`; sets `chunks['gflat_hwm_gb']`, `chunks['gflat_chunk_size']`. Round-4 fix note: `gflat_chunk_size` override now passed INTO the planner (was applied after → HWM mis-predicted).
5. Cutoff resolution (`_resolve_cutoff`, 681–689): `bare_coulomb_cutoff_ry` (V_q sqrt_v mask) and `zeta_cutoff_ry` (on-disk per-q ζ sphere / `ngk[q]`) both default `wfn.ecutwfc` (BGW `screened_coulomb_cutoff` default match), capped at `wfn.ecutrho`; enforces zeta_cutoff ≥ bare_coulomb_cutoff (V_q reads ζ̃(q+G) at every G in its sphere).
6. Charge ζ fit: `fit_zeta_to_h5(...)` → `{tmp_dir}/zeta_q.h5`; `write_ibz_only` gated on env `LORRAX_FORCE_FULL_BZ` (default IBZ-on; orbit closure checked downstream). Wrapped in `timing.section("gw_jax.zeta_fit_chunked")` + `jax_profile.trace_section("zeta_fit")`.
7. HWM print + γ = peak_gb/aot_peak_gb calibration log (729–742).
8. **Bispinor transverse branch (747–867):** loud-fail if `cfg.bispinor` and no `cfg.paths.centroids_file_current` (silent scalar-V_q fallback + shape-mismatch crash documented from 2026-05-14 CrI3 30 Ry KNOWN_SANDBOX_ERRORS). Loads current-density centroids (`file_io.centroids.load_centroids`), rebuilds `meta_curr` via `dataclasses.replace` with `n_rmu_curr`, `n_rmu_jax` (rounded to `meta.n_proc`), `n_rmu_padded` (rounded to `jax.device_count()`; comment: missing refresh tripped TypeError at isdf_fitting.py:1442). `meta_curr.sys_dim = meta.sys_dim` copied MANUALLY (dynamic attr set by gw_jax.main; Meta has no sys_dim field). Reloads centroid ψ via `load_centroids_band_chunked`, then loops `mu_L in (1,2,3)` calling `fit_zeta_to_h5(..., vertex_mu_L=mu_L)` → `zeta_q_mu{1,2,3}.h5`; before each, `_drop_traced_caches()` clears `isdf_fitting._fit_one_rchunk_cache` + `gc.collect()` (comment: cache key includes `id(psi_G_store)`, drop = memory hygiene, not tracer workaround; pair-density caches preserved to share compile). Returns `transverse_wfn_data = {psi_rmu_Y, psi_rmuT_X, meta, centroid_indices}` for the σ^B Wfns bundle.

Config consumed: `cfg.backend.gamma_contract_mode`, `cfg.memory.per_device_gb / use_aot_chunk_chooser / chunk_chooser_mode / chunk_target_utilization / r_chunk_override / band_chunk_size / gflat_chunk_size`, `cfg.bispinor`, `cfg.paths.centroids_file_current`, `cfg.head.bare_coulomb_cutoff`, `cfg.head.zeta_cutoff`, `cfg.backend.slab_io / cusolvermp_charge / cusolvermp_lu`, `cfg.gspace_mode`. Env: `LORRAX_FORCE_FULL_BZ`.

### `compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est, print_fn, bgw_v_grid_fn, sym, centroid_indices)` (872–1145)

Computes bare Coulomb `V_qmunu = ∫∫ ζ*_μ(q,r) v(r,r') ζ_ν(q,r')` (flat-q shape `(nq, μ, μ)`) from ζ on disk, plus `G0 = ζ_μ(G=0)` at q=0; writes `g0_mu` back into zeta_q.h5. Steps:
1. `os.sync()` on rank 0 + `sync_global_devices("zeta_flush")` barrier.
2. Budget = min(legacy planner's `available_vcoul_gb`, `common.gpu_utils.get_device_memory_info()['budget_gb']`) (silent `except Exception: pass`).
3. Re-resolves `vcoul_cutoff_ry` from `cfg.head.bare_coulomb_cutoff` (duplicating `_resolve_cutoff` logic from fit_zeta WITHOUT the ecutrho validation — comment at 906–911 says "hoist the resolved value... rather than re-resolving" but the code DOES re-resolve, just without validation; mild drift risk).
4. **Bispinor branch** (921–1084): `bispinor_ready` = cfg.bispinor AND all `zeta_q_mu{1,2,3}.h5` exist; if bispinor but not ready → prints warning and **silently falls back to scalar V_q** (contrast with fit_zeta's loud-fail; the loud-fail upstream should make this unreachable, but the fallback still contradicts the guard's philosophy). If ready:
   - Reads `zeta_layout` from `file_io.isdf_header.read_isdf_header(zeta_h5_path)`; `_is_g_flat = (layout == 'G_flat')`.
   - `n_rmu_T` from disk: `f['zeta_q_G'].shape[1]` (G-flat) or `f['zeta_q'].shape[-1]` (r-space).
   - IBZ gating: `_use_ibz_bispinor` from `LORRAX_FORCE_FULL_BZ`; reloads transverse centroid indices from text file (comment: "reloading is cheap... keeps the bispinor IBZ wiring local"; also notes orbit-closure check in `_resolve_ibz_q_list` "silently falls back to full-BZ on failure").
   - G-flat path: `gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` with four `file_io.zeta_reader.ZetaReader`s → `v_q_bispinor.h5`. Consumes `cfg.memory.vq_g_chunk_size`, `wfn.bdot` (0-D only), `meta.sys_dim`.
   - Legacy r-space path: `make_v_munu_chunked_kernel` (from `.compute_vcoul`) + `compute_V_q_bispinor_to_h5` with four `file_io.slab_io.SlabIO` handles. Consumes `cfg.head.mc_average_vcoul_body`.
   - Reads CC tile + g0 back via `BispinorVqReader.get_tile(0,0)` / `get_g0_CC()`; zero-pads V_q_raw/G0 from logical n_rmu to `meta.n_rmu_padded` (comment: pad rows of ψ are zero — "Phase 3a invariant" — so zero-padding V is exact).
5. **Scalar branch** (1085–1117): `compute_all_V_q` (from `.compute_vcoul`) via `ZetaReader` with `use_g_flat_zeta=True` ("ZetaReader is the V_q reader of record after C3 — the FFT moves out of the V_q kernel into the reader's read_zeta_G_slab"). Consumes `mc_average_vcoul_body`, `bare_coulomb_cutoff`, `bgw_v_grid_fn`, `sym`, `centroid_indices`, `vq_g_chunk_size`.
6. G0 write-back (1119–1128): `_slab_io_allgather._to_host` (private-module import across package boundary), rank-0 h5py append of `g0_mu` to zeta_q.h5, `sync_global_devices("g0_write")`.
7. `G0` dimensionality squeeze: `while G0.ndim > 1: G0 = G0[0]` — index gymnastics tolerating unknown leading axes.
Returns `(V_qmunu, G0)`.

### `build_wavefunction_bundle(wfn, sym, meta, band_slices, mesh_xy, *, psi_rmu_Y, psi_rmuT_X, enk_full=None, print_fn)` (1148–1169)

Thin wrapper: optional `get_enk_bandrange` then `gw.wavefunction_bundle.build_wavefunctions` → 4-copy sharded bundle (xn/xr/yr/yn) over [b0,b4).

### `prepare_isdf_and_wavefunctions(*, cfg, wfn, sym, meta, centroid_indices, band_slices, mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None, **_ignored)` (1172–1328)

Top-level orchestrator (7-step docstring). Non-restart path: `compute_optimal_chunks` → cap band_chunk by `cfg.memory.band_chunk_size` (UPPER cap; comment cites CrI3 6×6×1 80 Ry 121 GB OOM in to_rmu from unsharded FFT box) → `load_centroids_band_chunked` (ψ at centroids, [b0,b4), reused by ζ-fit and Wfns bundle) → `fit_zeta` → optional `LORRAX_EXIT_AFTER_ZETA` clean `SystemExit(0)` (profiling; pairs with `LORRAX_MAX_RCHUNKS`/`LORRAX_RCHUNK_DEBUG`) → `mem_probe("pre_v_q")` / `compute_V_q` / `mem_probe("post_v_q")` (P4/P5 probes, gated `LORRAX_MEM_DEBUG`) → `get_enk_bandrange` → `write_restart_state_to_h5(mode="w", V_qmunu, G0_mu_nu, enk_full, init_W0=True, kgrid=...)` → `build_wavefunction_bundle` (+ second transverse bundle for σ^B if bispinor) → `write_restart_state_to_h5(mode="a", psi_full_y=wfns.psi_yr)` → `save_restart_state_per_proc(tmp_dir/isdf_tensors, V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)`. Restart path (`cfg.restart`): `load_restart_state_from_h5` + rebuild bundle; `wfns_transverse = None` ("bispinor restart not-yet-supported... fail loud"). Returns `SimpleNamespace(V_qmunu, wf_bundle, wf_bundle_transverse)`.

`**_ignored` in the signature silently swallows unknown kwargs — refactor smell.

## I/O

| file | dir | format / datasets |
|---|---|---|
| `{tmp_dir}/zeta_q.h5` | W (via `fit_zeta_to_h5`), later A (adds `g0_mu` dataset rank-0 h5py); R (isdf_header `zeta_layout`, `ZetaReader`/`SlabIO` for V_q) | datasets referenced here: `zeta_q_G` (G-flat, shape[1]=n_rmu), `zeta_q` (r-space, shape[-1]=n_rmu), `g0_mu`, `isdf_header/zeta_cutoff_ry` (mentioned in comment) |
| `{tmp_dir}/zeta_q_mu{1,2,3}.h5` | W (bispinor transverse fits), R (V_q) | same layout as zeta_q.h5 |
| `{zeta_dir}/v_q_bispinor.h5` | W (bispinor orchestrators), R (`BispinorVqReader`: CC tile + g0_CC) | 7 polarisation tiles V^{μ_L,ν_L}_q |
| `{tensors_filename}` restart H5 | W mode="w" then mode="a" | V_qmunu, G0_mu_nu, enk_full, W0 placeholder (`init_W0`), kgrid, `psi_full_y` |
| `{tmp_dir}/isdf_tensors*` | W (`save_restart_state_per_proc`) | per-proc restart dump |
| `centroids_frac_*_current.txt` | R (`load_centroids`, twice: fit_zeta + compute_V_q bispinor branch) | text centroid fractions |

## Flags / env consumed

cohsex.in via LorraxConfig: `memory.per_device_gb, chunk_target_utilization, chunk_size, band_chunk_size, r_chunk_override, zct_stage_cap_gb, use_aot_chunk_chooser, chunk_chooser_mode, gflat_chunk_size, vq_g_chunk_size`; `backend.gamma_contract_mode, slab_io, cusolvermp_charge, cusolvermp_lu`; `head.bare_coulomb_cutoff, zeta_cutoff, mc_average_vcoul_body`; `bispinor`; `gspace_mode`; `restart`; `paths.centroids_file_current`.
Env: `LORRAX_FORCE_FULL_BZ` (3 read sites), `LORRAX_EXIT_AFTER_ZETA`, `LORRAX_MEM_DEBUG` (+ comment-mentioned `LORRAX_MAX_RCHUNKS`, `LORRAX_RCHUNK_DEBUG`).

## Dead suspects

1. `_solve_moment` (113) and `_gather_moment` (118) — zero call sites; `_eval_stages`/`_max_cr_per_stage` re-derive the same formulas inline. Grep: `grep -n '_solve_moment\|_gather_moment' src/**/*.py` → only definitions.
2. `get_bandranges` (519) — zero callers; `psp/get_DFT_mtxels.py:170` has its own duplicate copy it actually uses. Only the `gw/__init__.py` re-export references it.
3. `chunks['q_gather']` and `chunks['k_chunk']` outputs of `compute_optimal_chunks` — computed, returned, never consumed anywhere (grepped `q_gather`, `k_chunk` across src/tests/tools/scripts; only gw_init hits). `m_psi_G_bc` field of `_ChunkAlphas` likewise never read after construction.
4. `compute_optimal_chunks`'s `chunk_r`/`band_chunk` picks — live code but functionally superseded: `fit_zeta` lines 636–637 unconditionally overwrite them with `plan_gflat_chunks` output. Only `q_chunk`, `available_vcoul_gb`, and the pre-fit centroid-load `band_chunk` matter.
5. `_apply_aot_chunk_model` chooser mode (`use_aot_chunk_chooser=True`) — its chunk overrides are also clobbered by the gflat planner; effectively predict/log only.
6. Only external test of `compute_optimal_chunks` lives in `tests/archive/` (archived).

## Redundancy suspects

1. **Three stacked chunk planners** for the same kernel: legacy `compute_optimal_chunks` (5-moment closed-form), AOT `aot_memory_model` chooser/predictor, and `gflat_memory_model.plan_gflat_chunks` (4-peak, final authority). Classic parallel-old/new-paths cruft; refactor should collapse to one.
2. `get_bandranges` duplicated verbatim in `psp/get_DFT_mtxels.py:170`.
3. Cutoff resolution duplicated: `fit_zeta._resolve_cutoff` (validated) vs `compute_V_q` lines 912–915 (re-resolves without ecutrho validation, despite a comment claiming it "hoists" the resolved value).
4. Bispinor V_q dual path: legacy r-space `compute_V_q_bispinor_to_h5` + new G-flat `compute_V_q_bispinor_g_flat_to_h5`, dispatched on on-disk `zeta_layout` (982–1065).
5. Transverse centroid indices loaded twice from the same text file (fit_zeta:772 and compute_V_q:963) instead of being passed through.
6. `_ZCT_ADDITIONAL_COEF`/moment functions vs inline coefficient reconstruction in `_eval_stages` / `_max_cr_per_stage` (same denominators typed twice).

## Weird code

1. Line 613–619: hard-coded `target_utilization=0.80` for the gflat planner while cohsex.in's `chunk_target_utilization` (default 0.97) feeds only the legacy planner — two utilization knobs, one hidden.
2. Line 596: `_ngkmax_est = meta.ngkmax or int(0.06 * meta.n_rtot)` — magic 0.06 sphere-fill estimate.
3. Line 288: `k_batch = int(fft_head * 0.5 / fft_per_k)` — magic 0.5 safety factor, undocumented.
4. Lines 570–579: `_apply_aot_chunk_model` must run on EVERY rank — historical rank-0-only guard caused divergent band_chunk → NCCL hang; a comment is the only guard against regression.
5. Line 793: `meta_curr.sys_dim = meta.sys_dim` — `sys_dim` is a dynamic attribute monkey-patched onto Meta by gw_jax.main, so `dataclasses.replace` drops it; manual copy required. Fragile hidden contract.
6. Lines 937–943: bispinor-but-not-ready → prints a warning and silently falls back to scalar V_q, while fit_zeta (757) loud-fails for the equivalent misconfiguration — inconsistent failure philosophy. Comment at 959 also notes IBZ orbit-closure "silently falls back to full-BZ on failure".
7. Line 1138: `while G0.ndim > 1: G0 = G0[0]` — shape-agnostic squeeze suggesting unresolved layout contract for G0.
8. Line 1121: import of private `file_io._slab_io_allgather._to_host` across module boundary.
9. `fit_zeta` docstring return contract stale (says 2-tuple, returns 3-tuple).
10. Mixed indentation: functions from `get_bandranges` (519) onward use TABS while the top half uses spaces (visible in raw file); also a stray deep-indented `band_norms=_band_norms` at line 840.
11. Greek-letter identifiers (`α_pair` etc.) in `_ChunkAlphas` — intentional but grep-hostile.
12. `**_ignored` kwarg sink on `prepare_isdf_and_wavefunctions` (1174).
13. Lines 899–903: budget clamp via `get_device_memory_info` wrapped in bare `except Exception: pass`.
14. Line 887: rank-0 `os.sync()` (whole-node filesystem sync) as a zeta-flush barrier.

## Cross-module deps (imports)

`common.timing`, `common.jax_profile`, `common.isdf_fitting` (fit_zeta_to_h5, mem_probe, `_fit_one_rchunk_cache` private), `common.gamma_matrices.set_gamma_contract_mode`, `common.fft_helpers.query_fft_peak_bytes`, `common.load_wfns` (load_centroids_band_chunked, get_enk_bandrange), `common.gpu_utils.get_device_memory_info`, `gw.gw_config` (re-exports), `gw.aot_memory_model`, `gw.gflat_memory_model.plan_gflat_chunks`, `gw.compute_vcoul` (compute_all_V_q, make_v_munu_chunked_kernel), `gw.v_q_bispinor` (both orchestrators + BispinorVqReader), `gw.wavefunction_bundle.build_wavefunctions`, `file_io` (SlabIO, ZetaReader, centroids.load_centroids, isdf_header.read_isdf_header, restart-state trio, `_slab_io_allgather._to_host` private).
