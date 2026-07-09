# aot_memory_model core — detailed notes (gw refactor map, 2026-07-01)

Group: `src/gw/aot_memory_model/{__init__,core,chooser,presets,cost,sweep,doe,predict_cli}.py`
(kernels/ subpackage is a separate group; artifacts/ holds the fitted JSONs.)

## Big picture

Offline-calibrated **memory planner**: AOT-lower heavy jits (`jit(f).lower(specs).compile().memory_analysis()`)
over a small DoE, NNLS-fit peak bytes (and FLOPs) against declared "primitive" volume features, persist
the fit as JSON, and at runtime evaluate the closed-form fit to pick `(chunk_r, band_chunk)` under a
per-device budget. No physics; pure resource management.

**Critical production status finding:** the only in-tree consumer is
`gw_init._apply_aot_chunk_model` (src/gw/gw_init.py:407-501, called at :574). When
`use_aot_chunk_chooser=True` it writes `chunks['chunk_r']` / `chunks['band_chunk']`
(gw_init.py:484-485) — but the newer **gflat planner** (`gw.gflat_memory_model.plan_gflat_chunks`,
gw_init.py:587-642) then **unconditionally overwrites both** at gw_init.py:636-637. So this whole
module's chooser output is clobbered in production; the only surviving effect is the
γ-calibration log line (`aot_peak_gb`, gw_init.py:739-742: `gamma = peak_gb / aot_peak_gb`).

There are **three parallel memory-model modules** in the repo:
1. `gw/aot_memory_model/` (this group) — AOT-sweep NNLS fits + choosers, largely superseded;
2. `gw/gflat_memory_model.py` — the current four-peak (A/B/C/D) planner that wins the chunk decision;
3. `runtime/aot_memory.py` — cuFFT-workspace-aware `aot_kernel_peak_bytes` (used by `gw/v_q_tile.py:220`,
   tested by `tests/test_aot_memory.py` — note that test file does NOT test this package despite the name).

No pytest file imports `gw.aot_memory_model` (grepped `tests/` for `aot_memory_model`,
`choose_chunks`, `predict_kernel_peak`: zero hits; `tests/test_aot_memory.py` imports
`src.runtime.aot_memory`; `tests/test_planner_refit_2026-05-17.py` imports `gw.gflat_memory_model`).

External sandbox usage: `runs/MoS2/B_bispinor_profile/sweep_chunks.py:41` imports
`predict_kernel_peak, SysDims, MeshSpec, Knobs`. Reports under
`reports/aot_memory_model_poc_2026-04-20/` and `reports/zeta_rchunk_memory_model_2026-05-13/`
document the calibration campaign.

## I/O (whole group)

All under `src/gw/aot_memory_model/artifacts/` (JSON, checked into the tree):
- `<kernel>__<tag>__samples.json` — list of `{sys, knobs, mesh, meas{temp,argument,output,alias,total,flops,t_lower_s,t_compile_s}}` (written by `sweep.run_sweep` via `core.save_samples`).
- `<kernel>__<tag>__fit.json` — `core.Fit` dict `{kernel, feature_names, coefs, intercept, gamma, residual_rms, n_samples, notes}` (`core.save_fit`).
- `<kernel>__<tag>__cost_fit.json` — `cost.CostFit` dict (`cost.save_cost_fit`).

Tags on disk: `current` (production, 12 kernels), `ortho`, `prev` (fit_one_rchunk only).
Production default tag is `"current"` (`__init__.predict_kernel_peak`, chooser loads).

## Flags consumed

The module itself reads **no** cohsex.in flags. Via the gw_init caller:
`memory_per_device_gb` (`mem.per_device_gb`), `chunk_target_utilization` (budget = per_device_gb·1e9·util),
`use_aot_chunk_chooser` (gw_config.py:225, default False), `chunk_chooser_mode`
(gw_config.py:232, default "heuristic"; "analytic" swaps in the regressed chooser).

---

## 1. `__init__.py` (110 loc)

Re-exports the public API from core/cost/chooser/doe plus one convenience wrapper.

| symbol | lines | role |
|---|---|---|
| `_PREDICT_CACHE` | 77 | module-global `(kernel_name, tag) -> (kernel, fit)` memo |
| `predict_kernel_peak(kernel_name, sys, knobs, mesh, *, tag="current")` | 80-110 | one-call `load_fit` + `predict_peak`; raises if `mesh is None` |

Callers: `predict_kernel_peak <- gw_init._apply_aot_chunk_model` (predict-only γ-calibration path,
gw_init.py:490) and `runs/MoS2/B_bispinor_profile/sweep_chunks.py:41`.

## 2. `core.py` (377 loc)

Registry + AOT measurement + NNLS fit + JSON persistence.

| function/class | lines | role |
|---|---|---|
| `SysDims` (frozen dataclass) | 36-88 | physical dims: `kgrid, n_rmu, n_s, nspinor, n_b_v, n_b_c, n_b, n_b_sum, n_r, fft_grid`; props `n_k = prod(kgrid)` (69-72), `fft_shape` (falls back to kgrid, 81-83), `nb_sum` (= `n_b_sum` or `2·n_b`, 86-88) |
| `MeshSpec` | 91-102 | `(p_x, p_y)`, `n_dev = p_x·p_y` |
| `Knobs` | 105-131 | frozen sorted-tuple kv-bag; `Knobs.of(**kw)`; kernel-specific keys: `chunk_r, band_chunk, k_chunk, q_chunk, mu_chunk, nb_pad` |
| `AotKernel` | 138-156 | base class; subclass declares `name, SYSTEM_DIMS, KNOBS, PRIMITIVES{name->f(sys,knobs,mesh)->bytes}`, `build_specs`, `build_callable`; kernels live in `kernels/`, register via decorator |
| `register_kernel` / `KERNEL_REGISTRY` / `get_kernel` | 159-176 | registry; `get_kernel` lazy-imports `kernels` package |
| `aot_measure(kernel, sys, knobs, mesh, *, disable_remat=True)` | 183-236 | `fn.lower(*specs).compile(...)`; returns `{temp, argument, output, alias, total, flops, t_lower_s, t_compile_s}`; `total = temp + argument + output − alias`; deliberately does NOT double-jit (donation → alias accounting) |
| `Fit` | 243-257 | fit artifact; `gamma` field = runtime/AOT correction, default 1.0 |
| `fit_nnls(kernel, samples, include_intercept=True)` | 259-302 | scipy NNLS: `peak = intercept + Σ βᵢ·Tᵢ(sys,knobs,mesh)`, all β ≥ 0 (negative β ⇒ missed primitive, deliberately clipped); raises on underdetermined |
| `predict_peak(fit, kernel, sys, knobs, mesh)` | 305-316 | `(intercept + Σ β·T) · gamma`, returns bytes |
| `_sample_to_dict` | 326-332 | JSON row |
| `save_samples` / `load_samples` | 335-344 | samples.json |
| `save_fit` / `load_fit` | 347-359 | fit.json |
| `samples_to_fit_input` | 362-377 | JSON → `(SysDims, Knobs, MeshSpec, peak_total)` tuples; pops derived `n_k` |

Callers: `aot_measure, fit_nnls, save_*/load_* <- sweep.py`; `get_kernel, load_fit, predict_peak <- __init__.predict_kernel_peak, chooser.py, predict_cli.py`; `samples_to_fit_input <- sweep.fit_from_saved` (and near-duplicated inline in `cost.fit_cost_from_saved`).

**Weird:**
- **Doc/code drift at 191-198 vs 212-215**: docstring says `disable_remat=True` passes
  `compiler_options={'xla_disable_hlo_passes': 'rematerialization'}`; the code actually passes
  `{"xla_gpu_memory_limit_slop_factor": 10000}`. Hypothesis: the pass-disable option broke or was
  swapped for the slop-factor trick and the docstring was never updated.
- **`SysDims.nspinor` is a dead field**: no kernel primitive reads it (grep `nspinor` in
  `kernels/*.py`: `fit_one_rchunk.py:443` passes `nspinor=sys.n_s`, i.e. n_s doubles as spinor dim).
  `predict_cli --nspinor` is therefore a no-op knob.
- `Fit.gamma` is never written by any code path (grep `gamma` — only read/copied); calibration is
  by hand-editing the artifact JSON.
- Comment at 42-44: varying k in DoE must go through `kgrid`, and some kernels *repurpose* `kgrid`
  as the FFT box (`points_load_psi_rchunk_fft`, `points_vq_mu_chunk`) — overloaded field semantics.

## 3. `chooser.py` (595 loc)

**Three parallel choosers** for `(chunk_r, band_chunk)` for the ISDF ζ-fit (`fit_one_rchunk` kernel):

| function/class | lines | role |
|---|---|---|
| `ChunkChoice` | 38-51 | result dataclass: chunk_r, band_chunk, k_chunk, num_r/bc_chunks, peak_bytes, per_call/total_flops, budget_bytes, note |
| `_ceil_div` | 54-55 | util |
| `_p_divisible_candidates(max_val, p, min_val)` | 58-73 | sparse p-divisible grid (powers of 2 + endpoint) |
| `_enumerate_candidates` | 76-106 | (cr, bc) grid; cr p-divisible; bc grid = explicit list ∪ {n_b} |
| `choose_chunks_aot(sys, mesh, *, budget_bytes, kernel_name="fit_one_rchunk", tag="current", ...)` | 109-171 | **grid-search chooser**: feasibility via `predict_peak ≤ budget`, cost = `ceil(n_rtot/cr) · predict_flops_per_call`; tiebreak toward bigger (cr, bc). **Zero callers** (see dead suspects) |
| `AlphaFit` | 202-208 | regrouped coefficients `α₀, α_cr, α_bc, α_crbc` |
| `_group_alpha(kernel, mem_fit, sys, mesh)` | 211-244 | collapses the 7-primitive fit into 4 scaling classes via kernel `PRIMITIVE_CLASSES`; primitives evaluated at `chunk_r=1, bc=1`; applies γ uniformly |
| `_p_divisible_floor` | 247-250 | util |
| `_largest_even_divisor_leq(n, cap, p)` | 253-278 | largest divisor of n that is ≤ cap and p-divisible (one compile shape); fallback `(cap//p)·p` |
| `choose_chunks_analytic(...)` | 281-412 | **analytic chooser**: closed-form inversion `cr_max(bc) = (M − α₀ − α_bc·bc)/(α_cr + α_crbc·bc)`; 1-D search over p-divisible bc candidates; allgather bound `cr ≤ 0.8·budget/(16·q_gather_min·n_rmu)` (lines 349-354, 20% slack for NCCL temp); optional `fft_launch_overhead_flops` small-bc penalty (default 0) |
| `describe_chunks(choice)` | 415-423 | one-line log formatter |
| `choose_chunks_heuristic(sys, mesh, *, budget_bytes, wfn_workspace_frac=0.20, pair_temp_count=4, fft_workspace_multiplier=3, q_gather_min=1)` | 468-595 | **20/80 budget-split heuristic** (production default): `wfn_budget = 0.2·budget` sizes `(band_chunk × k_chunk)` via `per_wfn_bytes = 3·16·n_s·n_r/P`; remaining 80% sizes chunk_r via `cr_per_byte = 4·16·n_k·n_s²·μ/P` (n_s² for the rank-5 open-spin pair density `(n_k, n_s, n_s, μ, cr)` — see common.isdf_fitting) plus persistent base `base_centroid = 16·n_k·μ·nb_sum·n_s/p_x`, `base_psiG = 16·n_k·n_b·n_s·n_r/P`, `base_Lq = 16·n_k·μ²/P` |

Callers: `choose_chunks_analytic, choose_chunks_heuristic, describe_chunks <- gw_init._apply_aot_chunk_model`
(gw_init.py:472, 477, 481; mode selected by cohsex.in `chunk_chooser_mode`). `choose_chunks_aot <- nobody`.

Physically-motivated constants (from the calibrated small-kernel fits, documented in the 426-466
comment block): `β[psi_G]=3` (load_psi_rchunk_fft: input+output+cuFFT scratch),
`β[PrBr]=4` (zct_lr: 4 concurrent pair-sized ZCT temps), `16` bytes = complex128.

**Weird:**
- Line 104-105 (`_enumerate_candidates`): `if bc % p != 0 and bc < p: continue` — skips bc only when
  BOTH non-divisible and < p, i.e. a non-p-divisible bc ≥ p passes. Comment says "band-sharding
  requires bc ≥ p in most kernels"; deliberate but reads like a divisibility check that isn't.
- `choose_chunks_aot` total cost ignores bc entirely (`total = num_r · per_call`), so bc selection
  rests wholly on the fit's bc primitives; comment at 153-157 admits ties are common when β_bc = 0
  in lean DoE sweeps.
- Heuristic dead branch at 522-524: `if band_chunk <= 0` after `_largest_even_divisor_leq` —
  unreachable (that helper clamps `cap = max(p, int(cap))` and its fallback `(cap//p)·p ≥ p`).
- `base_centroid` divides by `p_x` only, `base_psiG`/`base_Lq` by `P = p_x·p_y` (lines 533-535) —
  intentional (centroids are x-sharded only) but a magnet for misreading.
- Magic slack numbers: `0.8` allgather headroom appears twice (354, 555), `wfn_workspace_frac=0.20`,
  heuristic `note` string encodes the whole budget-split rationale.
- Heuristic returns `per_call_flops=0.0, total_flops=0.0` (N/A markers) inside the same
  `ChunkChoice` type the FLOPs-driven choosers use.

## 4. `presets.py` (423 loc)

Per-kernel DoE point factories, `points_<kernel>(preset) -> [(SysDims, Knobs, MeshSpec), ...]`.
Dispatched **dynamically** from `sweep.main` via `getattr(presets, f"points_{args.kernel}")` —
grep for the individual names shows no static callers; they are reachable only through the sweep CLI.

| factory | lines | kernel | presets | knobs swept |
|---|---|---|---|---|
| `points_chi0_tau_step` | 18-97 | chi0 τ-step | `si444_60Ry`, `si444_60Ry_lean` (hand-picked identifiability points; n_s axis breaks Gbuf/chi collinearity), `mos2_3x3`, `si222_tiny` | none |
| `points_pair_density_traced` | 104-115 | pair density | (ignores `preset`) | none |
| `points_cct_lr` | 122-132 | C·Cᵀ | (ignores) | none |
| `points_zct_lr` | 139-149 | Z·Cᵀ | (ignores) | `chunk_r` |
| `points_solve_q` | 156-166 | per-q solve | (ignores) | `chunk_r, q_chunk` |
| `points_vq_mu_chunk` | 173-185 | V_q μ-chunk, 1 GPU, kgrid=FFT box | (ignores) | `mu_chunk` |
| `points_sigma_kij` | 192-214 | Σ k-ij | (ignores) | none; extra n_s=1 points documented as confirming the 2.67/2.33 Gmid/Vmid split is real XLA scheduling, not collinearity |
| `points_load_psi_rchunk_fft` | 221-251 | ψ(G)→ψ(r) FFT | (ignores; kgrid **repurposed** as FFT box, k count via `k_chunk` knob) | `k_chunk, band_chunk, chunk_r` |
| `points_load_psi_rchunk_reshard` | 258-274 | ψ reshard | (ignores) | `k_chunk, band_chunk, chunk_r` |
| `points_fit_one_rchunk` | 290-406 | composite driver jit (the production chooser target) | `mos2_ortho` (orthogonal DoE, per-primitive identifiability axes documented at 296-304), `mos2_3x3`, `si444_60Ry` | `chunk_r, band_chunk` |
| `points_slab_write` | 413-423 | HDF5 slab write | (ignores) | `chunk_r` |

**Weird:** 9 of 11 factories accept `preset` and never read it (uniform signature for the
dynamic dispatch). The `mos2_ortho` comment (296-299) records that the first DoE produced
non-integer β dumps (1.65/5.02) — i.e. earlier fits were degenerate; matches the MEMORY.md
"agent audit failure modes"/planner-refit history.

## 5. `cost.py` (191 loc)

FLOPs-target twin of the memory fit; per-call FLOPs → total cost = `num_r_chunks · per_call`.

| function/class | lines | role |
|---|---|---|
| `CostFit` | 48-64 | parallels `core.Fit` (no gamma), separate artifact file |
| `_kernel_flops_primitives` | 67-73 | requires kernel `FLOPS_PRIMITIVES` (only `fit_one_rchunk` declares them, kernels/fit_one_rchunk.py:363) |
| `fit_cost_nnls(kernel, samples, *, include_intercept=True)` | 76-130 | NNLS on `meas['flops']`; samples carry full meas dict (skips rows with flops=None) |
| `predict_flops_per_call(fit, kernel, sys, knobs, mesh)` | 133-144 | evaluate fit; per one jit call |
| `save_cost_fit` / `load_cost_fit` | 151-163 | `<kernel>__<tag>__cost_fit.json` |
| `fit_cost_from_saved(kernel_name, tag="current", *, save=True)` | 166-191 | reload samples.json → refit → write cost_fit.json |

Callers: `load_cost_fit, predict_flops_per_call <- chooser.choose_chunks_aot, chooser.choose_chunks_analytic`.
`fit_cost_nnls <- fit_cost_from_saved` only. `fit_cost_from_saved <- nobody in-tree` (offline
calibration helper; presumably run interactively).

**Redundancy:** lines 176-187 re-implement `core.samples_to_fit_input`'s JSON→dataclass
reconstruction verbatim (kgrid/fft_grid tuple-ification, `n_k` pop) just to keep the meas dict —
a "fetch_X_dyn next to fetch_X"-style copy. A `keep_meas=True` flag on the core helper would
delete it. `fit_cost_nnls` is likewise a structural near-clone of `core.fit_nnls`.

## 6. `sweep.py` (153 loc)

Offline sweep CLI: `lxrun python3 -m gw.aot_memory_model.sweep --kernel X --preset Y --tag Z
[--mode sweep|fit|both] [--dry-run]`.

| function | lines | role |
|---|---|---|
| `_make_mesh(mesh_spec)` | 29-38 | real `jax.sharding.Mesh` from first p_x·p_y devices |
| `_is_rank0` | 41-47 | `jax.process_index() == 0`, exception → True |
| `run_sweep(kernel_name, points, *, tag, verbose)` | 49-80 | all ranks `aot_measure` (deterministic compiler estimate); rank 0 collects + `save_samples`; per-point failures print and continue |
| `fit_from_saved(kernel_name, tag, *, save=True)` | 83-99 | rank-0: load samples → `fit_nnls` → `save_fit`; prints β table |
| `main(argv)` | 102-131 | argparse; dynamic `getattr(presets, f"points_{kernel}")` dispatch |
| `_init_jax_for_sweep` | 134-147 | x64 + optional `jax.distributed.initialize` (SLURM_NTASKS>1), bare `except: pass` |

Callers: CLI-only (`__main__` guard 150-153); no in-tree importer. Referenced in
`reports/aot_memory_model_poc_2026-04-20/report.md`.

**Weird:** `import sys` (line 14) is unused — bottom uses `sys_exit = main()` presumably to dodge
the shadow; `from .doe import build_doe_axes` (line 25) is also unused (presets does the DoE
building); `_init_jax_for_sweep` swallows every distributed-init exception silently.

## 7. `doe.py` (80 loc)

| function | lines | role |
|---|---|---|
| `build_doe_axes(baseline_sys, baseline_knobs, baseline_mesh, *, sys_axes, knob_axes, mesh_axes)` | 20-63 | one-at-a-time sweep: `1 + Σ N·(K−1)` points instead of `K^N`; skips values equal to baseline |
| `build_product_check(..., field_a, field_b, factor=2)` | 66-80 | two points `(2a, b/2)` and `(a/2, 2b)` — equal predicted peak ⇒ `a·b` enters primitives only as a product |

Callers: `build_doe_axes <- presets.points_chi0_tau_step, presets.points_fit_one_rchunk`
(and dead import in sweep.py, re-export in `__init__`). `build_product_check <- nobody`
(grep across src/tests/tools/scripts: definition + its own docstring mention only).

## 8. `predict_cli.py` (73 loc)

| function | lines | role |
|---|---|---|
| `_parse_tuple` | 17-21 | "4,4,4" → tuple |
| `main(argv)` | 24-69 | build SysDims/MeshSpec/Knobs from flags (`--knob name=int` repeatable), `load_fit` + `predict_peak`, print per-primitive β×T breakdown and gamma |

Callers: CLI-only (`python -m gw.aot_memory_model.predict_cli`); no importer, no run-script
reference found (grepped runs/ and reports/ for `predict_cli`: only its own docstring pattern).

**Weird:** `--nspinor` flag feeds the dead `SysDims.nspinor` field (no primitive reads it) — a
no-op knob. CLI can't set `fft_grid` or `n_b_sum`, so it cannot reproduce the exact SysDims
gw_init builds for production predictions (it predates those fields).

---

## Group-level suspect summary

### Dead suspects
- `chooser.choose_chunks_aot` — grepped `choose_chunks_aot` across src/, tests/, tools/, scripts/:
  only its definition, the `__init__` export, and a stale *comment* at gw_config.py:221. The
  analytic chooser superseded it.
- `doe.build_product_check` — zero call sites anywhere (same grep scope).
- `cost.fit_cost_from_saved` — exported but never called in-tree; offline-calibration one-off.
- `core.SysDims.nspinor` + `predict_cli --nspinor` — no primitive consumes it.
- `sweep.py` unused imports: `sys` (line 14 usage is only the `sys_exit` rename dance),
  `build_doe_axes` (line 25).
- Whole-module near-dead risk: with `use_aot_chunk_chooser=False` by default AND
  `plan_gflat_chunks` overwriting `chunk_r`/`band_chunk` unconditionally (gw_init.py:636-637),
  the module's only live production output is the γ log line — everything else (grid chooser,
  analytic chooser, cost model, 10 of 11 fitted kernels) is offline-calibration tooling.
- No tests import `gw.aot_memory_model` (test_aot_memory.py tests `runtime.aot_memory`).

### Redundancy suspects
- Three choosers in one file (grid / analytic / heuristic) with duplicated tiebreak + candidate
  logic; only heuristic (default) and analytic are reachable from gw_init.
- Three sibling memory planners across the repo: `gw/aot_memory_model`, `gw/gflat_memory_model`
  (wins in production), `runtime/aot_memory` (cuFFT-aware, used by v_q_tile). Names collide
  confusingly (`tests/test_aot_memory.py` is about the runtime one).
- `cost.fit_cost_from_saved` re-implements `core.samples_to_fit_input`; `fit_cost_nnls`
  near-clones `core.fit_nnls`.
- `chooser._ceil_div` re-defined locally inside `choose_chunks_heuristic` (`_ceil_div_local`,
  lines 575-576) three hundred lines below the module-level `_ceil_div`.

### Weird code (consolidated)
- core.py:212-215 vs docstring 191-198: `disable_remat` doc promises `xla_disable_hlo_passes:
  rematerialization`, code sends `xla_gpu_memory_limit_slop_factor: 10000`.
- `Fit.gamma` has no writer — hand-edited JSON is the calibration mechanism.
- chooser.py:104-105 `if bc % p != 0 and bc < p` — half-divisibility gate.
- chooser.py:522-524 unreachable `band_chunk <= 0` fallback.
- Heuristic magic constants: 0.20 wfn frac, β=3 FFT workspace, β=4 pair temps, 0.8 allgather
  slack (×2), 16-byte complex128 literals throughout; `base_centroid/p_x` vs others `/P`.
- presets: `preset` arg ignored by 9/11 factories; `kgrid` repurposed as FFT box in two kernels.
- gw_init.py:611-619 (caller): gflat planner hard-codes `target_utilization=0.80` while cohsex.in
  `chunk_target_utilization` (default 0.97) feeds only this module's legacy path — two
  utilization knobs, one dead-ended.
