# src/gw/ppm_pipeline.py — deep-read notes (2026-07-01)

409 LOC. Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D

## Purpose

Driver-level orchestration of the GN-PPM / HL-PPM dynamic Σ^c(ω) path. Sequences
χ₀(probe ω) → W(probe ω) → 2-point plasmon-pole fit (B_q, Ω_q) → precompile+run
Σ^c(ω,k,m,n) → analytic q→0 head injection → diag(Σ_c) interpolation at DFT
energies → sigma_mnk.h5 write. Contains no math kernels itself — kernels live in
`gw.ppm_sigma` (per-τ Σ_c kernel, PPM fit) and `gw.head_correction` (head fits,
analytic head Σ). Extracted from ~200 lines that were previously inlined in
`gw_jax.main` (per module docstring).

Category: **physics orchestration: PPM Σ_c stage driver (glue, not kernels)**.

## Entry points

| Symbol | Callers (grep over src/, tests/, tools/, scripts/) |
|---|---|
| `compute_ppm_sigma_pipeline` | `gw_jax.main` (src/gw/gw_jax.py:43 import, :425 call — one-shot dynamic modes, skipped when `config.self_consistent`); `sigma_dispatch.compute_sigma_xc` (src/gw/sigma_dispatch.py:155 import, :210 call — QSGW SC per-iteration path, with `write_sigma_omega_h5=False`) |
| `PPMOutputs` (dataclass) | consumed field-by-field at the "post-PPM seam" in gw_jax.py (~lines 440–458) and in sigma_dispatch.py |
| `_write_sigma_omega_h5` (private!) | `sc_iteration.dump_sigma_omega_h5_final` (src/gw/sc_iteration.py:664) — cross-module import of a private helper, with a `SimpleNamespace(sigma_kij_h5_path=None)` stub for the `sigma_omega` duck-typed arg |
| `_fit_head_correction`, `_inject_analytic_head`, `_eval_sigma_c_at_dft_energies` | internal only (grep found no external callers) |

No `python -m` usage of this module anywhere (grep `python -m .*ppm_pipeline` → none).

## Function-by-function

### `PPMOutputs` (dataclass, L43–57)
Frozen result bundle returned to the GW driver. Fields:
- `sigma_c_omega: jax.Array | None` — (n_omega, nk, nb, nb), Ry, device-resident sharded ω-tensor; `None` if Σ_c was streamed to disk instead of held in memory. Per gw_jax.py:461–467 comment this is "the only sharded object surviving past this point", collapsed to replicated by `build_qsgw_sigma_xc`.
- `sigma_c_at_dft_ev`, `sigma_xc_at_dft_ev`, `omega_dft_rel_ev`: np.ndarray (nk, nb), host, replicated across ranks.
- `efermi_dft_ev: float | None`.
- `sigma_omega_h5_path: str`.
- `ppm_options: PPMSigmaRuntimeOptions` (from `gw_driver_helpers.build_ppm_sigma_runtime_options`; carries `omega_grid_ry/ev`, `sigma_omega_batch_size`).
- `head_sigma_diag_w_kn_ry: np.ndarray | None` — (nω, nk, nb) band-diagonal of head-only Σ, kept separately for diagnostics (head is band-diagonal so decomposition is lossless).

### `_fit_head_correction(head_resolver, *, config, meta, probe_omega, print_fn)` (L60–106)
Fits the GN/HL-PPM scalar q→0 head from user-selected source. Three branches:
1. `config.ppm.head_omega_h_ry is not None` → `fit_head_with_fixed_omega_from_sample(head_static, omega_h_ry=...)` — user-supplied Ω_h (e.g. BGW analytic value); static W^c(0) head still LORRAX's.
2. HL mode (`config.compute_mode is ComputeMode.HL_PPM`) → analytic head pole:
   **Ω_h² = ω_p² / (1 − ε_head⁻¹)** with **ω_p² = 16π · N_e / V_cell** in Ry²
   (comment: Hartree-AU → Ry² carries factor 4 → 16π). Calls `fit_head_hl_analytic_from_sample`.
3. GN default → two-sample fit `fit_head_ppm_from_samples(head_static, head_probe, probe_omega=probe_omega)` where `head_probe = head_resolver.at(probe_omega)`.
Prints `format_head_diagnostics(head_gn, cell_volume=meta.cell_volume)`. Returns `head_gn`.
Uses local (deferred) imports from `.head_correction`.

### `_inject_analytic_head(sigma_c_omega, head_gn, *, ppm_options, band_slices, wfn, sym, meta, print_fn)` (L109–163)
Adds the analytic q→0, G=G'=0 head to Σ^c(ω). Returns `(None, None)` immediately if `sigma_c_omega is None` (streamed mode → **head is NOT injected into the streamed h5**, only the in-memory tensor gets it; see weird-code below).
- Gets DFT energies `enk_full` via `common.load_wfns.get_enk_bandrange(wfn, sym, band_slices.sigma_range, band_slices.sigma_range, nspinor=meta.nspinor)`.
- `efermi_ry = wfn.efermi` (canonical mid-gap E_F computed once at WFN load, band-window-independent — single source of truth per comment).
- `n_occ = min(meta.nelec, enk.shape[1])`.
- Head tensor: `head_sigma_kij_ry = compute_ppm_head_sigma_kij(head_gn, omega_grid_ry, enk_ry, efermi_ry, n_occ, cell_volume, nk_tot)` — host numpy.
- Diagnostic on-shell occupied-band value printed as
  **`on_shell_occ = -R_h / (Ω_h · V_cell · N_k) · RYD_TO_EV`** (i.e. Σ^head on-shell ≈ −R_h/(Ω_h V N_k)).
- Returns `sigma_c_omega + jnp.asarray(head_sigma_kij_ry, complex128)` (host→device add of the full (nω,nk,nb,nb) head — replicated add onto the sharded tensor) and `np.diagonal(head_sigma_kij_ry, axis1=2, axis2=3)`.

### `_eval_sigma_c_at_dft_energies(sigma_c_omega, sigma_omega, *, ppm_options, sig_x, band_slices, wfn, sym, meta, mesh_xy, print_fn)` (L166–233)
Interpolates diag(Σ_c)(ω) at each DFT energy, replicated on all ranks.
- E_F/VBM/CBM from `wfn.efermi/vbm/cbm` (single source of truth, don't recompute).
- `omega_dft_rel_ev = enk_dft_ev − efermi_dft_ev`.
- In-memory branch: `qsgw_utils.extract_sigma_diag_replicated(sigma_c_omega, mesh_xy)` — cheap allgather of the diagonal only (~MB) so result is rank-consistent.
- Streamed branch (`sigma_c_omega is None`): rank 0 opens `sigma_omega.sigma_kij_h5_path`, reads dataset `"sigma_c_kij_ry"` (complex128, (nω,nk,nb,nb)), takes np.diagonal(axis1=2,axis2=3); other ranks allocate zeros; then `jax.experimental.multihost_utils.broadcast_one_to_all`. **Note: streamed branch head never injected (see above), so streamed-mode diag(Σ_c) at DFT energies lacks the head unless it was baked in upstream.**
- `sigma_c_at_dft_ev = qsgw_utils.interp_along_omega(sig_c_diag, omega_ev, omega_dft_rel_ev)` (linear interp along ω of the (nω,nk,nb) diag at (nk,nb) targets).
- `sigma_xc_at_dft_ev = diag(sig_x)·RYD_TO_EV + sigma_c_at_dft_ev`.
Returns 4-tuple, all replicated host arrays.

### `_write_sigma_omega_h5(sigma_c_omega, sigma_omega, *, ppm_options, sig_x, sig_h, config, input_dir, meta, mesh_xy) -> str` (L236–278)
"One writer, two backends" for the canonical sigma_mnk.h5:
- Output path: `config.paths.sigma_omega_h5_file` (cohsex.in key `sigma_omega_h5_file`, default `"sigma_mnk.h5"`), joined onto `input_dir` if relative.
- In-memory backend: `file_io.write_sigma_omega_h5(out_path, ppm_options.omega_grid_ev, None, sigma_c_kij_ev=RYD_TO_EV*sigma_c_omega, sigma_sx_kij_ev=RYD_TO_EV*sig_x, hartree_kij_ev=RYD_TO_EV*sig_h, mesh=mesh_xy, backend=config.backend.slab_io)`. All ranks must enter (SlabIO handles rank-0 dispatch internally) — no rank guard.
- Streamed backend (rank 0 only, `sigma_omega.sigma_kij_h5_path` truthy): `file_io.copy_sigma_kij_h5_to_omega_h5(src, out_path, omega_grid_ev, sigma_sx_kij_ev=sig_x, hartree_kij_ev=sig_h, omega_batch_size=ppm_options.sigma_omega_batch_size)`. **Unit inconsistency suspect**: in-memory branch multiplies sig_x/sig_h by RYD_TO_EV; streamed branch passes them raw (presumably the copy helper converts internally — verify in file_io/sigma_output.py:340).
- `sigma_omega` param is duck-typed (`'object'`, noqa F821); only attribute consulted is `sigma_kij_h5_path`, and only in the streamed branch. sc_iteration stubs it with SimpleNamespace.

### `compute_ppm_sigma_pipeline(...) -> PPMOutputs` (L281–409)
The public entry point. Signature is kwargs-only bundle style: `wfns, V_q, W_static_q, W_probe_q, sig_x, sig_h, quad, e_ref, config, meta, mesh_xy, head_resolver, band_slices, wfn, sym, input_dir, write_sigma_omega_h5=True, print_fn=print`.
- Requires `config.do_screened` (raises ValueError otherwise).
- Both W's must be pre-computed by the caller (screening decoupled: in SC the caller is `gw.screening.compute_screening`; in one-shot main() same helper at the screening seam — "lets future Σ schemes (CD, spectral, …) share the same screening planner").
- `ppm_options = build_ppm_sigma_runtime_options(config, input_dir=input_dir)`.
- Probe ω convention: `probe_omega = complex(config.ppm.omega_p, 0)` for HL (real axis), `1j*config.ppm.omega_p` for GN (imaginary axis). Same convention the screening planner used to pick W_probe_q's point.
- Step 1: `ppm = fit_ppm(W_static_q, W_probe_q, V_q, probe_omega, mesh_xy, fallback_omega=config.ppm.fallback_omega, n_nodes_static=quad.node_count, print_fn, model_label)` (in gw.ppm_sigma).
- Step 2: `precompile_sigma(wfns, ppm, meta, mesh_xy)` under `timing.section("sigma.compile")`; then `sigma_omega = compute_sigma_c_ppm_omega_grid(wfns, ppm, meta, mesh_xy, ppm_options, sigma_window_quad=config.sigma_quadrature_config, print_fn)` under `timing.section("sigma.exec")` + `profile_section("sigma_ppm")`. `sigma_c_omega = sigma_omega.sigma_c_kij` (None if streamed).
- Step 3: `_fit_head_correction` + `_inject_analytic_head`.
- Step 4: `_eval_sigma_c_at_dft_energies` (docstring at L317 says "(rank-0 only)" but the helper is explicitly replicated-across-ranks — stale comment).
- Step 5: `_write_sigma_omega_h5` if `write_sigma_omega_h5=True`; else re-derive the absolute path inline (L394–398, duplicates path logic from L254–256) without writing (SC intermediate iterations; SC driver writes once at convergence via `sc_iteration.dump_sigma_omega_h5_final`).
- Unused-looking args: `e_ref` is accepted but never referenced in the function body (grep `e_ref` inside file: only the parameter and the docstring); `quad` is used only for `quad.node_count`. Also `from . import w_isdf` at L320 is imported "for ensure_compilation_cache + cache hit timings" but the name `w_isdf` is never used afterwards in the function — import-for-side-effect or leftover.

## Flags / config keys consumed

- `config.do_screened` (cohsex.in `do_screened`)
- `config.compute_mode` (HL_PPM vs GN_PPM pivot; cohsex.in mode key)
- `config.ppm.omega_p` ← `ppm_omega_p` (default 2.0 Ry)
- `config.ppm.fallback_omega` ← `ppm_fallback_omega` (default 2.0)
- `config.ppm.head_omega_h_ry` ← `ppm_head_omega_h_ry` (default None; BGW-comparison override)
- `config.paths.sigma_omega_h5_file` ← `sigma_omega_h5_file` (default `sigma_mnk.h5`)
- `config.backend.slab_io`
- `config.sigma_quadrature_config` (property on LorraxConfig)
- indirectly whatever `build_ppm_sigma_runtime_options` reads (ω-grid, batch sizes — see gw_driver_helpers.py:17/238)

## I/O

- **Writes** `sigma_mnk.h5` (name from `sigma_omega_h5_file`), eV units, via `file_io.write_sigma_omega_h5` (datasets: ω grid, sigma_c_kij_ev, sigma_sx_kij_ev, hartree_kij_ev; SlabIO backend) or via `file_io.copy_sigma_kij_h5_to_omega_h5` (streamed re-batch from an intermediate h5).
- **Reads** (streamed mode only) the intermediate `sigma_omega.sigma_kij_h5_path` h5, dataset `sigma_c_kij_ry` (complex128, (nω,nk,nb,nb)), rank 0 + MPI broadcast.

## Dead suspects

None. All five functions have callers (grepped `compute_ppm_sigma_pipeline|PPMOutputs|_write_sigma_omega_h5|_fit_head_correction|_inject_analytic_head|_eval_sigma_c_at_dft_energies` across src/, tests/, tools/, scripts/). The private helpers are called only within this file except `_write_sigma_omega_h5` (sc_iteration.py:664).

## Redundancy suspects

1. **Two call sites, parallel one-shot vs SC paths**: `gw_jax.main` (one-shot, skipped when `self_consistent`) and `sigma_dispatch.compute_sigma_xc` (SC) both call `compute_ppm_sigma_pipeline` with nearly identical kwarg blocks; the seam is deliberate but the duplicated ~15-kwarg call plumbing is refactor-map-relevant.
2. **Path resolution duplicated**: L394–398 (the `write_sigma_omega_h5=False` else branch) re-implements the abs-path join from `_write_sigma_omega_h5` L254–256.
3. **Two write backends in `_write_sigma_omega_h5`** (in-memory vs streamed copy) — documented as intentional "one writer, two backends", but the streamed branch passes sig_x/sig_h without the RYD_TO_EV factor the in-memory branch applies (conversion presumably inside copy helper; a unit-convention split across two code paths).
4. **Three E_F/enk fetches**: `_inject_analytic_head` and `_eval_sigma_c_at_dft_energies` each independently call `get_enk_bandrange(wfn, sym, band_slices.sigma_range, band_slices.sigma_range, nspinor=...)` and re-read `wfn.efermi` — same data fetched twice per pipeline run.

## Weird code

1. **L92**: magic `16π·N_e/V_cell` for ω_p² in Ry² (HL analytic head). Documented in comment (Hartree→Ry factor 4 → 16π) but a naked constant; hypothesis: correct BGW-style plasma frequency, keep but centralize.
2. **L126–127 + streamed h5 path**: in streamed mode (`sigma_c_omega is None`) the analytic q→0 head is silently NOT injected (`_inject_analytic_head` returns `(None, None)`), yet `_eval_sigma_c_at_dft_energies` and the streamed h5 copy proceed. Hypothesis: streamed Σ_c on disk and its DFT-energy diag lack the head correction — either the head is added elsewhere for streamed runs or this is a real gap for large streamed runs.
3. **L168/L238**: `sigma_omega: 'object'` with `# noqa: F821 (forward decl)` — duck-typed param whose real type lives in ppm_sigma; sc_iteration stubs it with `SimpleNamespace(sigma_kij_h5_path=None)`. Fragile implicit contract across three modules.
4. **L263–265 vs L274–275**: RYD_TO_EV applied to sig_x/sig_h in the in-memory branch but not the streamed branch (see redundancy #3).
5. **L317 docstring** says step 4 is "(rank-0 only)" but `_eval_sigma_c_at_dft_energies` docstring and code say replicated on all ranks ("required by the post-Σ flow in gw_jax which now runs on all ranks"). Stale comment.
6. **L320**: `from . import w_isdf  # for ensure_compilation_cache + cache hit timings` — imported name never used in the function; import-for-side-effect or dead import.
7. **`e_ref` parameter**: accepted by `compute_ppm_sigma_pipeline` and passed by both callers, never used in the body. Dead parameter (kept for signature symmetry with `compute_screening`?).
8. **sc_iteration.py:664** imports the private `_write_sigma_omega_h5` — privacy leak; the function is de facto public API.
9. **gw_jax.py:400–405 history note** (context, not in this file): the analytic head injected at the end of this pipeline was removed in commit 1542342 (Apr-10) and re-added 2026-04-25; ±W^c(0)/(2·V_cell·N_k) on-shell, ~1.24 eV/band on Si 4×4×4 60b — sign/convention churn history around `_inject_analytic_head`.

## Cross-module deps

`gw.gw_config` (ComputeMode, LorraxConfig), `gw.gw_driver_helpers` (PPMSigmaRuntimeOptions, build_ppm_sigma_runtime_options, profile_section), `gw.head_correction` (HeadResolver, fit_head_*, compute_ppm_head_sigma_kij, format_head_diagnostics), `gw.ppm_sigma` (fit_ppm, precompile_sigma, compute_sigma_c_ppm_omega_grid), `gw.qsgw_utils` (extract_sigma_diag_replicated, interp_along_omega), `gw.gw_output` (print_section), `gw.w_isdf` (side-effect import), `common.units` (RYD_TO_EV), `common.load_wfns` (get_enk_bandrange), `common.timing`, `file_io` (write_sigma_omega_h5, copy_sigma_kij_h5_to_omega_h5), `jax.experimental.multihost_utils` (broadcast_one_to_all), `h5py`.
