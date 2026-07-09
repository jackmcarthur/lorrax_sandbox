# src/gw/ppm_sigma.py — deep-read notes (2026-07-01)

LOC: 1702. One file in this group.

## Purpose

GN-PPM / HL-PPM plasmon-pole construction from precomputed W(0) and W(probe), plus the
frequency-integrated correlation self-energy Σ^c_nm(k, ω) evaluated on a real-ω grid via
minimax τ quadrature. Per the module docstring:

    Σ^c_nm(k, ω) = Σ_branches Σ_windows Σ_τ  α(τ) · e^{i·ω_sign·ω·τ} · project[σ^τ_nmk(τ)] · pref · scale
    σ^τ_nmk(τ)   = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)         = diag[e^{-i(E_A − E_ref_A)·τ}] · mask_A          (A = val or cond)
    W(τ)         = Σ_μν B_q · e^{-i(Ω_q − E_ref_B)·τ} · mask_B     (PPM pole sum)

Four branches (ω-sign × cond/val) cover ω ∈ ℝ using Σ_c(−ω) = −[Σ_c(ω)]^* (scale = −1
for the −ω half; kernel_sign flips between cond/val and between halves). Within a +ω
cond branch with nontrivial ω range, a 3-window decomposition is used: Laplace "core",
crossing "a_stripe" (HGL quadrature, keeps only Im[coeff·σ]), tail "b_slab".

**Category guess:** physics: PPM fit + Σ_c(ω) frequency-integration stage (τ-loop GPU
kernels + host/GPU/H5 accumulators).

## Entry points (grep across src/, tests/, tools/, scripts/)

| Symbol | Callers |
|---|---|
| `fit_ppm` | `gw/ppm_pipeline.py:341` (`compute_ppm_sigma_pipeline`, both GN and HL probe) |
| `precompile_sigma` | `gw/ppm_pipeline.py:351` (inside `timing.section("sigma.compile")`) |
| `compute_sigma_c_ppm_omega_grid` | `gw/ppm_pipeline.py:353`; docstring mention in `gw/gw_driver_helpers.py:71` |
| `fit_gn_ppm` | **no callers** (only its own def + file-header mention) — pipeline calls `fit_ppm` directly with `1j*omega_p` |
| `_make_project_ri_reduce_scatter` | `gw/aot_memory_model/kernels/sigma_kij.py:115` (memory-model mirror); pattern referenced by `bse/bse_simple.py:27` (comment only) |
| `minimax_tau_integrate_sigma` | internal only (`_integrate_tau_windows_for_branch`); named as sibling in `w_isdf.py:150` docstring |
| `PPMBuildResult` / `SigmaOmegaResult` | consumed by `ppm_pipeline` (fields `.B_q`, `.Omega_q`, `.valid_mask_q`, `.sigma_c_kij`, `.sigma_kij_h5_path`, `.omega_ry/.omega_ev`) |

No test file under tests/ (active/regression/archive included) references ppm_sigma or
any of its internals (grepped `ppm_sigma|fit_ppm|_project_tau_onto_omega|_build_three_sigma_windows|_build_single_sigma_window|minimax_tau_integrate_sigma`). The module is exercised only through
end-to-end GW runs.

## Function-by-function

### Enums / mode selection
- **`_AccumMode`** (L80–94): enum `KIJ_HOST` / `KIJ_STREAM`.
- **`_select_accum_mode`** (L97–130): maps `omega_accumulation` cohsex key
  (`auto|kij|kij_stream`) to a mode. `auto` → KIJ_HOST if no h5 path AND kij buffer
  ≤ 0.5 GiB; streaming requires a path AND single-process (multi-process h5
  read-modify-write "storm" fallback → KIJ_HOST). Note `kij_bytes` passed by the driver
  is the *global* n_ω·n_k·n_b² ·16 B, not per-rank.

### Data classes
- **`PPMBuildResult`** (L133–143): omega_p (probe magnitude, diagnostics only), W0_q,
  Wiwp_q, B_q, Omega_q, valid_mask_q — all flat-q (nq, μ, μ), sharded P(None,'x','y');
  unfulfilled_fraction; n_nodes_static.
- **`SigmaOmegaResult`** (L145–150): omega_ry/omega_ev (np), sigma_c_kij
  (n_ω, nk, nb, nb) c128 jax array or None if streamed, sigma_kij_h5_path.
- **`_SigmaWindow`** (L153–180): name, MinimaxNodes (t, alpha c128), mask_A (nk,nb),
  mask_B (unused in practice — mask_B_mode drives lazy materialization), E_ref_A/B,
  omega_sign, project ("full"→code 0 / "imag"→code 1), prefactor, mask_B_mode
  ("explicit"/"all"/"le_t"/"gt_t"), mask_B_threshold, crossing_kind.
- **`_SigmaBranch`** (L189–197): tag, E_A (nk,nb), base_mask_A, kernel_sign (±1),
  scale (±1), omega_abs, omega_idx.

### Branch enumeration
- **`_iter_branches`** (L200–237): builds ≤4 `_SigmaBranch`:
  (+ω,cond,ks=+1,scale=+1), (+ω,val,ks=−1,scale=+1), (−ω,cond,ks=−1,scale=−1),
  (−ω,val,ks=+1,scale=−1). Physics: +ω half is a Laplace transform on
  E_A = E_c − E_F; −ω half evaluated at |ω| via Σ_c(−ω) = −[Σ_c(ω)]^*.

### Host/device utilities
- **`_to_host_np`** (L240–248): process_allgather with bare `except Exception` fallback
  to device_get.
- **`_to_host_scalar`** (L251–254), **`_masked_stats_device`** (L257–265): masked
  count/min/max of Ω_q for window interval selection.
- **`_materialize_window_mask_B`** (L268–287): lazily builds B-side selector on device
  from mask_B_mode: "all" → base; "le_t"/"gt_t" → base & (Ω_q ≤/> T); "explicit" path
  exists but no current window builder uses it (all builders set mask_B=None + a mode).

### Physics-state prep
- **`_prepare_sigma_state`** (L313–356, @jax.jit): one fused trace deriving
  E_F (VBM or midgap = (VBM+CBM)/2 selected by traced bool `use_midgap`),
  E_cond = max(e_nk − E_F, 0), H_val = max(E_F − e_nk, 0), cond/val masks from
  occ > 0.5, Ω_abs = max(Re Ω_q, 0), B_corr = complex128(B_q),
  B_mask = (Ω_abs > 1e−14) & valid_mask_q, invalid-mode tallies. Sentinels ±1e30 for
  masked max/min.

### ω-projection kernels
- **`_combine_coeff_with_sigma_tau`** (L359–391): lax.switch on project_code —
  code 0 "full": (coeff_re + i·coeff_im)·(σ_re + i·σ_im); code 1 "imag":
  coeff_re·σ_im + coeff_im·σ_re (= Im[coeff·σ]). Docstring notes a historical "real"
  code path removed; 2-way switch kept so HLO matches the old lowering.
- **`_project_tau_onto_omega`** (L394–422, @jax.jit): contrib[ω,k,i,j] =
  pref · α_eff · exp(i·sign·ω·t) · P(σ_re, σ_im). σ^τ carried as (re, im) pair
  because the crossing window needs only Im[coeff·σ]; complex σ through the FFT stack
  would double HBM.
- **`_project_tau_onto_omega_np`** (L1008–1030): exact numpy mirror of the above for
  `_HostOmegaAccumulator` (host-side accumulate). Deliberate jax/np duplicate pair.

### Sharded projection / kernel factories
- **`_make_project_ri_reduce_scatter`** (L425–496): shard_map'd
  Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(k,s,μ)·σ(k,s,μ,s',μ')·ψ_n(k,s',μ').
  in_specs: ψ_xr P(None,None,None,'x') (nk,m,s,μ_X); σ P(None,None,'x',None,'y')
  (nk,s,μ_X,s',μ_Y); ψ_yn P(None,None,'y',None) (nk,s',μ_Y,n). out: two
  P(None,'x','y') (nk,m_X,n_Y) arrays (re, im done as separate channels).
  Einsums VERBATIM: `'kmsx,ksxty->kmty'` (with jnp.conj(psi_xr_local)) then
  psum_scatter('x', scatter_dimension=1, tiled=True); `'kmty,ktyn->kmn'` then
  psum_scatter('y', scatter_dimension=2, tiled=True). check_rep=False.
  Docstring TODO: requires m % p_x == 0 and n % p_y == 0; padding unhandled.
  Documented as drop-in replacement for `wavefunction_bundle.project_ri`
  (wavefunction_bundle.py:389 cross-references it).
- **`_get_sigma_kij_kernel`** (L499–558): cached on (id(mesh_xy), kgrid). Builds
  flat-k FFT helpers from `common.fft_helpers.make_flat_k_fftn/ifftn` with
  `G_FFT7D_SPEC` / `V_FFT5D_SPEC` from wavefunction_bundle, norm='ortho'.
  Inner jit `_sigma_kij_kernel` (donate_argnums=(8,) → W_q donated):
  G_k = build_G_tau(ψ_coh_xn, ψ_coh_yr, E_A, 1j·t_node, e_ref, mask);
  G_R = iFFT(G_k); V_R = iFFT(W_q)[:,None,:,None,:] (broadcast (nk,1,μ,1,μ));
  σ_k = FFT(G_R·V_R·inv_sqrt_nk) with **inv_sqrt_nk = −1.0/√Nk** (the Σ_c minus sign
  lives here); tail = reduce-scatter project → (re, im) each (nk, m_X, n_Y).
- **`_get_sigma_tau_kernel`** (L561–604): cached on (id(mesh_xy), kgrid). Inner jits:
  `_build_W_t_q`: W(τ) = where(mask_B, B_q·exp(−i(Ω_q − E_ref_B)·τ), 0), sharding
  pinned P(None,'x','y'); `_tau_kernel` chains it into `_sigma_kij_kernel`.
- **`precompile_sigma`** (L607–656): AOT lower+compile the τ kernel at real
  shapes/shardings. Careful committed-ness matching notes: E_A dummy must be
  device_put with NamedSharding(P(None,None)) to match `_prepare_sigma_state` jit
  output, masks/scalars left uncommitted. One compile covers all 4 branches
  (shape-invariant).

### PPM fit
- **`fit_ppm`** (L663–720): model-agnostic 2-point pole fit. Wc = W − V at ω=0 and
  ω=probe; delegates algebra to `minimax_screening.fit_gn_ppm_from_wc_pair(Wc0, Wci,
  z, fallback_omega)`; GN when probe = i·ω_p, HL when probe real Ω. Pins B/Ω/valid to
  P(None,'x','y'). Logs unfulfilled fraction. `omega_p` field carries |probe| for
  provenance only.
- **`fit_gn_ppm`** (L723–744): thin GN wrapper (`probe = 1j·omega_p`). **Zero callers.**

### Minimax window construction (host side)
- **`_build_single_sigma_window`** (L752–796): one Laplace window on
  x ∈ [max(S_min,1e−12), S_max(+ω_max if kernel_sign=−1)], S = E_A extremum + Ω_B
  extremum; `solve_laplace_minimax_interval`; nodes `to_minimax_nodes(time_axis='imag')`;
  prefactor = +1 (ks=+1) / −1 (ks=−1); mask_B_mode="all"; project="full".
- **`_build_three_sigma_windows`** (L799–897): T = ω_max + edge_factor·ξ where
  ξ = max(regularization_width_ry, 1e−12).
  - "core": mask_A = base & (E_A ≤ T), B-side "le_t"; crossing quadrature
    `solve_phase_minimax_bandwidth(A_core = max(2T/ξ, 1e−8), target_kind="hgl")`,
    nodes scaled t = τ/ξ, α = α/ξ; project="imag", prefactor = −1.
  - "a_stripe": mask_A = base & (E_A > T), B "le_t"; Laplace on
    x ∈ [max(S_min − (T − z_edge), z_edge, 1e−12), S_max]; project="full", pref=+1.
  - "b_slab": mask_A = base, B "gt_t"; same Laplace formula; project="full", pref=+1.
- **`_build_windows_for_branch`** (L1174–1255): host orchestration — allgathers
  E_A/base_mask_A, masked Ω stats on device, picks 3-window (kernel_sign=+1 &
  ω_max > 1e−14) vs single window, prints per-window summary. Contains 5
  `[DBG-PPM-WIN]` raw `print(..., flush=True)` debug lines (L1202, 1204, 1206, 1210).

### Accumulators
- **`_SigmaAccumulator`** (L908–931): protocol — begin_window(window, scale) /
  add_tau(σ_re, σ_im, t_c, α_eff_c) / end_window / finalize.
- **`_ReduceScatterGpuAccumulator`** (L934–1005): on-GPU Σ sharded P(None,None,'x','y');
  per-window buffer; add via `_project_tau_onto_omega` + sharding-pinned +=.
  **Never instantiated anywhere** (grep: only its def + comments at L45, 508, 1122,
  1338, 1686). Docstring records deleted τ-scan factory: lax.scan batching regressed
  MoS2 3×3 sigma_ppm by ~80%; and future _CollectiveFlushSlabIoAccumulator.
- **`_HostOmegaAccumulator`** (L1033–1111): the accumulator actually used in the
  non-streaming path. Per-rank numpy tile = local shard of (n_ω,nk,m_X,n_Y) via
  `sharding.shard_shape`; async D2H pipeline (`addressable_data(0)` +
  `copy_to_host_async`, deque with lag=2) overlapping GPU τ_{k+lag} with numpy
  accumulate of τ_k; finalize via `make_array_from_process_local_data`.
- **`_StreamedH5Accumulator`** (L1114–1164): projects each τ in ω-batches on GPU and
  hands (global ω idx, contrib) to a writer callable (rank-0 h5py read-modify-write).
  Long docstring about the future collective-flush SlabIO variant (cross-referenced
  from `file_io/slab_io.py:299` and `file_io/_slab_io_allgather.py:282`).

### τ loop / branch orchestration
- **`minimax_tau_integrate_sigma`** (L1258–1313): one window's τ sweep; Python loop
  (deliberately NOT lax.scan — per-τ body emits NCCL; scan regressed 80%).
  α_eff = α·exp(−i·E_ref_sum·t) absorbs the reference-energy shift so Laplace kernel
  sees nonneg (E_A, Ω) arguments. Sibling of `w_isdf.minimax_tau_integrate_chi`.
- **`_integrate_tau_windows_for_branch`** (L1316–1378): walks windows, materializes
  mask_B per window, binds closures, drives accumulator lifecycle; jax_profile
  annotations `sigma_branch[...]` / `sigma_window`; LoopProgress ticker.
- **`_run_sigma_branch`** (L1381–1481): per-branch orchestrator: build windows (host)
  → pick `_HostOmegaAccumulator` (stream_writer None) or `_StreamedH5Accumulator` →
  integrate → finalize. Returns (Σ_kij or zeros, windows). Two `[DBG-PPM]` prints
  (L1442, 1448).

### Top-level driver
- **`compute_sigma_c_ppm_omega_grid`** (L1488–1702): reads ppm_options via getattr
  with defaults (`sigma_regularization_ry` default = **0.018374661087827496 Ry**
  = 0.25 eV; `sigma_edge_factor` 1.5; `sigma_omega_batch_size` 4;
  `sigma_omega_accumulation` 'auto'; `sigma_kij_h5_path` None; `fermi_reference`
  'midgap'). Quadrature from `SigmaQuadratureConfig` (target_error 1e−6, max_nodes 64,
  crossing_max_nodes 500, crossing_eps_q 1e−3, use_shipped_tables) or hard-coded
  fallbacks. Splits ω grid into ± halves, `_prepare_sigma_state`, selects accum mode,
  optionally creates the stream h5 file (rank 0 only, chunks (min(4,nω_batch), min(4,nk),
  nb, nb)), runs 4 branches, sums cond+val per ω-half **before** host gather
  ("preserves traversal order so reduction stays bit-identical"), scatters halves into
  `sigma_kij_host[idx]`, then re-uploads the full host buffer as a replicated jnp array
  in the returned `SigmaOmegaResult`. `[DBG-PPM]` prints at L1587, 1672, 1674.

## Cross-module dependencies

- `common.jax_profile`, `common.units.RYD_TO_EV`, `common.progress.LoopProgress`,
  `common.fft_helpers.make_flat_k_fftn/ifftn`
- `gw.minimax_config` (`MinimaxConfig` — unused, `SigmaQuadratureConfig`)
- `gw.minimax_screening` (`MinimaxNodes`, `fit_gn_ppm_from_wc_pair`,
  `solve_laplace_minimax_interval`, `solve_phase_minimax_bandwidth`;
  `build_static_minimax_window_pair` + `solve_laplace_minimax_imag_interval` imported
  unused)
- `gw.w_isdf._ensure_compilation_cache` (3 call sites)
- `gw.wavefunction_bundle` (`G_FFT7D_SPEC`, `V_FFT5D_SPEC`; wfns bundle accessors
  `.xn/.yr/.xr/.yn`, `.slices.full/.sigma`, `.enk`, `.occ`)
- `gw.greens_function_kernel.build_G_tau`
- downstream consumers: `gw.ppm_pipeline` (only real caller),
  `gw.aot_memory_model.kernels.sigma_kij` (imports `_make_project_ri_reduce_scatter`),
  `file_io.sigma_output.copy_sigma_kij_h5_to_omega_h5` (copies the stream h5 into
  sigma_mnk.h5, called from gw_jax side)

## I/O

- **Writes** (stream mode only, rank 0, h5py): file at `paths.sigma_kij_h5_file`
  (`sigma_kij_h5_path`), datasets `omega_ry` (f64, n_ω), `omega_ev` (f64, n_ω),
  `sigma_c_kij_ry` (c128, (n_ω, nk, nb, nb), chunked, fillvalue 0), attr
  `layout="omega,k,i,j"`. Accumulation is per-(τ × ω-batch) read-modify-write.
- Reads nothing from disk. Minimax tables are read indirectly inside
  minimax_screening (use_shipped_tables flag threaded through).

## Flags / config keys consumed

Via `ppm_options` (PpmSigmaRuntimeOptions built in `gw_driver_helpers.build_ppm_sigma_runtime_options` from cohsex.in `ppm.*` block):
`omega_grid_ry` (← ppm ω grid in eV), `sigma_regularization_ry` (← ppm.regularization_ev),
`sigma_edge_factor` (← ppm.window_edge_factor), `sigma_omega_batch_size`
(← ppm.omega_batch_size), `sigma_omega_accumulation` (← ppm.omega_accumulation),
`sigma_kij_h5_path` (← paths.sigma_kij_h5_file), `fermi_reference` (← ppm.fermi_reference).
Via `SigmaQuadratureConfig` (← ppm_sigma_target_error, ppm_sigma_max_nodes,
crossing_max_nodes, crossing_eps_q, regenerate_tables). Via `fit_ppm` caller:
ppm.omega_p, ppm.fallback_omega. Mode gate upstream: `use_ppm_sigma` / `compute_mode`.

**Options set but never read by this module (or anywhere):** `ppm_sigma_scale`,
`ppm_sigma_flip_neg` (defined gw_config.py:322–323, threaded into
PpmSigmaRuntimeOptions gw_driver_helpers.py:270–271, zero consumers elsewhere);
`ppm_sigma_debug_static_norm` (gw_config.py:332/1045, gw_driver_helpers.py:277, zero
consumers).

## Dead suspects

1. **`fit_gn_ppm`** (L723) — grepped `fit_gn_ppm\b` across src/tests/tools/scripts:
   only its def and the file-header layout comment. `ppm_pipeline` calls `fit_ppm`
   directly for both GN and HL.
2. **`_ReduceScatterGpuAccumulator`** (L934) — grepped the class name repo-wide: only
   comments/docstrings reference it; `_run_sigma_branch` always instantiates
   `_HostOmegaAccumulator` or `_StreamedH5Accumulator`. ~70 lines of dead class.
3. **Unused imports** — `MinimaxConfig` (L68), `build_static_minimax_window_pair`
   (L71), `solve_laplace_minimax_imag_interval` (L74): imported, zero uses in file.
4. **`_SigmaWindow.mask_B` field + `_materialize_window_mask_B` "explicit" branch**
   (L153/276–279) — every window builder passes mask_B=None with a non-explicit mode;
   the explicit path is unreachable from current builders.
5. **Dead flags**: `ppm_sigma_scale`, `ppm_sigma_flip_neg`, `ppm_sigma_debug_static_norm`
   (see flags section; parsed from cohsex.in but consumed nowhere).

## Redundancy suspects

1. `_project_tau_onto_omega` (jax, L394) vs `_project_tau_onto_omega_np` (numpy,
   L1008) — deliberate mirror pair, but a classic fetch_X/fetch_X_dyn-style duplicate;
   any change must be made twice.
2. `_ReduceScatterGpuAccumulator` vs `_HostOmegaAccumulator` — same accumulate logic
   on GPU vs host; the GPU one is dead (see above), the file header (L45) still
   advertises it as one of the two accumulators and omits `_HostOmegaAccumulator`.
3. `_make_project_ri_reduce_scatter` vs `wavefunction_bundle.project_ri` — documented
   parallel old/new project paths (wavefunction_bundle.py:389 cross-ref); plus a third
   mirror copy in `aot_memory_model/kernels/sigma_kij.py` for the planner.
4. `compute_sigma_c_ppm_omega_grid` re-validates `omega_accumulation` and
   `fermi_reference` strings that `build_ppm_sigma_runtime_options` /
   `_select_accum_mode` already validate (triple validation of the same key).
5. Window-interval math duplicated between `_build_single_sigma_window` and the
   stripe/slab arm of `_build_three_sigma_windows` (S_min/S_max/x_min/x_max recipe).

## Weird code

1. **L1587**: `print_fn(...) if False else print(f"  [DBG-PPM] post-print rank=...", flush=True)`
   — a dead ternary whose `if False` arm keeps the print_fn call unreachable; pure
   leftover debug scaffolding in the production driver.
2. **`[DBG-PPM]` / `[DBG-PPM-WIN]` raw prints** at L1202/1204/1206/1210 (window build),
   L1442/1448 (_run_sigma_branch), L1672/1674 (branch loop) — bypass `print_fn`, fire
   on every rank, clearly a multi-process-hang debugging session that was never
   cleaned up.
3. **Magic constant L1526**: `getattr(ppm_options, 'sigma_regularization_ry',
   0.018374661087827496)` — 0.25 eV in Ry, undocumented at the use site.
4. **Sign convention L524**: `inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))` — the global
   minus sign of Σ_c is folded into the FFT normalization constant; easy to miss.
5. **Branch sign table** (`_iter_branches`, L200): kernel_sign/scale flips encode
   Σ_c(−ω) = −[Σ_c(ω)]^* and the val-side kernel sign; correctness rests entirely on
   the docstring derivation.
6. **Deleted-scan tombstone** (L948–956): docstring records that a τ-batching lax.scan
   factory was deleted after regressing sigma_ppm ~80% at MoS2 3×3 — do-not-reintroduce
   note that a refactor must preserve.
7. **Divisibility TODO** (L459): reduce-scatter project requires m % p_x == 0 and
   n % p_y == 0; unpadded inputs would break silently-by-assert at shard_map level.
8. **`_to_host_np` bare `except Exception`** (L247) — swallows any allgather failure
   and silently falls back to device_get; in multi-process this could return a local
   shard where a global array was expected.
9. **HLO-pinning 2-way lax.switch** (L376–382): `_combine_coeff_with_sigma_tau` keeps a
   lax.switch purely so generated HLO matches an older 3-way lowering ("no
   minimax-table consumer gets a silent behavior change").
10. **Final re-upload** (L1696): the non-streaming path converts the accumulated host
    numpy Σ back to a replicated device array (`jnp.asarray`) — the full
    (n_ω,nk,nb,nb) buffer lands on every GPU at the end despite all the effort to keep
    it off-GPU during the loop.
11. **`PPMBuildResult.omega_p` semantic drift** (L702–704): historically "imaginary-axis
    magnitude", now carries |probe| for HL too; comment-documented, diagnostics only.
