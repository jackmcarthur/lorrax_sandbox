# LORRAX GW — Complete Input-Parameter Surface

Audited 2026-07-01 against `sources/lorrax_D` (branch `main`).

> **Updated 2026-07-02** (branch `agent/memplanner-cleanup`, `e7b6c7d..HEAD`).
> Reconciled against the memplanner + V_q-tile delete-pass. Net changes to this
> catalog:
> - **2 keys deleted from `_DEFAULTS`:** `use_aot_chunk_chooser`,
>   `chunk_chooser_mode` (whole `aot_memory_model/` package removed; `_apply_aot_chunk_model`
>   gone). `MemoryConfig` no longer carries them. §1.5.
> - **1 key added:** `ecutrho` (density-grid cutoff for the psp tools only —
>   `kin_ion`/`dipole`; NOT read by the GW driver). §1.1.
> - **6 env vars now DEAD** (their modules were deleted): `LORRAX_V_Q_FFT_COEF`,
>   `LORRAX_V_Q_Q_CHUNK`, `LORRAX_V_Q_MU_CHUNK`, `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH`
>   (all with `v_q_tile.py` / the `compute_vcoul.py` strip), plus debug
>   `LORRAX_V_Q_AOT_VERBOSE`, `LORRAX_V_Q_TIME_STAGES`. §2.2/§2.4.
> - **`LORRAX_FORCE_FULL_BZ` is a DEV/gate seam only** — NOT a user-facing flag
>   (user decision). It backs `test_ibz_full_bz_equivalence`. §2.1.
> - The r-space V_q tile subsystem (`v_q_tile.py`, `make_v_munu_chunked_kernel`,
>   the legacy `compute_all_V_q` r-space tail, `compute_bare_coulomb_sphere_idx`)
>   is **gone**; the live V_q path is G-flat only (`v_q_g_flat` /
>   `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`). Consumer refs updated below.
> - `compute_all_V_q` lost `n_rmu`/`n_rtot`/`budget_bytes`/`use_g_flat_zeta`;
>   `mu_chunk_size`/`q_batch_size` are now inert legacy args.
> - Line numbers below are as of the 2026-07-01 audit base; `gw_init.py` /
>   `compute_vcoul.py` shifted substantially in the delete-pass — treat file paths
>   as authoritative, line numbers as approximate for those two files.

> **Updated 2026-07-03** (branch `agent/memplanner-cleanup`, `bb04399` HEAD).
> Reconciled against the parser fix + the ISDF stage move/library split + reader
> cleanup. No `_DEFAULTS` keys or env vars added/removed by user input this pass;
> the churn is **code-location + one silent-flag bug fix**:
> - **Parser inline-`#` fix (`61ae4b8`):** `read_lorrax_input` now passes
>   `inline_comment_prefixes=('#',)` (`gw_config.py:405`). Before this, a trailing
>   comment silently voided any flag — e.g. `cusolvermp_charge = off  # note`
>   parsed to the string `'off  # note'`, failed the `== 'off'` check, and fell
>   back to `auto`. **Every string-valued key in this catalog was exposed to that
>   footgun** (bools/ints/floats less so — configparser's getters tolerate trailing
>   text differently); it is now fixed. Unit-verified `'key = off  # c' → 'off'`.
> - **`isdf_fitting.py` MOVED `common/` → `gw/`** (`c9fb0e2`) and the neutral ISDF
>   **core primitives were split out to a new `src/isdf/core.py`** package
>   (`dfb6b90`, byte-identical relocation). `gw/isdf_fitting.py` is now the thin
>   ~1030-line GW orchestrator (`fit_zeta_to_h5`, `mem_probe`); `isdf/core.py`
>   (~1733 L) holds `fit_one_rchunk`/`solve_zeta`/`factor_c_q` + the 6 jit caches.
>   **All `isdf_fitting.py:NNNN` line refs in this catalog are stale** (the file
>   shrank from ~2700 → 1030 L); paths + line numbers below are re-pointed to
>   `gw/isdf_fitting.py` or `isdf/core.py` as the symbol dictates.
> - **`LORRAX_MEM_PROFILE` is GONE** — no longer read anywhere in `src/` (was
>   `isdf_fitting.py:21`). Removed from §2.4.
> - Reader cleanup Wave 1/2 (`94cc354`/`a697abc`/`842c3c6`/`04ff0a9…bb04399`)
>   single-sourced mf_header/ψ-unfold/small-component-lift and **deleted
>   `common/load_wfns.py`** — touches no input flags in this catalog.
>
> **Prior — Updated 2026-07-02** (memplanner + V_q-tile delete-pass) below still holds:

Sources of truth reconciled:

1. `src/gw/gw_config.py` — `_DEFAULTS` dict (85 keys) + `read_lorrax_input()` (the actual `cohsex.in` parser; `gw_init.read_cohsex_input` is a re-export alias, `gw_init.py:516`).
2. `src/gw/gw_config.py` — `LorraxConfig` dataclass + sub-dataclasses (`FilePaths`, `HeadConfig`, `ScreeningConfig`, `PPMConfig`, `MemoryConfig`, `BackendConfig`, `DebugConfig`, `BSEConfig`).
3. `docs/docs_gwjax/COHSEX_INPUT.md` (411 lines; substantially stale — see §6).
4. CLI tools (`centroid.kmeans_cli`, `psp.get_dipole_mtxels`, `gw.kin_ion_io`) + `grep os.environ` across `src/`.

Type coercion rule (`gw_config.py:~433-455`): the parser infers type from the `_DEFAULTS` value (bool→`getboolean`, int→`getint`, float→`getfloat`, `None`→nullable float, else str). Keys in `_NORMALIZE_STR` (`gw_config.py:342`) are lowercased/stripped. Inline `#` comments are now stripped (`inline_comment_prefixes=('#',)`, `gw_config.py:405`, fixed `61ae4b8` — previously they silently voided string flags). Unknown keys in `cohsex.in` are **silently ignored** (no strict-key check) except `use_shipped_minimax_tables` (hard error, `gw_config.py:410`) and `output_file`/`eqp_output_file` (DeprecationWarning, ignored, `gw_config.py:414`).

Legend — Category: **P** physics choice · **C** convergence parameter · **R** resource/memory knob · **IO** I/O path · **D** debug/diagnostic.
Status: ✅ alive · 💀 dead (parsed, never read) · 📕 doc-only (documented, not parsed) · 📄 undocumented (implemented, absent from COHSEX_INPUT.md).

---

## 1. `cohsex.in` `[cohsex]` keys (every key in `_DEFAULTS`)

### 1.1 System geometry & band window

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `nval` | 5 | C | `gw_jax.py:147` (`Meta.from_system` → band edges b1=nelec−nval); `kin_ion_io.py:103`; `get_dipole_mtxels.py:517`; `get_DFT_mtxels.py:812`; `htransform.py:590` | Documented. σ-window valence count. |
| `ncond` | 5 | C | same as `nval` (`gw_jax.py:147`, tools) | Documented. |
| `nband` | 100 | C | `gw_jax.py:147` (b4); `kin_ion_io.py:105`; `get_dipole_mtxels.py:521`; `get_DFT_mtxels.py:814`; `htransform.py:592` | Documented. **BGW gotcha:** must match `number_bands` in `epsilon.inp`/`sigma.inp` for fair comparison. |
| `sys_dim` | **2** | P | `gw_jax.py:150` → `meta.sys_dim` → `gw/coulomb/get_kernel`, `compute_vcoul`, `v_q_g_flat`, `v_q_bispinor`, ζ G-sphere (`gw/isdf_fitting.py:~550`, `build_static_*` w/ `sys_dim=int(meta.sys_dim)`); `kin_ion_io.py:96` | Documented (0/2 only in docs; 3 also valid — `coulomb/base.py:32`). **BGW gotcha:** default 2 = `cell_slab_truncation`; 3D bulk runs MUST set `sys_dim=3` (BGW default is untruncated). `kin_ion_io` has an unreachable fallback default of 3 (`params.get("sys_dim", 3)`) — dead code since `_DEFAULTS` always supplies 2. |
| `ecutrho` | `None` → **ecutwfc** | C | 📄 **NEW 2026-07-02.** psp tools only: `get_DFT_mtxels.py:779-783` (valence charge-density FFT grid). **Not read by the GW driver** and unrelated to the V_q cutoffs `bare_coulomb_cutoff`/`zeta_cutoff` (§1.6) — those default to `wfn.ecutwfc` too but are ceilinged *by* `wfn.ecutrho`. When None the psp tool falls back to `wfn.ecutwfc`. Undocumented. |

### 1.2 File paths (`config.paths`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `wfn_file` | `WFN.h5` | IO | `gw_jax.py:136`; all four preprocessing tools; `htransform.py:585` | Documented. GW k-grid comes from this file, never from K_POINTS. |
| `centroids_file` | `centroids_frac.txt` | IO | `gw_jax.py:138`; `htransform.py:587` | Documented (docs cite `centroid.kmeans_isdf`; user-facing CLI is `centroid.kmeans_cli`, see §4.1). |
| `centroids_file_current` | `""` (→None) | IO | `gw_init.py:757,769,962` (bispinor transverse ζ-fit + V_q IBZ); `sigma_x_bispinor.py:135` | 📄 Undocumented. **Required** when `bispinor=true` (loud ValueError `gw_init.py:758`). Produced by `kmeans_cli --density-mode current`. |
| `kin_ion_file` | `kin_ion.h5` | IO | `gw_jax.py:468` (diag T+V_ion+V_NL read) | 📄 Undocumented. Produced by `gw.kin_ion_io` (§4.3). |
| `sigma_diag_file` | `sigma_diag.dat` | IO | `gw_jax.py:938` | 📄 Undocumented (replaced legacy `output_file` 2026-05-04). |
| `eqp0_file` | `eqp0.dat` | IO | `gw_jax.py:939` (BGW-format eqp0) | 📄 Undocumented. |
| `eqp1_file` | `eqp1.dat` | IO | `gw_jax.py:940` (Z-linearized eqp1) | 📄 Undocumented. |
| `sigma_omega_h5_file` | `sigma_mnk.h5` | IO | `ppm_pipeline.py:254,395` | Documented but **doc default stale** (docs say `""`). |
| `sigma_kij_h5_file` | `""` | IO | `gw_driver_helpers.py:281` → `ppm_sigma.py:1530` (`kij_stream` accumulation target) | 📄 Undocumented (docs mention it only in passing under `sigma_omega_accumulation`). |

### 1.3 Core mode flags (top-level)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `restart` | `true` | IO | `gw_init.py:1191` (skip ISDF fit, load `tmp/isdf_tensors_*.h5`) | Documented. Must be false on first run / after nband change. |
| `compute_mode` | `auto` | P | `LorraxConfig.compute_mode` property (`gw_config.py:734`) → `gw_jax.py:313,316,406,411`; `ppm_pipeline.py:77,326,334`; `sc_iteration.py:282,295` | 📄 **Undocumented — this is the canonical mode axis** (`x_only`/`cohsex`/`gn_ppm`/`hl_ppm`). Docs still describe only the legacy trio. `auto` infers from `do_screened`/`use_ppm_sigma`/`ppm_model`. |
| `do_screened` | `true` | P | mode inference; `gw_jax.py:199,291,301,352-361,877`; `cohsex_sigma.py`; `ppm_pipeline.py:322` | Documented. Legacy; superseded by `compute_mode` but still load-bearing in several driver branches. |
| `bispinor` | `false` | P | `gw_jax.py:147,151`; `gw_init.py` (transverse ζ/V_q branch); `gflat_memory_model`; `kin_ion_io.py:106`; `get_dipole_mtxels.py:528`; `get_DFT_mtxels.py:815`; `htransform.py:593` | Documented (docs describe γ⁰ vertex; current code = 4-channel γ̃^{0,1,2,3} ISDF; **bare DHF+Breit exchange only, screened-W unbuilt**). |
| `do_G0` | `true` | P | `gw_jax.py:352` (head injection gate); `sigma_dispatch.py:140` | 📄 Undocumented. Disables q→0 head correction entirely when false. |
| `self_consistent` | `false` | P | `gw_jax.py:411,475,757,924`; `sc_iteration.py`; `gw_output.py:342` | Documented (docs claim "not compatible with use_ppm_sigma" — stale: SC now wraps any mode per `gw_config.py:50-52`, driver wires COHSEX + dynamic modes via `sc_iteration`). SC loop knobs are **env-only** (`LORRAX_SC_*`, §3). |
| `use_ppm_sigma` | `false` | P | only `LorraxConfig.compute_mode` property (`gw_config.py:745`) | Documented as primary key in docs; actually a legacy mirror — prefer `compute_mode=gn_ppm/hl_ppm`. |
| `no_degen_averaging` | `false` | P | `gw_jax.py:380,723` | 📄 Undocumented. **BGW gotcha:** default-on averaging mirrors BGW `Sigma/shiftenergy.f90`; disable to see raw QE-basis diagonals. |
| `degen_avg_tol_ry` | `1e-6` | C | `gw_jax.py:387,732` | 📄 Undocumented. Matches BGW `TOL_Degeneracy`. |

### 1.4 Backend / I-O selection (`config.backend`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `use_ffi_io` | `true` | R/IO | → `backend.slab_io` enum (`PHDF5_FFI`/`PHDF5_HOST`/`H5PY_ALLGATHER`); `gw_driver_helpers.py:154-167` (MPI pre-init); `SlabIO`/`ZetaReader` throughout `gw_init.py`, `ppm_pipeline.py:267`, `gw_jax.py:311,366,469`; legacy bool view via `LorraxConfig.use_ffi_io` property | 📄 Undocumented. Auto-downgrades on CPU backend (`gw_config.py:994-1018`). |
| `gspace_mode` | `host_cache` | R | `gw_init.py:720,842` → `fit_zeta_to_h5(gspace_mode=)`; enum `GspaceIO` (`host_cache`\|`file_reread`) | 📄 Undocumented. `file_reread` for ψ(G) too big for host RAM. |
| `isdf_memory_mode` | `auto` | R | → `backend.screening_solver` (`jax_native`\|`cublasmp_ffi`); `gw_jax.py:262,265`; `screening.py:164` → `w_isdf.solve_w` | 📄 Undocumented. Legacy strings: `auto`/`high_mem`→JAX_NATIVE, `low_mem`→CUBLASMP_FFI. Invalid value = hard error (`gw_config.py:854`). |
| `cusolvermp_charge` | `auto` | R | `gw_init.py:721,843` → `fit_zeta_to_h5` (`gw/isdf_fitting.py`) → `fit_one_rchunk`/`factor_c_q` (`isdf/core.py`) ζ-fit Cholesky path: cusolvermp vs sharded | 📄 Undocumented. 3-state `auto|on|off`; forced `off` on CPU. `auto` = cuSolverMp only on true 2D device meshes. **Was a footgun target of the inline-`#` bug (now fixed).** |
| `cusolvermp_lu` | `auto` | R | `gw_init.py:722,844` → `fit_zeta_to_h5` → `isdf/core.py` LU solve path | 📄 Undocumented. Same 3-state. Note bispinor μ_L=1 CCT is indefinite → LU, not Cholesky. |
| `gamma_contract_mode` | `take` | D/R | `gw_init.py:547` → `common/gamma_matrices.set_gamma_contract_mode` (`take|einsum|scan`) | 📄 Undocumented. Math-identical HLO variants of the γ̃ double-contract. |

### 1.5 Memory / chunking (`config.memory`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `memory_per_device_gb` | `0.0` (auto-detect) | R | `gw_config.py:861-867` (autodetect); `compute_optimal_chunks` (`gw_init.py:1199`); `plan_gflat_chunks` (`gw_init.py:610`); V_q budget (`gw_init.py:898`) | Documented. Perlmutter A100-40: 28; local 8 GB: 6.0. |
| `chunk_size` | `-1` | R | 💀 **DEAD.** `gw_jax.py:152` sets `meta.chunk_size = get_effective_chunk_size(...)`; `meta.chunk_size` is never read anywhere in `src/`. | Documented in docstring as "legacy band-chunk knob". Delete candidate. |
| `band_chunk_size` | `16` | R | `gw_init.py:1213-1215` (UPPER CAP on heuristic chooser); `gw_init.py:627-628` (`band_chunk_override` into gflat planner) | 📄 Undocumented. **Non-obvious:** default 16 > 0 means the cap is always active — the planner can never pick band_chunk > 16 unless the user sets this larger or ≤0. |
| `r_chunk_size` | `0` (auto) | R | `gw_init.py:624-625,1203` (`r_chunk_override`) | 📄 Undocumented. Forces ISDF r-chunk. |
| `gflat_chunk_size` | `0` (planner) | R | `gw_init.py:600,629-631` → `plan_gflat_chunks(gflat_chunk_size_override=)`; runtime `accumulate_rchunk_to_gflat` | 📄 Undocumented. `cohsex.in` > 0 wins over the planner. |
| `vq_g_chunk_size` | `0` (auto `_pick_g_chunk`) | R | `gw_init.py:1026-1027` (bispinor V_q), `compute_all_V_q(g_chunk_size=)` → `v_q_g_flat.compute_all_V_q_g_flat` | 📄 Undocumented. Per-q G-axis GEMM chunk. Now the **only** V_q working-set knob (the old byte-budget args on `compute_all_V_q` were deleted). |
| ~~`use_aot_chunk_chooser`~~ | — | — | ✅ **DELETED 2026-07-02.** Removed from `_DEFAULTS` and `MemoryConfig` (commit `554bcbe`); the whole `gw/aot_memory_model/` package (~8.7k L incl. fit artifacts) and `_apply_aot_chunk_model` are gone. `gflat_memory_model.py` is the sole planner. | No longer parsed. |
| ~~`chunk_chooser_mode`~~ | — | — | ✅ **DELETED 2026-07-02** (same commit). | No longer parsed. |

Non-key members of `MemoryConfig` sourced from env, not `cohsex.in`: `chunk_target_utilization` (← `ISDF_CHUNK_TARGET_UTILIZATION`), `zct_stage_cap_gb` (← `ISDF_ZCT_STAGE_CAP_GB`/`_FRAC`). Note `gw_init.py:619`: the gflat planner uses a **hard-coded 0.80** target utilization — the 0.97 knob feeds only the legacy `compute_optimal_chunks`.

### 1.6 Coulomb / head (`config.head`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `mc_average_vcoul_body` | `true` | P | `gw_init.py:949` → `compute_all_V_q(mc_average_vcoul_body=)` → `v_q_g_flat.py:512` (**only takes effect for `sys_dim==3`**) | 📄 Undocumented. Mini-BZ MC averaging of the V body (BGW-style cell averaging). Old `make_v_munu_chunked_kernel` r-space consumer deleted 2026-07-02. |
| `bare_coulomb_cutoff` | `None` → **ecutwfc** | C/P | resolved via `_resolve_cutoff` in `gw_init.py` (ζ-fit ~L578, re-resolved for V_q ~L794) → `bare_coulomb_cutoff_ry` sqrt_v(q+G) mask in `v_q_g_flat` / `v_q_bispinor` (r-space `compute_vcoul`/`v_q_tile` mask **deleted 2026-07-02**); stored in `isdf_header` | 📄 Undocumented. **BGW gotcha (verified):** code default = `wfn.ecutwfc` (matches BGW's `screened_coulomb_cutoff` default). **Ceiling = `wfn.ecutrho`** (hard error above — "the FFT grid can't represent G's past ecutrho"). Historical LORRAX default was 4·ecutwfc (MEMORY `project_bare_coulomb_cutoff_default`) — **always set explicitly in BGW comparisons**. Distinct from the psp-tools `ecutrho` key (§1.1). |
| `zeta_cutoff` | `None` → ecutwfc | C | resolved via `_resolve_cutoff` (`gw_init.py` ~L579) → `fit_zeta_to_h5(zeta_cutoff_ry=)` → on-disk per-q ζ G-sphere (`gw/isdf_fitting.py:~515-612`, builds `_gflat_sphere_idx_padded`/`_gflat_gvec_components`/`_gflat_ngk_per_q` + writes `isdf_header`); read back by `ZetaReader.read_zeta_G_slab` (`file_io/zeta_reader.py:188`) | 📄 Undocumented. Must be ≥ `bare_coulomb_cutoff` (hard error `gw_init.py` ~L581-587), ≤ `wfn.ecutrho`. V_q reads ζ̃ at every G inside its mask, so the ζ sphere must be at least as wide. |
| `use_bgw_vcoul` | `false` | D | `gw_driver_helpers.py:185-235` (`build_bgw_v_grid_fn`) → `compute_V_q(bgw_v_grid_fn=)` | 📄 Undocumented. Substitutes BGW's MC-averaged v(q+G) for bit-reproducible comparison. |
| `bgw_vcoul_file` | `""` | IO | `gw_driver_helpers.py:197-206` | 📄 Undocumented. Required when `use_bgw_vcoul=true` (hard error). |
| `bgw_vcoul_sym_wfn` | `""` | IO | `gw_driver_helpers.py:219-225` (pull 48-op sym group from aux WFN when main WFN is nosym) | 📄 Undocumented. |
| `wcoul0_source` | `s_tensor` | P | `head_correction.py:112` (`s_tensor|epshead`) | Documented. `s_tensor` needs `dipole.h5` from `psp.get_dipole_mtxels`. **BGW gotcha:** `epshead` is static-only → wrong PPM at ω≠0 (debug only). |
| `wcoul0_eta` | `0.0` | C | `head_correction.py:118` | 📄 Undocumented (regularization η in the mini-BZ average). |
| `vhead` | `None` | D | `head_correction.py:91-98` (override bypasses `determine_wcoul0`) | Documented. For injecting BGW epsilon.out head values. |
| `whead_0freq` | `None` | D | `head_correction.py:92-98` | Documented. Set together with `vhead`. |
| `whead_imfreq` | `None` | D | `head_correction.py:92-98` (ω=iω_p key) | Documented. PPM only. |

### 1.7 Screening / minimax (`config.screening`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `screening_method` | `minimax` | P | 💀 **DEAD.** Parsed into `ScreeningConfig.method`; never read (grep: no `screening.method` consumer). Only minimax exists. | Documented. Delete or validate-only. |
| `minimax_target_error` | `1e-6` | C | via `LorraxConfig.minimax_config` property → `w_isdf.build_static_quadrature` (`gw_jax.py:237`, `screening.py:153,157`), `minimax_screening.py` | Documented. |
| `minimax_max_nodes` | `64` | C | same path | Documented. |
| `regenerate_minimax_tables` | `false` | C/IO | `minimax_config` + `sigma_quadrature_config` → minimax disk-cache regeneration | 📄 Undocumented (docs still show the removed `use_shipped_minimax_tables`, which now raises ValueError `gw_config.py:419-422`). |
| `minimax_energy_reference` | `midgap` | C | `w_isdf.py:456` (`midgap|vbm`) | Documented (docs also claim numeric Ry accepted — **not implemented**, string only). |

### 1.8 PPM model & Σ_c quadrature (`config.ppm`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `ppm_model` | `gn` | P | `compute_mode` auto-inference (`gw_config.py:751-754`); PPMConfig.model | 📄 Undocumented (`gn|hl`; docs describe GN only). Superseded by `compute_mode=gn_ppm/hl_ppm`. |
| `ppm_omega_p` | `2.0` Ry | P/C | `gw_jax.py:317,320`; `screening.py:91,94` (probe-W eval point); `ppm_pipeline.py:336-337` | Documented. **BGW gotcha:** GN probe must match epsilon's second (imaginary) frequency, 2.0 Ry by default. HL: real Ω (typical 200 Ry per docstring). |
| `ppm_fallback_omega` | `2.0` | P | `ppm_pipeline.py:343` → `fit_ppm(fallback_omega=)` (`ppm_sigma.py:663-691`) | Documented, but the docs tie it to `ppm_invalid_mode="fixed_2ry"` which is now dead (below); it feeds the pole-fit fallback directly. |
| `ppm_head_omega_h_ry` | `None` | D | `ppm_pipeline.py:76` (override head pole Ω_h) | 📄 Undocumented. **BGW gotcha:** set to BGW's √(ω_p²/(1−ε_head⁻¹)) to remove head-averaging-convention disagreement. |
| `ppm_sigma_target_error` | `1e-6` | C | `sigma_quadrature_config` property → `compute_sigma_c_ppm_omega_grid(sigma_window_quad=)` (`ppm_pipeline.py:355`) | 📄 Undocumented. |
| `ppm_sigma_max_nodes` | `64` | C | same; also `crossing_max_nodes = max(500, ...)` (`gw_config.py:782`) | 📄 Undocumented. |

### 1.9 Σ_c(ω) output grid & accumulation (`config.ppm`, ω-grid half)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `sigma_omega_min_ev` | `-5.0` | C | `omega_grid_ry/ev` properties; `build_ppm_sigma_runtime_options` (`gw_driver_helpers.py:252-258`); `gw_jax.py:503` | Documented. |
| `sigma_omega_max_ev` | `5.0` | C | same; `gw_jax.py:504` | Documented. |
| `sigma_omega_step_ev` | `0.25` | C | same (validated > 0) | Documented. |
| `sigma_regularization_ev` | `0.25` | C | `ppm_sigma.py:1526` (crossing-window sine-sum ξ) | Documented. Node count ∝ range/ξ. |
| `sigma_window_edge_factor` | `1.5` | C | `ppm_sigma.py:1527` | 📄 Undocumented. |
| `sigma_omega_batch_size` | `4` | R | `ppm_sigma.py:1528`; `ppm_pipeline.py:276` | 📄 Undocumented. |
| `sigma_omega_accumulation` | `auto` | R | `ppm_sigma.py:1529` (`auto|kij|kij_stream`) | Documented. `kij_stream` needs `sigma_kij_h5_file`. |

### 1.10 PPM on-shell / evaluation-at-E_DFT options

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `ppm_sigma_scale` | `1.0` | D | 💀 **DEAD.** Parsed into `PPMSigmaRuntimeOptions.ppm_sigma_scale`; no reader anywhere. | Documented — doc-vs-code drift. |
| `ppm_sigma_flip_neg` | `false` | D | 💀 **DEAD** (same). | Documented. |
| `ppm_invalid_mode` | `static_limit` | P | 💀 **DEAD.** Parsed; never read. Invalid-GN-mode handling is now hardwired (`valid_mask_q` in `ppm_sigma.py`; static-limit path implicit). | Documented (incl. `fixed_2ry` option that no longer exists as a switch). |
| `fermi_reference` | `midgap` | P | `ppm_sigma.py:1531,1551-1561` (`midgap|vbm`; validated in both parser and kernel) | Documented. |
| `sigma_at_dft_extrapolate` | `false` | P | `gw_jax.py:673` (extrapolate Σ_c outside ω-grid vs NaN) | Documented. |
| `sigma_at_dft_energies` | `false` | P | 💀 **DEAD as a gate.** Parsed into options; `_eval_sigma_c_at_dft_energies` now runs **unconditionally** (`ppm_pipeline.py:372-381`). | Documented as required for `sigC_EDFT` — stale; always computed now. |

### 1.11 Debug flags (`config.debug`)

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `debug_hartree` | `false` | D | 💀 **DEAD.** Parsed to `DebugConfig`; never read. | Documented. |
| `debug_omega` | `None` | D | 💀 **DEAD.** | 📄 Also undocumented → double-dead. |
| `sigma_freq_debug_output` | `false` | D | `gw_jax.py:805` (writes per-band decomposition table) | Documented (docs' claim that it forces `sigma_debug_split_contrib` is stale — that key no longer exists). |
| `sigma_freq_debug_file` | `sigma_freq_debug.dat` | IO/D | `gw_jax.py:908-909` | 📄 Undocumented as a key (filename mentioned in docs). |
| `ppm_sigma_debug_static_norm` | `false` | D | 💀 **DEAD.** | Documented. |
| `ppm_static_cohsex_check` | `false` | D | 💀 **DEAD.** | Documented. |
| `sigma_debug_quadrature` | `false` | D | 💀 **DEAD.** | Documented. |
| `sigma_debug_quadrature_samples` | `200` | D | 💀 **DEAD.** | 📄 Undocumented. |
| `write_w_copies_debug` | `false` | D | 💀 **DEAD.** Parsed into options; W-snapshot writer no longer exists. | Documented with full h5 schema — fully stale. |
| `w_copies_debug_file` | `""` | IO/D | 💀 **DEAD.** | 📄 Undocumented. |
| `write_wfn_h5` | `true` | IO | `gw_jax.py:574,757` (end-of-run `WFN_qp.h5` BGW-format dump) | 📄 Undocumented. |

### 1.12 BSE interpolation setup (`config.bse`)

These are parsed into `BSEConfig` but the **GW driver never reads `config.bse`**; the consumer is the separate `bandstructure.htransform` CLI, which re-parses `cohsex.in` into a raw params dict.

| Key | Default | Cat | Consumers | Notes |
|---|---|---|---|---|
| `get_centroids_fi` | `false` | P | `htransform.py:897` | 📄 Undocumented. Master gate for fine-grid wfn recovery. |
| `wfn_fi_min` | `0` | C | `htransform.py:899` | 📄 Undocumented. |
| `wfn_fi_max` | `0` (=full window) | C | `htransform.py:900` | 📄 Undocumented. |
| `kgrid_fi` | `""` | C | `htransform.py:905` → `bse_setup._parse_kgrid_fi` | 📄 Undocumented. `"nx ny nz"` or comma form. |

### 1.13 Blocks & rejected keys

| Item | Behavior | Consumers |
|---|---|---|
| `K_POINTS {crystal_b}` block | Parsed into `params["kpoints_crystal_b"]` (`gw_config.py:459-491`); stripped before INI parse. | `htransform.py:451` only (band-plot segments). Does **not** affect the GW k-grid (from WFN.h5). Documented correctly. |
| `use_shipped_minimax_tables` | **Hard ValueError** (`gw_config.py:419-422`). | — |
| `output_file`, `eqp_output_file` | DeprecationWarning, ignored (`gw_config.py:423-435`). | Docs §9 still document `output_file` with default `eqp0_noqsym.dat` — stale. |

### 1.14 Doc-only keys (in COHSEX_INPUT.md, NOT parsed by the code)

| Doc key | Doc section | Reality |
|---|---|---|
| `x_only` | §3 | 📕 Never parsed. Use `compute_mode = x_only`. |
| `use_chunked_isdf` | §1 | 📕 Never parsed. Chunked ISDF is the only path now. |
| `write_no_head_vw` | §10 | 📕 Never parsed. |
| `sigma_debug_split_contrib` | §8 | 📕 Never parsed; split-contribution datasets no longer switchable via input. |
| `output_file` | §9 | Deprecated (warns + ignored); see 1.13. |

---

## 2. Environment variables (grep `os.environ` across `src/`)

### 2.1 GW-pipeline behavior (physics-relevant!)

| Var | Default | Cat | Read at | Notes |
|---|---|---|---|---|
| `LORRAX_FORCE_FULL_BZ` | `0` | D (gate seam) | os.environ reads: `gw_init.py:553,846`; `gw_jax.py:210`; `v_q_g_flat.py:174` (in `gw/isdf_fitting.py` it is now a **passed-in param**, not an env read — see comments at `gw/isdf_fitting.py:358,472`) | Disables the ζ/V_q/Σ_X IBZ-only cascade (forces full-BZ). **DEV/gate seam only — NOT a user-facing flag** (user decision 2026-07-02): it exists to back `test_ibz_full_bz_equivalence` (1-GPU MoS2, commit `1479162`), which asserts IBZ and full-BZ paths agree. Do not document as a tuning knob. IBZ path additionally gates on centroid orbit closure (kmeans `--orbit`). |
| `LORRAX_SC_MAX_ITER` | `20` | C | `gw_jax.py:536` | SC loop max iterations. Env-only, undocumented. |
| `LORRAX_SC_TOL_EV` | `1e-4` | C | `gw_jax.py:537` | SC convergence tolerance (eV). Env-only. |
| `LORRAX_SC_ACCEL` | `rcrop` | C | `gw_jax.py:538` | SC acceleration scheme. Env-only. |
| `LORRAX_SC_DEPTH` | `5` | C | `gw_jax.py:539` | SC mixing history depth. Env-only. |
| `LORRAX_SC_MIXING` | `1.0` | C | `gw_jax.py:540` | SC linear-mixing factor. Env-only. |

**Refactor flag:** the five `LORRAX_SC_*` knobs are the SC loop's entire convergence-control surface and live outside `cohsex.in`/`LorraxConfig` entirely — prime candidates for promotion to input keys.

### 2.2 Memory / chunking

| Var | Default | Read at | Notes |
|---|---|---|---|
| `ISDF_CHUNK_TARGET_UTILIZATION` | `0.97` (clamped 0.85–1.0) | `gw_config.py:872` → `MemoryConfig.chunk_target_utilization` | Feeds legacy `compute_optimal_chunks` only; gflat planner hard-codes 0.80 (`gw_init.py:619`). |
| `ISDF_ZCT_STAGE_CAP_GB` | unset | `gw_config.py:880` | Absolute soft cap on the ZCT stage. |
| `ISDF_ZCT_STAGE_CAP_FRAC` | unset (clamped 0.10–0.95) | `gw_config.py:881` | Fractional variant; GPU backends only; GB wins over FRAC. |
| ~~`LORRAX_V_Q_FFT_COEF`~~ | — | ✅ **DEAD 2026-07-02** | Was `v_q_tile.py:84`; `v_q_tile.py` deleted (~1.7k L, commit `8369ecc`). |
| ~~`LORRAX_V_Q_Q_CHUNK`~~ | — | ✅ **DEAD 2026-07-02** | Was `v_q_tile.py:1156`; module deleted. |
| ~~`LORRAX_V_Q_MU_CHUNK`~~ | — | ✅ **DEAD 2026-07-02** | Was `compute_vcoul.py:1039`; the r-space μ-chunk path was stripped (`compute_vcoul.py` 967→~270 L). |
| ~~`LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH`~~ | — | ✅ **DEAD 2026-07-02** | Was `compute_vcoul.py:900`; async-prefetch path removed. No longer read anywhere in `src/`. |
| `LORRAX_MAX_RCHUNKS` | unset | `gw/isdf_fitting.py:926` | Truncate r-chunk loop (profiling). Still live. |

### 2.3 I/O, caching, runtime

| Var | Default | Read at | Notes |
|---|---|---|---|
| `LORRAX_FFI_SO` | unset | `ffi/common/ffi_loader.py:65` | Override path of the phdf5/cuBLASMp FFI `.so`. |
| `LORRAX_PHDF5_STRIPE_COUNT` | `16` | `_slab_io_ffi.py:353`, `_slab_io_mpi_host.py:178`, `gw/isdf_fitting.py:622` | Lustre pre-striping. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS` | `4M` | same | Lustre stripe size. |
| `LORRAX_WRITE_NO_JIT` | unset | `_slab_io_ffi.py:608` | Disable jit in FFI write path. |
| `LORRAX_PHDF5_CLOSE_VERBOSE` | `1` | `_slab_io_ffi.py:744` | File-close logging. |
| `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | unset | `minimax_screening.py:52` | Disable minimax table disk cache. |
| `LORRAX_MINIMAX_CACHE_DIR` | unset | `minimax_screening.py:54` | Cache location override. |
| `ISDF_JAX_CACHE_DIR` | `~/.cache/isdf_jax_compilation` | `jax_compile_cache.py:82-84` | `""` = opt out of persistent XLA cache. |
| `ISDF_JAX_PROFILE_DIR` | unset | `common/jax_profile.py:12-17` | Enables jax-profiler trace sections. |
| `PF_ARTIFACTS_DIR` | `profile` | `gw_driver_helpers.py:81` | pf memory-snapshot/xprof artifact dir. |
| `JAX_PROCESS_COUNT` / `JAX_NUM_PROCESSES` / `JAX_PROCESS_INDEX` / `JAX_COORDINATOR_ADDRESS` | SLURM-derived | `runtime/__init__.py:64-104` | Multi-host JAX distributed init (fallbacks `SLURM_NTASKS`/`SLURM_PROCID`/`SLURM_NODELIST`). |
| `JAX_ENABLE_X64`, `JAX_PLATFORMS`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_ALLOCATOR`, `TF_GPU_ALLOCATOR` | set via `setdefault` | `gw/__init__.py`, `runtime/__init__.py`, `kin_ion_io.py:15-18`, `get_DFT_mtxels.py:22-28` | Import-time defaults, user-overridable. `gw_output.py:128-129` reads `XLA_PYTHON_CLIENT_PREALLOCATE`/`_MEM_FRACTION` for the run banner only. |

### 2.4 Debug / diagnostics

| Var | Read at | Notes |
|---|---|---|
| `LORRAX_EXIT_AFTER_ZETA` | `gw_init.py:1235` | Clean SystemExit after ζ-fit (profiling). |
| `LORRAX_MEM_DEBUG` | `gw_init.py:1256`; `gw/isdf_fitting.py:52,736,836,911` (`mem_probe` def at :34) | Per-stage HBM probes (P4/P5 etc.). |
| ~~`LORRAX_MEM_PROFILE`~~ | ✅ **GONE 2026-07-03** | Was `isdf_fitting.py:21`; the memory-profiling hook was dropped in the stage move/split (`c9fb0e2`/`62ce45e`). No longer read anywhere in `src/`. |
| `LORRAX_RCHUNK_DEBUG` | `gw/isdf_fitting.py:851`; `isdf/core.py:1685` | Per-r-chunk timing (the `fit_one_rchunk` half now lives in `isdf/core.py`). |
| ~~`LORRAX_V_Q_AOT_VERBOSE`~~ | ✅ **DEAD 2026-07-02** | Was `v_q_tile.py`; module deleted. |
| ~~`LORRAX_V_Q_TIME_STAGES`~~ | ✅ **DEAD 2026-07-02** | Was `v_q_tile.py:1264`; module deleted. |
| `LORRAX_FFI_DEBUG_SHARDS` | `_slab_io_ffi.py:583` | Shard-layout dump. |
| `LORRAX_SC_DUMP_DIR` | `sc_iteration.py:430` | Per-iteration SC state dumps. |
| `LORRAX_LU_DEBUG_DUMP` | `cusolvermp_solve_lu_test.py:157` | Test-only. |
| `STERN_DEBUG`, `KP2_DEBUG` | `sternheimer_solve.py:44`, `run_sternheimer.py:475` | Sternheimer module (not core GW). |

---

## 3. Preprocessing-tool CLI surfaces

### 3.1 `centroid.kmeans_cli` (`src/centroid/kmeans_cli.py:42-133`)

Note: COHSEX_INPUT.md cites `python -m centroid.kmeans_isdf`; `kmeans_isdf.py` is the library module (console entry `lorrax-centroids`), the maintained CLI is `centroid.kmeans_cli`.

| Arg | Default | Cat | Notes |
|---|---|---|---|
| `N_c` (positional) | `400` | C | Centroid count; recommendation 8–12×(nval+ncond). Bispinor: ~3:1 charge:transverse ratio. |
| `--seed` | `0` | C | k-means RNG. |
| `--plot` / `--plot-zoom` | off / `1.0` | D | Matplotlib visualization. |
| `--no-shard` / `--force-shard` | off | R | Device-mesh control; auto-gate at 100k points/shard. |
| `--rho-source` | `auto` (`qe_save|wfn_ibz`) | P | Weight-density source. |
| `--rho-power` | `1.0` | C | ρ^α weighting (Gersho scaling notes in help). |
| `--qe-save` | auto-detect | IO | Explicit QE `.save` path. |
| `--oversample` | `1.5` | C | Pivoted-Cholesky prune factor; `1.0` disables (OOM workaround for large μ). |
| `--prune-n-val` / `--prune-n-cond` | `wfn.nelec` / `=n_val` | C | Prune Gram band counts. |
| `--prune-window` | `v_x_vc` (`v_x_c|vc_x_vc`) | P | Gram band-window pair. |
| `--orbit` / `--no-orbit` | auto | P | **IBZ-cascade gotcha:** centroid orbit closure gates IBZ-only ζ/V_q; `--no-orbit` silently forces the slower full-BZ path downstream. |
| `--use-phdf5` | off | IO/R | Parallel-HDF5 wfn loader for huge WFN.h5. |
| `--density-mode` | `scalar` (`current`) | P | `current` = Gordon current density for bispinor transverse channels → `centroids_file_current`. |
| `--out-suffix` | `""` / `_current` | IO | Output filename suffix. |
| env: `--prune-mem-gb`? | — | — | Not present in this checkout (mentioned in MEMORY as a fix; verify branch). |

### 3.2 `psp.get_dipole_mtxels` (`get_dipole_mtxels.py:427-503`) — produces `dipole.h5` for `wcoul0_source=s_tensor`

Reads from `cohsex.in`: `wfn_file`, `nval`, `ncond`, `nband`, `bispinor`.

| Arg | Default | Cat | Notes |
|---|---|---|---|
| `-i/--input` | `cohsex.in` | IO | |
| `--out` | `dipole.h5` | IO | |
| `--vnl-mode` | `analytic` (`numeric`) | P | Nonlocal-velocity evaluation. |
| `--vnl-h` / `--vnl-h-rel` | `1e-5` / `0.0` | C | FD step for numeric mode. |
| `--vnl-num-scheme` | `naive` (`richardson`) | C | |
| `--skip-vnl` | off | P | **BGW gotcha:** p̂-only, matches BGW `use_momentum`. Note the dipole driver's p−vNL sign is a BGW convention; physical velocity is p+vNL. |
| `--divide-energy` / `--debug` / `--debug-kindex` | off / off / `1` | D | |
| `--with-finite-q` / `--iq-list` | off / all q | P | Finite-q SOS matrix elements. |

### 3.3 `gw.kin_ion_io` (`kin_ion_io.py:74-83`) — produces `kin_ion.h5`

Reads from `cohsex.in`: `wfn_file`, `sys_dim`, `nval`, `ncond`, `nband`, `bispinor`.

| Arg | Default | Cat | Notes |
|---|---|---|---|
| `-i/--input` | required | IO | |
| `-o/--output` | `kin_ion.h5` | IO | Must match `kin_ion_file` in `cohsex.in`. |
| `-n/--nb` | `nband` from input | C | Clamped to `wfn.nbands`. |
| `--sys_dim` | input-file value | P | CLI overrides input file. |
| `--pseudo_dir` | input dir | IO | Falls back to `../qe/scf`, `../qe/nscf`. |

---

## 4. Summary of defects found

**Resolved by the 2026-07-02 delete-pass (branch `agent/memplanner-cleanup`):**
`use_aot_chunk_chooser` + `chunk_chooser_mode` removed from `_DEFAULTS`/`MemoryConfig`;
6 env vars retired with their modules (`LORRAX_V_Q_FFT_COEF`, `LORRAX_V_Q_Q_CHUNK`,
`LORRAX_V_Q_MU_CHUNK`, `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH`, `LORRAX_V_Q_AOT_VERBOSE`,
`LORRAX_V_Q_TIME_STAGES`); `compute_all_V_q` byte-budget args
(`n_rmu`/`n_rtot`/`budget_bytes`/`use_g_flat_zeta`) deleted. `LORRAX_FORCE_FULL_BZ`
reclassified as a DEV/gate seam (not user-facing).

**Resolved by the 2026-07-03 parser/split pass (branch `agent/memplanner-cleanup`):**
inline-`#` comments no longer silently void string flags (`61ae4b8`) — this closed a
real footgun affecting every string-valued key (`cusolvermp_charge`, `wcoul0_source`,
`compute_mode`, all file paths, …). `isdf_fitting.py` relocated `common/`→`gw/` and its
neutral core carved into `isdf/core.py`; env-var/consumer line refs re-pointed above.
`LORRAX_MEM_PROFILE` retired (unused). No `_DEFAULTS` key or input-visible env var
added/removed.

**Dead keys STILL parsed-but-never-read — 15 (unchanged by this refactor; still delete candidates):**
`chunk_size` (`meta.chunk_size` set at `gw_jax.py:152`, never consumed), `screening_method`, `ppm_sigma_scale`, `ppm_sigma_flip_neg`, `ppm_invalid_mode`, `sigma_at_dft_energies` (eval now unconditional), `debug_hartree`, `debug_omega`, `ppm_sigma_debug_static_norm`, `ppm_static_cohsex_check`, `sigma_debug_quadrature`, `sigma_debug_quadrature_samples`, `write_w_copies_debug`, `w_copies_debug_file`; plus the `PPMSigmaRuntimeOptions` fields `omega_p_ry`/`ppm_fallback`/`sigma_freq_debug_output`/`sigma_at_dft_*` duplicating values the code reads from `config.*` directly (bundle-level redundancy). **→ This is the top still-open cleanup this catalog surfaces: a second delete-pass on `config.debug` / `config.ppm` dead keys, mirroring the memplanner pass.**

**Documented but not implemented — 5:** `x_only`, `use_chunked_isdf`, `write_no_head_vw`, `sigma_debug_split_contrib`, `output_file` (deprecated). Also `minimax_energy_reference` numeric values (doc-claimed, not implemented).

**Implemented but undocumented — ~36 keys** (see 📄 markers), most importantly: `compute_mode`, `do_G0`, `ecutrho` (new), `bispinor`-support keys (`centroids_file_current`), `bare_coulomb_cutoff`, `zeta_cutoff`, `mc_average_vcoul_body`, `use_bgw_vcoul(+file,+sym_wfn)`, `no_degen_averaging`/`degen_avg_tol_ry`, all of §1.4 backend keys, most of §1.5 memory keys, `ppm_head_omega_h_ry`, `ppm_sigma_target_error/max_nodes`, `write_wfn_h5`, all §1.12 BSE keys, plus the five `LORRAX_SC_*` env-only SC knobs. **→ Second still-open target: `COHSEX_INPUT.md` is ~1 mode-axis behind — it documents the legacy `do_screened`/`use_ppm_sigma` trio but not the canonical `compute_mode` axis (`x_only|cohsex|gn_ppm|hl_ppm`). The doc rewrite should lead with `compute_mode`.**

**BGW-comparison gotchas:**
1. `bare_coulomb_cutoff`: current default = ecutwfc (matches BGW per code comment); historic LORRAX default was 4·ecutwfc — set explicitly in every comparison.
2. `sys_dim` defaults to 2 (slab truncation); bulk comparisons need explicit 3.
3. `ppm_omega_p` must equal BGW epsilon's second frequency (2.0 Ry).
4. `nband` must equal BGW `number_bands`.
5. Degenerate-set averaging on by default (mirrors BGW `shiftenergy.f90`, tol=TOL_Degeneracy).
6. `use_bgw_vcoul`+`bgw_vcoul_file`(+`bgw_vcoul_sym_wfn` for nosym WFNs) for bit-reproducible v(q+G).
7. `ppm_head_omega_h_ry` override removes head-averaging convention differences.
8. `wcoul0_source=epshead` is static-only → wrong dynamic PPM heads (debug only).
9. `--skip-vnl` in `get_dipole_mtxels` matches BGW `use_momentum`; dipole p−vNL sign is BGW convention.
10. K_POINTS block never affects the GW k-grid (WFN.h5 header wins).
11. IBZ cascade silently gated by centroid orbit closure + `LORRAX_FORCE_FULL_BZ`; kmeans `--no-orbit` centroids silently force full-BZ.
