# LORRAX GW — exhaustive feature catalog

The features a Berkeley-physics owner would name, sorted into the merged teleology (MAP.md §0).
Each feature: **what** (with physics) · **where** (files/functions) · **flags** · **interactions** ·
**refactor note**. This is the document to load before adding a feature — it says which category a
new feature belongs to and what it must not break. Per-file detail: `files/*.md`. Every flag: `FLAGS.md`
(128 flags). Dead/buggy call-outs: `DEAD_CODE.md`.

Legend for footprint: 🟢 clean (one home) · 🟡 smeared (see refactor note) · 🔴 has a suspected bug.

> **Updated 2026-07-02** (branch `agent/memplanner-cleanup`, base `e7b6c7d`). Landed since the
> catalog was written: the r-space **V_q tile subsystem is deleted** (A3 is now G-flat only); the
> **`aot_memory_model/` package is deleted** — `gflat_memory_model.py` is the sole planner (C6);
> the cohsex.in parser is single-sourced (Tier D); the aot chunk-chooser + several config knobs are
> removed; 5 dead modules deleted (`chi_sos`, `cg_posdef`, `centroid_io`, `archive/projector_pipeline`,
> `archive/build_projectors`); **two gates added** — GN-PPM regression + IBZ-vs-full-BZ equivalence
> (B4/E). Entries below are annotated inline (✅ = done, → = new pointer). Line numbers reverified.
>
> **Updated 2026-07-03** (same branch, later commits `61ae4b8`…`bb04399`). Big structural moves this
> pass: (1) the **A2 ζ-fit stage moved `common/` → `gw/`** and its neutral core was extracted into a
> **NEW standalone `src/isdf/` mini-library** (`isdf/core.py`, 1733 L — 7 public primitives + private
> helpers + the 6 jit caches, byte-identical) with `gw/isdf_fitting.py` (now 1030 L) as the thin GW
> orchestrator consumer; a future direct-BSE path is the second intended consumer (target #2 DONE,
> and then some). (2) The **A1/C5 WFN reader family collapsed**: `common/load_wfns.py` is **DELETED**
> — its 5 ψ-loading helpers relocated into `common/wfn_transforms.py`, 22 sites repointed; the B3
> bispinor lift single-sourced to `common/bispinor_init.lift_to_4spinor`; the B4 ψ-unfold rule
> single-sourced to `symmetry_maps.trs_augment_U`/`tau_phase_row`; the eager whole-array `coeffs[:]`
> slurp made lazy (OOM fix); `get_umklapp_vector` promoted public (targets #4, #10 substantially DONE).
> (3) Two backlog **fixes**: config parser now strips inline `#` comments; `wfn_writer` nspinor=1
> occupation uses `nelec//2` (no more 2× over-occupation). (4) **First BGW-anchored 3D gate**: Si
> 4×4×4 no-SOC COHSEX (`tests/regression/si_cohsex_debug/`, `si_cohsex_3d` case, atol 1e-3 eV) — the
> 3D Coulomb/analytic-head path now has e2e coverage. Suite: 242 passed / 24 skipped / 0 failed.
> **Still open (deferred):** `zeta_loader`/`zeta_reader` merge (target #8 — genuine backend/valid_μ
> divergence, needs a padded-μ fixture; see `READER_CLEANUP_PLAN.md` step 4).

---

## Tier A — Pipeline stages

### A0. Preprocessing (input producers) 🟢
Standalone CLIs run once before `gw_jax`, each a driver over the C10 DFT stack.
- **ISDF centroid generation** — orbit-aware density/current-weighted k-means + pivoted-Cholesky
  pruning. `centroid/kmeans_isdf.py` (algorithm), `kmeans_cli.py` (CLI), `pivoted_cholesky.py`,
  `charge_density.py`/`current_density.py` (weights). Flags: `--density-mode charge|current`,
  `--orbit`/`--no-orbit`, `--seed`, N-centroids. Emits `centroids_frac_*.txt`. Interaction: the
  `--orbit` closure gates the B4 IBZ cascade (MAP.md §3). ⚠ docs/`v_q_bispinor.py:571` cite a
  nonexistent `--orbit-aware` flag (DEAD_CODE §1).
- **kin_ion.h5 generator** — diagonal T + V_loc + V_NL matrix elements. `gw/kin_ion_io.py` over
  `psp/get_DFT_mtxels.py`. 🟡 filed under `gw/` but is a psp preprocessing driver. ✅ the *duplicate*
  cohsex.in parser is gone — `kin_ion_io` and `get_DFT_mtxels` now both import the canonical
  `gw_config.read_lorrax_input` (aliased `read_cohsex_input`); the misplaced-driver 🟡 remains.
- **dipole.h5 generator** — velocity matrix elements 〈m|v|n〉 for the q→0 head (B2). `psp/get_dipole_mtxels.py`.
  Velocity-operator sign is the BGW `p−vNL` convention (memory `project_lorrax_velocity_sign`).
- **Sternheimer/DFPT head** — χ_{G'0}(q) columns without sum-over-states. `psp/run_sternheimer.py` +
  `solvers/sternheimer_solve.py`. 🔴 JVP `b_dot` sign flip (DEAD_CODE §1). Future screening path.
- **No-QE WFN generation** — Davidson NSCF + pseudobands. `psp/run_nscf.py`, `solvers/pseudobands{,_v2}.py`.
  🔴 `run_nscf.py:299-331` broken for nk>1. ✅ **FIXED (`61ae4b8`):** the `wfn_writer` nspinor=1
  factor-of-2 over-occupation — `n_occ`/`ifmax` now use `nelec//2` for nspinor=1 (2 e⁻/band) vs `nelec`
  for nspinor=2. Affects the no-QE/pseudobands writer path only (QE-fed mainline reads occ from QE).

### A1. WFN ingest 🟡🔴 ✅ **(reader de-smeared 2026-07-03)**
Load WFN.h5 → build the 4-copy `Wavefunctions` bundle (ψ(G), ψ(r)-slab, ψ-centroid, occ).
`file_io/wfn_loader.WfnLoader` (`backend='auto'` picks phdf5 async); the 5 ψ-loading helpers
(`get_enk_bandrange`, `load_kpoint_fftbox`, `read_Gvecs_to_devices`, `iter_psi_rchunk_bandwise`,
`load_centroids_band_chunked`) now live in **`common/wfn_transforms.py`** (their real home — `WfnLoader.load`
composed with the FFT-box/r-chunk/centroid transforms). Flags: `wfn_file`, phdf5 backend selection,
band/G padding. **Refactor note (targets #4/#10 — substantially DONE):** ✅ `common/load_wfns.py`
**DELETED** (was a ~522 L facade; all 22 consumers repointed to `wfn_transforms`). ✅ the two foreign
concerns single-sourced *out* to their axes — B3 bispinor lift → `common/bispinor_init.lift_to_4spinor`
(loader keeps only the sharding/jit-cache wrapper `_get_bispinor_lift_jit` + `_apply_bispinor_lift`
that calls it), B4 ψ-unfold rule → `symmetry_maps.trs_augment_U`/`tau_phase_row` (both the host
`unfold_psi` and the device phdf5 table build now call these; `get_umklapp_vector` promoted public).
✅ the unconditional eager `wfns/coeffs[:]` slurp is **gone** — only the eager backend keeps the h5
dataset *handle* (`_coeffs_ds`, `wfn_loader.py:184`) and hyperslabs the requested block per call in
`_eager_build`; the phdf5 backend never read it (latent multi-GB OOM fixed). Residual 🟡: the loader
still *owns* the bispinor/unfold jit-cache wrappers (the physics rules are single-sourced, the
device-kernel plumbing isn't).

### A2. ζ-fit (ISDF) 🟢 ✅ **(re-homed + library-split 2026-07-03 — target #2 DONE)**
Build interpolation vectors ζ_q: pair density M_μ = Σ_vc ψ*_v(r_μ)ψ_c(r_μ) at centroids →
CCT/ZCT metric (C_q = M M†) → distributed Cholesky (charge) / **LU** (bispinor: μ_L=i CCT is
indefinite, memory `project_bispinor_isdf`) → G-flat accumulate → `zeta_q.h5`. The stage moved
`common/` → `gw/`, then its **neutral core was extracted into a new standalone `src/isdf/`
mini-library**:
- **`src/isdf/core.py`** (1733 L, `import isdf` / `from isdf.core import …`) — the 7 public primitives
  `pair_density`, `gram_q0_from_pair`, `c_q_from_psi_sm`, `z_q_from_psi_sm`, `factor_c_q`,
  `solve_zeta`, `fit_one_rchunk` + ~14 private helpers + the **6 module-level jit caches**, moved
  byte-identical. Streaming core `z_q_from_psi_sm` (scan-inside-shard_map + io_callback + all_gather —
  memory `feedback_path_d_scaffolding_pattern`) lives here now. This is a reusable ISDF library — GW is
  the first consumer, a future direct-BSE path the second (see Tier C new home).
- **`gw/isdf_fitting.py`** (now 1030 L) — the thin **GW orchestrator**: `fit_zeta_to_h5` + the C6
  `mem_probe`/`_nvsmi_used_mb_local_gpu` lifecycle probes, driven by `gw_init.py`; imports what it
  consumes (`c_q_from_psi_sm`, `factor_c_q`, `fit_one_rchunk`, …) from `isdf.core`. ✅ the dead
  `_mem_report` probe was dropped and the jit caches docstringed (`62ce45e`).
Flags: n-centroids, `zeta_cutoff`, r-chunk/gflat-chunk knobs (C6), `LORRAX_MAX_RCHUNKS` (profiling break).
Bit-identity gate exists (`test_zq_from_psi_sm_bit_identity`, 6 sub-gates incl. bispinor γ̃≠I; now
imports from `isdf`). Residual 🟡: the C6 mem-probes still ride in the GW orchestrator (see C6 note).

### A3. V_q (bare Coulomb) 🟢 ✅ **(de-smeared 2026-07-02 — now G-flat only)**
V_q(μ,ν) = Σ_G ζ*_μ v(q+G) ζ_ν and g0_μ(q). The live path is **G-flat exclusively**:
`gw/compute_vcoul.py` (now 252 L, 3 funcs: v(q+G) factory `compute_v_q_per_G`, `build_v_head_miniBZ_avg_3d`,
and the `compute_all_V_q` **dispatcher** — hands off to G-flat, `raise NotImplementedError` on any other
layout since `fit_zeta_to_h5` writes G_flat exclusively), `v_q_g_flat.py` (per-q G-chunked GEMM + batched
IBZ pre-read; **now also home to the surviving `_unfold_g0_ibz_to_full`**, relocated here),
`v_q_bispinor.py` (`compute_V_q_bispinor_g_flat_to_h5`, 7-tile: CC charge + transverse). B2 supplies
v(q+G); B4 unfolds IBZ→full. Flags: `bare_coulomb_cutoff` (⚠ default 4·ecutwfc vs BGW ecutwfc — memory
`project_bare_coulomb_cutoff_default`), G-chunk (C6). ✅ **DELETED this branch:** all of `v_q_tile.py`
(1662 L, r-space + G-flat Case A/B tile kernels), `v_q_bispinor.compute_V_q_bispinor_to_h5` + 2 factories,
`compute_vcoul.make_v_munu_chunked_kernel` + the FFT-era trio (`fft_integer_axes`/`_v_q_per_q_g_chunked_jit`/
`compute_v_q_per_q_g_chunked`), the r-space tail of `compute_all_V_q`, the `gw_init` legacy else-branch, and
`coulomb_sphere.compute_bare_coulomb_sphere_idx`. With them went the 🔴 stale cache key and 🔴 Case-B chunk
overflow (both lived in the deleted tile code) and the C6 `_choose_v_q_chunks` chooser. `tile-mode` /
`_unfold_v_q_ij_ibz_to_full` no longer exist. Gates: charge + bispinor orchestrator value gates,
TRS-unfold gates, plus the new **IBZ-vs-full-BZ equivalence gate** (B4/E) that pins this unfold algebraically.

### A4. χ₀/W screening 🟡
χ₀(q,μ,ν) = Σ over minimax-τ nodes of G_v(τ)G_c(τ) (B1 supplies nodes) → Dyson W=(I−Vχ₀)⁻¹V via
C3 distributed solve. `gw/w_isdf.solve_w`; `greens_function_kernel.build_G`; request planner decides
which (q,τ) run. q→0 head/wing from B2. Flags: `do_screened`, `screening_method` (ctsp variant untested),
W-solve backend (`w_solve_mode`), minimax table selection. **Refactor note:** hosts B1 quadrature
builders (`build_static/imag/real_quadrature`) that belong in the minimax engine (target #9). 🔴 **bispinor
χ₀/W is UNBUILT** (screened-W δ−vχ inversion not implemented — memory `project_bispinor_screened_w_roadmap`);
the entire W stage is **ungated end-to-end for bispinor and for 3D** (GATE_AUDIT §3). This is where the
~3.5 meV silent drift lives (MAP.md §5).

### A5. Σ static (X / COHSEX) 🟢🔴
Σ_X = −Σ_occ G V (bare exchange); SX−X, COH (static COHSEX); bispinor transverse Σ^B.
`gw/cohsex_sigma.py`, `gw/sigma_x_bispinor.py`. Σ_μν → band basis via `wavefunction_bundle.project`.
Flags: `compute_mode=x_only|cohsex`, `no_degen_averaging`. Interaction: bispinor SC path 🔴 **silently
drops Σ^B** (memory `project_bispinor_screened_w_roadmap`); in-plane transverse unfold non-covariant
(memory `project_bispinor_tt_noncovariance`) — pragmatic fix = IBZ charge + full-BZ-direct transverse.
Gate: static-COHSEX e2e (2D MoS2 `cohsex` case) + ✅ **NEW `si_cohsex_3d` (2026-07-03)** — the first
**BGW-anchored** static-COHSEX gate (Si 4×4×4, sys_dim=3, MAE 0.12 meV vs BerkeleyGW, atol 1e-3 eV),
so the e2e value gate is no longer a re-frozen-LORRAX-only 2D freeze — plus Σ_X bispinor unit
(bookkeeping only, no Σ^B value gate).

### A6. Σ dynamic (PPM Σc) 🟡🔴
Fit 2-point plasmon-pole (GN or HL) from W(0)/W(probe) → evaluate Σc(k,ω) on a real-ω grid via
τ-quadrature with windowed decomposition. `gw/ppm_sigma.py` (1702 L), `ppm_pipeline.py`. Flags:
`compute_mode=gn_ppm|hl_ppm`, `ppm_model`, ω-grid/window knobs, `sigma_omega_h5_file`/`sigma_kij_h5_file`
(streamed accumulation). 🔴 **streamed mode silently skips the analytic q→0 head** (`ppm_pipeline.py:126`),
🔴 negative-Ω² head sign flip (`head_correction.py:320`). **Refactor note:** `compute_sigma_c_ppm_omega_grid`
dominates runtime on large grids; not band-sharded (CHANGELOG 2026-04-09 open item — **still open**).
✅ **GN-PPM gate #1 landed 2026-07-02**: `test_gw_jax_matches_reference` is now parametrized over
`("cohsex", "gnppm")` (fixtures in `tests/regression/gnppm_debug/`, ref `sigma_diag_gnppm_ref.dat`), and a
1-GPU (1×1 mesh) crash in `build_G_tau` was fixed (mask ndim) to make it runnable. The e2e Σc path is now
gated; band-sharding of `compute_sigma_c_ppm_omega_grid` remains the open perf item.

### A7. QP solve / eqp / SC (QSGW) 🟡🔴
Z-factor, eqp0/eqp1 linearization; QSGW: mode-orthogonal Σ^xc dispatch, H_qp build, diagonal Σ(E)
fixed point, Hermitise, band mixing. `gw/sc_iteration.py`, `sigma_dispatch.py`, `qsgw_utils.py`,
`scissor.py`, `band_partition.py`, `degen_average.py`, `eqp_bgw.py`, `mixing/acceleration.py`.
Flags: `self_consistent`, `scissor`, degen tol; **SC loop knobs are env-only** (`LORRAX_SC_*`, shadow
surface — DEAD_CODE §4, FLAGS §3). 🔴 `gw_jax.py:237` latent NameError, 🔴 `qsgw_utils` convergence
metric inconsistency, 🔴 SLATE-eigh eigenvectors (Hermitise path). **Refactor note (target #5):** eqp0/eqp1/Z
math duplicated ~4× across `eqp_bgw`/`gw_output`/`gw_jax`/`sigma_dispatch` — single-source it.
`degen_average`, `sc_iteration`, `sigma_dispatch` have **no dedicated test**.

### A8. Output writers 🟡
`eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `sigma.h5`, `qp_wfn.h5`. `gw/gw_output.py`, `eqp_bgw.py`
(BGW text writers, byte-identity gated), `file_io/sigma_output.py`, `qp_wfn.py`. Flags: `sigma_diag_file`,
`eqp0_file`, `eqp1_file`, `debug.write_wfn_h5`. 🔴 **`debug.write_wfn_h5` defaults true and crashes every
IBZ run** (full-BZ U vs `wfn.nkpts` — GATE_AUDIT §2, the current red gate). Z-factor central-difference
`dE=0.5 eV` vs BGW forward-difference (`gw_output.py:189`) — convention gotcha.

---

## Tier B — Variant axes (cut across A-stages)

### B1. Frequency treatment (minimax / PPM engine) 🟡
Emits (τ,α) node sets and PPM (B,Ω) params for every scheme: static COHSEX (1/x), imag-axis GN-PPM
probe (x/(x²+ω_p²)), real-axis HL windows, Σc windows. `common/minimax.py`, `gw/minimax_screening.py`,
`gw/minimax_config.py`. Flags: `use_shipped_minimax_tables` (hard-error if missing), n-τ, window params.
**Refactor note:** the axis is smeared — quadrature *builders* live in `w_isdf.py` (A4). Consolidate into
this engine (target #9). Gates: minimax-asset selection + real-axis quadrature (strong unit coverage).

### B2. Coulomb truncation + q→0 head/wing (3D/2D/0D) 🟡🔴
Owns v(q+G) boundary conditions and the G=0 singularity: dimension-dispatched kernels + mini-BZ MC
average + head-source resolution. `gw/coulomb/{base,bulk_3d,slab_2d,box_0d}.py`, `compute_vcoul_0d.py`,
`vcoul.py` (✅ **stripped 186→68 L** to its 2 live helpers `wrap_points_to_voronoi` + `compute_q0_averages`;
the legacy per-q V(q,G) builder + `compute_vcoul_comps_for_q`/`compute_wcoul0_with_S` are gone),
`head_correction.py`, `common/chi_from_dipole.py`. ✅ `common/chi_sos.py` **DELETED** (was unwired, 363 L).
Flags: `sys_dim` (0/2/3; ⚠ default 2 = slab-truncated vs BGW untruncated), `do_G0`/`do_head`, `epshead`,
`bare_coulomb_cutoff`, MC-average knobs. 🔴 0D volume convention split (`box_0d.py:29`), 🔴 negative-Ω²
head sign. ✅ **3D gate landed (2026-07-03):** the `sys_dim=3` Coulomb/analytic-head path is no longer
untested — `si_cohsex_3d` (Si 4×4×4 no-SOC COHSEX, native finite-q body + BGW q→0 head scalars,
BGW-anchored to ~few meV, atol 1e-3 eV) exercises it e2e. 0D still rides no fixture; 2D fixture also
remains.

### B3. Bispinor / Breit spin algebra 🟡🔴
Cross-stage spin machinery: γ̃^{0,1,2,3} vertices, (perm,phase) monomial contraction, small-component lift,
current-density Gordon weights. `common/gamma_matrices.py`, `common/bispinor_init.py`,
`centroid/current_density.py`. Flags: `bispinor`,
`centroids_file_current` (required when bispinor), `gamma_contract_mode` (🔴 silently inert — falls back
to `take` on the dominant path, `gamma_matrices.py:172`). Footprint: A2 (4-channel ζ), A3 (7-tile V_q),
A5 (Σ^B) — **A4 screened-W unbuilt**. Prod centroid ratio ~640 charge : ~200 transverse (memory
`feedback_bispinor_centroid_ratio`). CrI3 always 16 GPU (memory `feedback_cri3_always_16_gpus`).
**Refactor note (target #4 — ✅ DONE):** the (α/2)(σ·(k+G))ψ_L small-component lift is now
single-sourced into `common/bispinor_init.lift_to_4spinor` (k-batched, pure-jnp) + shared module-level
`HALFALPHA`; the WFN reader (`wfn_loader._apply_bispinor_lift`/`_get_bispinor_lift_jit`) keeps only the
sharding/jit-cache wrapper that *calls* it. `bispinor_init.get_small_psi_component` stays an
independent non-batched reference oracle for the bit-match test (deliberately not merged).

### B4. IBZ symmetry & BZ mapping 🟡🔴
The single place that should know how ψ/ζ/V_q/W/k/q transform under space-group + TRS: IBZ tables,
sym_perm/L_table, SU(2) spinor rotations, centroid/r-grid perms, IBZ↔full unfolds. `common/symmetry_maps.py`,
`centroid/orbit_syms.py`, `common/kq_mapping.py`. **Still smeared across ≥5 parallel unfold helpers**
(the `v_q_tile._unfold_*` pair is gone with the tile deletion — the survivor `_unfold_g0_ibz_to_full`
now lives in `v_q_g_flat`): `v_q_g_flat._unfold_g0_ibz_to_full` + `_resolve_ibz_q_list`,
`zeta_loader._unfold_q_full_bz`, `qe_save_reader._reduce_mp_to_ibz` (a *second* IBZ reduction),
`read_bgw_vcoul.find_q_index`. ✅ **the `wfn_loader` ψ-unfold is now single-sourced (2026-07-03):** the
spinor rule (spatial → U; TRS row → iσ_y·conj U) and the τ-phase rule live in
`symmetry_maps.trs_augment_U` / `symmetry_maps.tau_phase_row`, called by BOTH the host `unfold_psi`
and the device phdf5 table build (the copy-pasted rule is gone from `wfn_loader`; the phdf5 apply-kernel
stays inside its shard_map). Flags: cascade gates on centroid orbit closure (`--orbit`);
`LORRAX_FORCE_FULL_BZ` is a **DEV/gate seam only** — NOT user-facing (user decision 2026-07-02). 🔴
`symmetry_maps.py:1050` silent nearest-IBZ fallback (same class as the historical TRS-blind bug —
memories `project_trs_blind_sym_bug`, `project_lorrax_ibz_cascade`). **Refactor note (target,
cross-cutting step 3):** unify to ONE table + ONE sym-action helper (memory `feedback_unified_sym_action`);
retire the parallel helpers. ✅ **NEW guardrail:** `test_ibz_full_bz_equivalence` (1-GPU MoS2, 642
orbit-closed centroids) asserts the IBZ cascade matches forced full-BZ-direct (`LORRAX_FORCE_FULL_BZ=1`)
to ~1e-6 eV — a direct wrong-unfold detector. Plus the strong TRS-unfold / centroid-perm units — keep
all green through the unification.

---

## Tier C — Infrastructure

- **C1. Layout / sharding / padding contracts** 🟡 — the 4-copy `Wavefunctions` bundle + module-level
  `PartitionSpec` constants every kernel shares + `runtime/padding.py` (canonical pad, 1 caller) + `Meta`.
  `gw/wavefunction_bundle.py`, `common/meta.py`, `runtime/padding.py`. Refactor note: bundle also owns
  physics (`rotate_wavefunctions` for QSGW, `project` for Σ_mn) — layout hub shouldn't; split.
- **C2. Sharded FFT & G↔r transforms** 🟡 — physics-free layout transforms: local/sharded/flat-k FFT
  factories, ψ(G)↔box↔r-slab↔centroid, Bloch phases. `common/fft_helpers.py`, `wfn_transforms.py`,
  `gvec_fft_box.py`. Refactor note: `fft_helpers` also hosts a C6 memory probe + a C3 Cholesky block
  chooser (misplaced). Flat-k chi0 refactor is queued (memory `project_flat_k_chi0_pipeline`).
- **C3. Distributed dense linear algebra (FFI + in-tree)** 🔴 — potrf/potrs/LU/eigh/gemm/fused-W-solve
  behind shard_map wrappers + .so loading + XLA FFI registration. `ffi/{cusolvermp,cublasmp,cusolvermg,slate}/*`,
  `ffi/common/ffi_loader.py`, `common/cholesky_2d.py`. Backends chosen by `w_solve_mode`/backend flags;
  SLATE installed (memory `project_slate_install`). 🔴 `slate/eigh` eigenvectors wrong (known/unresolved).
- **C4. Parallel-HDF5 slab I/O** 🟢 — move sharded device arrays to/from HDF5 without host materialization;
  `SlabIO` facade + 3 backends (FFI collective / mpi4py host / allgather+rank0). `file_io/slab_io.py`,
  `_slab_io_{ffi,mpi_host,allgather}.py`, `ffi/phdf5/*`, `common/async_io.py`. phdf5 unified on Cray MPICH
  (memory `project_phdf5_mpich_default`). 🔴 `_slab_io_ffi.py:525` read-after-write hazard.
- **C5. Per-format readers/writers** 🟡 — one schema per format: WFN.h5, zeta_q.h5 (+isdf/mf_header),
  sigma outputs, QP-WFN, kin_ion.h5, BGW eps/vcoul, QE .save. `file_io/*`. ✅ **Reader boilerplate
  single-sourced (2026-07-03, `94cc354`):** the 35-field `mf_header` mirror → `mf_header.bind_mf_attrs`,
  an `isdf_header` binder, `_rank0`/`_barrier` → `_slab_io_ffi`, `kpt_starts` → `mf_header` (−59 LOC).
  ✅ the `wfn_writer` nspinor=1 occupation bug is FIXED (see A0). Refactor note: unfinished
  `zeta_loader` vs `zeta_reader` migration (target #8, **still open** — genuine backend/valid_μ
  divergence, deferred pending a padded-μ fixture, `READER_CLEANUP_PLAN.md` step 4); `qe_save_reader`
  embeds a second IBZ reduction (→B4). ✅ `centroid/centroid_io.py` orphan dup of `file_io/centroids`
  **DELETED** (55 L). NB the 5 ψ-loading helpers that were in `common/load_wfns.py` moved to
  `common/wfn_transforms.py` (C2) and the facade was deleted (see A1).
- **C6. Memory planner & chunk choosing** 🟡 ✅ **(collapsed to one model 2026-07-02)** — predict
  per-rank HBM peak, pick static chunk sizes before jit. **`gw/gflat_memory_model.py` is now the SOLE
  planner** (922 L); `runtime/aot_memory.py` is kept (live cuFFT scratch query, not a model). ✅ the
  entire `gw/aot_memory_model/` package **DELETED** (~8.7k L incl. 26 files of JSON fit artifacts) — with
  it the 🔴 `chooser.py` dead-by-clobber. ✅ planner self-check fixed to read `peak_bytes_in_use` (the
  faithful peak); `v_q_tile._choose_v_q_chunks` gone with the tile deletion. Planner faithful within ~14%
  (memory `project_planner_conservative_8x`). **Refactor note (cross-cutting step 3):** planner is *still*
  lightly smeared — `gw_init` stacks the gflat planner, the **`gw/isdf_fitting.py` GW orchestrator** hosts
  the C6 `mem_probe` lifecycle probes (they stayed in the orchestrator, NOT in the neutral `isdf/core.py`
  library — correct), `fft_helpers` hosts a probe; finish consolidating these into `gflat_memory_model`.
  CrI3 ceilings + `--prune-mem-gb` (memory `project_cri3_gnppm_maxbands`).
- **C7. Host caches & compile cache** 🟢 — ψ(G) host tiles served per-slice via io_callback
  (memory `feedback_iocallback_for_large_caches`); XLA persistent compile cache. `common/psi_G_store.py`,
  `jax_compile_cache.py`.
- **C8. Runtime bootstrap** 🟢 — env defaults before jax import, idempotent `jax.distributed` init on
  Perlmutter/SLURM, NCCL warmup, phdf5 MPI init. `runtime/__init__.py`, `gw/gw_driver_helpers.setup_runtime`.
  🔴 `runtime/__init__.py:198` CPU-fallback can't work on pinned jax≥0.9.
- **C9. Generic iterative solvers** 🟢 — physics-free Hermitian-matvec: Davidson, Lanczos, Chebyshev/KPM,
  DOS windowing, FEAST quadrature. `solvers/{davidson,lanczos,chebyshev,dos,quadrature}.py`. ✅ `cg_posdef`
  **DELETED** (191 L); `minres` still dead (DEAD_CODE §2). Refactor note: Sternheimer/DFPT physics
  (`sternheimer_solve`, `projectors`) sits in
  generic `solvers/` — move to a DFPT home.
- **C11. ISDF core mini-library** 🟢 ✅ **NEW (2026-07-03, `dfb6b90`)** — the neutral ψ + centroids → ζ
  interpolative-separable-density-fitting core, carved out of the A2 ζ-fit stage into a reusable package
  so it is no longer welded to GW. `src/isdf/core.py` (1733 L) + `src/isdf/__init__.py` (7-name public
  re-export: `pair_density`, `gram_q0_from_pair`, `c_q_from_psi_sm`, `z_q_from_psi_sm`, `factor_c_q`,
  `solve_zeta`, `fit_one_rchunk`). Byte-identical extraction incl. the 6 module-level jit caches
  (`isdf.core._fit_one_rchunk_cache` etc. — `gw_init` clears them via that path). Consumers: the
  `gw/isdf_fitting.py` GW orchestrator (A2) today; a **future direct-BSE path** is the intended second
  consumer. Physics-free of GW specifics — depends only on C1/C2/C3 layout+FFT+linalg primitives.
- **C10. DFT / pseudopotential operator stack** 🟢 — plane-wave KS Hamiltonian + derivatives (T/V_loc/V_H/
  V_xc/V_NL, KB projectors, radial→q, solid harmonics, dH/dk velocity), UPF loading. `psp/*`. Shared by A0.
  🔴 V_xc is spin-blind (memory `project_lorrax_vxc_spin_blind`) — blocks magnetic Sternheimer, core GW
  unaffected. 🔴 NLCC enum gate, density symmetrization phase, j-default (DEAD_CODE §1).

---

## Tier D — Config & flag surface 🟡
Single source of truth *should* be `gw_config.LorraxConfig` (frozen) + `read_lorrax_input` parser.
128 flags catalogued in `FLAGS.md`. **Defects still defining the refactor here:** (1) ✅ **FIXED** — the
cohsex.in parser is now single-sourced: `gw_config.read_lorrax_input` is canonical (aliased
`read_cohsex_input`), the `get_DFT_mtxels` copy is deleted and all consumers import the alias; and ✅
(2026-07-03, `61ae4b8`) the parser now sets `inline_comment_prefixes=('#',)` — previously
`key = off  # note` parsed to the value `'off  # note'` and silently voided the flag (fell back to auto); (2)
`PPMSigmaRuntimeOptions` (23 fields, `gw_driver_helpers`) mirrors config (**open**); (3) a **shadow env-flag
surface** (`LORRAX_SC_*`, `LORRAX_V_Q_*`, `LORRAX_MAX_RCHUNKS`, `LORRAX_NGPU`, …) lives outside
`_DEFAULTS` — a live TODO in `sc_iteration` acknowledges it (**open**). ✅ **Config knobs DELETED this
branch:** `use_aot_chunk_chooser`, `chunk_chooser_mode` (chooser gone with `aot_memory_model`);
`async_prefetch` (was a compat no-op); and dead `compute_all_V_q` params (`n_rmu`/`n_rtot`/`budget_bytes`/
`use_g_flat_zeta`). Docs (`COHSEX_INPUT.md`) are substantially
stale: `compute_mode` (the canonical mode axis) is undocumented; many IO-path keys undocumented; some
documented keys dead. **Gotcha flags** (BGW parity): `sys_dim`, `bare_coulomb_cutoff`, `nband`,
`no_degen_averaging` — all in FLAGS.md with the BGW-side value.

## Tier E — Diagnostics / instrumentation / benches 🟡
Observe, don't compute: timing/profiler/progress (`common/{timing,jax_profile,progress}.py`), ~20 FFI
correctness/bench CLIs (`common/*_{test,bench}.py` — cusolvermp/slate/cublasmp), plotting. Refactor note:
these ~20 benches make `common/` host a whole diagnostics category — move under `tools/` or `tests/`.
✅ **Regression gates added 2026-07-02** (`tests/test_gw_jax_regression.py`): the e2e value gate is now
parametrized over COHSEX **and** GN-PPM (`test_gw_jax_matches_reference`, fixtures under
`tests/regression/{cohsex_debug,gnppm_debug}/`), and `test_ibz_full_bz_equivalence` guards IBZ↔full-BZ
unfolds via the `LORRAX_FORCE_FULL_BZ` dev seam. Gate-0 crash fixed + reference re-frozen from main.
✅ **2026-07-03:** a third parametrized case `si_cohsex_3d` (`tests/regression/si_cohsex_debug/`, Si
4×4×4 no-SOC, sys_dim=3) adds the **first BGW-anchored** e2e gate (MAE 0.12 meV vs BerkeleyGW, atol
1e-3 eV) — 3D Coulomb/analytic-head coverage that no prior gate had. Also (`25067f4`) 4 container-JAX
failures are conditionally skip-marked ("red means red") and 3 golden gates wired into
`skills/checkpoint/SKILL.md`; suite = 242 passed / 24 skipped / 0 failed.

## Tier F — Archived / dead
Delete list — see DEAD_CODE.md §2. ✅ **Deleted 2026-07-02:** `psp/archive/{projector_pipeline,build_projectors}.py`,
`solvers/cg_posdef.py`, `centroid/centroid_io.py`, `common/chi_sos.py`, the whole `gw/aot_memory_model/`
package (incl. `chooser.py`), `gw/v_q_tile.py`; `gw/vcoul.py` stripped to 2 helpers. ✅ **Off the delete
list (2026-07-03):** `common/bispinor_init.py` is **no longer dead** — it now hosts the live,
single-sourced B3 lift `lift_to_4spinor` + shared `HALFALPHA` (its `get_small_psi_component` remains the
bit-match reference oracle). **Still on the list:** `gw/experimental/head_wing_schur.py` and ~180 dead
functions within live files.
