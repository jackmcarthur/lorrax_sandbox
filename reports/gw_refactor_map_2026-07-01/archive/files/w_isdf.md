# src/gw/w_isdf.py — deep-read notes (2026-07-01)

690 LOC. Module docstring: "Static χ₀ and W computation using ISDF + minimax quadrature.
All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers."

## Purpose / category

Physics: the chi0/W stage of the ISDF GW pipeline. Builds χ₀(q,μ,ν) in the ISDF-centroid
basis via imaginary-time (minimax Laplace) quadrature over Green's-function pairs, then
solves the Dyson equation W = (I − Vχ₀)⁻¹V per q. Also hosts the quadrature-builder
helpers (static / imag-freq / real-freq HL-PPM) that map band-energy intervals to
(τ_l, α_l) node sets, and the AOT-precompile twins of the two kernels.

Category guess: **physics: chi0/W stage** (with a slice of "resource mgmt": sharding
choreography, kernel caches, precompile helpers).

## Module-level state

- `_chi_minimax_kernel_cache: dict` (line 32) — keyed `(id(mesh_xy), kgrid)`.
- `_w_solve_cache: dict` (line 33) — keyed `(id(mesh_xy), nq, n_rmu)` for JAX-native,
  `("low_mem_fused", id(mesh_xy), nq, n_rmu, dtype)` for FFI.
- Both keyed on `id(mesh_xy)` — id-reuse hazard if a mesh is GC'd and a new object
  lands on the same address (weird_code W1).

## Function-by-function

### `_ensure_compilation_cache()` — lines 38-40
Thin wrapper delegating to `common.jax_compile_cache.ensure_jax_compile_cache()`.
Kept so in-module callers keep working (per comment). External callers:
`ppm_sigma.py:520,573,622` call `w_isdf._ensure_compilation_cache()` directly;
`ppm_pipeline.py:320` imports `w_isdf` "for ensure_compilation_cache + cache hit timings".
`ffi/phdf5/ARCHITECTURE.md:240-244` documents its history (used to activate the cache
only inside w_isdf/ppm_sigma).

### `_get_chi_minimax_kernel(mesh_xy, kgrid)` — lines 47-196
Factory returning cached jitted `minimax_tau_integrate_chi`. Contents:

- Builds three flat-k FFT helpers via `common.fft_helpers.make_flat_k_fftn`:
  `_Gv_fftn`, `_Gc_fftn` (spec `G_FFT7D_SPEC`, norm='ortho') and `_chi_fftn_local`
  (spec `CHI_Q_SPEC`). Note `_Gv_fftn` and `_Gc_fftn` are constructed identically
  (same mesh/kgrid/spec/norm) — two instances of the same helper (weird_code W2).
- Sharding specs imported from `wavefunction_bundle`: `G_FFT7D_SPEC`, `G_FLATK_SPEC`,
  `CHI_Q_SPEC`, `CHI_R_SPEC`, `PSI_XN_SPEC`, `PSI_YR_SPEC`.
- Long convention comment (lines 58-79): historically Gv used ifftn (+ikR) and Gc fftn
  (−ikR) with einsum `'Rambn, Rbnam -> Rmn'` which forced resharding. Now exploits per-k
  Hermitian property `G_k(μ,ν) = G_k(ν,μ)*` ⇒ after FT `G_R(μ,ν) = G_{-R}(ν,μ)*`, so both
  use fftn with a `jnp.conj` at the call site; chi0 einsum collapses to element-wise
  product + spin sum with identical index order, no reshard. "Verified to machine
  precision against the original formulation." chi_R inherits `P(_, 'x', 'y')` aligned
  with V for the W-solve.

#### inner `_build_Gv_Gc(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn, enk_v, enk_c, tau_scalar, vmax, cmin)` — lines 101-124
jit with explicit in/out shardings (ψ_xn spec, ψ_yr spec, replicated 1-D energies,
replicated scalars → `G_FLATK_SPEC` × 2). Physics:
- `phases_v = exp(-τ (vmax - e_v))` → `build_G_tau(psi_v_xn, psi_v_yr, enk_v, -τ, e_ref=vmax)`
- `phases_c = exp(-τ (e_c - cmin))` → `build_G_tau(psi_c_xn, psi_c_yr, enk_c, +τ, e_ref=cmin)`
Returns `jnp.conj(Gv_k), jnp.conj(Gc_k)` — the Hermitian-swap conj, deliberately placed
at the call site "NOT inside build_G_tau" (line 122-123).
Depends on `gw.greens_function_kernel.build_G_tau`.

#### inner `minimax_tau_integrate_chi(nodes, psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn, enk_v, enk_c, vmax, cmin)` — lines 132-191
jit; `nodes` is a `MinimaxNodes(t, alpha)` pytree, both replicated, complex128.
Full τ sweep via `jax.lax.scan` accumulating χ_R, then one R→q FFT — no Python loop.
Docstring: sibling of `ppm_sigma.minimax_tau_integrate_sigma` (same nodes slot).
For chi0, τ is purely real (`time_axis='real'`), α complex with Im=0; α already includes
the chi0 prefactor `-2·α_quad·exp(-τ·E_gap)` (folded upstream in `compute_chi0`).
Scan body:
1. `tau_real = jnp.real(t_scalar).astype(jnp.float64)` — comment ties this cast to
   "the exact numerical path that produced the locked MoS2 3×3 regression hash" (W3).
2. `Gv_k, Gc_k = _build_Gv_Gc(...)`; FFT both k→R.
3. Contraction (VERBATIM einsum):
   `jnp.einsum('Rambn,Rambn->Rmn', Gc_R, jnp.conj(Gv_R), optimize=True)`
   i.e. `chi_R(m,n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))` (a,b = spinor axes).
4. `chi_R_acc + alpha_scalar * chi_tau` (complex·complex; comment notes Im(α)=0 makes
   it hardware-identical to float·complex).
5. Final `_chi_fftn_local(final_R)` — R→q.
`static_argnums=()` on the jit is a no-op (W4). Lines 193-194 comment: "Minimax
quadrature always delivers ≥1 node — the compiled scan handles any n≥1 without a
short-circuit wrapper."

Arrays crossing: ψ_xn/ψ_yr device-sharded per `PSI_XN_SPEC`/`PSI_YR_SPEC` (from
wavefunction_bundle); G arrays 5-D `P(_, _, 'x', _, 'y')`; chi_R `(nk, μ, μ)` in
`CHI_R_SPEC` = `P(None,'x','y')`; output χ₀(nq, μ, μ) in `CHI_Q_SPEC`.

Note: `gw/aot_memory_model/kernels/chi0_tau_step.py` re-implements a single τ step of
this kernel for the AOT memory model — a documented mirror, must be kept in sync
(redundancy R5, by design).

### `_get_w_solve_fn(mesh_xy, nq, n_rmu)` — lines 205-266
Factory for the JAX-native W solve. `W = (I − Vχ)⁻¹V` via q-parallel shard_map.
- `q_shard = P(('x','y'), None, None)`; `reshard_mid = P('x', None, 'y')` — the
  documented two-stage reshard from `P(None,'x','y')` to q-parallel. Comment records a
  measurement: Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh): via fully-replicated
  intermediate peak 2.95 GB/dev (SPMD "Involuntary full rematerialization") vs
  1.11 GB/dev via `P('x',None,'y')` — 62% reduction (deliberate perf hack, W5).

#### inner `_solve_w(V_flat, chi_flat, pref)` — lines 230-263
jit with `donate_argnums=(1,)` — χ₀ is donated; comment ties this to `gw_jax.main`'s
`del chi0_q` inside the `W.exec` timing block. V is NOT donated (reused by COHSEX
Σ_SX/Σ_COH/Σ_X and PPM's Wc = W − V).
Steps: `chi_scaled = pref * chi_flat`; pad nq up to a multiple of device count;
double `with_sharding_constraint` (mid then q_shard) on both V and χ; shard_map'd
`_local_solve` runs a `fori_loop` over local q:
`A = I − V_local[iq] @ chi_local[iq]`, `lu, piv = lu_factor(A)`,
`W[iq] = lu_solve((lu,piv), V_local[iq])`. Unpad, then constrain W to fully
replicated `P(None,None,None)`.

### `_get_w_solve_fn_low_mem(mesh_xy, nq, n_rmu, dtype)` — lines 269-301
Low-mem backend: single fused cuBLASMp + cuSOLVERMp FFI call
(`ffi.cublasmp.batched_fused_w_solve`). Algorithm per docstring:
```
v = X X†                 (cusolverMp potrf)
H = I − X† (pref·χ) X    (2 cublasMp gemms + identity kernel)
L_H = chol(H)            (cusolverMp potrf)
W = X H⁻¹ X†             (2 cublasMp trsms + 1 cublasMp gemm)
```
Requires χ such that I − X†χX is PD (Cholesky, not LU — contrast with the JAX-native
LU path). `pref` must be a Python complex scalar (compile-time FFI attr, not a jnp
array). Inputs constrained to `P(None,'x','y')`; no reshard needed since compute_chi0
now emits that sharding. Stale cross-references elsewhere: `ffi/cusolvermp/batched.py:187,324`
and `ffi/cusolvermp/cpp/batched_solve_lu_ffi.cc:23` still cite "`w_isdf.solve_w_low_mem`",
a function name that no longer exists here (R2).

### `_w_solve_pref_scalar(meta) -> float` — lines 304-311
Dyson prefactor `pref = 2 / (√N_k · n_spin · n_spinor)` with `N_k = meta.nk_tot`,
`nspin = max(1, meta.nspin)`, `nspinor = max(1, meta.nspinor)` (getattr defaults 1).
Variable is named `nq` but holds nk_tot (W6, cosmetic). `scripts/checks/sigma_direct_check.py:436`
mirrors this convention ("Match w_isdf Dyson prefactor convention") with its own
`_solve_w_from_chi_direct` — independent numpy reimplementation for checking (R4, by design).

### `_normalize_screening_solver(solver_or_mode)` — lines 314-329
Accepts `ScreeningSolver` enum or legacy strings "high_mem"/"low_mem"/"auto" via
`gw_config._LEGACY_ISDF_MEMORY_MODE` ("auto"→JAX_NATIVE, "high_mem"→JAX_NATIVE,
"low_mem"→CUBLASMP_FFI). "Kept narrow so the only place legacy strings cross over is here."

### `_resolve_w_solve_fn(meta, mesh_xy, *, solver, dtype, n_rmu)` — lines 332-351
Single dispatch point for JAX-native vs cuBLASMp-FFI fork. Returns `(solve_fn, pref)`:
FFI gets `complex(pref)`, native gets `jnp.asarray(pref, complex128)`. Both `solve_w`
and `precompile_solve_w` route through it (though precompile's FFI branch bypasses the
returned fn — see below).

### `solve_w(V_q, chi0_q, meta, mesh_xy, *, solver=None, memory_mode=None)` — lines 354-379
Public Dyson solve. `W(q) = (I − Vχ₀)⁻¹V`, flat-q (nq,μ,μ). Dual kwargs: new `solver`
enum or legacy `memory_mode` string (`chosen = solver if solver is not None else memory_mode`)
(R1). Wraps execution in `jax_profile.annotation("W_solve")`.
Callers: `gw_jax.py:264` (main driver), `screening.py:162` (`compute_screening`, per
ScreeningRequest role), `common/w_solve_modes_test.py:88,90` (compares high_mem vs
low_mem on random inputs; notes χ donation forces `_fresh()` copies).
`experimental/head_wing_schur.py:130-135` bypasses `solve_w` and imports
`_get_w_solve_fn` directly (private-API reach-in, W7).

### `resolve_minimax_energy_reference(enk_v, enk_c, *, reference="midgap", reference_fn=None) -> float` — lines 382-418
Resolves the minimax band-energy zero: `reference_fn` callback > None→0.0 >
numeric→float > "none"/"raw"/"zero"→0.0 > "midgap"→(VBM+CBM)/2 > "vbm" > "cbm" > raise.
Shift is algebraically neutral for χ₀/W (only E_c−E_v enters) but "keeps reference
conventions explicit and synchronized with sigma paths". Pulls energies to host via
`jax.device_get`. Only caller found by grep across src/tests/tools/scripts:
`build_static_quadrature` in this same file. The `reference_fn` hook has zero users
anywhere (D1).

### `flatten_V_qmunu(V_qmunu)` — lines 425-443
Back-compat shim: 8-D legacy `(1, npol, npol, nkx, nky, nkz, μ, μ)` → `arr[0,0,0].reshape(-1, μ, μ)`;
6-D transitional `(1, npol, npol, nq, μ, μ)` → `arr[0,0,0]`; 3-D flat-q passes through.
Docstring: "In the new flat-q world it's a no-op"; kept only for restart files written
under the old layout. Sole caller `gw_jax.py:198`, whose surrounding comment (line 195)
says V_q is bound directly and flatten is "kept" for back-compat (R3). PERFORMANCE.md:143
notes it emits eager gathers (jit-cache misses), tiny cost.

### `build_static_quadrature(wfns, minimax_config, *, print_fn=None)` — lines 446-459
Builds `(quad, e_ref)`: static minimax quadrature for 1/x on the band-energy interval
via `minimax_screening.build_static_minimax_window_pair`, plus the energy zero from
`resolve_minimax_energy_reference(... reference=minimax_config.energy_reference)`.
Slices `wfns.enk[:, s.val]` / `[:, s.cond]` from the bundle. Callers: `gw_jax.py:237`;
referenced in `sigma_dispatch.py:144` docstring ("once per W solve") and
`screening.py:108` (quad arg provenance).

### `build_imag_quadrature(quad, omega_p, minimax_config, *, print_fn=None)` — lines 462-478
Imag-frequency minimax quadrature for `x/(x²+ωp²)` on the same [x_min,x_max] interval,
via `minimax_screening.solve_laplace_minimax_imag_interval`. Prints R, node count, err.
Consumed by `screening.compute_screening` (screening.py:151) for PPM ω on imag axis.
Flags: `minimax_config.target_error`, `.max_nodes`.

### `build_real_quadrature(quad, Omega, minimax_config, *, print_fn=None)` — lines 481-565
Real-frequency (HL-PPM) χ₀(Ω) quadrature with no new minimax kernel. Decomposition
(verbatim from docstring):
```
x / (x² - Ω²) = (1/2)·[ 1/(x−Ω) + 1/(x+Ω) ] = -(1/2)/(Ω−x) + (1/2)/(Ω+x)
```
Requires `Ω > quad.x_max` (raises otherwise). Two standard 1/y minimax solves via
`solve_laplace_minimax_interval` on shifted intervals:
- (Ω+x) branch: `alpha_plus = +0.5·α·exp(−τ·Ω)`, τ_plus = τ.
- (Ω−x) branch: `1/(Ω−x) ≈ Σ α e^{−τ(Ω−x)} = Σ [α e^{−τΩ}] e^{+τx}` → cast into the
  kernel's `e^{−τ'x}` form by `τ' = −τ`; `alpha_minus = −0.5·α·exp(−τ_raw·Ω)`.
Result: fused `LaplaceMinimaxQuadrature` with mixed-sign τ_l — exactly the
`Σ_l α_l e^{−τ_l x}` form `compute_chi0` consumes. Stability argument documented:
HL regime Ω≈200 Ry, x_max≈5 Ry ⇒ R'≈1.03, 1-3 nodes/branch, residual exponent ≈0.025.
`err_combined = 0.5·(err+ + err−)`. Callers: `screening.py:155`,
`tests/test_real_axis_quadrature.py` (dedicated sanity test, 4 test cases incl. the
Ω ≤ x_max ValueError).
Note the printed "+branch R'" expression `Omega/quad.x_min + quad.x_max/quad.x_min`
(line 561) is not the ratio of the +branch interval, which would be
`(Ω+x_max)/(Ω+x_min)` — looks like a diagnostic-print bug, print-only (W8).

### `compute_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=0.0)` — lines 568-616
Public χ₀ driver. Physics (verbatim docstring):
```
χ₀ = -2 Σ_ℓ α_ℓ Σ_{v,c} |M_vc|² exp(-τ_ℓ (E_c - E_v))
```
where quad approximates 1/x (static) or x/(x²+ωp²) (imag-freq) with x = E_c−E_v.
Steps: activate compile cache; kgrid from `meta.nkx/nky/nkz`; slice enk_v/enk_c;
subtract `energy_reference`; compute host-side vmax/cmin/E_gap; prefold
`alpha_chi = -2 · quad.alpha · exp(-τ·E_gap)` into complex128 `MinimaxNodes`;
fetch cached kernel; call with `wfns.xn(s.val), wfns.yr(s.val), wfns.yr(s.cond),
wfns.xn(s.cond)` + shifted energies + vmax/cmin. Returns flat-q (nq, μ, μ).
Default `energy_reference=0.0` here vs `None` in precompile_chi0 (both coerce None→0.0).
Callers: `gw_jax.py:242`, `screening.py:159`.

### `precompile_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=None)` — lines 619-655
AOT `.lower(...).compile()` twin of `compute_chi0` at real shapes/shardings, to
separate compile from exec in the timing report (`chi0_W.chi.compile` section).
Body copy-pastes compute_chi0's entire setup (eref, vmax/cmin, E_gap, alpha_chi,
nodes) — ~20 duplicated lines (R6). Contains `if len(tau) == 0: return` with comment
"compute_chi0 falls through to a static-zeros path — nothing to compile" — but
compute_chi0 has NO zero-node path, and the kernel factory comment (line 193) asserts
quadrature always delivers ≥1 node: stale comment / dead guard (W9/D2).
Caller: `gw_jax.py:239`.

### `precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, solver=None, memory_mode=None)` — lines 658-690
AOT twin of `solve_w`. FFI branch does NOT use `_resolve_w_solve_fn`'s returned fn:
it builds `ffi.cublasmp.batched_fused_w_solve_jit(dtype, nq, n, pref=complex(...), mesh)`
and lowers `(V_q, chi0_q)` (pref folded into compile-time attrs) — documented as
deliberate ("thin per-solver branch"). Native branch calls `_resolve_w_solve_fn` with
solver hard-set to `ScreeningSolver.JAX_NATIVE` and lowers `(V_q, chi0_q, pref)`.
Also primes the cuBLASMp context handle. Caller: `gw_jax.py:261`.

## Entry points (grep evidence across src/, tests/, tools/, scripts/)

| symbol | callers |
|---|---|
| `solve_w` | gw_jax.main (gw_jax.py:41,264), screening.compute_screening (screening.py:135,162), common/w_solve_modes_test.py:38,88,90 |
| `compute_chi0` | gw_jax.main (gw_jax.py:37,242), screening.compute_screening (screening.py:134,159) |
| `precompile_chi0` | gw_jax.main (gw_jax.py:39,239) |
| `precompile_solve_w` | gw_jax.main (gw_jax.py:40,261) |
| `build_static_quadrature` | gw_jax.main (gw_jax.py:36,237) |
| `build_imag_quadrature` | screening.compute_screening (screening.py:132,151) |
| `build_real_quadrature` | screening.compute_screening (screening.py:133,155), tests/test_real_axis_quadrature.py:39 |
| `flatten_V_qmunu` | gw_jax.main (gw_jax.py:38,198) — only caller |
| `resolve_minimax_energy_reference` | internal only (build_static_quadrature) |
| `_ensure_compilation_cache` | ppm_sigma.py:520,573,622; ppm_pipeline.py:320 (private cross-module use) |
| `_get_w_solve_fn` | experimental/head_wing_schur.py:130 (private cross-module use) |
| `minimax_tau_integrate_chi` (inner) | mirrored (not called) by aot_memory_model/kernels/chi0_tau_step.py |

## Cross-module deps

- `common` (Meta, jax_profile), `common.jax_compile_cache`, `common.fft_helpers.make_flat_k_fftn`
- `gw.minimax_config.MinimaxConfig`; `gw.minimax_screening` (LaplaceMinimaxQuadrature,
  MinimaxNodes, build_static_minimax_window_pair, solve_laplace_minimax_imag_interval,
  solve_laplace_minimax_interval)
- `gw.greens_function_kernel.build_G_tau`
- `gw.wavefunction_bundle` (G_FFT7D_SPEC, G_FLATK_SPEC, CHI_Q_SPEC, CHI_R_SPEC,
  PSI_XN_SPEC, PSI_YR_SPEC; wfns bundle API: .slices, .enk, .xn(), .yr())
- `gw.gw_config` (ScreeningSolver, _LEGACY_ISDF_MEMORY_MODE)
- `ffi.cublasmp` (batched_fused_w_solve, batched_fused_w_solve_jit)

## Flags / config keys consumed

- `minimax_config.energy_reference` ← cohsex.in `minimax_energy_reference` (midgap|vbm|cbm|none|float, default midgap)
- `minimax_config.target_error` ← `minimax_target_error` (default 1e-6)
- `minimax_config.max_nodes` ← `minimax_max_nodes` (default 64)
- `solver` / `memory_mode` ← `config.backend.screening_solver` ← cohsex.in
  `isdf_memory_mode` (auto|high_mem|low_mem; auto & high_mem → JAX_NATIVE, low_mem → CUBLASMP_FFI)
- `meta.nk_tot`, `meta.nspin`, `meta.nspinor` (Dyson prefactor), `meta.nkx/nky/nkz` (FFT grid)

## I/O

None. No file reads or writes. Only side channel is the JAX compilation cache
(activated via `common.jax_compile_cache`) and profiler annotations.

## Suspects

### dead_suspects
- **D1 `resolve_minimax_energy_reference` `reference_fn` kwarg**: grepped
  `resolve_minimax_energy_reference` and `reference_fn` across src/tests/tools/scripts;
  the function's only caller is `build_static_quadrature` in this file, which never
  passes `reference_fn`. Public-API surface with zero external users.
- **D2 `precompile_chi0` zero-node guard (line 638-639)**: `if len(tau)==0: return`
  claims "compute_chi0 falls through to a static-zeros path" — no such path exists in
  compute_chi0 (grep for zeros/short-circuit); factory comment at line 193 says
  quadrature always yields ≥1 node. Dead branch + stale comment.
- **D3 `flatten_V_qmunu` 8-D/6-D branches**: only caller gw_jax.py:198 binds
  the already-flat compute_all_V_q output per its own comment; legacy branches only
  reachable via old-layout restart files.

### redundancy_suspects
- **R1 dual solver kwargs**: `solve_w`/`precompile_solve_w` accept both `solver` enum
  and legacy `memory_mode` string — a parallel old/new API kept alive by
  `_normalize_screening_solver`. Legacy strings still used by `w_solve_modes_test.py`.
- **R2 stale name `solve_w_low_mem`**: `ffi/cusolvermp/batched.py:187,324` and
  `ffi/cusolvermp/cpp/batched_solve_lu_ffi.cc:23` reference `w_isdf.solve_w_low_mem`,
  which does not exist (now `_get_w_solve_fn_low_mem`). Docs drift.
- **R3 `flatten_V_qmunu`**: self-described no-op shim in the flat-q world.
- **R4 `scripts/checks/sigma_direct_check.py:_solve_w_from_chi_direct`**: independent
  numpy re-derivation of the Dyson solve + pref convention (deliberate check harness,
  but a convention fork to keep in sync).
- **R5 `aot_memory_model/kernels/chi0_tau_step.py`**: standalone re-implementation of
  one τ step of `minimax_tau_integrate_chi` for the memory planner; documented mirror
  that must track the real kernel.
- **R6 `precompile_chi0` vs `compute_chi0`**: ~20 lines of setup (eref/vmax/cmin/
  E_gap/alpha_chi/nodes) copy-pasted between the pair; same for the precompile/execute
  twin pattern generally (`precompile_solve_w` partially bypasses `_resolve_w_solve_fn`
  on the FFI branch).
- **R7 `_Gv_fftn` / `_Gc_fftn`**: two identical `make_flat_k_fftn(mesh, kgrid,
  G_FFT7D_SPEC, norm='ortho')` instances (lines 89-90); one would do unless distinct
  compiled instances are intended (not documented as such).

### weird_code
- **W1 (lines 53, 209, 285)**: caches keyed on `id(mesh_xy)` — id can be recycled
  after GC; hypothesis: safe in practice because the mesh lives for the whole run,
  but fragile for refactors that rebuild meshes.
- **W2/W7**: `experimental/head_wing_schur.py` imports private `_get_w_solve_fn`;
  `ppm_sigma`/`ppm_pipeline` call private `_ensure_compilation_cache` — private-API
  reach-ins that constrain renames.
- **W3 (lines 168-173)**: complex→real τ cast justified by "the locked MoS2 3×3
  regression hash" — a numerics-freezing comment; changing the cast path is treated
  as a regression even if algebraically equivalent.
- **W4 (line 143)**: `static_argnums=()` no-op jit arg.
- **W5 (lines 215-223)**: `reshard_mid = P('x',None,'y')` with hard-coded measured
  numbers (2.95→1.11 GB/dev, 62%) — deliberate SPMD-planner workaround for
  "Involuntary full rematerialization"; sensitive to XLA version changes.
- **W6 (line 308)**: `_w_solve_pref_scalar` names the variable `nq` but it is
  `meta.nk_tot`; the prefactor `2/(√N_k·nspin·nspinor)` is a convention magic
  constant matched by sigma_direct_check.
- **W8 (line 561)**: `build_real_quadrature` diagnostic print computes "+branch R'"
  as `Omega/x_min + x_max/x_min`, not the interval ratio `(Ω+x_max)/(Ω+x_min)`;
  print-only, no numerical effect. Hypothesis: typo of the intended ratio formula.
- **W9 (lines 638-639)**: stale "static-zeros path" comment (see D2).
- **W10 (lines 226-229, 261-263)**: `donate_argnums=(1,)` on χ₀ couples this kernel
  to caller behavior (gw_jax's `del chi0_q`; test's `_fresh()` copies) — an implicit
  ownership contract across modules.
- **W11 (lines 539-543)**: sign gymnastics in the (Ω−x) branch (τ'=−τ, −1/2 weight,
  exp uses τ_raw not τ') — correct per the derivation shown, but a classic
  sign-flip hotspot; the dedicated test file exists precisely to pin it.
