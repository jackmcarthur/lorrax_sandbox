# src/gw/aot_memory_model/kernels/ — detailed notes (2026-07-01)

Group: AOT memory-model kernel registry. Each module registers one (or two) `AotKernel`
subclasses via `@register_kernel` (`gw/aot_memory_model/core.py:162`). A kernel = a
mirror/stand-in of a production heavy jit that can be AOT-lowered
(`jit(f).lower(specs).compile().memory_analysis()`) without real data, plus a set of
"primitive" byte-formula functions `_T_*(sys, knobs, mesh)` whose NNLS-fitted coefficients
β predict peak device memory. Some kernels also carry `_F_*` FLOPs primitives (cost model).

**Kernels are never called by physics code at runtime.** Consumption paths:
1. Offline calibration: `python -m gw.aot_memory_model.sweep --kernel <name> --preset <p>`
   (sweep.py imports `kernels` package to populate registry; presets.py has a
   `points_<kernel>` DoE generator per kernel; fits land in
   `src/gw/aot_memory_model/artifacts/<kernel>__<tag>__{fit,samples}.json`).
2. Offline prediction: `python -m gw.aot_memory_model.predict_cli --kernel <name>`.
3. Online production: `gw/gw_init.py:431-498` (`compute_optimal_chunks` wiring) calls
   `predict_kernel_peak("fit_one_rchunk", ...)` and `chooser.choose_chunks_analytic/aot`
   (chooser default `kernel_name="fit_one_rchunk"`). **Only `fit_one_rchunk` is consumed
   online**; every other kernel is offline-calibration-only.

All artifacts exist for every kernel (`artifacts/` has `__current__fit.json` for all 11
registered names; fit_one_rchunk also has `__ortho__`, `__prev__`, and a cost_fit).
No tests import `aot_memory_model` (grepped tests/ for `aot_memory_model|get_kernel|
predict_kernel_peak`: zero hits).

**Systemic redundancy risk (applies to the whole group except fit_one_rchunk):** each
kernel *re-implements* the production jit body by hand instead of calling the production
factory, so the copies can silently drift from production sharding/algebra.
chi0_tau_step even says so explicitly ("must update in lockstep"). fit_one_rchunk is the
one exception — it imports and calls `common.isdf_fitting._make_fit_one_rchunk_kernel`
directly. A refactor should either route all kernels through production factories
(fit_one_rchunk-style) or accept the drift risk deliberately.

Other group-wide notes:
- `_B = 16.0` (bytes/complex128) re-defined in every file.
- `sys.kgrid` field is overloaded: k-grid in most kernels, but repurposed as the 3-D
  **FFT box** in `vq_mu_chunk` (explicit comment) and effectively in `load_psi_rchunk`
  (build_specs/build_callable compute `n_r = prod(sys.kgrid)`).
- None of these files do file I/O themselves; JSON artifact read/write lives in core.py.

---

## __init__.py (11 loc)

Pure side-effect registry: imports the ten kernel modules so `register_kernel` populates
the registry dict. Imported by `core.get_kernel` (line 172) and `sweep.py:113`.
No dead code.

---

## fit_one_rchunk.py (488 loc)

**Purpose.** Driver-level composite AOT kernel for the per-r-chunk zeta-fit body —
mirrors `common.isdf_fitting.fit_one_rchunk` (streaming shard_map+scan pair-density,
ZCT, reshard, Cholesky solve in one HLO). Models *coexisting* buffers that the per-stage
kernels cannot see. The only kernel consumed by the online chunk chooser
(`gw_init.compute_optimal_chunks`, `chooser.choose_chunks_analytic/aot`).

**Functions/classes**
- `_AotStubPsiGStore` — stand-in for `common.psi_G_store.PsiGStore`; `_slice_local_tile_bc`
  returns a zeros slab `(nk, _bpd_max, ns, ngkmax)`; `g_index`/`kvecs_frac` lazy
  device-put zeros; `begin_rchunk/end_rchunk/close` no-op interface stubs.
- Knob accessors: `_Br` (chunk_r), `_bc` (band_chunk, default 16), `_nbc` (# band chunks).
- Memory primitives (in `PRIMITIVES`): `_T_pair`, `_T_psiG_bc`, `_T_psiY_bc`,
  `_T_centroid`, `_T_Lq_sharded`.
- Memory primitives **defined but removed from PRIMITIVES** (dead): `_T_psiG_cache`
  (lines ~193-200; removal reason: post-io_callback refactor there is no resident ψ(G)
  device cache — leaving it extrapolated a phantom ~3.3 GB), `_T_Lq_rep` (lines ~244-248;
  HLO audit showed no fully-replicated L_q buffer; was absorbing collinear residual →
  phantom 3.7 GB at Si 10³).
- FLOPs primitives: `_F_pair_density`, `_F_bc_fft`, `_F_zct`, `_F_solve`; `_F_fixed`
  defined but not in `FLOPS_PRIMITIVES` (self-described "kept here for naming only").
- `FitOneRChunkKernel(AotKernel)` — `PRIMITIVE_CLASSES` maps primitives to
  (const/cr/bc/crbc) scaling classes consumed by the analytic chooser's closed-form
  feasibility inversion. `build_specs` = 9-arg production `_kernel` signature
  (psi_l_X, psi_r_X, L_q, norms×2, r_start, gamma_perm(4,), gamma_phase(4,),
  cct_trace_per_q). `build_callable` calls the **production factory**
  `common.isdf_fitting._make_fit_one_rchunk_kernel` with a synthetic
  `common.load_wfns.Meta` and the stub PsiGStore, so it tracks production bit-for-bit.

**Deps.** common.isdf_fitting, common.load_wfns.Meta, ..core; jax.experimental.io_callback
and shard_map imported but **unused in code** (docstring references only).

**Weird / suspects**
- Unused imports: `io_callback` (l.65), `shard_map` (l.66), `MeshSpec` (l.69, specs use
  jax Mesh), `partial` (l.60).
- Dead in-file helpers: `_T_psiG_cache`, `_T_Lq_rep`, `_F_fixed` (grepped src/ — only
  comment references in presets.py:300,348).
- Duplicate statement: `kx, ky, kz = sys.kgrid` at both l.431 and l.450.
- Magic default: `ngkmax_aot = sys.n_g or max(1, n_r_total // 16)` — //16 "keeps AOT
  conservative", empirical ratio.
- Synthetic `Meta` populated with `cell_volume=1.0`, `npol=1`, zeros elsewhere — relies
  on the factory only reading a documented subset of fields; fragile if factory grows.
- γ̃ pair (`gamma_perm/gamma_phase`) always passed even for charge channel where body
  resolves them to None at trace — spec/arg coupling to `gamma_matrices.gamma_perm_phase`.
- Extensive tombstone comments for removed primitives (good history, but the dead
  function bodies should go with them in a refactor).

---

## load_psi_rchunk.py (221 loc) — registers TWO kernels

**Purpose.** Models the two production sub-jits inside
`common.load_wfns.get_sharded_wfns_rchunk_slice`: (1) `load_psi_rchunk_fft` =
IFFT3D + k·r phase + contiguous r-slice, band-sharded XY output; (2)
`load_psi_rchunk_reshard` = {-,XY,-,-} → {-,X,-,Y} → {-,-,-,Y} reshard (all_gather-x +
all_to_all-y), whose stage-Y intermediate is the binding peak. Split into two jits in
production so SPMD doesn't rematerialize the FFT; modeled separately here for the same
reason.

**Functions**
- `_k`, `_b` — effective k_chunk / band-count (nb_pad overrides band_chunk) per jit call.
- Primitives: `_T_fft_full`, `_T_psi_G` (documented as **exactly collinear** with
  fft_full — identical formula; NNLS puts all weight on one), `_T_rchunk_xy`, `_T_rchunk_y`.
- `LoadPsiRchunkFftKernel.build_callable` — hand-built `_fft_and_rslice` using
  `common.fft_helpers.make_jittable_local_ifftn_3d`, zero kvecs (phase = 1), √n_r norm,
  shard_map'd `dynamic_slice_in_dim` with `r_start_arr=jnp.array([0])`.
- `LoadPsiRchunkReshardKernel.build_callable` — one-liner jit with
  `with_sharding_constraint` to stage `P(None,'x',None,'y')` and
  `out_shardings=P(None,None,None,'y')`; comment ties it to the 2026-04-21 reshard
  audit ("y-first ordering + out_shardings forces the final all_gather-x").

**Deps.** common.fft_helpers, ..core.

**Weird / suspects**
- `sys.kgrid` used as the FFT box: `nx,ny,nz = sys.kgrid; n_r = nx*ny*nz` in both
  build_specs and build_callable (kernel 1), while the `_T_*` primitives use `sys.n_r`.
  Consistent only if presets set kgrid = fft_grid and n_r = prod — same field-overloading
  hazard as vq_mu_chunk, undocumented here.
- Spec G-dim = n_r ("upper bound of n_gmax"; fit coefficient absorbs the ~4-8× ratio) —
  deliberate over-bound baked into calibration.
- Unused imports: `numpy as np`, `MeshSpec`, `shard_map` is used (kernel 1) — ok.
- Reshard kernel re-implements rather than importing the production
  `common.load_wfns` jit — drift risk (production form is described, not shared).

---

## chi0_tau_step.py (177 loc)

**Purpose.** AOT model of the chi0 per-τ-step jit — the OOM bottleneck at Si 4×4×4
60 Ry / 16 GB. Mirrors `gw.w_isdf.minimax_tau_integrate_chi` (`_get_chi_minimax_kernel`):
build Gv/Gc Green's functions with τ-exponential phases, flat-k IFFT/FFT to R, einsum
`'Rambn,Rbnam->Rmn'`, accumulate into donated `chi_R_acc`.

**Functions**
- Primitives: `_T_Gbuf` (nk·ns²·μ²/P — the dominant one; handoff profile shows β≈4),
  `_T_chi` (donated, β≈0 expected), `_T_psi` (sum of all four ψ input shards, x+y copies).
- `Chi0TauStepKernel.build_specs` — 11 args (chi acc, 4 ψ arrays, 2 enk, τ, prefactor,
  vmax, cmin). `KNOBS = ()` — no chunk knob exists in the implementation.
- `build_callable` — hand-copies production `_tau_step` (docstring: "duplicates the
  construction pattern ... because that factory returns only the full compiled scan
  ... must update in lockstep — the validation point (gamma calibration vs runtime)
  will catch drift"). Uses `gw.greens_function_kernel.build_G`,
  `common.fft_helpers.make_flat_k_{i,}fftn`, `donate_argnums=(0,)` matching production.

**Deps.** common.fft_helpers, gw.greens_function_kernel, ..core.

**Weird / suspects**
- Explicit acknowledged copy-paste of production `_tau_step` (redundancy by design;
  candidate for refactor: have `w_isdf` expose `_tau_step` from its factory).
- Header comments carry an empirical allocation dump (21.97 GiB = 4×Gbuf etc.) keyed to
  a specific run dir — useful but stale-prone.
- Unused imports: `numpy as np`, `MeshSpec`, `partial` IS used (l.177).
- Gv/Gc conjugation (`jnp.conj`) and opposite x/y shardings for Gv vs Gc
  (`_Gv_out_flatk` P(None,None,'x',None,'y') vs Gc swapped) — physics-convention detail
  that must match production exactly; no cross-check besides γ calibration.

---

## sigma_kij.py (137 loc)

**Purpose.** AOT model of the Σ_kij kernel: G(R)·W(R)/√Nk in real space plus the
two-stage reduce-scatter tail projection. Mirrors `gw.ppm_sigma._get_sigma_kij_kernel`.

**Functions**
- `N_RS_STAGES = 4` module constant (re+im branches × 2 psum_scatters) — **defined but
  never referenced anywhere** (grepped whole src/: only its own definition + comment).
  Documentation-as-code; also oddly placed *between the docstring and the imports*.
- Primitives: `_T_Gmid` (nk·ns²·μ²/P), `_T_Vmid` (nk·μ²/P), `_T_psi_X`, `_T_psi_Y`,
  `_T_rs_left` (**identical formula to `_T_psi_Y`** — collinear pair), `_T_rs_mn`
  (nk·nb²·μ/P; note docstring says (nk, m/p_x, n, μ/p_y) so formula includes μ — check
  against docstring shape which reads as nb·nb·μ bytes: formula has n_rmu factor, matches).
- `SigmaKijKernel.build_callable` — hand-built `_sigma_kij_kernel` using production
  helpers `gw.greens_function_kernel.build_G`, `gw.ppm_sigma._make_project_ri_reduce_scatter`,
  flat-k FFT helpers; `inv_sqrt_nk = -1.0/√nk` (note the **minus sign** — Σ sign
  convention folded into the normalization constant).

**Deps.** common.fft_helpers, gw.greens_function_kernel, gw.ppm_sigma, ..core.

**Weird / suspects**
- Dead constant `N_RS_STAGES` (and placed before imports — lint-hostile).
- `_T_rs_left` byte-identical to `_T_psi_Y` (redundant primitive; NNLS cannot split).
- `-1.0/sqrt(nk)` sign flip embedded in a "normalization" variable named `inv_sqrt_nk`.
- Body is a partial re-implementation (imports production sub-factories but rebuilds the
  outer jit) — moderate drift risk.
- Docstring: "n_b represents m_coh == n_proj for this POC" — POC-level simplification;
  production m_coh ≠ n_proj generally.

---

## vq_mu_chunk.py (87 loc)

**Purpose.** AOT model of one V_q μ-chunk: ζ(r) FFT → weight by √v(G) → self-contract
V[μ,ν]. Mirrors `make_v_munu_chunked_kernel.fft_weight_contract_diag` (used by
`compute_all_V_q_from_zeta_h5`). Single-GPU scale (p_x = p_y = 1); tests whether
MEMORY_MODEL.md's "3 concurrent μ_chunk×n_r buffers" concurrency is real.

**Functions.** Primitives `_T_zeta`, `_T_vphase`, `_T_out`; `VqMuChunkKernel` with knob
`mu_chunk`; `build_callable` hand-builds `_fft_weight_contract_diag` using
`common.fft_helpers.make_sharded_fftn_3d` (replicated box).

**Weird / suspects**
- Explicit field overload: "we reuse the SysDims.kgrid field as the 3-D FFT box here"
  (comment at l.65-68) — SYSTEM_DIMS lists `fft_grid` but code reads `sys.kgrid`.
- `sys.kgrid if sys.kgrid else (1,1,1)` fallback silently degrades to a 1-point box.
- Unused import `partial`.
- Re-implementation of the production inner kernel (drift risk).
- `n_G` computed then only used in reshape — fine, but `norm=None` on FFT vs production
  normalization is unverified here.

---

## slab_write.py (85 loc)

**Purpose.** AOT model of one shard_map'd SlabIO FFI write
(`file_io._slab_io_ffi._FfiBackend.write_slab`) — the MPI-IO collective HDF5 write of a
zeta (nq, μ, B_r) slab. FFI itself can't be AOT-lowered; kernel models only the per-rank
slab footprint. The async queue holding (K+2)×slab across writes is a Python-level
effect handled by MEMORY_MODEL.md's orchestrator meta-formula, which consumes this
kernel's β.

**Functions.** Primitive `_T_slab` (16·nq·μ·B_r/P). `SlabWriteKernel.build_callable`
returns a jit'd shard_map of `_per_rank_identity` with `donate_argnums=(0,)`,
`check_rep=False`.

**Weird / suspects**
- Anti-DCE trick: `return A_local + jnp.zeros_like(A_local)` — "prevents it from being
  elided as dead" (l.73-77). Fragile against smarter XLA simplification.
- `_T_slab` uses `sys.n_k` for the q-count (n_k≡n_q assumption, Γ-centered grid; shared
  with several kernels but implicit).
- `knobs.get("chunk_r", sys.n_r or 1)` fallback-to-1.
- Unused import `partial`.
- This kernel models an I/O op but performs no I/O; the real format
  (zeta_q HDF5 via SlabIO FFI / MPI-IO) lives in file_io.

---

## zct_lr.py (78 loc)

**Purpose.** AOT model of ZCT_LR: Z_q(μ,r) = FFT[conj(IFFT(P_l)) ⊙ IFFT(P_r)] over the
k-grid, for one r-chunk. Mirrors `common.isdf_fitting.compute_ZCT_from_left_right_zchunk`
including the production two-sub-jit split (donation through the IFFT). Verifies
MEMORY_MODEL.md's "4 pair-sized temps + 1 output" concurrency claim.

**Functions.** Single primitive `_T_PrBr` (16·nk·μ·B_r/P). `ZctLrKernel.build_callable`
builds `_left_ifft_conj` (donate 0) + `_right_ifft_mul_fft` (donate 0,1) and wraps in an
outer jit with `donate_argnums=(0,1)`; uses `common.fft_helpers.make_flat_k_{i,}fftn`
with `norm="forward"`.

**Weird / suspects**
- Nested jits inlined by an outer jit — donation semantics through inlined sub-jits are
  subtle; the comment asserts JAX "traces the two sub-jits and inlines them" (behavior
  the model depends on).
- Re-implementation of the production routine (drift risk).
- Only one primitive → all concurrency lands in one β (fine for NNLS but no
  decomposition).
- Unused: `Mesh` import? (`Mesh` used in type hints only via signature default — actually
  unused symbol along with `partial` IS used). `MeshSpec` unused.

---

## solve_q.py (77 loc)

**Purpose.** AOT model of the batched Cholesky triangular solve
zeta_q = L^{-H} L^{-1} Z_q across all q. Mirrors `common.isdf_fitting._solve_all_at_once`
(inside `solve_zeta_from_L_q`): L_q replicated per-device inside a shard_map, Z_col
sharded on the flattened ('x','y') last axis.

**Functions.** Primitives `_T_Zcol`, `_T_Lrep` (q_chunk replicated panels), `_T_Lfull_rep`
(full nq·μ² replicated). `SolveQKernel.build_callable` — shard_map'd vmap of double
`solve_triangular` with `with_sharding_constraint` to replicate L first.

**Weird / suspects**
- `_T_Lrep` with default `q_chunk = sys.n_k` is **identical** to `_T_Lfull_rep` unless
  the knob is set — near-duplicate primitives, collinear at default (NNLS can't split);
  presets must set q_chunk to break it.
- Full L_q replication on every device (`P(None,None,None)` in_spec) is the modeled
  production behavior — a known memory hot spot worth flagging for the refactor itself.
- `knobs.get("chunk_r", sys.n_r or 1)` fallback-to-1; `Mesh`, `MeshSpec`, `jnp` unused
  imports (jnp unused — only jax.scipy used).
- Re-implementation of production routine (drift risk).

---

## pair_density.py (63 loc)

**Purpose.** AOT model of the traced pair density P_k(μ,ν) = Σ_{n,s} ψ*ψ — a single
einsum `'kmns,knsv->kmv'`. Mirrors `common.isdf_fitting._compute_P_traced`.

**Functions.** Primitives `_T_P`, `_T_psiL`, `_T_psiR`; `PairDensityKernel`
(registry name `pair_density_traced` — **class/file name vs registry name mismatch**,
grep for "pair_density" alone misses the registry key). `build_callable` is a one-line
jit'd einsum with explicit in/out shardings.

**Weird / suspects**
- Registry name `pair_density_traced` ≠ module name `pair_density` (minor discoverability
  trap; presets function is `points_pair_density_traced`).
- Note: this models the OLD non-streaming pair density; the production fit path now
  streams band-chunks inside fit_one_rchunk (see fit_one_rchunk docstring "Round 6
  streaming-scan rewrite"). This per-stage kernel may be a stale-stage model kept for
  calibration history — candidate for retirement if the standalone `_compute_P_traced`
  path is gone from production (verify against common/isdf_fitting.py).
- Unused import `Mesh`? (used in signature annotations only), `MeshSpec` unused.

---

## cct_lr.py (62 loc)

**Purpose.** AOT model of CCT_LR: C_q(μ,ν) = FFT[conj(IFFT(P_l)) ⊙ IFFT(P_r)] over the
k-grid on full μ×μ pair densities. Mirrors `common.isdf_fitting.compute_CCT_from_left_right`
(`_compute_CCT_LR` inner jit with donate_argnums=(0,1)).

**Functions.** Single primitive `_T_Pq`; `CctLrKernel.build_callable` — jit with
donation, two IFFTs + conj-multiply + FFT via `make_flat_k_{i,}fftn(norm="forward")`.

**Weird / suspects**
- **Docstring/code mismatch:** header lists two primitives `T_Pq` and `T_fft`
  ("cuFFT workspace lives here", expected β[fft]≈1) but `PRIMITIVES = {"Pq": _T_Pq}`
  only — T_fft was dropped (byte-identical to T_Pq) without updating the docstring.
- Near-twin of zct_lr (same algebra, μ×μ vs μ×B_r shapes, single-jit vs two-sub-jit
  structure) — inherent to modeling two production routines that are themselves near
  twins; refactor of production could collapse both.
- Unused imports `Mesh` (annotation only), `MeshSpec`.

---

## Cross-cutting refactor observations

1. **Mirror-copy drift** is the group's defining hazard: 9 of 10 kernel bodies re-derive
   production jits by hand. Only fit_one_rchunk consumes the production factory. If
   production factories were refactored to expose their inner step functions
   (e.g. `w_isdf` exposing `_tau_step`, `isdf_fitting` exposing the ZCT/CCT/solve jits),
   every kernel here could become a thin spec-provider and the lockstep-maintenance
   comments could be deleted.
2. **Per-stage kernels vs composite:** fit_one_rchunk's docstring explains the per-stage
   kernels "cannot model" coexisting buffers; pair_density/zct_lr/solve_q/cct_lr model
   stages that are now *inside* the composite fit_one_rchunk jit. Their standing value is
   (a) DoE calibration of small-integer β concurrency claims from MEMORY_MODEL.md and
   (b) history. Candidates for pruning if the composite + chi0/sigma/load/slab set covers
   the planner's needs.
3. **Collinear primitive pairs** deliberately kept for naming (psi_G/fft_full,
   rs_left/psi_Y, Lrep/Lfull_rep, and the removed Lq_rep) — a refactor could formalize
   "primitive aliases" instead of duplicating formulas.
4. **knob vocabulary** (chunk_r, band_chunk, k_chunk, nb_pad, mu_chunk, q_chunk) is
   stringly-typed via `knobs.get`, with per-file re-implementations of default/clamp
   logic (`_Br`, `_bc`, `_k`, `_b`) — small consolidation win.
