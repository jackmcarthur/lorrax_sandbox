# src/gw/v_q_tile.py — deep-read notes (gw_refactor_map 2026-07-01)

**LOC:** 1662. **Category:** physics: bare-Coulomb V_q(μ,ν) construction stage (ISDF centroid basis), with an embedded per-kernel memory planner.

**Purpose.** Single source of truth for building the bare-Coulomb matrix in the ISDF
centroid basis, `V_q(μ,ν) = Σ_G conj(ζ_{q,μ}(G)) · v(q+G) · ζ_{q,ν}(G)`, plus the head
vector `g0_μ(q) = ζ_{q,μ}(G=0)`. Module explicitly documents itself as the unification
that replaced the old `_make_V_q_caseA_kernel` / `_make_V_q_caseB_kernel` pair and the
two outer drivers `_compute_all_V_q_sharded` / `_compute_all_V_q_replicated` in
`compute_vcoul.py`. Handles scalar-charge V^{0,0} (`same_zeta=True`) and bispinor
V^{μ_L,ν_L} L≠R tiles (`same_zeta=False`, signed transverse-projector weight folded
into `v_per_G_fn`), Case A (q-batched, full μ) and Case B (single q, μ×ν tiled), plus
IBZ→full-BZ unfold helpers for the transverse tensor tile and g0.

---

## Function-by-function

### Module constant `_Q_COMPUTE_COEF_FFT` (L84)
`float(os.environ.get('LORRAX_V_Q_FFT_COEF', '5.0'))`. Flat heuristic coefficient for
the FFT-stage memory footprint (cuFFT scratch guess), doubled to effectively 10× for
distinct-ζ tiles (L372). Only the fallback when the AOT model is unavailable.

### `_gather_stage_per_q_bytes(n_rmu, n_G, p_x, p_y)` — L87–108
Analytical per-rank per-Q peak of the gather+contract stage:
`16 · n_rmu · n_G · (1/p_x + 1/p_y + 1/(p_x·p_y))` — "three slabs alive at contract
time" (post-sphere ζ_G on P_prod + one-axis-gathered ζ_μ_X and ζ_ν_Y). Comment records
this replaced the legacy empirical `_Q_COMPUTE_COEF_GATHER = 4.4`. Callers: internal
only (`_choose_v_q_chunks`, 3 call sites L374/413/461).

### `_aot_full_kernel_peak(...)` — L115–233
AOT-compiles the *full* production kernel via `_make_V_q_tile_kernel(...).lower(*specs)
.compile(compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})`, then reads
per-rank peak = XLA `memory_analysis()` buffers + cuFFT plan scratch through
`runtime.aot_memory.aot_kernel_peak_bytes`. Cached in `_v_q_full_kernel_aot_cache`
keyed by full shape tuple. Returns `None` on any exception (bare `except Exception`).
Verbose print gated on env `LORRAX_V_Q_AOT_VERBOSE`. Caller: `_choose_v_q_chunks`
(shrink-retry loop L515). Cost ~1–2 s compile per unique shape.

### `_drop_unused_v_q_kernel_cache_entries(keep_cache_keys)` — L236–255
Scrubs `_v_q_tile_kernel_cache` entries not in `keep_cache_keys` — cleans up kernels
compiled for rejected q_chunk candidates during shrink-retry. Note admits JAX's own
pjit compile cache still holds the artifacts. Caller: `_choose_v_q_chunks` L549.

### `_aot_fft_model(n_rmu, fft_grid, mesh_xy, same_zeta)` — L258–323
Measures the FFT stage alone: jits `make_sharded_fftn_3d` (same primitive as the
production kernel, explicitly NOT `make_jittable_local_fftn_3d`), compiles at Q=1 and
Q=2, returns `(slope, intercept)` s.t. `peak_fft(Q) = slope·Q + intercept`.
`slope *= 2` when `same_zeta=False`. Cached in `_v_q_aot_cache`. Returns None on
failure. Caller: `_choose_v_q_chunks` L384.

### `_choose_v_q_chunks(...)` — L326–557
The memory-model chooser. Picks `(q_chunk, mu_chunk)` given `budget_bytes`:
1. Reserve accumulators: `v_ref = 16·n_q·μ²/p_prod`, `g0_ref = 16·n_q·μ/p_x`.
2. Per-q cost = `max(gather_stage, fft_stage)`, FFT stage preferring the AOT
   slope/intercept model over the flat `_Q_COMPUTE_COEF_FFT` heuristic.
3. If even one q doesn't fit → **Case B**: q_chunk=1, μ-tiled; picks μ_chunk that
   divides `gcd(N_μ/P_x, N_μ/P_y)` AND is a multiple of `P_x·P_y` ("aligned"), else
   snaps down to a multiple of `p_x·p_y`.
4. Else **Case A**: maximize q_chunk, then verify with `_aot_full_kernel_peak` and
   shrink-retry (≤4 attempts, factor 0.7×) until predicted peak ≤ budget; scrub the
   losing kernel-cache entries.
Returns dict `{q_chunk, mu_chunk, n_mu_blocks, tiled, aligned, per_rank_peak,
ref_bytes}`. Callers: `compute_V_q_tile` (internal, when `chooser_choice is None`);
re-exported by `compute_vcoul.py:518` "for tests/other modules" — **grep of
src/tests/tools/scripts finds zero consumers of that re-export** (dead re-export).

### `_make_V_q_tile_kernel(...)` — L567–783 (cache `_v_q_tile_kernel_cache`, L564)
Unified r-space inner kernel builder. jit-compiled, donated accumulators
(`donate_argnums=(0,1)`), cached by
`('unified', id(mesh_xy), q_chunk, mu_size, nu_size, n_rmu_L, n_rmu_R, n_G_sph,
fft_shape, id(sphere_idx), same_zeta, write_g0)`.

Inner helper `_zeta_disk_to_G(zeta_disk, qvec_batch_frac)` (L664–694):
ζ_q,μ(r) → ζ_q,μ(G): transpose disk layout `P(None,None,('x','y'))` →
`P(None,('x','y'),None)`, reshape to `(Q, μ/rank, nx, ny, nz)`, apply separable Bloch
phase `exp(-2πi q·r)` via `common.wfn_transforms.apply_bloch_phase(..., sign=-1)`
(three 1D-broadcast multiplies — scratch `Q·(nx+ny+nz)` not `Q·nx·ny·nz`), sharded 3D
FFT (`make_sharded_fftn_3d`), flatten, `g0_blk = zeta_box[:, :, 0]`, sphere gather
`jnp.take(zeta_box, sphere_idx, axis=-1)`.

Kernel body `_kernel_body` (L704–740):
- physics: `V_block(μ,ν) = Σ_G conj(ζ_L(G)) · v(K) · ζ_R(G)`; one-sided v(K) multiply
  on the L side (handles signed transverse weights; "bit-identical to symmetric-√v for
  V^{0,0}").
- einsum VERBATIM: `jnp.einsum('qmG,qnG->qmn', jnp.conj(zeta_mu_X), zeta_nu_Y,
  optimize=True)` with ζ_μ resharded to `P(None,'x',None)`, ζ_ν to `P(None,'y',None)`,
  V_block to `P(None,'x','y')`.
- `dynamic_update_slice` into V_acc at `(q_lo, mu_lo, nu_lo)`; optional g0 DUS at
  `(q_lo, mu_lo)` when `write_g0` (asserted `same_zeta` only).
Two jit signatures (same/distinct ζ, arg-count difference only). Attaches convenience
shardings `.zeta_disk_sh/.V_sh/.g0_sh`.

Key arrays: ζ_disk `(Q, n_rtot, μ_size)` c128 device `P(None,None,('x','y'))`;
v_per_G `(Q, n_G_sph)` c128 replicated `P(None,None)`; qvec_frac `(Q,3)` f64;
V_acc `(nq_total, n_rmu_L, n_rmu_R)` c128 `P(None,'x','y')` donated;
g0_acc `(nq_total, n_rmu_L)` c128 `P(None,'x')` donated.

### `_make_V_q_tile_kernel_Gflat(...)` — L789–938 (cache `_v_q_tile_gflat_kernel_cache`, L786)
G-flat input variant: identical math, but FFT + Bloch phase + sphere gather happen
upstream in `file_io.zeta_reader.ZetaReader.read_zeta_G_slab`. Kernel = shard casts +
one-sided v(K) multiply + same VERBATIM einsum `'qmG,qnG->qmn'` + DUS. g0 comes from
`zeta_L_G[:, :, int(g0_sphere_idx)]` (driver always passes `g0_sphere_idx=0`). Cache
key tagged `'gflat'`, includes g0_sphere_idx only when write_g0 (else −1).

### `_make_g0_dummy(mesh_xy, nq_total, n_rmu_L)` — L941–955
Sharded zero g0 slab reused across the tile loop for tiles without a head, so the
donated position-1 argument always exists. Caller: `compute_V_q_tile` L1196.

### `compute_V_q_tile(...)` — L963–1434  **[main public entry point]**
Outer driver covering V^{0,0} + bispinor tiles, Case A/B, PHDF5/allgather backends,
r-space and G-flat read paths.

Callers (grep evidence):
- `src/gw/compute_vcoul.py:517` `from .v_q_tile import compute_V_q_tile as
  _compute_V_q_tile`, called at `compute_vcoul.py:1054` (the scalar/charge pipeline;
  passes `bgw_v_grid_overlay_fn`, `q_list_kgrid_int`, `use_g_flat_zeta`).
- `src/gw/v_q_bispinor.py:53` import, called at `:396` per (μ_L,ν_L) tile (BGW overlay
  only for the (0,0) CC tile, `v_q_bispinor.py:381`).
- `use_g_flat_zeta=True` originates at `gw_init.py:1115` → `compute_vcoul.py:1072`.

Flow:
1. q-list: default full-BZ nested loop over kgrid in canonical flat order; or caller's
   `q_list_kgrid_int` (IBZ subset via `SymMaps.irr_idx_q/sym_idx_q`; disk zeta_q must
   be indexed in the same order).
2. μ-padding invariant (L1100–1126): logical centroid counts rounded up to
   `p_x·p_y`-divisible (`_round_up_to_mesh`); trailing pad zero-filled via SlabIO
   `valid_shape=`; V_acc returned at PADDED extent (mirrors Meta.b_id_4_user pattern,
   comment cites common/meta.py:99-100 + load_wfns.py:952-959).
3. Chooser (or caller-supplied `chooser_choice`); env `LORRAX_V_Q_Q_CHUNK` force
   override for Case A (L1156).
4. Allocate/donate V_acc, g0_acc (g0 auto-allocated for same_zeta; dummy otherwise).
5. `_qvec_wrap` (L1202): BGW wrap `q > kgrid/2 → q − kgrid` (referenced from
   `common/isdf_fitting.py:2194`). `_v_qvec_batch` (L1206): builds
   `v_per_G = v_per_G_fn(qvec_np)` + optional `bgw_v_grid_overlay_fn` (byte-reproducible
   BGW `use_bgw_vcoul=true` comparisons) + fractional qvec for apply_bloch_phase.
6. Nested loop q_batches × mu_blocks × nu_blocks; `on_diag` (same_zeta ∧ same block) →
   single-read path; else second ζ_R read. Reads via
   `SlabIO.read_slab('zeta_q', shape=(Q, n_rtot, μ), offset=(q_flat0, 0, mu_lo),
   partition_spec=P(None,None,('x','y')))` or
   `ZetaReader.read_zeta_G_slab(q_offset, q_count, mu_offset, mu_count,
   qvec_batch_frac, sphere_idx, mesh, valid_mu)`. No `.block_until_ready()` between
   iterations — read/compute overlap by async dispatch.
7. Env `LORRAX_V_Q_TIME_STAGES=1` → per-stage host-blocked timing breakdown.
Returns `(V_acc, g0_acc_or_None)` (padded μ extent).

### `_unfold_v_q_ij_ibz_to_full(V_q_ij_ibz, ...)` — L1452–1558
Transverse-channel IBZ→full-BZ unfold. Physics (verbatim from docstring):
`v_ij(RK) = R_{ia} R_{jb} v_ab(K)`, so
`V^{ij}_{Sq}(π_S μ, π_S ν) = Σ_{a,b} R^{ia}(S) R^{jb}(S) V^{ab}_q(μ, ν)`; τ-phases
cancel by bilinearity; TRS conj irrelevant "because the kernel itself is even in K".
Steps: q-gather by `full_to_irr_idx`; centroid double-permute via
`inv_perm = argsort(sym_perm)` with `jnp.take_along_axis(..., mode='promise_in_bounds')`
(workaround for XLA s32/s64 take_along_axis verifier failure under shard_map+x64,
comment cites commit `49b7f84`); polarization mixing einsum VERBATIM:
`jnp.einsum('qia,qjb,qabmn->qijmn', R_q, R_q, V_perm_nu, optimize=True)`.
In/out: `(n_q_ibz|n_q_full, 3, 3, μ, μ)` c128 `P(None,None,None,'x','y')`.
Callers: **ONLY `tests/test_v_q_transverse_unfold.py:30`** — zero production callers
(grepped src/tests/tools/scripts for the name). Consistent with the known finding that
the transverse IBZ-unfold gives wrong in-plane Σ^B (CrI3 C3 gate) and production moved
to full-BZ-direct transverse; this is a dead-in-production Phase-D path.

### `_unfold_g0_ibz_to_full(g0_ibz, ...)` — L1561–1662
g0 IBZ→full-BZ unfold. g0 transforms as a single ζ leg:
`g0_full[q, π_s(μ)] = e^{-i(Sq)·τ_s} · g0_ibz[i(q), μ]`, but the τ-phase is
deliberately **omitted** — docstring argues the only downstream consumer is the Γ slot
in `head_correction.py` where S=identity ⇒ phase=1. TRS rows get plain conjugation
`g0 → conj(g0)` gated on `n_sym_spatial`; hard-fail guard if `full_to_irr_sym`
contains TRS-augmented indices but sym_perm lacks the `extend_trs=True` rows
(mirrors the TRS-blind-sym-bug fix). Trivial-IBZ short-circuit returns input
unchanged. μ-pad logic duplicated from the ij unfold. Uses same
`mode='promise_in_bounds'` workaround.
Callers: `src/gw/compute_vcoul.py:519` (import) + `:1102` (call);
`src/gw/v_q_g_flat.py:310` (import) + `:456` (call).

---

## I/O
No direct file I/O in this module. Boundary datasets:
- **reads** `zeta_q` dataset via SlabIO handle: shape `(n_q, n_rtot, n_rmu)` c128,
  offsets `(q_flat0, 0, mu_lo)`, zero-fill past `valid_shape` (padded μ).
- **reads** G-flat ζ via `ZetaReader.read_zeta_G_slab` (WFN.h5-style per-q sphere
  layout; FFT/phase/sphere done in reader).
- **writes** nothing; V_acc/g0_acc returned in-memory (callers persist, e.g.
  V_qmunu.h5 written elsewhere).

## Flags consumed
Env vars only (no cohsex.in keys read directly):
- `LORRAX_V_Q_FFT_COEF` (L84) — FFT heuristic coefficient, default 5.0.
- `LORRAX_V_Q_AOT_VERBOSE` (L222, L393, L496) — AOT chooser prints.
- `LORRAX_V_Q_Q_CHUNK` (L1156) — force Case A q_chunk, bypass chooser.
- `LORRAX_V_Q_TIME_STAGES` (L1264) — per-stage timing.
- (`LORRAX_V_Q_MU_CHUNK` is documented here L1061 but consumed in
  `compute_vcoul.py:1039`, not in this file.)
Upstream cohsex.in keys arriving as arguments: `use_bgw_vcoul` (→
`bgw_v_grid_overlay_fn`), memory budget (→ `budget_bytes`), G-flat opt-in
(→ `use_g_flat_zeta`).

## Cross-module deps
- `common.fft_helpers.make_sharded_fftn_3d` (kernel FFT + AOT model)
- `common.wfn_transforms.apply_bloch_phase` (in-kernel Bloch phase, sign=-1)
- `runtime.aot_memory.aot_kernel_peak_bytes` (chooser AOT peak)
- `common.timing`, `common.progress.LoopProgress` (driver)
- duck-typed `SlabIO` / `file_io.zeta_reader.ZetaReader` handles
- consumed by: `gw.compute_vcoul` (scalar pipeline + re-exports),
  `gw.v_q_bispinor` (per-Lorentz-tile driver), `gw.v_q_g_flat` (imports only
  `_unfold_g0_ibz_to_full`; replaces the rest wholesale)

## Dead suspects
1. `_unfold_v_q_ij_ibz_to_full` — grepped
   `_unfold_v_q_ij_ibz_to_full` across src/, tests/, tools/, scripts/: only
   `tests/test_v_q_transverse_unfold.py` imports it; no production caller. Matches the
   known bispinor-TT in-plane unfold covariance problem (production uses full-BZ-direct
   transverse instead).
2. `_choose_v_q_chunks` **re-export** at `compute_vcoul.py:518` ("for tests/other
   modules") — grepped `choose_v_q_chunks` across src/tests/tools/scripts: no consumer
   of the re-export. (Internal use inside `compute_V_q_tile` is live.)
3. Docstring-only ghost: `_unfold_v_q_ibz_to_full` (scalar V_q unfold) is referenced in
   docstrings here (L1509, L1542, L1595, L1604, L1654), in `zeta_reader.py:34`,
   `orbit_syms.py:264`, `v_q_g_flat.py:28` — but has **no definition anywhere**
   (`grep -rn "def _unfold_v_q_ibz_to_full"` → nothing). The functionality now lives at
   `common.symmetry_maps.unfold_v_q` (L161). All these are stale references.

## Redundancy suspects
1. **Three parallel V_q compute paths**: (a) r-space `_make_V_q_tile_kernel` (FFT
   in-kernel), (b) `_make_V_q_tile_kernel_Gflat` (same driver, `use_g_flat_zeta=True`),
   (c) `gw/v_q_g_flat.py::_make_per_q_kernel` — a separate orchestrator whose header
   says it "replaces compute_V_q_tile wholesale" for G-flat on-disk ζ. Same einsum
   physics three times; classic old/new-path pile-up flagged for consolidation.
2. `_unfold_g0_ibz_to_full` duplicates the μ-pad + TRS-guard + promise_in_bounds
   machinery of `common.symmetry_maps.unfold_v_q` (and of `_unfold_v_q_ij_ibz_to_full`
   in this same file) — three hand-rolled copies of the "one canonical sym-action"
   logic.
3. Chooser stacks three memory models: flat `_Q_COMPUTE_COEF_FFT` heuristic → AOT FFT
   slope/intercept (`_aot_fft_model`) → full-kernel AOT shrink-retry
   (`_aot_full_kernel_peak`). The first two are fallback tiers of the third; comments
   themselves call the flat 5× "over-conservative".

## Weird code
- L84: magic constant 5.0 (`_Q_COMPUTE_COEF_FFT`), silently ×2 for distinct-ζ (L372);
  env-tunable A/B knob.
- L216, L306: `compiler_options={"xla_gpu_memory_limit_slop_factor": 10000}` — huge
  slop factor so AOT probing compiles never fail on memory limits.
- L232, L322: bare `except Exception: return None` around AOT compiles — silently
  degrades the memory model with no logging.
- L536: shrink-retry factor 0.7 and 4-attempt cap — magic tuning constants.
- L458: Case B fallback `mu_chunk = max(snap, mu_chunk_max - (mu_chunk_max % snap))` —
  when `mu_chunk_max < snap` this picks `snap` (> budget-derived max), i.e. can exceed
  the modelled budget; hypothesis: accepted because μ_chunk < p_x·p_y can't shard.
- L621–623, L508–514: kernel cache keys use `id(mesh_xy)` / `id(sphere_idx)` — id-based
  keys can alias after GC of the original objects; also the chooser must hand-replicate
  the exact key tuple to scrub the cache (fragile duplication of key construction).
- L1138 vs L1181: `wants_g0` computed twice with different formulas
  (`(g0_acc is not None) or same_zeta` for the chooser, then reassigned
  `g0_acc is not None` before the auto-alloc branch) — works, but the double definition
  is a trap for refactoring.
- L1206–1222: `_v_qvec_batch(..., pad_to=actual)` — the pad branch (`n_actual <
  pad_to`) is unreachable since the only call site passes `pad_to=actual`; leftover
  from a fixed-Q-batch padding design.
- L1167–1170: reads chooser keys `n_mu_blocks_L`/`n_mu_blocks_R` with fallback to
  `n_mu_blocks` — the chooser never emits the L/R variants; speculative API for
  asymmetric bispinor tiling that nothing produces.
- L1541–1547, L1655: `mode='promise_in_bounds'` on take_along_axis — workaround for an
  XLA s32/s64 broadcast verifier failure under shard_map+x64 (commit `49b7f84`); if the
  perm tables ever contain OOB indices (the TRS bug class) this silently reads garbage
  instead of failing.
- L1582: docstring of `_unfold_g0_ibz_to_full` says "set `include_tau_phase=True`" for
  future consumers — **no such parameter exists** in the signature; τ-phase is simply
  omitted (justified only for the Γ-slot consumer in head_correction.py).
- L1509 etc.: stale docstring references to the nonexistent `_unfold_v_q_ibz_to_full`
  (moved to `common.symmetry_maps.unfold_v_q`).
