# RULES_seed.md — design-requirement registry, mined 2026-08-05

Raw material for the rules registry proposed in report.md §"single source of
truth" discussion: every prescriptive design requirement found in the lorrax
repo (GitHub origin @ 2026-07-22 — the ~150 unpushed Frontera commits are NOT
covered; re-sweep after consolidation) plus the sandbox ledgers. Each row:
the rule, where it is stated, how it is currently enforced, and a proposed
enforcement tier for the registry (GATE = AST/grep gate is feasible now;
RUNTIME = refusal in code; REVIEW = needs judgment, goes to TASTE.md;
LADDER = caught by fastloop/HLO/job-level checks).

Status: DRAFT — being populated from the doc and source sweeps.

## 0. Meta-finding

The registry this file seeds already half-exists as dangling references:
six named pins (`feedback_zero_replicated_intermediates_principle`,
`feedback_path_d_scaffolding_pattern`, `feedback_unified_sym_action`,
`feedback_no_new_api_layers`, `feedback_no_isdf_rank_excuse`,
`feedback_iocallback_for_large_caches`) are cited across AGENTS.md,
docs/theory/isdf-zeta-vq.md, docs/theory/symmetry.md and
docs/architecture/memory-model.md as if `MEMORY.md` / `feedback_*.md` files
held them — no such file exists in the repo. Rule one of the registry:
every named principle gets exactly one home row here.

## 1. FFT / array layout

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| One FFT path: all G↔r transforms through the `common/fft_helpers.py` factories; a raw `jnp.fft.*` in a stage kernel is a bug (keeps sharding/box/Bloch phases consistent; FFT→NUFFT upgrade in one place) | AGENTS.md "One FFT path" | prose + partial: eager-FFT allowlist ratchet (`test_fft_shardmap_context.py`, per sandbox KNOWN bse row) | GATE (banned-token, allowlist ratchet) |
| k/q dimensions are FLAT leading axes, never folded/reshaped into the FFT grid (keeps flat-k batching and NUFFT drop-in) | AGENTS.md "k/q dimensions are FLAT axes" | prose only | GATE (partial: ban reshape-into-box idioms) + REVIEW |
| Python-unrolled inner loops inside jit pile up N× unsharded slots: use `scan(unroll=1)` INSIDE `shard_map(check_rep=False)`, not naive `fori_loop` (fori SPMD-replicates the sharded carry; unroll>1 preallocates N× temps) | AGENTS.md; docs/theory/isdf-zeta-vq.md ~L966; docs/architecture/memory-model.md ~L171 (`feedback_path_d_scaffolding_pattern`) | prose + one structural fix landed | REVIEW + LADDER (HLO slot count) |

## 2. Sharding / memory

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| No replicated large intermediates; an op that rematerializes a large array on a subset of processors is a defect to fix, not a budget to work around | AGENTS.md (`feedback_zero_replicated_intermediates_principle`); sandbox INVARIANTS row 6 (no N_mu² tile on any rank) | prose + HLO probes + some AST gates (sandbox) | LADDER (per-stage `--hlo-forbid`) |
| Never hard-code mesh shapes; refer to mesh axes by name ('x','y'); always `NamedSharding`/`PartitionSpec` | AGENTS.md (twice) | prose only | GATE (grep for tuple-mesh literals) |
| Let XLA move data: no `np.concatenate`, no host-side gathers in stage code | AGENTS.md | prose only | GATE (banned-token per module, allowlist) |
| Big read-only host caches go through `io_callback` (`common/psi_G_store.py`), never jit args (jit args replicate on every device → OOM) | AGENTS.md (`feedback_iocallback_for_large_caches`); memory-model.md ~L369 | prose + the deleted `psi_G_device_full` precedent | GATE (ban large-cache identifiers as jit args) + REVIEW |
| OWNER ADDITION 2026-08-05: all device data ingress through the FFI I/O layer; never `jax.device_put` | owner, this session | none yet | GATE (banned-token outside file_io/ffi, allowlist ratchet) |

## 3. Symmetry

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| One IBZ table + one sym-action helper: every unfold routes through canonical `SymMaps`; no per-object "rotate X at q" variants (historically ≥6, being retired) | AGENTS.md; docs/theory/symmetry.md ~L690 (`feedback_unified_sym_action`) | prose; consolidation in progress | GATE (ban new unfold_* siblings) + REVIEW |
| TRS index handling must be explicit; never silently clip or nearest-fallback an unmapped k (the TRS-blind bug) | AGENTS.md | prose only | REVIEW + test |

## 4. Code structure / abstraction policy

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| Procedural on plain arrays: no new classes/dataclasses/wrapper API layers for what a function on arrays does; no `SymAction` object — augment existing bundle/table with an accessor (`feedback_no_new_api_layers`) | AGENTS.md; docs/theory/symmetry.md | review only | REVIEW (TASTE.md) |
| `main()` reads as a physics outline: drivers are sequences of named stage calls; machinery lives in stage helpers | AGENTS.md | review only | REVIEW |
| Minimal signatures: pass `(wfns, meta, config)` bundles, not >~6 positional arrays | AGENTS.md | review only | REVIEW (lint-able: positional-arg count) |
| Single source of truth, no parallel old/new paths: never `fetch_X_dyn` beside `fetch_X`, no deprecated facades left on the import path; delete the old routine in the same change; duplicated logic is a defect to collapse | AGENTS.md | review only | REVIEW + GATE (name-sibling heuristic) |
| NumPy-style docstrings documenting shapes, units, and shardings for array parameters; physics functions reference the equation they compute | AGENTS.md "Coding standards" | review only | REVIEW (lint-able partially) |

## 5. Physics reporting / epistemics

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| Don't blame residuals on "ISDF rank" without evidence; plateau-shaped LORRAX-vs-BGW disagreement rules out basis error — chase algorithm/convention difference (`feedback_no_isdf_rank_excuse`); convention gotchas in FLAGS.md (missing from origin) | AGENTS.md | prose only | TASTE.md |
| Every claim carries a jobid and on-disk artifact; outputs read from disk, never predicted | sandbox AGENTS.md rule 1 | ledger convention | ledger lint |
| Verify the instrument before trusting its verdict | sandbox rule 2 | convention | selftest (report.md §3.6) |
| A perf candidate states its scaling over the design envelope (natoms→hundreds, N_mu→tens of thousands, P→thousands, both backends) BEFORE implementation | sandbox INVARIANTS row 9 | owner rule, prose | claim-file template field (predicted impact) |

## 6. Environment / process (sandbox-side, for completeness)

| Rule | Stated in | Current enforcement | Proposed tier |
|---|---|---|---|
| Jobs read frozen source bundles, never the live tree | sandbox INVARIANTS row 5 | bundle script + template | keep |
| Collective/HLO tables cache-cold only (`ISDF_JAX_CACHE_DIR=""`) | sandbox INVARIANTS row 4 | convention | RUNTIME (analyzer refuses warm dumps — promotion proposed in review) |
| `dipole.h5` regenerated on any band-window change | sandbox INVARIANTS row 3 | convention | RUNTIME (stamp + refusal, ζ-provenance pattern) |
| centroid band window spans the sigma band window | sandbox INVARIANTS row 1 | RUNTIME (`rank_criterion.py` refusal) | keep |
| `warm_mesh_cliques(mesh)` before first jitted mpi collective | sandbox INVARIANTS row 2 | RUNTIME (refusal) + `initialize_communicator_stack` | keep |
| commit with explicit pathspecs; no `__pycache__`/`.venv`/cache dirs; run pytest after long branches | AGENTS.md "Before committing" | prose only | pre-push hook / CI |

## 7. From the doc sweep (origin @ 2026-07-22)

Canonical rules block: AGENTS.md:93-165 ("enforced by review and the
regression gate, not by ceremony. When a convention forces a bigger change
than the task, flag it — don't silently violate it"). ~90 prescriptive
rules found; compressed here as rule → source (→ enforcement if any).
Sections overlapping §1–6 are not repeated.

### 7a. Structure / abstraction (beyond §4)

- One eigh dispatcher (`ffi.common.dispatch.dispatch_eigh`); "do not
  reintroduce a parallel dispatcher" (FFI arms 11-41x slower) →
  src/bse/STATUS.md:156.
- Unsupported paths REFUSE loudly; "a config key that is parsed and
  quietly ignored has cost this project multiple days" →
  src/bse/STATUS.md:168. (Same principle as ppm direct-attribute rule §8c.)
- FFI subpackages: copy-with-edits until a THIRD consumer exists; extract
  to common/ only then → src/ffi/TEMPLATE.md:188. Do not reinvent the
  shared FFI primitives → TEMPLATE.md:22.
- New freq-integration engine must reuse get_windows/hgl_quadrature — "not
  contain its own GL/HGL sizing formulas"; each formula appears exactly
  once; layouts.py is the single source of PartitionSpecs →
  docs/dev/plans/FREQ_INTEGRATION_REWRITE_PLAN.md:505/18/529.
- `_resolve_ibz_q_list` is the single source of unfold_v_q inputs →
  docs/theory/symmetry.md:452.
- Functions <100 lines target, none >200 without justification; no
  module-level global mesh (pass mesh as parameter) →
  docs/dev/notes/AGENT_TODO.md:93/78 (weaker register: "suggested").

### 7b. FFT (beyond §1)

- norm='ortho' for physics FFTs; 'forward' for the CCT/ZCT convolution
  identity → codebase.md:282, physics.md:50.
- Sparse-G → FFT box via precomputed inverse-index GATHER, never
  `.at[].set()` scatter on GPU ("catastrophically slow", 800→90 ms);
  sentinel scheme requires cnk_padded zero slot →
  src/common/GVEC_FFT_BOX_GATHER.md:13-103.
- Chunk FFTs only over local batch axes, never spatial; chunk_count static
  with hard ==1 fast path → docs/dev/plans/PLAN_zeta_g_flat_migration.md:118.
- The 3-D (nkx,nky,nkz) shape appears only INSIDE fft_helpers →
  codebase.md:271.

### 7c. Sharding/memory (beyond §2)

- Exactly one 2-D mesh ('x','y'); there is NO 1-D 'bands' mesh;
  `_build_mesh()` is the single constructor → codebase.md:215.
- One canonical sharding per object type: "whatever axis the next consumer
  needs sharded on 'x' is sharded on 'x' going in"; canonical spec table
  at codebase.md:229-252; specs live in wavefunction_bundle, not
  per-consumer → isdf-zeta-vq.md:816.
- Full ζ never in memory (exists only as gflat_acc, read one q at a time)
  → isdf-zeta-vq.md:115.
- IFFT-before-gather (gather-first forces ~80 GB box); gather-then-slice
  (slice-first = silent wrong numerics, Round-6 Bug B, f567aa0);
  symmetric front+back pad because dynamic_slice_in_dim silently clamps
  OOB (BLOCKER c796420); n_zchunk % p_y == 0 pre-flight →
  memory-model.md:211-237. All four are the "wrong-but-plausible" class —
  top candidates for invariance gates.
- scan(unroll=1) slot-aliasing invariant: "every transient inside the scan
  body aliases to a single slot" (the solve_zeta 88 GB OOM) →
  memory-model.md:193.
- No explicit all-gather/all-reduce in the τ loop; scan carry is the only
  live accumulator → FREQ_INTEGRATION_REWRITE_PLAN.md:526/308.
- Chunk-size precedence: r_chunk first, then band_chunk, then
  gflat_chunk_size → isdf-zeta-vq.md:961.
- Explicit with_sharding_constraint on _chi_R_spec (stops XLA replicating
  the 23 GB χ₀) → codebase.md:257.

### 7d. Symmetry (beyond §3)

- BGW r-action convention r' = mtrx⁻¹·r + τ is the source of truth;
  everything composes with BGW's choice; enforced by 3 numeric gates
  (MoS2/CrI3/Si ≤0.09 meV) → symmetry.md:5/562.
- translations stored in raw BGW form (2π·τ_frac); every consumer divides
  by 2π itself, unfold_psi the one exception → symmetry.md:48/220.
- Exactly one place holds TRS algebra: unfold_psi → symmetry.md:251.
- Centroid sets must be orbit-closed (else multi-eV C3 splits); recovered
  density point group adopted only when it strictly CONTAINS the stored
  group — never downgrade → symmetry.md:340/352.
- BGW-convention inverse permutation direction, never argsort of forward
  (silent ~4 eV gap error on order-3 groups, 0735c2a) → symmetry.md:286.
- q-wrap convention q > kgrid/2 → q − kgrid shared by unfold and V_q;
  G=(0,0,0) always slot 0 on every Coulomb sphere; pad slots
  [ngk[q], ngkmax) zeroed → isdf-zeta-vq.md:247/238/232.

### 7e. I/O, HDF5, FFI (beyond §8c)

- Collective HDF5 paths byte-identical on every rank — broadcast from rank
  0; rank-local state (pid, mkstemp) in a path requires broadcast
  (H5Fcreate hangs insidiously) → src/ffi/phdf5/ARCHITECTURE.md:221.
- Caller-varying values are runtime FFI Args, never compile-time Attrs
  (each distinct Attr forces a ~400 ms recompile) → ARCHITECTURE.md:150.
- No per-call cudaEventCreate/Destroy on the hot path — pool on ctx
  (800 ms stalls) → ARCHITECTURE.md:132.
- Layout preconditions validated Python-side BEFORE invoking FFI (SLATE
  OpenMP-task exceptions std::terminate all ranks) →
  src/ffi/slate/README.md:110; SLATE mesh/tile rules (p==q or q==1;
  default nb the only layout-consistent value) → README.md:83-110.
- One JAX process per GPU, always — the process model; the cuSOLVERMg
  backend built on the alternative was deleted → src/bse/STATUS.md:149.
- Build-time and runtime MPI stacks must match (CMake prints resolved
  paths as evidence) → docs/installation/ffi-native-libs.md:101.
- zeta_q.h5 layout (nq, n_rtot, n_rmu) — n_rmu innermost (old layout 8x
  slower) → codebase.md:414; enforced by test_file_io.
- New FFI target = fixed 7-step registration checklist →
  src/ffi/AGENTS.md:135; stage scripts idempotent + readelf -d check.

### 7f. Docstrings / units / naming

- NumPy docstrings with shapes, units, shardings; match existing
  formatting, no unrelated reformats; physics functions cite their
  equation → AGENTS.md:89-91.
- Ry internally, eV on output unless labeled; ambiguous input keys carry
  units in the name (sigma_omega_min_ev, degen_avg_tol_ry); energies
  measured from chemical potential (ε_v < 0 < ε_c) →
  manual/01_introduction/1.4:3-7.
- Manual prose register governed by manual/STYLE.md ("never overexplain";
  no em dashes; bold only for paths/keys; diff drafts against §1.1) —
  11 sub-rules, see STYLE.md.

### 7g. Testing practice (beyond §8b)

- Tier-2 doctrine: invariance gates self-checking, no frozen refs →
  tests/README.md:30.
- Gate metrics must MOVE when the thing under test moves — perturb the
  input and confirm response before trusting a gate ("a gate returning a
  bit-identical number across two different fH windows is not measuring
  the window") → src/bse/STATUS.md:172. [= sandbox rule 2, independently
  stated repo-side]
- Do not gate off-grid exciton bands on absolute 2nd differences ("a
  few-meV target in that metric is unreachable and inviting fabrication");
  use the reference-free point-group symmetry gate →
  src/bse/EXCITON_BANDS.md:81.
- ongrid mode cannot validate interp (exercises none of it) →
  EXCITON_BANDS.md:21.
- Gates must not require 16 GPUs; single-device must keep working →
  STATUS.md:153.
- Numerical-equivalence smokes mandatory before calling chi-pipeline /
  compile-cache changes done → src/psp/PERFORMANCE.md:115.
- Single-compile property: whole Q path through one lax.scan (Python loop
  over Q recompiles per point) → EXCITON_BANDS.md:173.
- Measure ISDF column-space rank; "do not trust the nominal" →
  EXCITON_BANDS.md:108.

### 7h. Physics-reporting / comparison norms (beyond §5)

- bare_coulomb_cutoff ALWAYS explicit in BGW comparisons (LORRAX default
  4·ecutwfc ≠ BGW default) → symmetry.md:517, manual/06_coulomb/6.3:28.
- Comparison runs share nval/ncond, cutoffs, memory budget, n_rmu AND the
  physical centroid file (never two independent kmeans draws) →
  symmetry.md:512-531.
- Six BSE-vs-BGW conventions, "skipping any produces silent O(1) errors"
  → src/bse/BGW_COMPARE.md:1-36.
- Compare total Σ_c only, never branch-by-branch → manual/07/7.6:29.
- Gauge-invariant scalars only when comparing eigenvectors →
  STATUS.md:70.
- Two-sums rule: nothing assembled by summing valence-conduction pairs;
  every object from single-occupancy-class band sums at O(N⁰) shared
  nodes → manual/04/4.2:43. [architectural identity of the whole code]
- Windowed Σ^c: per-window Re/Im projection BEFORE band projection (the
  reverse is not equivalent) → physics.md:589.
- Band chunks summed INTO Q before the outer product — backwards is a
  silent ~1.7 eV drift, Cholesky still succeeds → EXCITON_BANDS.md:28.
- Velocity convention: physical v = p + vNL; the BGW-matching p − vNL
  flip must not be used for orbital magnetization →
  src/psp/orbital_magnetization_THEORY.md:163.

### 7i. Where the named pins actually live

NOWHERE ON DISK. The six `feedback_*` pins and `MEMORY.md` are cited
across AGENTS.md / theory docs / memory-model.md but the files exist in
neither repo nor sandbox git history. However, sandbox commit `6324f27a`
(pre-purge) holds `reports/gw_refactor_map_2026-07-01/` including
**SHARDING_RULES.md — a full prior rules registry ("every rule cites
file:line", the "memory spine")** plus MAP.md, archive/FLAGS.md,
archive/FEATURES.md, archive/GATE_AUDIT.md. The registry effort should
mine SHARDING_RULES.md (git show 6324f27a:reports/gw_refactor_map_2026-07-01/SHARDING_RULES.md)
and give the six pin names real homes here, closing the dangling
references.

## 8. From the source/test sweep (origin @ 2026-07-22)

Headline: **origin has ZERO AST gates and zero allowlist/ratchet machinery**
— `test_layering.py`, `test_crossfile_requests.py`, `test_env_registry.py`,
`test_fft_shardmap_context.py`, `rank_criterion.py`, `resolve_mesh`,
`warm_mesh_cliques`, `contract_bands` all absent (they exist only in the
unpushed Frontera commits). On origin, enforcement is: 2 source-inspecting
tests, ~18 behavioral invariance gates, ~29 runtime refusals, ~30
single-source-of-truth helper contracts stated in docstrings, and 2 escape
hatches. Sections 1–6 above should read "Frontera tree only" wherever they
say an AST gate exists.

### 8a. Source-inspecting gates that DO exist on origin

| Rule | Where | Mechanism |
|---|---|---|
| `_phdf5_build` must keep calling the per-rank clamp helper | tests/test_wfn_loader_eager.py:335 | `inspect.getsource` substring check, "brittle by design" |
| Head/wing Schur kernels compile to pure-local HLO — no collectives | tests/test_head_wing_schur.py:252 | compile + grep HLO text for banned op substrings |

The second is the in-repo precedent for per-stage `--hlo-forbid`.

### 8b. Behavioral invariance gates (the "two paths must agree" layer)

tests/test_invariance_gates.py:1 states the doctrine: "every 2026 bug class
= two paths that must agree, run FROM PREPARED STATE" — self-checking, no
frozen refs. Members: μ-pad flip invariance at fixed P (a result that moves
under the pad knob reads the pad extent — also stated as contract in
src/runtime/padding.py:52); bispinor pad flip bit-identical; kij vs
kij_stream accumulators identical ("a dropped head is 4.13 eV"); charge
ζ-Cholesky mesh-invariance (test_zeta_mesh_invariance.py); restart written
at one P reads bit-correct at any other (test_restart_pad_roundtrip.py);
padded FFI solve == logical solve with exact-zero pad rows
(test_ffi_linalg_contract.py:447); planner is the single source of chunk
sizes, user hints floored/rounded (test_band_chunk_size_floor.py:11); Si 3D
BGW anchor "IRREPLACEABLE — do not shrink or re-freeze casually"
(test_gw_jax_regression.py:9); IBZ cascade asserted on the RUN LOG because
frozen values cannot see a silently deactivated cascade (:17); open
symmetry orbits raise loudly (test_symmetry_unfold.py:10); mini-BZ head
MC-seed-stable (test_minibz_average.py); BSE matvec peak temp flat in
n_trials via `memory_analysis()` (test_bse_stack_matvec.py:13); auto never
picks SLATE Cholesky / ScaLAPACK LU, unknown eigh_backend fails loudly
(test_ffi_linalg_contract.py:826/648/924).

REGISTRY LESSON: this "two paths from prepared state" gate family is the
repo's strongest native idiom — new invariants should prefer it over frozen
references wherever both paths exist.

### 8c. Runtime refusals encoding design rules (selection; full list in sweep)

Layout/mesh: SLATE tile layout refuses p!=q and 1x{q} meshes ("SLATE
silently assembles a permuted global matrix", src/ffi/slate/context.py:96);
block-cyclic FFI refuses non-divisible N/NRHS (cusolvermp/batched.py:130,
scalapack/solve_lu.py:85); n_rmu_padded % (Px·Py) refusal under the "never
exceed 1x single-tile per rank" memory contract (symmetry_maps.py:420-431).
Symmetry: centroid orbit-closure hard refusal with regeneration recipe
(centroid/orbit_syms.py:498); sym rows must be true permutations (:510);
TRS index vs spatial-only sym_perm raises, never OOB-clamps
(symmetry_maps.py:309, v_q_g_flat.py:619); μ-extent must match EXACTLY
between tables and operands — no per-site re-padding (symmetry_maps.py:340);
k−q not on grid raises, no nearest-fallback (symmetry_maps.py:1457).
Provenance: ζ files must match write_ibz_only / zeta_cutoff_ry / q-layout
(v_q_g_flat.py:333,341); mf_header refuses overwrite (mf_header.py:221).
Config: bare_coulomb_cutoff ≤ zeta_cutoff (gw_init.py:100); PPM knobs use
direct attribute access — "a stale/typo'd name must raise, not silently
default" (ppm_sigma.py:582); fixed_point × kij_stream rejected ("previously
that pair silently degraded", gw_config.py:1041); head-less Σ_c never
silently produced (ppm_pipeline.py:172); PPM crossing-window bandwidth
floored or Σ_c blows to O(1e5) eV mesh-dependently (ppm_windows.py:67);
Tikhonov ridge "PERTURBS the physical result … hence opt-in, a physics
call" (gw_config.py:319).

### 8d. Comment-only rules — highest-value promotion candidates

These are load-bearing and protected by nothing but a comment:

| Rule | Where | Why it matters |
|---|---|---|
| Occupation build stays numpy — "DO NOT \"fix\" back to jnp" (1.79 s cross-device scatter) | wavefunction_bundle.py:190 | classic well-meaning-cleanup bait |
| DO NOT unroll the bc-scan — aliasing depends on per-iter sequential lifetime; scan(unroll=8) OOMs, refuted | isdf/core.py:702, :1876 | perf-refactor bait |
| Donation discipline: no `_chi_sec.watch()` (keeps bound method alive, blocks donation); donated χ₀ invalid after solve_w | screening.py:180, :212 | silent perf/correctness trap |
| Pad rows EXACTLY zero: np.zeros never np.empty — "math-neutrality of pad rows depends on it" | psi_G_store.py:322 | wrong-but-plausible class |
| Never fall back to single-device under multi-host (other ranks deadlock on collectives) | kmeans_cli.py:206 | distributed hang class |
| Never device_get whole coeffs; ψ(G) host-only, one band-chunk on device via io_callback | wfn_loader.py:179, gw_config.py:271 | the io_callback rule, stated at 2 sites |
| Rank-0-only .dat/.png writes (shared-FS race) | exciton_bands.py:763 | the htransform bug class, pre-stated |
| .at[].add() not .set() under ngkmax padding | psp/dft_operators.py:696 | pad-correctness class |
| Do not rename `_fit_one_rchunk_cache` (cleared by name across runs) | isdf/core.py:71 | rename-refactor bait |
| Refuted-idea blocklists: V_q model "do NOT improve with: pinned real-space moments (refuted twice), SVD multipoles…"; never validate finite-q W by sym-unfolding between q's | bse/vq_interp.py:73, bse_w_exact.py:592 | negative results encoded at point of use — the pattern TASTE.md generalizes |
| LORRAX_EXTRA_MU_PAD never in production; disk stores LOGICAL extent; ngkmax ragged padding is a DIFFERENT convention, do not unify | runtime/padding.py:57, :19 | conventions collide silently |

### 8e. Single-source-of-truth helper contracts (the registry's §1 backbone)

Explicit "single source" docstrings found (rule = all call sites go through
me): `make_flat_k_fft` + fft_helpers factories (k-axes replicated in spec,
fft_helpers.py:379); Bloch-phase formula "the ONLY place in LORRAX"
(wfn_transforms.py:1294); WfnLoader "single entry point for ψ(G) loading,
both backends byte-identical" (wfn_loader.py:1); SlabIO "replaces the
ad-hoc process_allgather → rank-0 h5py patterns" (slab_io.py:1); PsiGStore
io_callback host-cache pattern (psi_G_store.py:1); SymMaps trs_augment_U /
tau_phase_row / kgrid_shift_map ("the ONE place k+q integer arithmetic
lives", symmetry_maps.py:23/664/708); kq_mapping umklapp+phase ("both
pipelines used to inline their own", kq_mapping.py:12); units.py Ry↔eV
("previously inlined at ~25 sites"); coulomb_sphere radius condition;
runtime.padding round_up ("THE spelling"); runtime/__init__ env+distributed
init ("five modules had drifting copies"); gw_config._DEFAULTS ("every
input key"); screening_requests_for; _resolve_w_solve_fn;
_resolve_solver_kind; eqp_bgw math+formatter; ppm_accumulators ω-projection;
coulomb/base mini-BZ sampler ("reimplementing it locally ate a bug");
wavefunction_bundle canonical sharding specs ("reshard mismatch caught at
import time"); psp dft_operators / radial_tables Hankel / xc one-path;
bse_io._load_layout_shim; isdf_fitting mem_probe; provenance_header (stamp
only — NO mismatch refusal yet, promotion candidate).

Nearly every one of these docstrings cites a bug or drift incident as its
rationale — the registry inherits its evidence column for free.

### 8f. Registries and escape hatches

Implicit env-var registry: 32 LORRAX_* vars by grep, no closure test on
origin (the missing test_env_registry.py is the obvious gate — it exists on
the Frontera tree). Config keys: `_DEFAULTS` single source, but unknown
keys WARN rather than refuse (gw_config.py:550) — tighten per the typo'd-
name-must-raise principle the PPM path already adopted. Escape hatches to
carry into the registry as flagged exceptions: LORRAX_SKIP_VQ_GATES=1
(vq_interp.py:995), LORRAX_FORCE_FULL_BZ=1 (v_q_g_flat.py:173).

### 8g. Stated-but-unenforced gaps (the sweep's own gate shortlist)

No gate anywhere for: raw `jnp.fft.*` outside fft_helpers (trivial grep
gate); "no parallel old/new paths" (deprecated shims live right now at
gw_config.py:933 and wfn_loader.py:37 `get_gvec_nk`); hard-coded mesh
shapes / np.concatenate / host gathers (`_to_host_np` helpers at
ppm_windows.py:179, minimax_screening.py:33 need allowlisting or retiring);
k-reshape-into-FFT-box (docstring only); import-direction layering
(asserted in prose at ppm_accumulators.py:11, no test on origin).
