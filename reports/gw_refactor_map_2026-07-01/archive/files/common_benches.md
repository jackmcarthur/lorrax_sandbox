# GW refactor map — group: `src/common/*_test.py` / `*_bench*.py` (standalone diagnostic scripts)

Cataloged 2026-07-01 from `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Group-level summary

All 20 files are **standalone `python -m common.<name>` CLI scripts**, NOT pytest tests.
They live in `src/common/` next to production modules (`isdf_fitting.py`, `symmetry_maps.py`, ...),
which is itself refactor-relevant: they are exercised only manually under `lxrun`/Shifter
(pytest is not available inside the Shifter runtime — see docstring of
`wfn_loader_backend_parity_test.py` and KNOWN_SANDBOX_ERRORS 2026-05-06).

**Grep methodology for callers:** grepped each module name across `src/`, `tests/`, `tools/`,
`scripts/`, `docs/` of lorrax_D. Findings: only doc references (`src/ffi/AGENTS.md`,
`src/ffi/PORTING.md`, `src/ffi/TEMPLATE.md`, `src/ffi/slate/README.md`,
`docs/architecture/codebase.md`, `docs/dev/plans/phdf5_cray_mpich_migration.md`) and
comment cross-references (`tests/test_head_wing_schur.py:43`, `tests/test_wfn_loader_eager.py:128`).
No script imports another script; all are `__main__`-only.

### Group-wide redundancy (the big finding)

Copy-paste boilerplate repeated across nearly every file with slight drift:

1. **`_init()` / `_maybe_init_jax_distributed()` / `_bootstrap_jax_distributed()` /
   `_init_distributed()`** — the JAX distributed bootstrap appears in ~15 files under 4
   different names and ≥4 behavioral variants (with/without `lrx_slate_init_mpi()` pre-call,
   with/without `local_device_ids=[0]` CUDA_VISIBLE_DEVICES sniffing, with/without
   JAX_PROCESS_COUNT fallbacks). One shared helper would delete ~200 LOC.
2. **Random HPD matrix builders** — `make_hermitian`, `make_hpd`, `build_hpd`,
   `build_hpd_batch` (x2 near-identical copies in slate_batched_test and
   cusolvermp_batched_test), `make_A`, `build_general_batch`, `build_hermitian`, plus the
   V/chi0 synthetic builders duplicated between `w_solve_modes_test` and
   `cublasmp_w_solve_test`. Same "0.5(Z+Z^H) + n*I" pattern ~10 times.
3. **`_log(s): if jax.process_index()==0: print`** — in every file.
4. **`_parse_mesh("PxQ")`** — 6 copies.
5. **RHS builders** `build_rhs`/`build_rhs_batch`/`make_rhs` — 5 copies.

### Group-wide dead-backend observation

Production LORRAX (grep of `src/` excluding `*_test/_bench/_sweep` files and `ffi/` itself):

- `ffi.cusolvermp` **batched** ops ARE used in production: `isdf_fitting.py:1103/1286/1316`
  (batched cholesky/potrs/solve_lu).
- `ffi.cublasmp.batched_fused_w_solve` IS used: `gw/w_isdf.py:282`.
- `ffi.slate.*` (distributed_cholesky/trsm/eigh, batched variants): **zero production
  callers** — exercised only by the slate_* scripts here.
- `ffi.cusolvermg.eigh_mg`: **zero production callers** — only cusolvermg_eigh_test +
  eigh_benchmark.
- `ffi.cusolvermp.distributed_eigh` (non-batched): **zero production callers** — only
  cusolvermp_eigh_test, eigh_benchmark, eigh_block_sweep, slate_vs_cusolvermp_bench.
- `ffi.cublasmp.batched_distributed_gemm`: **zero production callers** — only
  cublasmp_gemm_test.

So the SLATE backend + cusolverMg backend + non-batched Mp eigh + standalone gemm are
evaluation-era artifacts; their test scripts inherit that status.

---

## 1. `src/common/cusolvermp_eigh_test.py` (228 LOC)

**Purpose:** Multi-process (4-GPU, 2x2 grid) correctness smoke for
`ffi.cusolvermp.distributed_eigh` (complex128 Hermitian + float64 symmetric paths);
compares eigenvalues against `np.linalg.eigvalsh`, prints PASS/FAIL, spot-checks
eigenvector residuals.

**Category:** diagnostic/bench script (distributed linalg FFI smoke).

**Functions:**
| function | role |
|---|---|
| `_maybe_init_jax_distributed()` | env-sentinel-guarded `jax.distributed.initialize`, handles 1-GPU-visible ranks |
| `make_hermitian(n, seed)` | deterministic random Hermitian + 2n*I diagonal dominance |
| `shard_matrix(A, mesh)` | device_put with P('x','y') |
| `gather_to_numpy(x)` | `multihost_utils.process_allgather(tiled=True)` wrapper |
| `_log(msg)` | rank-0 print |
| `run_complex_test(n, mesh)` | c128 path, tol 1e-8*n, residual spot check on 3 columns |
| `run_real_test(n, mesh)` | f64 path, tol 1e-9*n |
| `main()` | argparse (-n, --grid P Q, --skip-complex/--skip-real), grid auto-pick |

**Entry points / callers:** `main` <- CLI only (`python -m common.cusolvermp_eigh_test`);
referenced by docs `src/ffi/AGENTS.md`, `src/ffi/PORTING.md`, `src/ffi/TEMPLATE.md`
("mirror this file"), `docs/dev/plans/phdf5_cray_mpich_migration.md` (regression gate).

**I/O:** none (synthetic matrices; stdout only).

**Flags/env:** CLI `-n --grid --skip-complex --skip-real`; env `JAX_ENABLE_X64`,
`JAX_PLATFORMS`, `JAX_PROCESS_COUNT`/`JAX_NUM_PROCESSES`/`SLURM_NTASKS`,
`CUDA_VISIBLE_DEVICES`, sentinel `_LORRAX_JAX_DISTRIBUTED_DONE`.

**Weird:** bootstrap swallows all exceptions from `jax.distributed.initialize` (bare
`except Exception: pass`) then sets the DONE sentinel anyway — a failed init looks like
single-process mode. This exact pattern is cloned everywhere. Also this file's bootstrap
is the only one consulting `JAX_PROCESS_COUNT`/`JAX_NUM_PROCESSES` (drift vs siblings).

**Dead/redundancy:** file itself is alive as the canonical FFI smoke (docs point at it).
`gather_to_numpy` duplicates the allgather one-liner used inline elsewhere.

---

## 2. `src/common/slate_batched_test.py` (217 LOC)

**Purpose:** Correctness test for `ffi.slate.batched_distributed_cholesky` +
`batched_distributed_trsm` on P('x',None,'y')-sharded batches; residuals vs numpy
Cholesky per batch slice, both op='N' and op='C' solves.

**Category:** diagnostic/bench script (SLATE batched FFI correctness).

**Functions:** `_init` (bootstrap incl. `lrx_slate_init_mpi()` eager MPI init),
`_log`, `build_hpd_batch`, `build_rhs_batch`, `_parse_mesh`, `main` (argparse
--nbatch -n -m --mesh --dtype --nb --seed).

**Entry points:** CLI only. Referenced in `src/ffi/slate/README.md:37,171`,
`docs/architecture/codebase.md:484`.

**I/O:** none (synthetic; stdout).

**Weird:** comment at line ~182 "L is implicit (handle.raw is layout-mangled bytes)" —
correctness of the distributed L itself is never checked, only solves against a numpy
reference factor; the trsm checks would pass even if SLATE's L differed by a unitary.
No PASS/FAIL exit code from residuals (always returns 0) — unlike its cusolvermp twin.

**Redundancy:** near-clone of `cusolvermp_batched_test.py` (build helpers differ only in
sharding spec P('x',None,'y') vs P(None,'x','y')); SLATE backend has no production callers.

---

## 3. `src/common/slate_chol_trsm_bench.py` (195 LOC)

**Purpose:** Timing benchmark of SLATE `distributed_cholesky` + `distributed_trsm`
(non-batched, single big matrix, P('x','y')) across mesh shapes; warmup + repeated
timed iterations with `sync_global_devices` fences, correctness check on last iter via
`L_handle.to_jax_lower()`.

**Category:** diagnostic/bench script (SLATE perf).

**Functions:** `_init`, `_log`, `make_hpd`, `make_rhs`, `_parse_mesh`, `main`
(--mesh required, -n --dtype --nb --repeats --seed).

**Entry points:** CLI only; `src/ffi/slate/README.md:38`.

**I/O:** none.

**Weird:** none beyond the shared boilerplate; nb default documented as "n/p (cholesky
default)" — magic layout coupling to SLATE tile grid.

**Redundancy:** duplicates make_hpd/make_rhs of `slate_cholesky_trsm_test.py` verbatim;
bench vs test split is itself a "parallel old/new path" smell (one script with `--repeats`
would cover both).

---

## 4. `src/common/slate_cholesky_trsm_test.py` (187 LOC)

**Purpose:** Correctness test for non-batched SLATE `distributed_cholesky` +
`distributed_trsm` on a 2x2 mesh; compares canonical L (via `handle.to_jax_lower()`)
against `np.linalg.cholesky`, checks forward (op='N') and adjoint (op='C') solves.

**Category:** diagnostic/bench script (SLATE FFI correctness).

**Functions:** `_init`, `_log`, `build_hpd`, `build_rhs`, `main` (-n -m --dtype --nb --seed).

**Entry points:** CLI only; `src/ffi/PORTING.md:103`, `src/ffi/slate/README.md:36`.

**I/O:** none.

**Weird:** lines 149-153 — hard-coded layout forensics: "In SLATE GridOrder::Col with
default nb=n/p, rank 2 owns SLATE tile (0,1) = strict upper... When gathered, that's JAX
block (1,0) = bottom-left" then inspects `L_raw_np[n/2:, :n/2]`. Debug residue from the
SLATE layout-debugging campaign; only meaningful for exactly 2x2 mesh + nb=n/p. Hard
`world != 4` requirement. Always returns 0 (no FAIL exit).

**Redundancy:** superseded-ish by `slate_chol_trsm_bench.py` (same math + timing);
build helpers cloned in 3 slate files.

---

## 5. `src/common/cusolvermp_solve_lu_test.py` (173 LOC)

**Purpose:** Correctness test for `ffi.cusolvermp.batched_distributed_solve_lu` (general
non-Hermitian distributed LU solve used by w_isdf's low_mem path); residual |AX-B|/|B|
per batch slice with PASS/FAIL at 1e-10; optional `--identity` sanity mode (A=I → X=B).

**Category:** diagnostic/bench script (cuSOLVERMp LU FFI correctness).

**Functions:** `_init`, `_log`, `build_general_batch` (diag-dominant non-Hermitian,
`identity_only` flag), `build_rhs_batch`, `_parse_mesh`, `main`
(--nbatch -n --nrhs --mesh --dtype --seed --identity).

**Entry points:** CLI only. Cross-referenced from `tests/test_head_wing_schur.py:43`
(comment). The production consumer of the op is `isdf_fitting.py:1316,1341`.

**I/O:** none.

**Flags/env:** extra env `LORRAX_LU_DEBUG_DUMP` — gates a zero-cell mask ASCII dump of
X[0] (lines 157-165), a leftover debugging aid from an LU-layout bug hunt (the mask
visualizes which cells came back zero when A=I).

**Weird:** the LORRAX_LU_DEBUG_DUMP block; sharding here is `P(None,'x','y')` (matrix
distributed, batch replicated) unlike the potrf sibling's `P('x',None,'y')` — intentional
(matches the low_mem sub-comm layout) but undocumented in-file and easy to misread.

---

## 6. `src/common/cusolvermp_batched_test.py` (171 LOC)

**Purpose:** Correctness test for `ffi.cusolvermp.batched_distributed_cholesky` +
`batched_distributed_potrs` on P('x',None,'y')-sharded HPD batches; |AX-B|/|B| residual,
PASS/FAIL at 1e-10.

**Category:** diagnostic/bench script (cuSOLVERMp batched FFI correctness).

**Functions:** `_init`, `_log`, `build_hpd_batch`, `build_rhs_batch`, `_parse_mesh`,
`main` (--nbatch -n --mrhs --mesh --dtype --nb --seed).

**Entry points:** CLI only; docstring of cusolvermp_solve_lu_test says "mirrors" this file.
Production consumer of the ops: `isdf_fitting.py:1103-1104, 1286-1301`.

**I/O:** none.

**Weird:** `--nb` block_size plumbed through but divisibility checks don't account for it.

**Redundancy:** near-clone of `slate_batched_test.py` (different backend + sharding);
`potrs_rhs_test.py` is a smaller near-duplicate of this file.

---

## 7. `src/common/eigh_benchmark.py` (170 LOC)

**Purpose:** Eigh timing shootout at n≈2048: `jnp.linalg.eigh` (1 GPU) vs
`ffi.cusolvermg.eigh_mg` (single-process multi-GPU) vs `ffi.cusolvermp.distributed_eigh`
(multi-process 2x2). Two phases selected by `--mode single|mp` because the backends need
different process affinity.

**Category:** diagnostic/bench script (eigh backend perf comparison).

**Functions:** `_log`, `make_symmetric`, `time_call(fn,args,repeats,name)` (warmup+stats),
`run_single_process(n,repeats)`, `run_multiprocess(n,repeats)`, `main`.

**Entry points:** CLI only. No doc references found outside its own docstring
(grepped `eigh_benchmark` across src/tests/tools/scripts/docs — only self-hits).

**I/O:** none.

**Flags/env:** `--mode --repeats -n`; docstring prescribes `CUSOLVERMP_FORCE_NCCL=1`,
`XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_PREALLOCATE` for phase B.

**Weird:** magic `tile = 256` for the Mg path (line 94) with no sweep or comment;
real-symmetric only (never benches c128, which is the production-relevant dtype).

**Dead suspects:** both benched backends (Mg eigh, Mp distributed_eigh) have zero
production callers — this whole benchmark is evaluation-era.

---

## 8. `src/common/slate_eigh_test.py` (169 LOC)

**Purpose:** Correctness smoke for `ffi.slate.distributed_eigh` (slate::heev) on square
process grids; eigenvalues vs numpy, |HQ-QW|/|H| residual, plus a `--kind diag` mode
(A=diag(1..n) so Q should be I) for diagnosing SLATE's eigenvector storage layout.

**Category:** diagnostic/bench script (SLATE heev FFI correctness/layout diagnosis).

**Functions:** `_init` (with `lrx_slate_init_mpi()` eager MPI, extensive comments),
`_log`, `make_hermitian(n,seed,dtype,mesh,kind)`, `main` (-n --dtype --seed --nb --kind).

**Entry points:** CLI only.

**I/O:** none.

**Weird:** line 135 `W_np = ...[:args.n]  # in case replica factor > 1` — silently trims
gathered eigenvalues, papering over an allgather replication ambiguity; `--kind diag`
plus the "|Q[i,j]| > 0.5 at:" probe are leftovers of a layout bug hunt (like
slate_cholesky_trsm_test's raw-tile probe). Requires world in {1,4,9,16} because "slate
heev needs p==q".

---

## 9. `src/common/wfn_loader_backend_parity_test.py` (166 LOC)

**Purpose:** P2 contract test asserting `file_io.wfn_loader.WfnLoader` outputs are
bit-equal between the `eager` (h5py) and `phdf5` (FFI) backends for the same WFN.h5:
`load(k='ibz')`, `load(k='full_bz')`, and `gvecs(k='full_bz')` metadata. Standalone
because pytest doesn't run under Shifter.

**Category:** I/O contract test (WFN loading parity), not a linalg bench.

**Functions:** `_bootstrap_jax_distributed`, `_log`, `_parse_mesh` (also handles `×`),
`_build_mesh`, `_replicate_to_host` (allgather + strip leading process axes),
`_check(name, eager, phdf5, atol)`, `main` (--wfn --mesh --bands --atol).

**Entry points:** CLI only; cross-referenced from `tests/test_wfn_loader_eager.py:128`
(comment: this script is the phdf5-vs-eager complement to the pytest eager tests).

**I/O:** **reads** WFN.h5 (BGW wavefunction HDF5; via WfnLoader both through h5py-eager
and phdf5-FFI paths). Writes nothing.

**Weird:** `_replicate_to_host` strips leading axes with `while h.ndim > arr.ndim: h = h[0]`
— assumes replicated (not tiled) gather; default `--atol 1e-12` contradicts the
docstring's "byte-for-byte" claim (argparse default overrides the `_check` default of 0.0,
so bit-equality is NOT actually asserted unless --atol 0 is passed).

**Cross-module deps:** `file_io.wfn_loader.WfnLoader` (production).

---

## 10. `src/common/slate_vs_cusolvermp_bench.py` (148 LOC)

**Purpose:** Head-to-head eigh timing, SLATE heev vs cuSOLVERMp syevd, 4 GPUs / 2x2,
one backend per invocation (`--backend slate|cusolvermp`) because each needs different
CUDA process affinity.

**Category:** diagnostic/bench script (distributed eigh backend shootout).

**Functions:** `_log`, `_init_distributed(backend)` (per-backend affinity), `build_hermitian`,
`main` (--backend -n --dtype --repeats --seed --nb).

**Entry points:** CLI only.

**I/O:** none.

**Weird:** unlike every sibling, `_init_distributed` does NOT wrap
`jax.distributed.initialize` in try/except for the slate branch — inconsistent drift.
Backend selection duplicates identical `kw` construction in both branches (copy-paste
within the file). No `lrx_slate_init_mpi()` eager call in the slate branch, unlike
slate_eigh_test — likely a latent difference in MPI init behavior.

**Redundancy:** subsumes/overlaps eigh_benchmark's mp phase and slate_eigh_test's timing.

---

## 11. `src/common/cublasmp_gemm_test.py` (118 LOC)

**Purpose:** Correctness test for `ffi.cublasmp.batched_distributed_gemm` with arbitrary
transa/transb and complex alpha/beta; residual vs numpy einsum-style reference,
PASS/FAIL at 1e-10 (c128).

**Category:** diagnostic/bench script (cuBLASMp gemm FFI correctness).

**Functions:** `_init`, `_log`, `main` (inline `make()` builder; --nbatch -m -n -k
--mesh --dtype --transa --transb).

**Entry points:** CLI only.

**I/O:** none.

**Weird:** magic alpha=2.0+0.5j, beta=0.3-0.1j (fine for a test); f64 tolerance is 1e-6
vs 1e-10 for c128 (line 111) — suspiciously loose for f64, suggests the real path was
observed to be less accurate and the tolerance was widened rather than investigated.

**Dead suspects:** `batched_distributed_gemm` itself has zero production callers
(grep across src excluding tests/ffi) — only `batched_fused_w_solve` is used
(w_isdf.py:282). Test + op pair are evaluation-era.

---

## 12. `src/common/w_solve_modes_test.py` (110 LOC)

**Purpose:** A/B parity test of the production `gw.w_isdf.solve_w` in `high_mem` vs
`low_mem` memory modes on synthetic SPD V and negative-semidefinite chi0; asserts
relative diff < 1e-10.

**Category:** physics: chi0/W stage (mode-parity test of production solver).

**Functions:** `_init`, `_log`, `main` (no argparse — hard-coded nq=2, n=64, 2x2 mesh).

**Entry points:** CLI only. Exercises production `gw.w_isdf.solve_w`.

**I/O:** none.

**Cross-module deps:** `gw.w_isdf.solve_w` (production W solve).

**Weird:** `meta = SimpleNamespace(nk_tot=nq, nspin=1, nspinor=1)` — hand-rolled mock of
the meta bundle (the user's memory notes explicitly warn mocks like this can hide
bispinor scaling); `_fresh = jax.jit(lambda x: x)` to defeat solve_w's donated χ₀ buffer
(lines 85-90) — documents that solve_w donates arg 1, a sharp API edge worth noting in
the refactor; hard-coded problem size, no CLI knobs.

---

## 13. `src/common/cusolvermg_eigh_test.py` (105 LOC)

**Purpose:** Single-process multi-GPU correctness smoke for `ffi.cusolvermg.eigh_mg`
(cusolverMgSyevd); f64 symmetric only; eigenvalue check vs numpy + row-vs-column
eigenvector residual probing.

**Category:** diagnostic/bench script (cuSOLVERMg FFI smoke).

**Functions:** `main` only (-n --tile --max-gpus). No distributed bootstrap (single
process by design).

**Entry points:** CLI only; docs `src/ffi/AGENTS.md:10,93`, `docs/architecture/codebase.md`.

**I/O:** none.

**Weird:** lines 80-96 — explicit admission of an unresolved layout convention: "the
returned Q ... cuSOLVERMg computed the eigenvectors of A^T (since we fed a row-major
buffer interpreted as col-major)... for real-symmetric it doesn't matter"; the residual
check then tries BOTH row-vectors and column-vectors and prints both, deciding nothing.
For a complex Hermitian matrix this row/col-major ambiguity would be a real transpose
bug — the FFI wrapper's layout contract is untested for c128 (test is f64-only).

**Dead suspects:** `eigh_mg` has zero production callers.

---

## 14. `src/common/cublasmp_w_solve_test.py` (103 LOC)

**Purpose:** Correctness test for `ffi.cublasmp.batched_fused_w_solve` — the fused
cuBLASMp gemm + cuSOLVERMp LU pipeline computing W = (I - V·pref·chi)^(-1) V — against a
per-slice numpy direct solve; PASS/FAIL at 1e-10.

**Category:** diagnostic/bench script (fused W-solve FFI correctness; closest of the
FFI tests to the physics W equation).

**Functions:** `_init`, `_log`, `main` (inline `make()`; --nbatch -n --mesh
--pref_re --pref_im).

**Entry points:** CLI only. Production consumer of the op: `gw/w_isdf.py:282`.

**I/O:** none.

**Weird:** default `--pref_re 0.67` — a magic default approximating a spin/degeneracy
prefactor (production pref would be e.g. 2/nk or similar); comment "V is donated" —
another donated-buffer sharp edge, snapshot-before-solve pattern required.

---

## 15. `src/common/slate_trsm_isolated_test.py` (101 LOC)

**Purpose:** Layout-debug harness for SLATE `distributed_trsm` with a hand-built known L
and known X (B = L@X): sweeps ALL side/uplo/op combos {L,R}x{L,U}x{N,T} and prints which
combination recovers X (and whether transposed), to pin down SLATE's convention mapping.

**Category:** diagnostic/bench script (one-off layout-forensics harness).

**Functions:** `_init`, `_log`, `main` (-n --seed; default n=4 so values printable).

**Entry points:** CLI only. No doc references (grepped `slate_trsm_isolated_test` across
src/tests/tools/scripts/docs — only self-hit).

**I/O:** none.

**Weird:** the whole file is a debugging artifact: checks `|X - X_true|` AND `|X.T - X_true|`
for every combo (transpose ambiguity hunting); special-cases printing X for
side='R', uplo='U', op='N' (line 95) — the combo that was presumably misbehaving during
the hunt. Real f64 only (no c128, no op='C').

**Dead suspects:** entire file — the layout question it answered is settled (production
handle carries `to_jax_lower()`); candidate for deletion in refactor.

---

## 16. `src/common/eigh_block_sweep.py` (101 LOC)

**Purpose:** Sweep `block_size` for cuSOLVERMp `distributed_eigh`, timing steady-state
per block and checking eigenvalue error; documents that block < n/p yields permuted
eigenvectors but correct eigenvalues.

**Category:** diagnostic/bench script (FFI tuning sweep).

**Functions:** `_log`, `main` (-n required, --blocks list required, --repeats).

**Entry points:** CLI only. No doc references found.

**I/O:** none.

**Weird:** docstring admits a known correctness hazard shipped as-if-benign:
"Block sizes < n/p produce a ... layout mismatch — the eigenvectors will be permuted
relative to the input basis" — i.e., the FFI returns silently wrong eigenvectors for
small blocks; only eigenvalues validated here. If distributed_eigh were ever used in
production with tuned block_size this would be a landmine. Divisibility guard
`n % (block*p) != 0` skips instead of fixing layout.

**Dead suspects:** whole file (non-batched Mp eigh has no production callers).

---

## 17. `src/common/potrs_rhs_test.py` (83 LOC)

**Purpose:** Minimal isolation test: is `batched_distributed_potrs` correct on its own
for A = LL^H and generic complex RHS; residual vs `np.linalg.solve`.

**Category:** diagnostic/bench script (cuSOLVERMp potrs isolation; bug-hunt residue).

**Functions:** `_init`, `_log`, `main` (hard-coded nq=1, n=64, 2x2 mesh; only `--nrhs`
via `parse_known_args`).

**Entry points:** CLI only. No doc references found.

**I/O:** none.

**Weird:** argparse inside `main` with `parse_known_args()` (tolerates stray args — quick
hack); stale comment lines 58-62 about "B: Hermitian (same as X_dagger in w_isdf — comes
from conj-swap of a lower-tri matrix, so B is upper-tri. But potrs should work on any B;
test with generic complex for now)" — ties this file to a specific historical w_isdf bug
hunt; no PASS/FAIL, prints residuals and always exits 0.

**Redundancy:** strict subset of `cusolvermp_batched_test.py` (same ops, fewer checks) —
merge/delete candidate.

---

## 18. `src/common/chol_natural_test.py` (77 LOC)

**Purpose:** Verify `ffi.cusolvermp.cholesky_handle_to_natural_L` converts the
layout-mangled batched-Cholesky handle back to a natural lower-triangular L with
LL^H = A and zero strict-upper; also diffs against numpy Cholesky.

**Category:** diagnostic/bench script (handle-layout conversion correctness).

**Functions:** `_init`, `_log`, `main` (no argparse; hard-coded nq=2, n=64, 2x2).

**Entry points:** CLI only. `cholesky_handle_to_natural_L` is exported by
`ffi/cusolvermp/__init__.py` and defined at `ffi/cusolvermp/batched.py:258`; grep shows
no production caller of it outside this test (isdf_fitting keeps handles opaque and uses
potrs) — the accessor + this test may both be dead.

**I/O:** none.

**Weird:** reference comparison only against `L_ref` of batch slice 0; no PASS/FAIL exit
code; no tolerance assertion at all (prints numbers, returns 0).

---

## 19. `src/common/isdf_zeta_mode_test.py` (175 LOC)

**Purpose:** Mode-parity test of the production ISDF fit path
`common.isdf_fitting.factor_c_q` + `solve_zeta` in `high_mem` (replicated-L vmap trsm)
vs `low_mem` (batched cuSOLVERMp potrf+potrs) on random HPD C and RHS Z; checks
C@zeta≈Z per q and cross-mode agreement to 1e-9.

**Category:** physics: ISDF zeta-fit stage (memory-mode parity test of production code).

**Functions:** `_init`, `_log`, `build_hpd_batch`, `build_rhs_batch`, `_parse_mesh`,
`_run_mode(mode, A, B, mesh)` (factor+solve+gather with per-mode reshard), `main`
(--nq -n --mrhs --mesh --seed).

**Entry points:** CLI only. Exercises production `isdf_fitting.factor_c_q` (line 1003)
and `solve_zeta` (line 1214).

**I/O:** none (synthetic).

**Cross-module deps:** `common.isdf_fitting` (production zeta-fit).

**Weird:** comment at line 98-99 "matches fit_zeta_chunked_to_h5's convention at line
~1350" — brittle line-number coupling to production source; per-mode RHS pre-reshard
logic (`P('x',None,'y')` for low_mem vs `P(None,None,('x','y'))` for high_mem) is
duplicated knowledge of solve_zeta's internal expectations — a refactor should make
solve_zeta own its input resharding; magic `q_chunk_size=1024`.

---

## 20. `src/common/symmetry_test.py` (62 LOC)

**Purpose:** CLI debug utility validating WFN symmetry metadata: runs
`SymMaps.validate_atomic_symmetries` and `SymMaps.validate_kgrid_unfolding` against a
WFN.h5 and prints a compact PASS/FAIL report with example failures.

**Category:** symmetry machinery (diagnostic front-end).

**Functions:** `_status_line(label, passed, detail)`, `run_symmetry_test(wfn_path, tol)`,
`main()` (argparse: positional `wfn`, `--tol`).

**Entry points:** `run_symmetry_test` <- CLI only. Grepped `symmetry_test` and
`run_symmetry_test` across src/tests/tools/scripts/docs: zero external callers or doc
mentions. The validators it fronts (`symmetry_maps.py:1092,1125`) are production-adjacent.

**I/O:** **reads** WFN.h5 via `file_io.WfnLoader` (aliased `WFNReader` — legacy name kept
alive here).

**Weird:** only file in the group with no JAX/distributed boilerplate (pure host);
`from file_io import WfnLoader as WFNReader` perpetuates the legacy WFNReader name;
relative import `.symmetry_maps` (others use absolute) — minor inconsistency.

**Dead suspects:** `run_symmetry_test`/whole file — zero grep hits anywhere; likely still
useful as a manual debug tool but unreferenced by any doc or skill.

---

## Refactor recommendations (group)

1. Extract shared `common/_dist_test_utils.py` (or move all of these to `tools/diag/`):
   bootstrap, `_log`, `_parse_mesh`, HPD/RHS builders, allgather-compare helpers.
2. Delete or archive: `slate_trsm_isolated_test.py`, `potrs_rhs_test.py`,
   `eigh_block_sweep.py` (bug hunts concluded); fold `slate_cholesky_trsm_test.py` into
   `slate_chol_trsm_bench.py`.
3. Decide fate of unused backends (slate non-batched+batched, cusolvermg, Mp
   non-batched eigh, cublasmp standalone gemm) before porting their tests anywhere.
4. Keep and possibly promote to gated CI: `cusolvermp_batched_test`,
   `cusolvermp_solve_lu_test`, `cublasmp_w_solve_test`, `w_solve_modes_test`,
   `isdf_zeta_mode_test`, `wfn_loader_backend_parity_test` — these guard production paths.
5. Fix `wfn_loader_backend_parity_test` --atol default (1e-12) vs advertised bit-equality.
6. Standardize PASS/FAIL exit codes (several always return 0).
