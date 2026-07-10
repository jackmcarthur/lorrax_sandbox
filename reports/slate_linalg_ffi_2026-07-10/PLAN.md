# Overnight program — block-cyclic linalg FFI stability + SLATE portability

_2026-07-10 (overnight). Branch `agent/slate-linalg-ffi` on lorrax_D (base `9605a0f`).
Owner directive: test + improve stability of the FFIs, particularly linear algebra with
block-cyclic distribution; make SLATE work on BOTH CPU and GPU Perlmutter nodes (the
portability path to Frontier/Aurora/non-NVIDIA), integrated as an OPTIONAL dependency
selectable from the input file._

## Starting state (recon)

- `src/ffi/slate/` EXISTS: potrf/trsm/eigh + batched wrappers, context caching,
  README claims 1–3e-16 residuals for cholesky/trsm on any p×q mesh; **known defect:
  eigh eigvec layout artifact** (eigh.py). Smoke tests: `common.slate_cholesky_trsm_test`,
  `common.slate_batched_test`, `common.slate_chol_trsm_bench`.
- SLATE built once at `$HOME/software/slate` (+ blaspp/lapackpp in install/); evaluation
  build, Cray wrappers + libsci, `-DSCALAPACK_LIBRARIES=""` gotcha (memory note).
  CPU capability UNVERIFIED.
- Input-file linalg axes today: `cusolvermp_charge` / `cusolvermp_lu` (off | on-ish
  values), `screening_solver`. SLATE is NOT reachable from cohsex.in.
- Other block-cyclic FFIs: `src/ffi/cusolvermp` (charge Cholesky, LU), `src/ffi/cublasmp`,
  `src/ffi/cusolvermg`. In-tree fallbacks: `common/cholesky_2d`, per-q `jnp.linalg`.

## Phases

- **P1 (agent, GPU)** — stability sweep of ALL block-cyclic FFIs: mesh shapes (1×1, 2×2,
  4×1, 1×4), dtypes (c128 primary), sizes incl. non-divisible + padded extents
  (LORRAX_EXTRA_MU_PAD analogue: logical < padded), repeat-runs for determinism.
  Deliverable: failure catalog + `tests/test_ffi_linalg_contract.py` (skipif per-lib),
  fixes for what breaks.
- **P2 (agent, login+GPU)** — SLATE build hardening: scripted reproducible builds
  (commit scripts to `src/ffi/slate/scripts/`): (a) `gpu_backend=cuda` build, (b) CPU
  path — either the same build exercised host-only or a `gpu_backend=none` second build
  (consult SLATE INSTALL.md + NERSC docs; decide + document). Verify smoke tests vs both.
- **P3 (me, after P1 releases the worktree)** — config integration. REVISED after P2:
  introduce the PORTABLE axis `distributed_cholesky = off | cusolvermp | slate`
  (deprecated alias: `cusolvermp_charge`, via the existing deprecation machinery) and
  `distributed_lu = off | cusolvermp` (alias `cusolvermp_lu`; SLATE getrf wrapper doesn't
  exist yet, so `slate` is NOT a valid LU value tonight).  Graceful absence: probe the
  .so at config resolve, fall back to in-tree with a printed note.  Wire `slate` at the
  cholesky_2d/charge dispatch point.  Unit contract tests only — no golden re-freezes.
- **P4 — REVISED SCOPE (P2 verdict: the FFI is GPU-only today)**: CPU validation of the
  SLATE *library* is DONE (P2 testers on a Milan node).  A CPU-capable FFI is a ~1-2 day
  port (host handler variants + CUDA-free .so target + per-platform loader) — documented
  in the report, NOT tonight.  P4 tonight = GPU-node e2e validation of
  `distributed_cholesky = slate` through the input file on the gnppm fixture (Σ within
  gate tolerance vs the default backend) + absence-fallback behavior check.
- **P5** — full suite, checkpoint (report + CHANGELOG), leave branch unpushed for owner
  review in the morning.

## Ground rules

Never main; suite before each commit (plain 1-GPU invocation, ~4 min); eigh eigvec
artifact is a REAL open defect — root-cause it in P1 if time permits (it blocks using
slate eigh for the QSGW H-build later); no 16-GPU requirements; optional-dependency
semantics = absence must not break any default path or test.
