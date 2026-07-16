# BSE cleanup log — CLEANUP_LOG.md

Running log for the `agent/bse-cleanup` program. Records landed cleanups and,
under "Deferred consolidations", forest-not-trees consolidations that a scoped
change surfaced but must NOT be half-done inside that change (mandate #1).

## Deferred consolidations

Surfaced 2026-07-15 by the recover-or-generate-V/W design scout
(`archive/designs/recover_or_generate_vw.md`). None actioned — recorded only.

- **BSE config → `gw_config.read_lorrax_input`**: the RECOVER path still reads
  cohsex.in through bse_io's private `_parse_head_overrides`/`_parse_wfn_path`
  (bse_io.py:667-695, 584-600), bypassing the canonical parser. The GENERATE
  path already forces `LorraxConfig.from_input_file`. Unify BSE's cohsex.in reads
  onto `gw_config` so there is one config surface. Larger than the seam change;
  belongs with the loader-repair/config pass (MAP §4 item 7).
- **Mesh builder unification**: `gw_jax._build_mesh` and
  `bse_ring_comm.create_mesh_2d` compute the *same* x×y factorization
  (largest divisor ≤ √ndev) in two places. Collapse to one shared builder so
  GENERATE (gw) and BSE agree by construction rather than by coincidence.
- **`gw_jax.main()` setup-prefix extraction** (`build_gw_restart`, gw_jax:187-287):
  needed by the recover-or-generate seam so bse/ calls the SAME producer chain
  with zero duplicated orchestration. Behaviour-preserving move; land as its own
  commit gated by the fresh-restart e2e gate (MAP §5 gate 2 / §7 step 1).

---

## Phase-1 executor pass — 2026-07-15 (`agent/bse-cleanup`, worktree lorrax_D_bse_cleanup)

Base `origin/main` c7a30ff. 13 commits, most-recent-first:

| hash | one-liner |
|------|-----------|
| 6bd4dc9 | archive feast_*_sweep.py experiment harnesses out of src/ |
| bd3ff6d | delete dead container + preconditioner-assembly scaffolding |
| ce724ab | delete dead matvec variants (module-level trio, ring pair, symmetrize_W_q) |
| 3f30db3 | delete superseded write_eigenvectors.py |
| eb316f4 | consolidate BSE eigenvector writer + kill test_bse's stale loader |
| 402e1cc | fix STATUS.md default-matvec drift + bse_feast docstring drift |
| adfeb9a | restore use_nohead kwarg + loud bare-V-fallback warning |
| 0495a44 | restore lost density drive/readout operators (pseudopoles wiring) |
| 41db774 | drop phantom v_couples_k kwarg (kpm + pseudopoles) |
| 12829d1 | single-source the --eqp band-window re-slice (B7) |
| b22f682 | _pad_axis_to_multiple returns padded extent, not pre-pad size (B6) |
| 25c3248 | fix B-side transition einsum (unbound output index 't') (B2) |
| 0a6d407 | single-source V/W μν layout shim; fix flat-q sharded/ring loaders (B3/B4/B5) |

### Completed
All VERIFIED work items acted on except the two marked `skip` in the brief:
loader normalization (B3/B4/B5), B2 einsum, B6 pad-extent, B7 eqp single-source,
v_couples_k removal (kpm+pseudopoles), pseudopoles density-operator wiring
restore, use_nohead + loud W0 bare-V warning, STATUS/docstring drift; dead-core
deletes (bse_jax matvec trio, bse_ring_comm ring pair, bse_serial symmetrize_W_q,
bse_io BSEData, bse_preconditioner 7-symbol scaffolding); write_eigenvectors.py
+ generate_kpts_grid consolidation + test_bse stale-loader removal; feast_*_sweep
archival.

### Skipped (per brief)
- **bse_feast_dense_debug.py** — left in src/ (untouched). Its inline-quadrature
  dedup + pytest-fixture repurposing is a design task, not a repair-item swap
  (see deferred item below).
- **v_couples_k restoration** — the dead kwarg was *removed* to unblock the
  crash; the physics it historically gated is NOT restored (see below).

### Runtime-check status: PENDING
No GPU pool reachable this session: `squeue -u jackm` empty; `module load
lorrax_agent` refuses without a base `lorrax_X` module + LORRAX_ROOT; no
lxstatus/lxattach/lxrun on PATH. Per mandate #4, no bare `python3` was run on the
login node — verification was grep/read-level only (zero-caller re-greps at HEAD
before every deletion; per-element einsum index math for B2; import-consistency
greps after each repair). Import-level (`import bse.bse_io, bse.bse_ring_comm,
bse.bse_kpm, bse.bse_pseudopoles`) and the plain 1-GPU pytest suite remain to be
run once a pool exists.

### Deferred consolidations surfaced this pass (NOT actioned)
- **B6 padded-slot spurious eigenvectors (Lanczos/Davidson)** — `_pad_axis_to_multiple`
  now returns the true padded extent, but padded band slots carry eps=0 →
  spurious ~zero-energy transitions. KPM/FEAST mask these; the Lanczos/Davidson
  Krylov start vectors do not. The fix (thread an n_cond/n_val mask into
  `solve_bse_sharded` and `solvers.lanczos` start-vector RNG) is a deeper,
  physics-observable change into shared solver code — needs GPU verification of
  spurious-eigenvector behaviour. Not bundled into the one-line B6 fix.
- **B2 einsum M/N-assignment open question** — `kvtM,bvksN->bMNtsk` is the
  minimal fix that makes the trace legal, but whether output slots M/N draw from
  psi_v vs the carried buffer in the convention `apply_W_from_T` expects is
  UNVERIFIED. On a square (px==py, n_rmu==n_rnu) mesh a wrong-axis swap cannot
  raise a shape error, and the gather/ring sibling shares the bug so they can't
  cross-check. Needs a dense reference on GPU.
- **v_couples_k / exchange-kernel k-contraction (B1)** — restoring the historical
  `v_couples_k`→`apply_V_ring(couple_k=...)` Σ_k-summing behaviour is walled off
  under mandate #3. Whoever lands the B1 exchange-kernel adjudication owns the
  v_couples_k restoration too — it is bundled with that fix, not a loose end.
- **apply_q0_head_rank1 (gw/head_correction.py) now dead** — B5 switched the
  single-device ring loader to `apply_q0_head_rank1_sharded` (its layout matches
  the post-shim V_q0/W_q). The k-first-layout `apply_q0_head_rank1` now has no
  callers (only doc/comment mentions in gw_output.py:189, tagged_arrays.py:167).
  Deleting it is GW-module scope, not BSE — deferred for a GW-side pass.
- **Ring-loader W0-fallback head semantics** — when W0 is not ready, the ring
  loader's W_q (bare-V fallback) no longer receives the q=0 head (previously it
  implicitly got vhead because injection ran on the shared raw V array before the
  W_src split). This only affects the explicitly-unphysical bare-V debug path
  (now loudly warned). Behaviour on the real W0-ready path is unchanged.
- **bse_feast_dense_debug.py quadrature dedup** — cannot naively `from
  solvers.quadrature import feast_ellipse_quadrature`: `solvers/__init__.py`
  eagerly imports jax-heavy submodules (davidson/lanczos/chebyshev/dos/
  pseudobands), converting a documented jax-free numpy debug script into a
  hard-jax-dependent one. Dedup needs `solvers/__init__` restructured to lazy-
  import, or a jax-free leaf target; the pytest-fixture repurposing is a separate
  design task.
- **bse_preconditioner.py now single-function** — after deleting the 7-symbol
  scaffolding, the module holds only `energy_diff_cv_k`. Could fold into
  bse_serial (its main consumer) to drop a file, but that touches bse_feast's
  import too; minor, left as-is.
