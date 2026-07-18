# Exciton-bandstructure pipeline perf session (agent/bse-bands-perf, 2026-07-18)

Task: profile the NEW exciton-bandstructure pipeline at 12×12 (owner: "seems
really slow"), fix the top findings, validate values, re-profile.
Worktree: `sources/worktrees/lorrax_A_bands_perf` @ agent/bse-bands-perf
(branched from agent/bse-exciton-bands @ 9fca293; liblorrax_ffi.so copied per
the fresh-worktree trap).  GPU: own 1-node 4×A100-40G interactive alloc
56095430 (m2651_g), runner `lxrun_perf.sh` (module-free srun+shifter,
OMP_NUM_THREADS=8, distinct job name; sibling alloc untouched).
Inputs: READ-ONLY symlinks into `04_mos2_12x12_bands_2026-07-18` (restart
isdf_tensors_640.h5 + zeta_q.h5 + nw smoke/path inputs) under `12x12/`.

## Timeline

- 12:01 worktree + FFI .so; read PHASE2_LOG/JOINT_FINDINGS/new code.
- 12:08 12×12 input scaffold (symlinks), baseline smoke launched.
- 12:09-12:12 BASELINE (3-pt, 4 GPU, census): 124 s total; census finds
  `_clean_split` compiled 15× (per sphere size).  Log: baseline_smoke_4gpu.log.
- 12:13-12:17 profile_stages.py sub-splits: vq 52.9 s (gates 18.9 /
  prep 17.5 [eigh only 3.3] / cq 6.7 / nulls 6.9), ψ_cQ 4.3 s/path-Q
  (replicated eigh), solve 118 ms + 115 ms/iter.  Log: prof_stages_4gpu.log.
- 12:18 FIX A (bse_setup q-batch sharding): 911→346 ms/batch verified
  (prof_ht_fixA_4gpu.log).  Commit 18f5cfb.
- 12:20-12:30 FIX B (vq trainer batched): EighResult pytree trip fixed;
  values identical; prepare 17.5→4.3, nulls→0.77, cq→0.66
  (prof_vq_fixB2_4gpu.log).  Commit 809b0fb.
- 12:2x probe_gates.py: the residual 13.7 s = XHX gate (4.0 host einsum loop
  + 9.7 s serial-BLAS zgemm — container numpy BLAS ~3 GFLOPS).
- 12:28 FIX D driver (device stacks + Γ-gate slice) + e2e AFTER smoke:
  124→62.9 s, .dat IDENTICAL, all gates identical (after_smoke_4gpu.log).
  Found+fixed my own scale hazard: first device XHX materialized X = 14.7 GB
  (nb=80 fit window) → 30.8 GiB executable remat warning → k/q-chunk-
  accumulated Gram forms (commit 58e140e; run_gates 18.9→2.4 s, remat gone).
  Driver commit b352f64.
- 12:3x probe_solve.py: bare matvec 123 ms ≈ full per-iteration cost (solve
  is 100% matvec, at the audit's bandwidth envelope); evs(80)-evs(40) =
  0.000000 meV, evs(40)-evs(20) ≤ 0.0004 meV → max_iter=40 has 2× headroom
  (default left; --max-iter exists).
- 12:4x pytest gates (3×3 fixture, 1 GPU): 4/4 passed in 25 s
  (pytest_gates.log).
- 12:5x 40-pt end-to-end after-run: 432 s TOTAL of which 183 s is the
  driver's built-in diagnostic warm re-scan → --skip-rerun-check flag
  (962737e, default unchanged).  40-pt values: Γ endpoints + M row
  bit-identical to the smoke rows (after_40pt_4gpu.{log,dat,png}).
- 12:4x-12:5x full suite #1: 234 passed / 12 skipped / 25 deselected in
  6:29 (pytest_full.log).
- 12:43 both-mode 3×3 on my branch OOM'd (16.76 GiB) where pre-fix code
  passed → root-caused to a PRE-EXISTING unclamped chunk pad in
  gflat_to_rmu/accumulate_rchunk_to_gflat (N=720 rows padded to the
  HBM-budget cs=6103 → 8.5× FFT box; the failing run was also sharing
  GPUs with the concurrent suite via srun --overlap).  One-line
  cs=min(cs,N) clamp in both mirrors (2e90edb).
- 12:5x both-mode 3×3 EXCLUSIVE rerun post-clamp: SUCCESS, 30.4 s;
  .dat (interp + refit rows) IDENTICAL to the pre-fix branch point
  (both_3x3_v2.log / both_3x3_OLD.log A/B via SRC_OVERRIDE to the
  sibling worktree, read-only).
- 13:0x full suite #2 post-clamp (pytest_full_postclamp.log) + PHASE2_LOG
  + CHANGELOG appended; allocation released.

## Key findings (detail in PHASE2_LOG § Bandstructure pipeline profiling)

1. `compute_wfns_fi` eigh batch was REPLICATED across devices (idiom breach
   vs `_kpath_batch`); ~170 s of a 40-pt path at 12×12.  Fixed → sharded.
2. vq trainer was serial-per-q with host round-trips, 15 shape recompiles,
   and serial-BLAS host GEMMs (gates/nulls/build_cq/XHX).  Fixed → q-batched
   device pipeline, chunk-accumulated Grams, ≤2 compiles, values identical
   to every printed digit.
3. The solve scan is matvec-bound at the audit's bandwidth envelope —
   no code lever; iteration count has 2× headroom (owner's flag).
4. LEFT (documented): bare-shard_map FFT helper (one-time), refit ψ_r
   17.3 GB unsharded (refit is representation-limited at 12×12 anyway),
   max_iter default.

## Environment note

The container numpy BLAS is effectively SERIAL: a 30 GFLOP zgemm took
9.7 s.  Any host-side linalg in this pipeline is ~3 GFLOPS — device-batch
or thread it.  Also: srun steps without `--cpus-per-task` throttle host
stages (the sibling smoke's vq 169 s vs my 53 s baseline is this).
