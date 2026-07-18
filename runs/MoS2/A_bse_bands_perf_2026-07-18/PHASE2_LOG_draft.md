# DRAFT — to append to reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md

## Bandstructure pipeline profiling (2026-07-18, agent/bse-bands-perf, lorrax_A worktree)

Owner: "it does to my intuition seem to be really slow" at 12×12.  Profiled
the exciton-bandstructure pipeline end-to-end at 12×12 (the sibling
production run's restart, `04_mos2_12x12_bands_2026-07-18` nw config,
READ-ONLY), fixed the top findings on `agent/bse-bands-perf` (branched from
`agent/bse-exciton-bands` @ 9fca293 — merge point for the sibling), and
re-profiled.  Own 1-node 4×A100-40G interactive alloc 56095430; all
harness/probe/log artifacts in `runs/MoS2/A_bse_bands_perf_2026-07-18/`.

### Where the time went — BEFORE (3-pt smoke, 12×12 nw, 4×A100, warm FS cache)

124 s total; per-stage (driver tick + `profile_stages.py` sub-split):

| stage | before | what it is |
|---|---:|---|
| solve_scan (warm) | 4.65 s/Q | 118 ms fixed + **115 ms/iteration** × 40 |
| htransform ψ_c(k+Q) | 4.3 s/Q | nQ·nk rank-1152 eighs (28 ms each, batch 32) — **replicated on all 4 GPUs** |
| vq_prepare | 52.9 s | run_gates 18.9 (XHX gate **13.7** = 4.0 host-einsum loop + 9.7 serial-BLAS zgemm) · prepare_coarse 17.5 (eigh only 3.3; 14.2 = host S-rebuild GEMMs + per-q device_get syncs + **15 `_clean_split` recompiles**, one per sphere size) · build_cq 6.7 (host) · run_nulls 6.9 (host) · load_zeta 2.3 |
| load_bse + ht setup | ~11 s | one-time |
| vq_eval | 10 ms/Q warm | dispatch-only — the per-q-recompile lesson held |

40-pt extrapolation BEFORE: **~435 s** ≈ solve 184 + ψ_cQ 170 + vq 53 + fixed 30.
(The sibling smoke's slower numbers — solve 12.45 s/Q, vq 169 s — are CPU
binding: the container numpy BLAS is effectively SERIAL and their srun step
had no `--cpus-per-task`; host stages are thread-starved without it.)

### Solve attribution (`probe_solve.py`, 4 GPU)

- bare stack matvec (bs=8) warm = **123 ms/call** ≈ the whole per-iteration
  cost → the scan solve is 100 % matvec; QR/full-reorth/α overhead
  unmeasurable at this shape.
- The matvec sits at the audit's bandwidth envelope (JOINT_FINDINGS §2):
  T = (μ_loc,ν_loc,ns,ns,nk) = 944 MB/device/trial at 12×12, ~7 HBM
  round-trips ≈ 4.3 ms/trial floor; measured 15.4 ms/trial ≈ 3.6× floor
  incl. 2×2 collectives — same class as the audit's 2.5× at 1×1.  No
  solver-side lever left (c64 stays owner-vetoed).
- **Iteration count is the lever**: Ritz drift bs=8, n_flat=2304:
  max|evs(80)−evs(40)| = 0.000000 meV, max|evs(40)−evs(20)| ≤ 0.0004 meV
  over the lowest 8.  `--max-iter 20` (Krylov 160) halves the solve stage
  below print precision.  Default LEFT at 40 (values-preserving; owner's
  call) — the flag already exists.

### Fixes (commits on agent/bse-bands-perf)

| commit | what | measured |
|---|---|---|
| 18f5cfb | `compute_wfns_fi` q-batch sharded P(('x','y')) (the `_kpath_batch` idiom; was replicated) | 911→346 ms per 32-q batch; ψ_cQ 12.8→4.8 s per 3 Q |
| 809b0fb | vq trainer q-batched on device: ngkmax zero-pad (≤2 compiles, was 15), q axis sharded in chunks of 48, ONE host round-trip per chunk, S off-device; `eigh_backend=auto` = batched native (kills the single-process cusolverMp trap); build_cq/makeVq-gate/nulls-F-rebuild as batched device einsums | prepare_coarse 17.5→4.3 s; build_cq 6.7→0.7 s; nulls 6.9→0.8 s |
| 58e140e | XHX gate on device, k-chunk-accumulated Gram (donated accumulator); build_cq P_R q-chunk-accumulated — peak device residency P_R + one chunk, never full-ψ(9.4 GB)/full-X(14.7 GB at nb=80) | run_gates 18.9→2.4 s; zero remat warnings |
| b352f64 | driver: conduction stacks as ONE jitted pad+reshard (no ~1.7 GB host round-trip); Γ-gate slices before device_get | htransform_psi_cQ tick 13.0→6.0 s |

### AFTER (same 3-pt smoke, 4×A100) — before → after

| stage | before | after |
|---|---:|---:|
| vq_prepare | 52.9 | **9.7** (final split: load 2.1 · cq 0.9 · gates 2.4 · prep 4.3 · nulls 0.8) |
| htransform_psi_cQ | 13.0 | **6.0** |
| solve_scan warm | 13.9 (4.65 s/Q) | 13.9 (unchanged — matvec-bound by design) |
| TOTAL (3-pt) | **124** | **63** |

**Measured 40-pt end-to-end AFTER (4×A100): 432 s** — ψ_cQ 36.4 (5760 q),
vq 9.6, solve cold 185.5 (incl. the ONE compile; 4.58 s/Q) + the driver's
DIAGNOSTIC warm re-scan 183.1 + fixed ~14.  Two further owner levers
measured & wired: (i) the warm re-scan is 42 % of production wall →
`--skip-rerun-check` (962737e; default keeps the assert), (ii)
`--max-iter 20` halves the solve below print precision (probe above).
Both together: 40-pt ≈ **155 s**, vs the pre-fix same-structure
equivalent ≈ 620 s (both-scan) / 435 s (single-scan).  Non-solve stages:
253 → 66 s (3.8×).

**Values**: the 3-pt `.dat` is bit-identical to the pre-fix baseline at
every printed digit; the 40-pt Γ/M rows match the smoke rows and both Γ
endpoints bit-identically; every trainer gate/null reproduces to every
printed digit (makeVq 5.281e-09, XHX 1.714e-11, recon 2.311e-16, stencils
2.197e-15 / 2.251e-15, F-rebuild 1.871e-09); htransform@Γ 0.000 meV /
min-sval 0.8852 (12×12 nw) and 0.000 meV / 0.9432 (3×3).  Gates:
`test_bse_vq_interp.py` + `test_exciton_bands.py` **4/4 passed** on the
3×3 fixture (1 GPU, 25 s); full suite 234 passed / 12 skipped / 25 deselected — TWICE (pre- and post-clamp, 6:29 / 4:55).

### Compile census AFTER (cold cache, JAX_LOG_COMPILES)

`solve_path` 1 · `eval_vq` 1 · `_q_batch` 1 · `_clean_split` 1 (was 15) ·
`_eigh_batch` 1 · `_cq`/`_xhx`/`_chunk`/`_rebuild_chunk`/`_stacks` 1 each ·
~150 one-time tiny eager ops (unchanged class).  Zero per-Q marginal
compiles, zero smoke↔full-path shape retraces.

### Deviations-from-idiom census (dispositions)

FIXED: replicated `_q_batch` eigh (idiom: `_kpath_batch` batch sharding);
`prepare_coarse` serial per-q sync loop + shape-recompiles (runtime-args /
one-compile lesson, applied via ngkmax pad); host serial-BLAS reliance in
gates/nulls/build_cq; full-bundle host round-trip in
`build_conduction_stacks`; whole-stack device_get in the Γ gate.

FOUND PRE-EXISTING (refit-path chunking, task suspect d):
`gflat_to_rmu` (wfn_transforms:867) sets `cs = chunk_size or N` WITHOUT
clamping to the actual row count, then zero-pads N → ⌈N/cs⌉·cs — when the
HBM-budget cs exceeds N (every small-problem galerkin) the per-iteration
FFT box inflates by cs/N: at the 3×3 nb=80 refit galerkin, N = 720 rows
padded to 6103 (8.5×, an 8.4 GB box / 16.76 GiB fused alloc for a
~1 GB-of-data transform).  This is why the refit path is memory-fragile.
Fixed with a one-line clamp `cs = min(cs, N)` (no-op at production GW
scale where cs < N; values identical — pad rows are zeros truncated at
out_flat[:N]).  Commit 2e90edb; 3×3 both-mode .dat (interp + refit rows)
IDENTICAL to the branch point post-clamp.

LEFT, with reasons:
1. `make_sharded_ifftn_3d` returns a BARE shard_map (the W-column-profiling
   anti-pattern) — driver calls it once for W_R (~1-2 s one-time re-lower);
   every hot-path consumer wraps it in jit.  A helper-level fix touches
   gw_jax-wide call sites — out of scope here.
2. `refit_prepare` holds ψ_r = 17.3 GB UNSHARDED on one device and slices
   it as jit args (`feedback_iocallback_for_large_caches` candidate).
   Refit is representation-limited at 12×12/640μ (sibling pivot, WORKLOG
   11:42) and fixture-scale runs fine; flagged for the refit's next
   lifecycle, not fixed blind.
3. Driver default max_iter=40 = 2× the converged Krylov at this shape —
   left (values); probe documented above.
4. The solve's 115 ms/iteration is structural (bandwidth + ns²·nk T-tensor);
   accepted per the audit's verdicts.

### Artifacts

`runs/MoS2/A_bse_bands_perf_2026-07-18/`: `lxrun_perf.sh` (runner),
`profile_stages.py` (sub-stage harness), `probe_solve.py` (matvec/drift),
`probe_gates.py` (gate attribution), `trace_solve_4gpu/` (xprof of one warm
40-iter 3-Q scan), `baseline_smoke_4gpu.log` / `after_smoke_4gpu.log`
(census runs), `prof_*.log`, `12x12/` input scaffold (symlinks into the
READ-ONLY sibling restart).
