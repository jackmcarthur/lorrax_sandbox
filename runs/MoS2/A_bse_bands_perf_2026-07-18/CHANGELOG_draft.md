# DRAFT — to prepend to lorrax_sandbox/CHANGELOG.md

## 2026-07-18: Exciton-bands pipeline perf pass — 12×12 e2e 2×, trainer 5×, values bit-identical [agent/bse-bands-perf, lorrax_A worktree, source, NOT pushed]

Profile-first perf pass on the new exciton-bandstructure pipeline at 12×12
(owner-flagged "really slow"; sibling restart `04_mos2_12x12_bands_2026-07-18`
read-only).  Three structural findings fixed, one structural cost accepted:

- `compute_wfns_fi` (the ψ_c(k+Q) htransform) ran its rank-1152 eigh batches
  REPLICATED on every device → q-batch sharded P(('x','y')) (the
  `_kpath_batch` idiom): 911→346 ms per 32-q batch on 4 GPUs (18f5cfb).
- vq_interp trainer was serial-per-q (host syncs, 15 per-sphere-size XLA
  recompiles, host S-rebuild GEMMs) with its gates/nulls/build_cq on an
  effectively SERIAL container BLAS (30-GFLOP zgemm = 9.7 s) → q-batched
  device pipeline, ngkmax zero-pad = ≤2 compiles, chunk-accumulated Grams
  (never materializes full-ψ 9.4 GB / full-X 14.7 GB), `eigh_backend=auto`
  = batched native (single-process cusolverMp trap gone):
  trainer 52.9→9.7 s (809b0fb, 58e140e).
- Driver conduction stacks + Γ gate moved ~1.7 GB through the host → one
  jitted pad+reshard (b352f64).
- Solve scan: measured 100 % stack-matvec (123 ms/call = the whole
  115 ms/iteration) at the audit's bandwidth envelope — accepted; Ritz
  probe: evs(80)−evs(40) = 0.000000 meV, evs(40)−evs(20) ≤ 0.0004 meV, so
  `--max-iter 20` halves the solve below print precision (default left 40).

3-pt smoke 124→63 s (4×A100); measured 40-pt end-to-end 432 s of which
183 s is the driver's diagnostic warm re-scan (now optional:
`--skip-rerun-check`, default keeps it) — production 40-pt ≈ 250 s, ≈155 s
with the validated `--max-iter 20`.  Also found+fixed a PRE-EXISTING
refit-path memory hazard: `gflat_to_rmu`/`accumulate_rchunk_to_gflat`
padded the flat axis up to an UNCLAMPED HBM-budget chunk (N=720 rows →
6103 at the 3×3 nb=80 refit galerkin — an 8.5× FFT box, 16.76 GiB fused
alloc); one-line `cs = min(cs, N)` clamp, no-op at production scale.
The .dat is IDENTICAL to the pre-fix baseline at every printed digit;
every trainer gate/null value reproduces to every printed digit; gates
`test_bse_vq_interp` + `test_exciton_bands` 4/4, full suite 234 passed /
12 skipped (twice: pre- and post-clamp).
PHASE2_LOG §"Bandstructure pipeline profiling" has the hot-spot tables,
idiom-deviation census (incl. the deliberately-left items: bare-shard_map
FFT helper one-time cost, refit ψ_r 17.3 GB unsharded — refit is
representation-limited at 12×12 anyway, max_iter default).  Run dir:
`runs/MoS2/A_bse_bands_perf_2026-07-18/` (WORKLOG, probes, harness).
Merge point: agent/bse-bands-perf branched from agent/bse-exciton-bands
@ 9fca293.
