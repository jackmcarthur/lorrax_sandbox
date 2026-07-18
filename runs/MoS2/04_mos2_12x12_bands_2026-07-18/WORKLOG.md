# MoS2 12x12 exciton bandstructure — session worklog (agent A, 2026-07-18)

Owner-requested production run: regenerate the exciton bandstructure with a
12x12 coarse grid (vs the delivered 3x3-fixture-trained
`B_exciton_bands_2026-07-17`) + denser Gamma-M-K-Gamma path; overlay both.
Owner addendum: per-stage wall-time table (GPU counts per stage) is a
first-class deliverable.

Code: `sources/worktrees/lorrax_A_exciton_bands` @ agent/bse-exciton-bands
(159447c + this session's `--refit-r-chunk` plumb).

## Plan (tracked steps)

1. [x] Run dir + inputs: qe/nscf (144 k, nbnd=82, SCF save COPIED from
       00_mos2_3x3_cohsex/qe/scf — never mutate the parent), pw2bgw (nk=12),
       00_lorrax_cohsex (fixture cohsex.in verbatim, k-grid via WFN.h5;
       WFN/kih symlinks; centroids_frac_640.txt symlink to the SAME file as
       the 3x3 trainer — geometry-only per skills/execute_workflow, and its
       orbit-closure failure is what keeps zeta storage full-BZ, which
       vq_interp requires), 01_lorrax_exciton_bands (path 15/8/16 = 40 pts).
2. [x] Allocation: sbatch sleeper 1 node 4x A100-80GB (hbm80g needed: refit
       state psi_r+B_full ~ 20+ GB at 12x12, + Xf 9.4 GB + chunk temps),
       qos regular 6 h, jid 56081997 (submitted 02:41 PDT).
3. [x] Code: `--refit-r-chunk` CLI on exciton_bands.py -> refit_prepare
       (default 2048 unchanged; 12x12 needs 512: the refit Z-build temp
       rho is (nk,nb,nb,r_chunk) c128 = 30 GB at r_chunk=2048, OOM even on
       80 GB).
4. [ ] NSCF -> pw2bgw -> wfn2hdf (4 GPUs QE, timed).
5. [ ] dipole + kin_ion (1 GPU each) + gw_jax 4-GPU COHSEX+do_screened ->
       144-q restart; verify zeta_q.h5 full-BZ (nq=144) + W0_qmunu.
6. [ ] Driver smoke: 3-pt path interp-only (census runner) at new sizes;
       max_iter 40-vs-80 convergence probe.
7. [ ] diag_loo.py: trainer gates/nulls + subsampled LOO (every 6th q).
8. [ ] Final run: 40-pt path, --vq-mode both, refit spot checks
       3,8,12,15,18,28,34 (15 = M node, on-grid floor reference; rest
       off-grid), census; exciton_bands_12x12_GMKG.{dat,png}.
9. [ ] plot_compare.py overlay vs delivered 3x3 .dat; PHASE2_LOG
       §"12x12 bandstructure" (incl. the full timing table), CHANGELOG,
       manifest, commits (code on branch; artifacts on sandbox main).

## Convention notes

- Path counts 15/8/16 ~ segment-length ratio (delivered 12/6/13), 40 total
  points, nodes at 0/15/23/39.
- On the 12x12 grid the M (iQ 15) and K (iQ 23) nodes are ON-grid training
  points; refit list includes M as the on-grid floor row.  Off-grid spot
  checks avoid the accidental on-grid path points (Gamma-M j=5,10; M-K j=4;
  K-Gamma j=4,8,12).
- Same 640-centroid file as the 3x3 trainer => same mu set; comparability
  of tiles/B-blocks across grids is exact at shared q (3x3 grid subset of
  12x12: yes — 1/3 = 4/12).

## Timing ledger (raw; table in PHASE2_LOG at the end)

- 02:41 PDT sbatch 56081997 submitted (queue wait excluded from grand
  total, reported separately).

## Log

- 2026-07-18 11:21 PDT OWNER REDIRECT: interactive QOS instead of the regular
  queue.  56081997 (regular sleeper, 8.7 h queued, never started) scancelled
  — queue wait excluded from the timing table; interactive wait ~= 0.
- 11:23 hbm80g x4 interactive NOT granted in 160 s (Resources) — cancelled
  per plan; plain-gpu fallback GRANTED in ~40 s: jid 56094475, 4 nodes
  nid[001097,1156-1157,1160], 16x A100-40G (all hbm40g), 4 h.
  Consequence: refit sized to 40 GB HBM (--refit-r-chunk 256; psi_r 17.3 +
  Xf 9.4 + B_full + rho(256) 3.8 ~= 35 GB — MARGINAL, flagged as risk).
  Stage layout per redirect: NSCF 4 GPU / GW 16 GPU (4x4) / driver 4 GPU
  single-node.
- 11:27 NSCF launched (4 GPU, npools 4) + worktree pytest (test_bse_vq_interp
  + test_exciton_bands) concurrently on the allocation.
- 11:29 QE chain COMPLETE: NSCF 35 s (144 k, converged, avg 38.6 diag iters),
  pw2bgw 47 s (CPU workaround), wfn2hdf 27 s.  WFN.h5 740 MB, kih/vxc ok.
  pytest 4/4 in 44 s; --refit-r-chunk commit 9fca293 on the branch.
- 11:31 GW chain launched: dipole + kin_ion (1 GPU each) then gw_jax
  COHSEX+do_screened on 16 GPUs (4 nodes, 4x4 mesh).
- 11:35 GW chain COMPLETE rc=0: dipole 152 s, kin_ion 43 s, gw_jax 150 s
  wall (recorded 98.6 s: zeta_fit 80.7 / V_q 9.2 / chi0_W 1.3 / sigma 2.6).
  Restart verified: isdf_tensors_640.h5 2.1 GB (16 rank shards, 4x4 mesh),
  zeta_q.h5 2.9 GB = 144 of 144 FULL-BZ spheres (orbit-closure fallback as
  designed), W0_qmunu + heads persisted (vhead 9315.306, whead0 3499.067),
  eqp0.dat 144 k.
- 11:37 driver smoke launched: 3-pt path, 4 GPUs (--px 2 --py 2, first
  multi-GPU driver run — owner-directed), census cache empty, max_iter 40.
- 11:38 SMOKE FAILED at the htransform@Gamma gate: max|d_eps_c| = 955.369
  meV, conduction-subspace min-sval = 0.2175 (assert > 0.5).  Root cause
  identified from the smoke's own SVD line "SVD of (11520, 1280):
  rank=1280": the htransform alpha-basis is capped at ns*n_mu = 1280
  fields, and the driver hands initialize_wfns the SIGMA window (nval=26/
  ncond=54 -> 144 k x 80 b = 11520 states).  At 3x3 (720 states) the basis
  was over-complete and the gate read 0.000 meV / 0.943; at 12x12 the same
  640 centroids CANNOT carry the 80-band window.  A representation limit,
  not a bug.
- 11:40 1-GPU discriminator (same input, --px 1 --py 1): IDENTICAL numbers
  bit-for-bit (955.369 meV / 0.2175) => NOT a multi-GPU/mesh artifact; the
  4-GPU driver layout stays.
- 11:42 Fix: NARROW-WINDOW driver inputs (*_nw.in: nval=2/ncond=6/nband=8,
  window bands 24-32 -> 1152 states <= 1280, full-rank regime restored; the
  BSE windows only need conduction 26-30 + guards).  Loader-safe: n_occ
  from WFN ifmax; b_min=nval_in=2 is window-relative (abs band 26).
  CONSEQUENCE: htransform-refit ground truth (nb=80 window asserted by
  refit_prepare) is unavailable at 12x12/640mu — the off-grid refit m-leg
  is representation-limited (same 0.2175-sval physics).  Spot-check pivot:
  (a) subsampled LOO (24 held-out q, tile+B relF vs stored) — off-grid fit
  quality; (b) DENSE on-grid exciton ground truth at the 8 on-grid path
  points (iQ 5,10,15=M,19,23=K,27,31,35): dense eigh (n_flat=2304) with the
  STORED disk tile vs the interp tile (stored psi, stored W0 — no
  htransform anywhere), joined against the driver's Lanczos rows.
- 11:41-11:55 narrow-window smoke v1 hit a SECOND trap: rank=0 (all-zero
  psi at centroids).  Probe ladder (probe_nw{1..5}.py): raw wfn.load
  (24,32) fine; gflat_to_rmu fine at any chunk size; cross test pinned it
  to META — Meta.b_id_4_user (= input nband) is the ABSOLUTE band-budget
  top, and load_centroids_band_chunked force-zeroes bands >=
  b_id_4_user (wfn_transforms.py:1870 "zero user-band-pad rows"): with
  nband=8 the absolute window (24,32) is entirely "pad".  Fix: nband=32
  in the _nw inputs (nband deviates from nval+ncond BY DESIGN here;
  documented in the input header).  No code change needed.
- 11:50 THIRD trap (after gates passed): --eigh-backend auto on the 2x2
  SINGLE-PROCESS mesh dispatches prepare_coarse's eigh to cusolverMp,
  which requires one JAX process per device ("mesh 2x2 does not cover all
  1 JAX processes").  Driver runs single-process/4-device; fix:
  --eigh-backend off (640x640 native eigh, trivial).
- 11:56 SMOKE PASS end-to-end (smoke_12x12_nw3.log, 286 s total, 4 GPU):
  htransform@Gamma gate 0.000 meV / min-sval 0.8852 (3x3 class was 0.943);
  wrapfix relabeled 4/144 half-boundary q (the KSE sphere-wrap trap,
  handled); ALL trainer gates OK (makeVq-vs-disk 5.3e-9 all-q max); nulls
  OK (1.2e-15/1.9e-9); gset(0.3)=337 G, 0 zero-filled channels; census =
  1 solve_path compile; warm 12.45 s/Q (Krylov 320 = 8x40 unclamped,
  n_flat 2304); vq_prepare (trainer at 144 q) 169 s.
- 11:57 launched concurrently: max_iter 40-vs-80 probe (4 GPU) +
  diag_loo.py LOO/dense-truth (1 GPU, other node).
- 12:05 solver probe: max|dE(it80-it40)| = 0.000000 eV on all 3 smoke rows
  (dat precision 1e-6 eV) — Krylov 320 converged; production max_iter=40.
  (it80 warm 48.9 s/Q vs it40 12.4 — full-reorth O(M^2).)  FINAL 40-pt run
  launched (4 GPU, census).
- 12:07 diag_loo COMPLETE (627 s, 1 GPU): LOO(24 held-out q) tile relF
  median 6.85e-2 / max 1.28e-1 (argmax q=0 — held-out Gamma hardest, head
  channel); physical B-block relF median 3.04e-3 / max 2.01e-2 (q=0).
  DENSE on-grid exciton truth (8 path points, stored psi + stored W0,
  n_flat 2304 dense): interp-vs-stored |dE| <= 0.034 meV over all 64
  states — V_Q interpolation is exciton-level exact on-grid.  FIRST
  PHYSICS: 12x12 exciton bands sit at ~1.61-1.97 eV (K-point E_1 1.609,
  M E_1 1.683) vs the 3x3-era ~0.18-0.40 eV — a >1.2 eV k-convergence
  shift (deep-binding artifact of the 3x3 BZ sampling), to be shown in
  the overlay.
- 12:25 ALLOCATION 56094475 LOST mid-final-run (the background salloc
  holder task was killed — same failure mode as the 2026-07-17 session's
  56074608; final run died at rc=143 ~20 min in, no .dat).  Replacement
  56095826 (1 node nid001097, 4 GPU, 3 h interactive) granted in ~40 s
  via a DETACHED holder (nohup setsid salloc ..., pid 689652 — not a
  harness-tracked task, immune to task reaping).  ~1 min compute gap.
  Final rerun launched 12:26.
- 12:3x coordinator FYI (perf agent, branch agent/bse-bands-perf, 7 commits
  18f5cfb..2e90edb, branched at my 9fca293; merge AFTER this run, no
  rebase mid-run): (a) my stage timings are largely ENVIRONMENT-bound —
  container numpy BLAS ~3 GFLOPS single-threaded host linalg, and srun
  steps without --cpus-per-task are core-throttled — interpretation note
  for the timing table, absolute walls here are NOT algorithmic limits;
  (b) its trainer (52.9->9.7 s; serial per-q loop + 15 per-sphere
  recompiles + host BLAS) and htransform-batch (4.3->2.0 s/Q; replicated
  eigh -> sharded) fixes are BIT-IDENTICAL on .dat vs my branch point, so
  production numbers here stand and speedups land at merge; (c) validated
  knobs on that branch: --skip-rerun-check (drops a 183 s always-on
  re-scan diagnostic) and --max-iter 20 (<=0.0004 meV vs 40) — noted for
  the report, not used mid-run.
- 12:51 FINAL RUN COMPLETE rc=0 (final_12x12_v2.log): 40 pts, 1474 s on 4
  GPU, census = 1 solve_path compile, warm 13.7 s/Q.
  exciton_bands_12x12_GMKG.{dat,png} written.  Overlay PNG rendered
  (12x12 sits ~0.7-1.0 eV above 3x3 — the k-convergence shift; 12x12
  visibly smoother).
- 13:0x diag_loo's dense table was missing the q=0 W-HEAD injection
  (loader does W0[q=0] += whead0/V_cell * conj(g0) g0^H; head-less dense
  sat ~470 meV HIGH).  diag_dense_head.py redoes dense truth with the
  head + joins driver rows: iQ 5 — driver-vs-dense-stored -0.019 meV
  (E_1), interp-vs-stored <= 0.025 meV; upper states few-meV
  (representation floor), E_8 +11.5 meV worst.
- 13:0x DIP investigation (iQ 9, Q=(0,0.3), lowest 2 states ~170 meV
  below neighbors): diag_dip.py micro-scan shows the free-pair floor
  D_min(Q) itself dips ~350 meV centered Q_y~0.29 (smooth across
  0.28-0.31, argmin-k jumps) — a conduction-energy kinematics feature
  (Lambda-valley-like), NOT a V_Q/W glitch.  Window-guard A/B in flight
  (window 24-32 vs 25-33) to discriminate physical vs window-truncation.

## Agent-B session (2026-07-18 afternoon) — SP bands + 1000-centroid variant

- 14:5x Session start (agent B, own alloc 56101152, 1 node 4x A100-40G,
  interactive, holder detached).  Deliverables: (1) htransform SP
  bandstructure + free-pair floor D_min(Q) [05_htransform_spbands]; (2)
  1000-centroid exciton variant [02_/03_..._1000c].
- 15:0x sp_bands v1 (windows (24,32)+(20,28)): on-grid gate EXACT (0.0000
  meV band 25 vs stored DFT); D_min(Gamma)=D_min(K)=1.7001 eV (direct gap);
  D_min shows the iQ 6 and iQ 9 dips.  BUT cross-window overlap gate max
  2.5 eV -> window-quality investigation.
- 15:1x gap_scan.py (stored enk_full): Kramers pairs (even,odd) EXACTLY
  degenerate at 12x12; safe window boundaries only between pairs, min-gaps:
  23|24 241 meV, 25|26 1700 (the gap), 33|34 2194, 27|28 75.8, 21|22 80.7,
  19|20 64.7, 29|30 37.6, 31|32 5.9 (!), 24|25 = 0 (pair).  A boundary that
  cuts a pair = eV-scale off-grid failure (probe: w2533 band 26 1.5 eV off;
  explains the sibling's w2331 A/B blowing up ~2 eV — that A/B window CUTS
  the (22,23) pair, so its ARTIFACT? flags were measuring its own breakage).
  Same mechanism class as the Si degeneracy root-cause (window truncation
  of degenerate multiplets, commit 73e58f79).
- 15:3x probe_spike.py fine scans + sp_bands v3 (windows (20,28) valence /
  (26,34) conduction [nval=0, a_band=28] / (24,32) D_min ref):
  * cond(26,34) — both boundaries at the two LARGEST gaps — is SMOOTH
    everywhere; ref(24,32) (5.9 meV 31|32 top boundary) RINGS at off-grid q:
    isolated 100-1000 meV excursions (e.g. band 28 -0.47 eV mid M->K leg),
    exactly bracketed by on-grid agreement; val(20,28) has its own isolated
    off-grid blemishes (band 24 ~250 meV @iQ 20, band 25 ~110 meV @iQ 30).
    Cross-window medians 1.3-4 meV = healthy interpolation floor.
  * D_min A/B (24,32)-vs-(26,34) over the full 40x144 k+Q set: median 9.7
    meV, on-grid rows EXACT, max 316.6 meV AT iQ 9.  The iQ 6/9/16-17
    "dips" exist ONLY in the driver-window curve; the clean-window floor is
    smooth there.  The sibling's exciton E_1 dips TRACK the driver-window
    curve => those exciton wiggles are HTRANSFORM WINDOW-CACHE ARTIFACTS
    (inherited through eps_c/psi_c(k+Q)), not Lambda-valley kinematics, not
    ISDF-basis error.  PREDICTION for the 1000c run: same window verbatim
    -> same iQ 6/9 dips (fH_k smoothness is window physics, not basis size).
  * DELIVERED: 05_htransform_spbands/sp_bands_12x12_GMKG.{dat,png} (2-panel:
    SP bands + D_min both-window overlay w/ exciton E_1), dmin_12x12_GMKG.dat.
    Caveat: plotted valence 22-25 (val(20,28)) carries the isolated iQ 20/30
    blemishes; conduction 26-31 (cond(26,34)) is clean.  Total 287 s (1 GPU).
- 15:2x kmeans 1000c attempt 1 (1 GPU) OOM in pivoted-Cholesky Gram
  (pair_density 20.7 GB @ M=1500, nk=144) -> rerun on 4 GPUs (sharded).
- 15:27-15:31 kmeans trap ladder: (i) 4-GPU run silently fell back to
  single-device (P/4 = 11520 < 100k per-shard floor) -> --force-shard;
  (ii) 2x2 mesh rejected (nb_total=26 % 4 != 0) -> 2 GPUs (1x2 mesh);
  (iii) still OOM at 10.5 GB under default BFC pool -> platform allocator
  envs (same as run_wt runners).  SUCCESS: centroids_frac_1000.txt (1000
  unique, seed 42, --no-orbit literal, prune left=(0,26)/right=(0,52) =
  the 640 convention), 19 s.
- 15:31 alloc swap: 56101152 released; 4-node interactive 56101959 granted
  in ~20 s (nid[002917,002920,002925,003917], holder detached).  GW chain
  launched 15:31 (16 GPU, 4x4 mesh, cohsex.in verbatim except
  centroids_file; dipole/kin_ion reused from 00 — centroid-independent).
  gw.out confirms: orbit closure FAILED (879/2000) -> FULL-BZ zeta on disk,
  as vq_interp requires (matches the 640 run's convention).
- 15:34 GW 1000c COMPLETE rc=0: wall 195 s (16 GPU), recorded 129.9 s
  (zeta_fit 106.8 / V_q 11.7 / chi0_W 1.6 / sigma 4.4) vs 640c (150 s wall,
  98.6 recorded: 80.7/9.2/1.3/2.6) — only 1.3x, well under the ~2.5x
  estimate.  Restart: isdf_tensors_1000.h5 4.99 GB, zeta_q.h5 4.53 GB
  (full-BZ), eqp0.dat 144 k, heads MATCH the 640 run (vhead 9315.306,
  whead0 3499.067).  15:35 launched concurrently: driver smoke 3-pt
  (4 GPU, nid002917) + v4 SP-bands (24,36)@1000c (1 GPU, nid002920).
- 15:41 v4 SP-bands COMPLETE (302 s, 1 GPU): (24,36)@1000c removes the
  val(20,28) ringing on bands 24/25 (365-392 meV @iQ 16/18); conduction
  agrees with (26,34)@640c to <=58 meV (median ~1 meV) across windows AND
  bases; clean D_min floors agree median 6.8 meV (max 113 meV @iQ 17 — the
  M-K half-integer-coordinate rows are the residual uncertainty everywhere).
  Deliverable PNG regenerated with >50 meV window-A/B uncertainty circles
  (valence M-K leg; BSE uses valence ON-grid only, where htransform is
  exact).  Committed a78606d8.
- 15:44 driver smoke 1000c PASS end-to-end (572.6 s, 4 GPU): htransform@
  Gamma 0.000 meV / min-sval 0.9010 (640c: 0.8852 — richer basis helps);
  wrapfix 4/144 (same rows); ALL trainer gates OK (makeVq-vs-disk 5.0e-9
  all-q); nulls OK; census = 1 solve_path compile; vq_prepare 397.5 s
  (640c: 168 — pre-perf trainer is the n_mu^2 hotspot); warm 20.0 s/Q
  (640c 13.7).  EARLY 640-vs-1000 SIGNAL (smoke rows vs 640c final rows):
  |dE| ~ 4-12 meV per state (Gamma E_1 -11.5 meV, M E_1 -5.6 meV) — the
  basis effect is small and uniform, no restructuring.  15:45 FINAL 40-pt
  1000c run launched (4 GPU, census, ~45 min ETA).
- 16:23 FINAL 1000c COMPLETE rc=0 (final_1000c.log): 2272 s on 4 GPU,
  census = 1 solve_path compile, warm 20.7 s/Q (ψ_cQ 184 / vq_prepare
  394.5 / cold 825.2 / warm 827.3).  exciton_bands_1000c.{dat,png} written.
- 16:2x OVERLAY + VERDICT (exciton_bands_640c_vs_1000c_GMKG.png):
  per-state |dE(1000c-640c)| median 9.7 / mean 11.0 / max 46.5 meV
  (@iQ 11); artifact rows iQ 6/9/16-17 shift by the SAME ~10 meV (max
  41.8) — the dips PERSIST (iQ 9 E_1 depth 180 -> 207 meV).  ANSWER to
  the owner: a more converged ISDF basis does NOT smooth the exciton
  bands; the wiggle mechanism is the (24,32) htransform window (see
  05_htransform_spbands), and the smoothing lever is a clean-boundary
  driver window ((22,34) fits at 1000c capacity; follow-up).
  diag_dense_head_1000c (dense on-grid stored-tile spot checks) running.
- 16:35 diag_dense_head_1000c COMPLETE (549 s, 1 GPU): 8 on-grid path
  points, dense-interp vs dense-stored max|dE| <= 0.056 meV over all 64
  states (V_Q interpolation exciton-level exact at 1000 mu, matching the
  640c 0.034 meV); driver rows within 1.4-8.5 meV of dense-stored
  (htransform-cache + Lanczos representation floor, same class as 640c).
- 16:3x Session wrap: PHASE2_LOG §"1000-centroid variant + SP bands",
  CHANGELOG, manifest -> complete; allocation 56101959 released after
  final commit.  Follow-up filed in PHASE2_LOG: clean-boundary driver
  window (22,34)@1000c as the actual smoothing lever for iQ 6/9/16-17.
