# b600 / P=64 scale shakeout — MoS2 4x4 deck

Full driver chain (centroid -> dipole -> kin-ion -> gw -> htransform x2) at
600 bands, 4775 charge centroids, 64 ranks (8x8 mesh), Frontera CLX.
2026-08-01.  Artifacts under `/scratch2/08271/jackmc/b600_p64/`.

## Configuration

| item | value |
|---|---|
| source | `/work2/08271/jackmc/frontera/lorrax` @ 1fc2759, frozen to `lorrax_frozen/{src,config}` (byte-verified copy; no job read the live checkout) |
| bundle | `bundle_1fc2759/lorrax_cpu_bundle.tar` (409 MB, byte-compiled), job 7885312 |
| geometry | 32 nodes x 2 ranks/node x 28 threads = P=64, 8x8 square mesh, `development` |
| transport | `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, patched thread-MULTIPLE MPIwrapper, provider `mlx` (banner-confirmed on every leg) |
| compile cache | ON (`$SCRATCH/lorrax_jax_cache/np64`) — no `ISDF_JAX_CACHE_DIR=""` opt-out |
| FFI | `build_host_ONE/liblorrax_ffi_host.so`; flat-k MKL FFT + vendor GEMM engaged (banners); `slab_io=auto` -> PHDF5_FFI collective writer (NOT the allgather fallback) |
| WFN | `mos2_4x4_test/WFN_b1024.h5` (1024 bands, same 30 Ry / 4x4x1 lineage as WFN_b300; 0.624 MB/band in both) |
| deck | `deck_b600.in`: nval 26, ncond 18 (UNCHANGED from b300), nband 600, memory_per_device_gb 40 |
| centroids | 4775 (requested 4800), orbit-closed, rank gate 431/431 PASS, selection window (0,26)x(0,600) |
| mu_pad | 4800 = round_up(4775, 64) — automatic via `Meta.n_rmu_padded`; NO divisibility ladder was needed |

No NSCF was run: `WFN_b1024.h5` already covers 600 bands on the identical
deck, and `nband=600` is honest — `load_centroids_band_chunked` zeroes
psi(G) on bands [b_id_4_user=600, b_id_4=640) even though the file holds
them.  The QE->pw2bgw certification side-deliverable is therefore NOT
triggered by this campaign and stays open in KNOWN_LORRAX_ISSUES.

## Instrument note (read before believing any memory number)

`sacct` MaxRSS is NOT usable for this harness.  Job 7885315: sacct
reports MaxRSS 15800K / 26928K / 18336K for the three steps while the
per-rank `/proc/<pid>/status` VmHWM sampler reports 10.61 / 2.10 / 2.15
GiB — a ~700x undersample.  Cause: this harness runs python as a CHILD
of the container shell (so a sampler can read its /proc), so the srun
task Slurm accounts is the `apptainer` wrapper.  The certified
`gw_dev.sbatch` template `exec`s python, which is why the b300 sacct
numbers quoted below (4.63 GiB) ARE real RSS peaks and directly
comparable to VmHWM.  All b600 memory numbers below are VmHWM at 5 s
sampling; a peak confined to the last 5 s would be missed.

## 1. Chain status

| driver | job | geometry | wall (leg) | recorded | per-rank peak VmHWM max/mean | verdict |
|---|---|---|---|---|---|---|
| centroid.kmeans_cli 4800 --orbit | 7885315 | 32x2, P=64 | 51 s | 32.6 s | 10.61 / 9.05 GiB | rc=0 — 4775c, rank gate 431/431 PASS |
| psp.get_dipole_mtxels | 7885315 | 32x2, P=64 | 22 s | 7.0 s | 2.10 / 1.25 GiB | rc=0 — dipole_b600.h5 322 MB |
| gw.kin_ion_io -n 600 --hartree | 7885315 | 32x2, P=64 | 17 s | 8.5 s | 2.15 / 0.66 GiB | rc=0 — kin_ion_b600.h5 184 MB |
| gw.gw_jax (default zeta tier) | 7885316 | 32x2, P=64 | 250 s | 240.6 s | 16.24 / 16.17 GiB | rc=0 — all five outputs written |
| bandstructure.htransform (dft) | 7885322 | 32x2, P=64 | 28 s | 12.1 s | 2.31 / 2.29 GiB | rc=0 |
| bandstructure.htransform (qp) | 7885322 | 32x2, P=64 | 18 s | 12.1 s | 2.22 / 2.20 GiB | rc=0 — EQP override PROVEN taken |
| gw.gw_jax (distributed zeta tier) | 7885323 | 32x2, P=64 | 194 s | 186.4 s | 16.22 / 16.19 GiB | rc=0 — diagnostic leg, §6 |

No OOM, no refusal, no traceback anywhere in the chain.  Total machine
time for the whole b600/P=64 chain: ~7 min of 32-node allocation.

Physics sanity: E_DFT columns of `eqp1.dat` are EXACT-0 against the b300
run (same WFN lineage, as they must be); Eqp1 shifts by mean 0.197 eV /
max 0.321 eV going 300 -> 600 chi/W bands — the expected direction and
magnitude for a band-sum convergence step, not a defect.  The K-point
gaps barely move (DFT 1.72 eV both; G0W0 2.43 -> 2.42 eV), i.e. VBM and
CBM shift together: the GAP is already converged in the chi/W band sum
at 300 bands even though the absolute QP levels are not.

Harness defect found and fixed inside this campaign: the first plot
(job 7885322) was titled "300 bands" because `plot_bandstructure.py` was
copied verbatim from the b300 worked example, which hard-codes the count.
Curves and printed gaps were always the b600 data; only the label was
wrong.  Title is now read from `LORRAX_PLOT_NBAND` and the PNG was
regenerated (job 7885324).

## 2. GW stage-by-stage, b600/P=64 vs b300/P=16

b300 baseline = job 7885024 (q-parallel local tier, GW wall 214.5 s,
sacct MaxRSS 4.63 GiB/rank).  b600 = job 7885316 (wall 240.6 s, VmHWM
16.24 GiB/rank).  Overall wall ratio 1.12x for 1.6x mu, 2.0x bands and
4x ranks.

| section | b300/P16 [s] | b600/P64 [s] | ratio | expected | reading |
|---|---|---|---|---|---|
| TOTAL | 214.47 | 240.63 | 1.12x | — | |
| gw_jax.isdf | 114.96 | 105.12 | 0.91x | — | net win |
| . load_centroid_wfns | 1.48 | 7.20 | 4.86x | ~1x | **worse than expected** |
| . . loader_load | 0.19 | 5.05 | **26.46x** | ~1x | **worst scaling in the run** |
| . zeta_fit_chunked | 109.73 | 92.49 | 0.84x | — | |
| . . zeta_fit.CCT | 1.91 | 1.52 | 0.79x | mu^2/P = 0.65x | as expected |
| . . zeta_fit.cholesky | 11.76 | 53.71 | **4.57x** | mu^3 = 4.13x | as expected, but see §4.2 |
| . . chunk_loop | 93.00 | 33.48 | 0.36x | — | strong scaling |
| . . . z_q_build | 76.44 | 22.84 | 0.30x | nb*mu/P = 0.80x | better than expected |
| . . . chunk.solve | 16.02 | 10.20 | 0.64x | mu^2/P = 0.65x | as expected |
| gw_jax.V_q_compute | 1.21 | 1.52 | 1.25x | | |
| gw_jax.screening | 17.96 | 31.02 | 1.73x | | |
| . chi0_W | 9.71 | 11.33 | 1.17x | | |
| . . chi.exec | 4.99 | 4.64 | 0.93x | nb*mu^2/P = 1.29x | better than expected |
| . . W.exec | 4.05 | 5.21 | 1.29x | mu^3 (no P gain) = 4.13x | much better than expected |
| . chi0_W_probe | 8.08 | 16.35 | **2.02x** | | **worse than the real pass** |
| . . probe W.exec | 3.99 | 11.92 | 2.99x | mu^3, 16 q, no P gain | |
| gw_jax.persist_w0 | 2.63 | 4.05 | 1.54x | mu/P | |
| gw_jax.sigma | 71.97 | 73.67 | 1.02x | | |
| . sigma.exec | 70.69 | 70.85 | **1.00x** | | flat — best-scaling stage |
| . . sigma.tau.host_accum | 63.19 | 65.56 | 1.04x | | device-bound, per-rank work constant |

Planner accuracy at b600/P=64: estimate 16.43 GB/dev, measured 17.44 GB
(16.24 GiB) = 1.06x.  At b300/P=16 the same planner estimated 33.99
GB/dev against a measured 4.63 GiB — a 7.3x OVER-estimate.  The planner
is far better calibrated at this shape than at the b300 one; the b300
r_chunk was budget-limited to 2 chunks on an estimate that was 7x too
big, which is part of why b300's chunk_loop was so expensive.

## 3. Non-GW drivers, b600/P=64 vs b300/P=16

b300 P=16 baseline = scorecard BD.2 (jobs 7884867 / 7884870).

| driver | b300 P=16 wall / recorded | b600 P=64 wall / recorded | recorded ratio |
|---|---|---|---|
| kmeans (3000c/300b -> 4800c/600b) | 31 s / 25.1 s | 51 s / 32.6 s | 1.30x |
| dipole (300b -> 600b) | 18 s / 5.7 s | 22 s / 7.0 s | 1.23x |
| kin-ion (300b -> 600b) | 14 s / 6.7 s | 17 s / 8.5 s | 1.27x |
| htransform dft (2979c -> 4775c) | 27 s / 18.4 s | 28 s / 12.1 s | 0.66x |

dipole and kin-ion cannot use more than 16 ranks on this deck (their
sweeps are over nk=16), so the 4x rank increase buys them nothing and
1.23-1.27x for 2x bands is BETTER than the naive 2x.  htransform is
faster in absolute terms at 1.6x the centroids — the 62ba395 Gram-eigh
rework doing what it was built for.

## 4. Findings, ranked by how much they cost at this scale

### 4.1 kmeans is dominated by an unscaled replicated weight build
`setup.weight` is **19.99 s of the 32.58 s** kmeans recorded total
(61.4%).  Scorecard BD.2 measured the same section at 7.1 s (P=1) and
7.0 s (P=16) on the 300-band deck — i.e. it has never scaled with P, and
it grows with the band window (7.0 s at 300 bands -> 20.0 s at 600).
This is the open KNOWN_LORRAX_ISSUES centroids row "weight field built
replicated on full r-grid"; at b600/P=64 it is now the single largest
term in the driver.  Everything else in kmeans scales: Lloyd 4.36 s,
pivoted-Cholesky prune 6.01 s (Gram 4.77 s) for a 6697-candidate pool.

### 4.2 The zeta charge factor saturates at P = nq_ibz, not at P
`zeta_fit.cholesky` is 53.71 s = 22.4% of the GW wall.  The driver
announces the reason itself:

    [zeta factor] replicated plan, q-parallel execution: nq=10 scattered
    over 64 devices (ceil(nq/P)=1 whole (4800,4800) tile(s)/device, q-pad 54)

**54 of the 64 ranks are idle for the whole stage.**  The 854af1f
q-parallel fold reached its ceiling at P=16 already (nq=10 <= 16), so
going 16 -> 64 ranks bought this stage exactly nothing and the 4.57x is
pure mu^3.  Adding nodes cannot help it; only `distributed_zeta_solve =
distributed` (pzheevd over the whole mesh) can.  This is not a defect —
it is the documented local-plan contract — but it IS the stage that
decides whether P=64 is worth buying on a 10-q deck, and it should be
stated that way in `large_nmu_operation.md`, which currently presents
the q-parallel fold without naming its P = nq ceiling.

Same ceiling, smaller absolute cost, on the W Dyson local plan:
`ceil(nq/P)` = 1 tile/rank at both P=16 and P=64, so W.exec gets no
P benefit either (48 ranks idle on the 16-q probe pass).

### 4.3 load_centroids.loader_load: 26.5x, the worst scaling in the run
0.19 s -> 5.05 s.  Bytes read grew ~8x (2x bands x 4x ranks, since each
rank reads the whole window) but time grew 26.5x — superlinear Lustre
contention from 64 concurrent whole-file h5py reads of a 609 MB WFN.
This is the open KNOWN_LORRAX_ISSUES row "full WFN.h5 h5py read PER RANK
at P>1" (filed under dipole/kin-ion; the same loader serves the GW
centroid path).  Absolute cost is still only 5 s at P=64, but the
exponent is the problem: it is the one term in the run that gets worse
faster than P grows.

### 4.4 chi0_W_probe now costs 2x the real screening pass
16.35 s vs the real pass's 11.33 s, because the probe's Dyson solve runs
the FULL BZ (16 q) while the real pass runs the IBZ wedge (10 q), and
neither gets P-parallelism past nq.  At b300 the probe was 0.83x the
real pass; at b600/P=64 it is 1.44x.  The existing KNOWN row
("chi0_W_probe re-runs chi+W nearly in full") is confirmed and its cost
share is growing with mu.  Note the shipped `ppm_probe_chi_reuse=auto`
opt-in only folds the chi sweep (3.88 s here), not the 11.92 s Dyson
solve, so it would not fix this.

### 4.5 Per-rank memory grew 3.5x while ranks grew 4x
4.63 GiB/rank (b300/P=16) -> 16.24 GiB/rank (b600/P=64), and the spread
across the 64 ranks is 16.13-16.24 GiB, i.e. perfectly flat (no owner
rank, no gather hotspot).  The named driver of the growth is the
replicated zeta back-solve gather, which the run announces as **3.69
GB/rank** (`nq*mu^2*16` at nq=10, mu_pad=4800) and which does not divide
by P — the standing item 3 of `large_nmu_operation.md`'s honest list.
It sat just under the 4 GiB `LORRAX_ZETA_GATHER_CAP_GIB`, so `auto`
resolved `replicated` rather than `per_q`; at ~5000 centroids this deck
is one step from flipping tiers.  Against 96 GB/rank there is no OOM
risk here, but the trend is the thing: per-rank memory is not falling
with P.

### 4.6 Deck-key deprecation notice fires on the certified deck
`[config] use_ffi_io=true is deprecated and redundant: slab_io=auto
already routes to the best available parallel writer.  Remove the key.`
Inherited from `deck_b300.in`.  Cosmetic, but the b300 deck in the
worked example still carries it.

## 5. What was NOT found

* No OOM, no `RESOURCE_EXHAUSTED`, no refusal, no traceback in any leg.
* No `nq*mu^2` BAND-gather term: per-rank memory is flat across all 64
  ranks in GW, and the dipole/kin-ion owner-gather shows exactly the
  expected one-hot profile (rank 0 at 2.10/2.15 GiB, others at
  0.46-0.47 GiB).  The class the task warned about as a regression is
  not present.
* No thread-main refusal from kmeans or htransform at P=64 with no
  `LORRAX_MPI_FORCE_THREAD_MAIN`: the e97e8ed fix (mesh + warm-up from
  `initialize_communicator_stack`) holds at 64 ranks.  BD.3's retry
  lever is no longer needed.
* `slab_io=auto` engaged the PHDF5_FFI collective writer, not the
  announced allgather demotion.

## 6. Diagnostic leg: `distributed_zeta_solve = distributed` (job 7885323)

Same deck, same geometry, one key changed.  Tier resolved as announced
(`path=distributed_rank_truncate`, "distributed tier gathers NO (mu,mu)
object").

| section | local (7885316) | distributed (7885323) | ratio |
|---|---|---|---|
| TOTAL (wall) | 240.63 s | 186.42 s | 0.77x |
| TOTAL minus startup | 214.75 s | 177.58 s | **0.83x** |
| zeta_fit.cholesky | 53.71 s | 32.74 s | **0.61x** |
| zeta_fit.chunk.solve | 10.20 s | 6.68 s | 0.65x |
| zeta_fit.chunk.z_q_build | 22.84 s | 21.90 s | 0.96x |
| screening | 31.02 s | 24.05 s | 0.78x |
| . probe W.exec | 11.92 s | 9.51 s | 0.80x |
| sigma.exec | 70.85 s | 71.08 s | 1.00x |
| per-rank peak VmHWM | 16.24 GiB | 16.22 GiB | **1.00x** |

Startup must be netted out: the local leg compiled 201 modules cold
(`runtime_stack.compile_cache` 12.58 s) while the distributed leg ran on
a warm cache (31 compiles, 2.88 s).  The honest figure is the
minus-startup row: **17% faster end to end, 1.64x faster on the charge
factor.**

Two conclusions, one of them contrary to the documented expectation:

1. **The wall crossover has already been passed at this shape.**
   `large_nmu_operation.md` records the distributed tier as 2.1x SLOWER
   than the local q-parallel factor at b300/P=16 (24.25 vs 11.8 s) and
   places the wall crossover far out at mu=10015 on 64 ranks (4712 s vs
   236 s).  At mu=4775 / P=64 the distributed tier is already 1.64x
   FASTER on the factor and 1.17x faster end to end.  The reason is §4.2:
   the local plan can only use nq=10 of the 64 ranks, so its advantage
   evaporates as soon as P exceeds nq by much.  The crossover is
   governed by P/nq, not by mu alone.

2. **It buys no memory at this shape.**  Peak per-rank VmHWM is
   unchanged (16.22 vs 16.24 GiB) even though the local tier's
   announced 3.69 GB/rank replicated `(nq,mu,mu)` gather is entirely
   absent from the distributed one.  The peak is set by the
   `C_fit_one_rchunk` transient (planner binder, 16.43 GB estimate),
   not by the zeta gather, so removing the gather moves nothing.  The
   docs' framing — "the tier buys memory shape and P-scaling, not wall
   time" — is exactly backwards at this shape: here it bought wall and
   no memory.

Parity local vs distributed (the documented ~kappa*eps gauge
difference; the run reports kappa/q = 9.98e7):

| file | max abs delta |
|---|---|
| eqp0.dat | 1.43e-05 eV |
| eqp1.dat | 1.10e-05 eV |
| eqp_g0w0.dat | 1.50e-05 eV |
| sigma_diag.dat | 1.40e-05 eV |

Same class as the b300 measurement (8.6e-6 - 9.0e-6 eV), scaled up with
mu as expected, and ~70x under the 1e-3 eV significance bound.

## 7. Artifacts worth keeping

| path | contents |
|---|---|
| `/scratch2/08271/jackmc/b600_p64/deck_b600.in` | the 600-band deck |
| `.../centroids_b600.txt` -> `run_deck/centroids_frac_4775_b600_c4800.txt` | the 4775-centroid orbit-closed set, 600-band selection window |
| `.../run_deck/{dipole_b600.h5, kin_ion_b600.h5}` | 322 MB / 184 MB operators at 600 bands |
| `.../run_gw/` | local-tier GW outputs (eqp0/eqp1/eqp_g0w0/sigma_diag/sigma_mnk.h5 45 MB, qp_wfn_rotations.h5) |
| `.../run_gw_dist/` | distributed-tier GW outputs (same five) |
| `.../run_ht_{dft,qp}/bandstructure.dat`, `mos2_4x4_b600_bandstructure.png` | band structures |
| `.../logs/` | every driver log + per-rank VmHWM sample dirs, per jobid |
| `.../harness/` | the five sbatch legs + shared env/run_leg/inner_common |
| `.../lorrax_frozen/`, `.../bundle_1fc2759/` | frozen src@1fc2759 + the bundle every job read |
| `.../baselines/` | b300/P=16 (7885024) and AQ 4962c/P=64 (7877789) timing tables |
