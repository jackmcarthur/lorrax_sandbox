# Agent A — Si non-bispinor μ-sweep + HLO calibration

**Branch:** `agent/si-nonbispinor-mu-sweep` on lorrax_A, tip `dc0b254`
(adds two nspinor=1 loader fixes on top of `0f355b7` Round-10 main).
**System:** Si 4×4×4 25 Ry **non-bispinor scalar** (`noncolin=.false., lspinorb=.false.`).
WFN: `nspinor=1, nspin=1, mnband=102, ngkmax=588, fft=24³, nelec=4 (ifmax=4)`.
**Hardware:** 4 GPUs on hbm40g (1 node, 2×2 mesh), JID 53097982.
**cohsex.in (per μ):** `bispinor=false, nval=4, ncond=96, nband=100, x_only=true,
do_screened=false, memory_per_device_gb=70.0`, **`cusolvermp_charge=off,
cusolvermp_lu=off`** (sharded-cholesky fallback — see §4 below).
**Probe envs:** `LORRAX_MEM_DEBUG=1 LORRAX_RCHUNK_DEBUG=1 LORRAX_MAX_RCHUNKS=3
LORRAX_EXIT_AFTER_ZETA=1`.
**Run dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/MU_nonbispinor_2026-05-18/`

## Headline verdict

**The two HLO-calibrated planner constants `pair_density_slots = 3` and
`fft_box_factor_D = 2.0` both hold *exactly* at ns=1.** Per-slot sizes
match `_bytes_c128(nk, 1, 1, mu_padded, r_chunk, shard=p_xy)` bit-for-bit
across μ = 408 and 1176 (HLO M1/M2 reproduction at ns=1 — see §3).

**But the planner-vs-mem_stats faithfulness pattern is structurally
different from bispinor:**

* μ=192:  HWM_pred=2.81 GB, mem_stats peak=8.02 GB → **+185 % UNDER-prediction**
* μ=384:  HWM_pred=5.99 GB, mem_stats peak=8.03 GB → **+34 % under**
* μ=768:  HWM_pred=11.17 GB, mem_stats peak=8.38 GB → **−25 % over** (CONSERVATIVE)
* μ=1200: HWM_pred=17.50 GB, mem_stats peak=13.04 GB → **−25 % over**
* μ=1800: HWM_pred=26.51 GB, mem_stats peak=19.60 GB → **−26 % over**

The bispinor sweep was monotonically under-predicting in the **−0.5 % to
−10.8 %** band.  Here at ns=1 the planner has a **constant ~8 GB/dev
floor** (CUDA context + framework overhead, sticky once allocated) that
dominates at small μ, then the Peak C model takes over and **over-predicts**
by ~25 % at large μ because at 1 r-chunk total the model conservatively
assumes 3 concurrent pair-density slots throughout the run, but the actual
peak is dominated by 2 slots (one is overlap with output, see HLO in §3).

**No leaks, no OOMs, no crashes** at any μ.  Planner picks a single
r-chunk (r=13824=n_rtot) at every μ, no last-chunk stub aliasing
(unlike Si bispinor μ=768 −10.8 % case).

## Headline table

| μ requested | μ realised (orbit-pruned) | r_chunk | n_chunks | HWM_pred (GB/dev) | mem_stats peak BFC+95 (GB/dev) | %-err (HWM − peak)/peak | nvsmi peak platform_false (GB/rank) | live_total worst (GB global) |
|---|---|---|---|---|---|---|---|---|
| 192  | 192  | 13824 | 1 |  2.81 |  **8.02** | **+184.7 %** | 3.19 | 0.48 |
| 384  | 408  | 13824 | 1 |  5.99 |  **8.03** |  +34.1 %     | 1.95 | 1.03 |
| 768  | 756  | 13824 | 1 | 11.17 |  **8.38** | **−25.0 %**  | 2.22 | 1.94 |
| 1200 | 1176 | 13824 | 1 | 17.50 | **13.04** | **−25.5 %**  | 2.55 | 3.08 |
| 1800 | 1764 | 13824 | 1 | 26.51 | **19.60** | **−26.0 %**  | 3.03 | 4.75 |

(Centroid pruning is normal: `centroid.kmeans_cli` oversamples and prunes by
orbit closure; e.g. requested 768 yields 756.  μ=192 happens to be orbit-closed
already.)

Compare to bispinor Si μ-sweep (`agent_t_si_bispinor_sweep.md`, ns=4, p_xy=4):

| μ | HWM_pred (GB) | mem_stats peak (GB) | %-err |
|---|---|---|---|
| 384  | 56.00 | 56.30 | −0.5 % |
| 768  | 55.99 | 62.78 | −10.8 % |
| 1200 | 55.97 | 59.03 |  −5.2 % |
| 1800 | 56.00 | 57.28 |  −2.2 % |

Bispinor was strongly compute-bound and the planner picked multi-chunk plans
(2, 3, 4, 7 chunks) hitting Peak C near the budget every time.  Non-bispinor
fits in 1 chunk at every μ ≤ 1800, and the static %-err is dominated by a
**~8 GB CUDA/framework floor** plus an **over-prediction in Peak C's
slot count** when no concurrent r-chunks ever fly.

## Per-chunk timings (LORRAX_MAX_RCHUNKS=3, but only 1 chunk per μ)

| μ | fit (ms) | write (ms) | total (ms) | z_q_build / solve breakdown (ms) |
|---|---|---|---|---|
| 192  | 1569 | 545 | 2113 | 764 / 654  |
| 384  | 1664 | 358 | 2022 | 905 / 758  |
| 768  | 1991 | 351 | 2342 | 1077 / 911 |
| 1200 | 1845 | 358 | 2203 | 1083 / 757 |
| 1800 | 2143 | 358 | 2501 | 1191 / 915 |

End-to-end ~20 s per μ (single channel, since non-bispinor has only the
charge channel — no transverse γ̃¹²³ channels like bispinor's
4-channel sweep).  Wall scales sub-linearly with μ; XLA compile is the
dominant cost at this scale.

## Planner breakdown (verbatim from gw.out, μ=1800)

```
G-flat memory model — chunk plan + HWM estimate
  band_chunk         = 128
  r_chunk            = 13824  (1 chunks)
  gflat_chunk_size   = 100
  budget             = 70.00 GB/dev
  HWM estimate       = 26.51 GB/dev (38% of budget) [bottleneck: C_fit_one_rchunk]
  peak totals (GB/dev):
    C_fit_one_rchunk........   26.51
    D_accumulate............    7.82
    B_CCT_chol..............    3.55
    E_v_q...................    1.39
    A_centroid..............    0.22
  per-peak components (GB/dev):
    [A]
      fft_box...............   0.113
      centroid_out_filling..   0.090
      phase_table...........   0.014
      sphere_idx_replicated.   0.004
    [B]
      P_l_plus_P_r_open_spin   1.593
      C_q...................   0.797
      L_q...................   0.797
      centroids_persistent..   0.361
      sphere_idx_replicated.   0.004
    [C]
      P_pair_concurrent_slots  18.728
      zeta_out..............   6.243
      L_q...................   0.797
      gflat_acc.............   0.374
      centroids_persist.....   0.361
      sphere_idx_replicated.   0.004
    [D]
      ... (accumulate side, sub-dominant)
    [E]
      ... (v_q side, small at COHSEX gate, do_screened=false)
```

`P_pair_concurrent_slots` is the dominant term (18.73 / 26.51 = 71 % of Peak C
at μ=1800).  Per-slot bytes match the formula
`_bytes_c128(nk=64, ns_a=1, ns_b=1, mu_padded=1764, r_chunk=13824, shard=p_xy=4)`
= 6.243 GB; planner uses 3 slots × 6.243 GB = 18.73 GB exactly.

Across all 5 μ values the formula is bit-exact (no ns² hidden factor):

| μ    | mu_padded | per-slot pred (GB) | 3-slot pred (GB) | planner reports |
|------|-----------|--------------------|--------------------|-----------------|
| 192  |  192      | 0.679              |  2.038            | 2.038 ✓ |
| 408  |  408      | 1.444              |  4.332            | 4.332 ✓ |
| 756  |  756      | 2.675              |  8.026            | 8.026 ✓ |
| 1176 | 1176      | 4.162              | 12.485            | 12.485 ✓ |
| 1764 | 1764      | 6.243              | 18.728            | 18.728 ✓ |

## 3. HLO calibration — `pair_density_slots = 3` at ns=1 confirmed

Profile via `scripts/profiling/run_profiled.py --out profile --no-trace
-m gw.gw_jax -i cohsex.in`, dump location:
`mu384/profile/xla_dump/` and `mu1200/profile/xla_dump/`.

### 3.1 `fit_one_rchunk` jit at μ=408 (= mu384/profile/xla_dump/module_0297.jit_fn.*-memory-usage-report.txt)

```
Total: 4.75 GiB
  allocation 12: 3.36 GiB, preallocated-temp
    offset       1152  →  1.34 GiB:  4×c128[64,1,6912,204,1] + c128[6912,204,4,4,4]
                                       + c128[1,6912,204,1,4,4,4] + c128[64,6912,204]
    offset 1443890304  →  1.34 GiB:  c128[64,25,1,24,24,24] + c128[64,100,1,13824]
                                       + c128[64,25,1,588] + 2×c128[1410048,64] + ...
    offset 2859467904  →  675.0 MiB: c128[25,64,1,13824] + c128[100,64,1,6912]
    offset 1797784704  →  337.5 MiB: c128[64,25,13824]
    + smaller (phase tables, indices, etc.)
  allocation 0: 1.34 GiB, output c128[64,204,6912]  ← ALSO aliases 4×c128[64,1,6912,204,1]
```

**Pair-density buffer shape:** `c128[64, 1, 6912, 204, 1]` = `c128[nk, ns_a, r_loc, mu_loc, ns_b]`
(ns_a = ns_b = 1).

Per-buffer bytes: `64 · 1 · 6912 · 204 · 1 · 16 = 1,443,791,872 B = 1.344 GiB` ✓
matches the planner's `_bytes_c128(nk, 1, 1, mu_padded, r_chunk, shard=p_xy)`.

**Slot count of pair-density-shaped buffers = 3** (three distinct 1.34-GiB
slots: offsets 1152 + 1443890304 + allocation 0 output).
Each slot also aliases:
* The FFT box `c128[1, 6912, 204, 1, 4, 4, 4]` (post-IFFT reshape).
* The zeta_chunk output `c128[64, 25, 1, 13824]`.
* The rank-7 P_l_3d shape.

Slot size set by largest aliased variant = pair-density at 1.34 GiB — same
"pair-density dominates" pattern as bispinor (agent_d M1).

**Confirmed:** `pair_density_slots = 3` at ns=1 (matches `pair_density_slots_charge`
default; same as ns=4 bispinor and ns=2 SOC non-bispinor in agent_d M4).

### 3.2 `accumulate_rchunk_to_gflat` jit (μ=408, cs=100, module_0347.jit__kernel)

```
Total: 419.74 MiB
  allocation 8: 240.26 MiB, preallocated-temp
    offset          384  →  189.84 MiB: c128[900,13824]      (scan-carry: nq_disk_pad × n_rtot)
    offset    199065984  →   21.09 MiB: c128[100,24,24,24]   (FFT box, spatial)
    offset    221184384  →   21.09 MiB: c128[100,13824]      (FFT box, flat)
    offset    243302784  →    8.07 MiB: 5×c128[900,588]
    + indices, sphere tables (≤ 200 KiB)
  allocation 0: 172.12 MiB, parameter 0 c128[8,102,13824] (zeta_chunk slab input)
  allocation 1: 7.32 MiB, parameter 1 c128[8,102,588] (gflat_acc slab + output)
```

**FFT-box slot pair:** spatial reshape (`c128[cs, nx, ny, nz]`) + flat (`c128[cs, n_rtot]`).
At cs=100:
* Per-slot bytes (spatial): `100 · 24³ · 16 = 22,118,400 B = 21.09 MiB` ✓
* Per-slot bytes (flat):    `100 · 13824 · 16 = 22,118,400 B = 21.09 MiB` ✓
* **Empirical factor_D = 2.0 confirmed at ns=1.**

The cuFFT plan workspace folds into the two box-sized slots — no separate
"cuFFT scratch" allocation.  Same structure as agent_d M2 at bispinor cs=360
(2 × 6.03 GiB) and M3 at cs=1 (2 × 17.17 MiB).

### 3.3 mu=1176 confirmation (mu1200/profile/xla_dump/module_0297.jit_fn)

```
Total: 11.74 GiB
  allocation 12: 7.75 GiB, preallocated-temp  (= 2 slots × ~4 GiB pair-density + zeta_out)
  allocation 0: 3.88 GiB, output c128[64,588,6912]  (1 more slot, aliased)
  allocation 1+2: 2×57.42 MiB, parameter centroid slabs
```

Per-slot at mu_padded=1176: `64 · 1 · 1176 · 13824 · 16 / 4 = 4.162 GB`.
3 slots × 4.162 GiB = 12.486 GiB; planner reports
`P_pair_concurrent_slots = 12.485 GB` — exact.

## 4. New bugs found (and fixed in-branch)

Both bugs were latent until a true `nspinor=1` WFN was loaded in this
sweep.  Pre-existing tests cover only `nspinor=2` (bispinor or
SOC-non-bispinor).

### 4.1 `unfold_psi` ns=1 silent spinor-axis broadcast (eager backend)
`src/common/symmetry_maps.py` — `unfold_psi()` always built a 2×2
`U_eff` regardless of input ψ's spinor axis.  When `cnk.shape[1] == 1`,
the einsum `"jk,nkl->njl"` silently broadcast cnk's spinor axis from 1 → 2,
then the caller (`WfnLoader._eager_build`) failed copying back into its
ns=1 host buffer.

The docstring already claimed "for ns=1, U_eff is the 1×1 identity, this
einsum is a no-op" but the code never enforced it.

**Fix (commit `8c18925`):** when `cnk.shape[1] == 1`, return after the
τ-phase + complex-conjugation (TRS branch) and skip the spinor mix.
Effect: 11 LOC change, no behavior change for ns=2, all 44 loader/unfold
tests still pass.

Surfaced by: `centroid.kmeans_cli 192 --qe-save ...` →
`pivoted_cholesky.build_gram_q0_via_loadwfns` → `WfnLoader.load(bispinor=False)`
→ `_eager_build`.

### 4.2 `WfnLoader._ensure_phdf5_static` ns=1 silent spinor-axis broadcast (phdf5 backend)
`src/file_io/wfn_loader.py` — same pattern as 4.1 but in the multi-rank
phdf5 device-side unfold kernel.  `U_per_spatial = sym.U_spinor[s_spatial_idx]`
unconditionally has shape `(nk_full, 2, 2)`, then the device einsum
`"kac,bckg->bakg"` broadcasts the ns=1 ψ's spinor axis to 2.
`psi_G_store._populate_from_loader` failed copying back into the ns=1
host tile.

**Fix (commit `dc0b254`):** when `self.nspinor == 1`, build
`U_per = ones((nk_full, 1, 1))`.  Einsum becomes a no-op
multiplication by 1.  TRS branch's complex conjugation is already
handled by the `where(tr_mask, conj(cnk), cnk)` step upstream
(iσ_y has no meaning for a 1-component spinor).
Effect: 19 LOC change, no behavior change for ns=2.

Surfaced by: `gw.gw_jax -i cohsex.in` → ζ-fit → `psi_G_store` populate
→ `WfnLoader.load(bispinor=False)` on the phdf5 path.

## 5. Cusolvermp Cholesky failure at BFC + MEM_FRACTION=0.95 on 2×2 mesh (sandbox-environment)

The default-mode bispinor sweep ran on `1×4` mesh
(`agent_t_si_bispinor_sweep.md`) so `_resolve_solver_kind_charge`
picked the in-tree `sharded_cholesky` path (cuSolverMp is only used
on true 2D meshes `px≥2 AND py≥2`).  At 4 GPUs / 1 node here we get a
**2×2 mesh by default**, so `auto` picks cuSolverMp's distributed
Cholesky.

Under `XLA_PYTHON_CLIENT_ALLOCATOR=default + PREALLOCATE=true +
MEM_FRACTION=0.95`, `cusolverMpPotrf(q=0) failed: status=7`
(INTERNAL_ERROR — NCCL user-buffer-registration failure when BFC has
preallocated all 95 % of HBM, leaving no room for cuSolverMp's
collective comm buffers).

**Workaround applied:** added `cusolvermp_charge = off, cusolvermp_lu = off`
to each `mu*/cohsex.in` so the in-tree sharded path (same as bispinor
sister sweep) is used at all μ.  After the fix all 5 BFC runs completed
cleanly.  This is **not a planner issue** — the planner's HWM is
computed without any cusolverMp footprint.

(Lowering `MEM_FRACTION` to 0.80 also works as a second-line workaround,
already wired into `_run_gw.sh` as the `bfc_pre80` variant.)

## 6. Structural comparison — Peak C across ns

| ns | system | μ | nk | p_xy | r_chunk | mu_padded | per-slot (GB) | 3-slot Peak C (GB) | per-slot HLO bytes |
|----|--------|---|----|----|---------|-----------|-----------------|--------------------|--------------------|
| 1 (this work) | Si 4×4×4 25 Ry | 1800 | 64 | 4 | 13824 | 1764 | 6.243 | 18.73 | 6.243 GiB c128[64,1,6912,441,1] |
| 2 (M4 prior)  | CrI3 6×6 80 Ry SOC | 1504 | 36 | 16 | 73328 | 1504 | 15.88 | 47.64 | 14.79 GiB c128[2,6892832,2,36] |
| 4 (M1 prior)  | CrI3 6×6 80 Ry bispinor | 1520 | 36 | 16 | 24576 | 1520 | 21.50 | 64.50 | 20.04 GiB c128[36,4,6144,380,4] |

`pair_density_slots = 3` is **invariant** across all three ns regimes
(ns=1, ns=2, ns=4) — confirmed bit-exact at HLO level.  The byte formula
`_bytes_c128(nk, ns_a, ns_b, mu_padded, r_chunk, shard=p_xy)` correctly
accounts for ns² in the spinor axes (charge = identity contracts to ns=ns,
bispinor = full ns²=16, SOC = ns²=4).

## 7. The +185 % under-prediction at μ=192 — what is the 8 GB floor?

At μ=192 the planner predicts Peak C = 2.81 GB but mem_stats reports
**8.02 GB at the very first `zeta_fit_start` probe** — *before* the
r-chunk loop even begins.  Subsequent `pre_rchunk_loop`, `rchunk_start`,
`after_fit_one_rchunk`, and `zeta_fit_end` probes all read **the same
8.02 GB peak**.

This indicates a **sticky framework/CUDA-context floor** that BFC sees
once (at JIT compilation + first device put) and never releases.
Likely contributors:
* XLA JIT cache (~3000 helper jits compiled in the run; PTX/SASS payload
  pinned per-process).
* CUDA context init + cuBLAS/cuFFT plan caches.
* NCCL user-buffer-pool reservation (cuFFT and sharded_cholesky paths
  both initialise NCCL even at this small scale).

This floor is NOT in the planner's model (correctly so — it's
framework overhead, not LORRAX-controllable).  It dominates whenever
Peak C drops below ~8 GB.  At μ=192 Peak C = 2.81 GB → floor wins,
planner under-predicts by +185 %.  At μ=1200 Peak C = 17.5 GB → planner
wins, dominant model is faithful (but slightly conservative — 25 % over).

## 8. Over-prediction at μ≥768 — likely cause

At single-chunk runs, the planner conservatively assumes 3 concurrent
pair-density slots (per `pair_density_slots = 3`).  HLO shows 3 slots
**ARE allocated**, but only 2 of them are actually live simultaneously
inside the JIT (one slot's lifetime is wholly contained in the OUTPUT
slot's lifetime via aliasing).  So the **runtime peak corresponds to
~2 slots, not 3**, when only 1 r-chunk is dispatched and no inter-chunk
state is carried across calls.

Specifically:
* μ=1800: 3-slot prediction = 18.73 GB; observed BFC peak = 19.60 GB.
  19.60 / 18.73 = 1.046; floor + driver overhead = ~6.6 GB.
* If we drop to 2-slot effective:  2 × 6.243 = 12.49 GB + 6.6 GB
  floor → 19.1 GB, matches better.

The planner is correctly conservative for a multi-chunk plan (each
r-chunk re-allocates its slot triplet at the JIT boundary) but
over-counts in the single-chunk degenerate case.  For an OOM-relevant
HWM the over-prediction is the *safer* direction.

## 9. Reproducer

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/MU_nonbispinor_2026-05-18/

# QE reference (one-shot, ~3 min)
./_run_qe.sh

# Centroids per μ
./_kmeans_run.sh 192       # ~15 s
./_kmeans_run.sh 384       # ~20 s   (yields _408)
./_kmeans_run.sh 768       # ~30 s   (yields _756)
./_kmeans_run.sh 1200      # ~40 s   (yields _1176)
./_kmeans_run.sh 1800      # ~50 s   (yields _1764)

# Probe runs (~20 s each)
for MU in mu192 mu384 mu768 mu1200 mu1800; do
  ./_run_gw.sh $MU platform_false
  ./_run_gw.sh $MU bfc_pre95
done

# HLO dumps at μ=384 + μ=1200 (~25 s each)
cd mu384 && rm -rf profile && \
  LORRAX_SHIFTER="$LORRAX_SHIFTER --env=LORRAX_MEM_DEBUG=1 --env=LORRAX_RCHUNK_DEBUG=1 \
    --env=LORRAX_MAX_RCHUNKS=3 --env=LORRAX_EXIT_AFTER_ZETA=1" \
  lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out profile --no-trace -m gw.gw_jax -i $(pwd)/cohsex.in
```

## 10. Artifacts

| Path | Purpose |
|---|---|
| `qe/{scf,nscf}/` | Si 4×4×4 25 Ry scalar QE reference (WFN.h5, vxc.dat, kih.dat) |
| `mu{192,384,768,1200,1800}/cohsex.in` | Per-μ COHSEX configs |
| `mu*/centroids_frac_*.txt` | Per-μ centroid files (orbit-pruned counts) |
| `mu*/gw_platform_false.out` | Production-allocator run logs |
| `mu*/gw_bfc_pre95.out` | BFC + preallocate + MF=0.95 (mem_stats peak ground truth) |
| `mu{384,1200}/profile/xla_dump/` | HLO dumps for `fit_one_rchunk` + `accumulate_rchunk_to_gflat` |
| `_run_qe.sh`, `_kmeans_run.sh`, `_run_gw.sh` | Reproducer scripts |

## 11. Commits on `agent/si-nonbispinor-mu-sweep`

| commit | summary |
|---|---|
| `8c18925` | `fix(unfold_psi): handle nspinor=1 wavefunctions` — eager backend |
| `dc0b254` | `fix(WfnLoader._ensure_phdf5_static): handle nspinor=1 wavefunctions` — phdf5 backend |

Branch tip = `dc0b254`, rebased on `origin/main` at `0f355b7`.

## 12. Open questions / suggested follow-ups

1. **Quantify the 8 GB framework floor.**  Is it ~constant across systems
   (CrI3 80 Ry HLO peak_heap_bytes is ~66 GB and mem_stats peak = 76 GB;
   the implied floor there is ~10 GB).  If the floor scales with kgrid or
   nbnd it would explain part of the bispinor μ-dependent error pattern.
2. **3 vs 2 effective slot count when n_chunks=1.** At single-chunk
   degenerate runs the planner over-predicts by ~25 %.  A natural extension
   is `effective_slots = min(3, ceil(n_chunks·1.5))` — but this needs
   verification with a system that does sit in the n_chunks=1 regime at
   production scale (Si 8×8×8 25 Ry might).
3. **cuSolverMp + BFC + MEM_FRACTION=0.95 incompatibility on 2D meshes.**
   The status=7 failure is reproducible on Si 4×4×4 / 2×2 mesh / 40 GB
   HBM.  Should this be documented in `KNOWN_SANDBOX_ERRORS.md` as a
   guideline ("for OOM-relevant measurements on 2D meshes, force
   `cusolvermp_charge=off`")?  See §5.
