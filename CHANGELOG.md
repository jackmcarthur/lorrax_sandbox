# Changelog

## 2026-05-20: full COHSEX (do_screened=true) validated on CPU MPI [coord]

End-to-end full-COHSEX (`x_only=false, do_screened=true, screening=cohsex`)
on CPU n=4 (Milan node) at Si μ=384 production config completes **on the
first attempt with no additional source patches required**. Tests Σ_SX +
Σ_COH (static screened-exchange + Coulomb-hole) on top of the bare Σ_X
that the prior x_only validation exercised.

| | wall | Σ at (k=0, band=1) | eqp0.dat vs GPU |
|---|---|---|---|
| CPU 4 ranks × 8 threads | ~50 s | −16.478 eV (was −8.915 for x_only — δ ≈ −7.56 eV is Σ_C) | byte-identical except timestamp |
| GPU 4×A100 hbm80g | comparable | identical to CPU | reference |

pytest on lorrax_B against modules we patched: **63 passed, 15 skipped,
0 failed**.

Branch ready for merge to origin/main. The five-commit branch tip is
89690f0; see lorrax_B `git log origin/main..agent/jax-09-cpu-compat`.

## 2026-05-20: `_to_host` refactored to metadata dispatch (lorrax_B 89690f0) [coord]

Subagent design review of the 4-case shape-switch `_to_host` introduced
in 565750a flagged it as accidental complexity: 3 of 4 branches were
defending against return shapes that the JAX 0.9 source provably cannot
produce for the inputs `_to_host` actually receives. Empirically
characterized the sharding inventory at every gather call site (debug
run, Si μ=384 x_only CPU n=4) — confirmed:

* Every NamedSharded `mesh_xy` array (gflat_acc, G0_all, V_qmunu, etc.)
  is non-fully-addressable → `process_allgather(tiled=True)` Path (B)
  returns shape exactly `A.shape`.
* The lone Path-(D) failure case (`enk_full`) is `SingleDeviceSharding`
  with `is_fully_replicated=True`, which can be short-circuited via
  `A.addressable_data(0)` without ever calling `process_allgather`.

Replaced the 50-line 4-case switch with a 30-line 2-case dispatch on
stable `jax.Array` metadata (`is_fully_replicated`). Folded the inline
gathers in `gw_init.py:1121` and `isdf_fitting.py:2685` into `_to_host`
calls; dropped their legacy `shape[0] == 1` post-process guards (dead
code under the current geometry — G0 is 2D, gflat_acc is 3D, the guards
checked for 5D/4D leftovers from an old V_q bispinor layout).

Net: −24 LOC, no dead branches, no shape arithmetic, dispatches on
documented public API. End-to-end Si μ=384 x_only CPU n=4 `eqp0.dat`
**byte-identical** to the 565750a baseline (timestamp only differs).
Max rank-0 RSS unchanged at 26.6 GB.

Design review report:
`reports/memory_model_nonbispinor_kgrid_2026-05-18/PROCESS_ALLGATHER_DESIGN_REVIEW_2026-05-20.md`

## 2026-05-20: CPU port + planner backend-aware pair_density_slots [coord]

End-to-end CPU MPI port of the GW driver on Si 4×4×4 μ=384 non-bispinor.
Three JAX-0.9 strictness fixes were needed to get past the multi-process
code path (lorrax_B branch `agent/jax-09-cpu-compat`, commit `c7e6695`):
`cholesky_2d.py` panel_init `lax.pcast(('x','y'), to='varying')`,
`_slab_io_allgather.py` + `isdf_fitting.py` `tiled=True`. Backend-agnostic
fixes — GPU back-compat verified byte-identical (HWM 20.12, peak 20.13,
−0.05% — same as 2026-05-19 reference).

Planner finding (lorrax_B `5c2dae7`): CPU XLA's BufferAssignment schedules
**4 concurrent pair-density slots** in `fit_one_rchunk` where GPU XLA
schedules 3. Per-slot bytes match the existing `_bytes_c128(nk, ns², mu,
r_chunk, /p_xy)` formula exactly on both backends; only the slot count
differs. The +30% RSS excess over HWM_pred on CPU is **exactly this one
extra slot**. New helper `_default_pair_density_slots()` in
`gflat_memory_model.py` resolves the value via `jax.default_backend()` at
function-call time.

HLO evidence — robust across:
* Scalar non-bispinor (`module_0342.jit_fn`): 4 × 5.70 GiB
* Bispinor charge (`module_0360.jit_fn`): 4 × 16.92 GiB
* Bispinor transverse (`module_0413.jit_fn`): 4 × 16.92 GiB
* FFT-scratch hypothesis tested + REJECTED: at band_chunk ∈ {32, 64, 120}
  slot count + per-slot bytes invariant; FFT-box shapes alias into slots
  but don't size them.

Post-fix predictor accuracy on CPU at Si μ=384:
* n=1: HWM_pred 73.92 vs RSS 71.89 → **+2.8%** over
* n=2: HWM_pred 52.39 vs RSS 53.06 → **−1.3%** under
* n=4: HWM_pred 26.24 vs RSS 26.64 → **−1.5%** under

(was −24% to −33% across the same configs before the fix.)

Profiling stack: `scripts/profiling/pf.py` (+40 LOC) gains a CPU-backend
branch — psutil RSS fallback when `device.memory_stats()` returns None,
peak_rss_bytes tracking, pre-import of `jax.profiler` to dodge a
JAX-0.9 lazy-import race that crashes the sampler on CPU. GPU path
unchanged. New `skills/profiling_stack/cpu_addendum.md` documents the CPU
launch recipe + what's empty + HLO naming conventions.

Reports:
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_VALIDATION_2026-05-20.md`
  — initial CPU port (n=1/2/4 RSS vs planner)
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_OVERHEAD_DECOMP_2026-05-20.md`
  — subagent decomposition of the +6.5 GB excess into HLO-evidenced
  contributors; identified the 4-slot finding
* `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_PLANNER_LANDED_2026-05-20.md`
  — this session's FFT-test + bispinor confirmation + planner-landed note

Run dirs:
* `runs/Si/NONBISPINOR_CPU_2026-05-20/{mu384,mu384_decomp,mu384_fft_probe,mu384_bispinor}/`
* `runs/Si/NONBISPINOR_BUDGETSWEEP_2026-05-20/` — earlier today: planner
  budget-fill behavior at memory_per_device_gb ∈ {25, 35, 50, 70} GB at
  both μ=384 (single-chunk) and μ=1200 (multi-chunk). Picker fills 80%
  of budget when binding, sits at single-chunk floor when loose, never
  exceeds budget; mem_stats peak tracks HWM_pred within +1% across the
  sweep.
* `runs/Si/NONBISPINOR_PROD_2026-05-19/` — 2026-05-19 GPU production
  redo + this session's GPU back-compat smoke test.

Allocations released (CPU 54411765, GPU 54411976).

## 2026-05-19: Non-bispinor planner audit — production-config redo, planner is faithful [coord]

Re-did the 2026-05-18 non-bispinor audit at the production configuration
(`noncolin=.true., lspinorb=.true.`, FR-ONCVPSP PBE pseudo, `bispinor=false`,
cuSOLVERMp default-on, hbm80g + BFC+0.95) after the 2026-05-18 sweep was
flagged as scope-erroneous (Agent A built `nspinor=1`; agents disabled
cuSOLVERMp instead of using `hbm80g` per env docs).

Two μ values matching the bispinor sister sweep on JID 53207377:

| μ | r_chunk × n_chunks | HWM_pred | mem_stats peak | %-err |
|---|---|---|---|---|
| 384 | 13824 × 1 | 20.12 | 20.13 | **−0.05%** (bit-exact) |
| 1200 | 13468 × 2 | 55.99 | 55.74 | **+0.45%** (slightly conservative) |

Both inside (and on the optimistic side of) bispinor's [−0.5%, −10.8%] band.

**Falsifies the all_gather-slab planner refinement** Agent C proposed at the
scope-erroneous config: it would have shown up here as a measurable
under-prediction in either μ data point — neither does. **Stand down on
the planner edit.**

The 2026-05-18 scope-error addendum: `nspinor=1` is unsupported (production
always uses FR pseudo + noncolin=true → nspinor=2); `cusolverMp status=7`
under BFC+0.95 is hbm40g vs hbm80g, not a sandbox bug (documented in
`ENVIRONMENT_COMPREHENSIVE.md` §3.2 + §8.3). The two `nspinor=1` loader
"fixes" landed on `agent/si-nonbispinor-mu-sweep` (`8c18925`, `dc0b254`)
were reset out — branch back to `origin/main`. Same for the
`d4cb599` cherry-pick on `agent/si-band-sensitivity`. Misleading
KNOWN_SANDBOX_ERRORS.md entries WITHDRAWN with cross-refs to the real docs.

Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/REDO_PROD_2026-05-19.md`.
Run dir: `runs/Si/NONBISPINOR_PROD_2026-05-19/` (qe/ symlinks to
`runs/Si/05_si_4x4x4_sym/qe/`).

## 2026-05-18: Memory-model non-bispinor + k-grid robustness — SYNTHESIS [coord]

Cross-cut synthesis of three parallel sub-agents (A: μ-sweep + HLO at scalar
ns=1; B: k-grid 2³→6³ scaling; C: nb=100 vs nb=200 sensitivity) on Si non-bispinor
against the bispinor-calibrated planner from `memory_model_refit_2026-05-17`.

**Headline.** Planner constants survive ns ∈ {1, 2, 4} bit-exact (A HLO),
per-term kgrid scaling matches analytic prediction within 5–6% on all 4 kgrids
(B), no leaks across r-chunks/bc-chunks/sym-channels/kgrids. Production-scale
non-bispinor configs (Si 4×4×4 nb=100 −0.8%, Si 6×6×6 nb=100 −6.1%, Si 4×4×4
nb=200 −3.2%) sit inside bispinor's [−0.5%, −10.8%] under-prediction window.

**Three biases identified, each with a quantitative mechanism:**
1. ~5–8 GB CUDA/JIT/NCCL framework floor dominates whenever the algorithmic
   peak is small (Si 2×2×2 → +96.5%, Si 3×3×3 → +52.9%, Si 4×4×4 μ=192 → +185%).
   Additive, NOT multiplicative — already user-deferred per §6.2.
2. Single-r-chunk degenerate configs over-predict by ~25% because the planner
   reserves 3 pair-density slots but only ~2 are concurrently live when
   n_chunks=1 (Si scalar μ≥768 in A).
3. **NEW**: Si 3×3×3 nb=200 breaks the bispinor window at −13.9%. Root cause
   identified — unmodeled `c128(nk, band_chunk, ns, r_chunk/p_y)` all_gather
   slab on `psi_l_X`/`psi_r_X` inside `z_q_from_psi_sm._local`. Documented in
   `docs/MEMORY_MODEL.md` §R-Chunk but absent from `_peak_C_fit_one_rchunk`.
   Same shape as bispinor Si μ=768 −10.8% and CrI3 80Ry −8.5% gaps.

**Highest-leverage open work (synthesis §5.1):** add `M_all_gather_slab` to
Peak C; Agent C estimates it lands the 3×3×3 nb=200 outlier at −5.8% and
likely cuts CrI3 80Ry from −8.5% to −5%. Needs HLO calibration of slab
coefficient first (1× vs 2×, with/without aliasing).

**Latent ns=1 bugs fixed in-branch.** Agent A: `unfold_psi` and
`WfnLoader._ensure_phdf5_static` both silently broadcast nspinor=1 → 2 via
2×2 spinor-rotation einsums. Commits `8c18925`, `dc0b254` on
`agent/si-nonbispinor-mu-sweep` (lorrax_A), 30 LOC, all 44 loader/unfold
tests pass at ns=2 — clean merge candidates for origin/main.

**New sandbox bug.** `cusolverMpPotrf` returns status=7 INTERNAL_ERROR under
BFC + PREALLOCATE=true + MEM_FRACTION=0.95 on a 2D mesh. Workaround:
`cusolvermp_charge=off, cusolvermp_lu=off` in cohsex.in. Logged at
`KNOWN_SANDBOX_ERRORS.md:117`.

**Bottleneck-flip risk.** B_CCT_chol hits 71% of binding Peak C at Si 6×6×6 μ=1348.
Will flip to bottleneck at larger μ or under bispinor 4-channel cascade. No
r-chunk knob to mitigate — remedy is smaller μ or larger mesh.

Synthesis: `reports/memory_model_nonbispinor_kgrid_2026-05-18/SYNTHESIS.md`.
Per-agent reports + JSON data + run dirs preserved under same dir + `runs/Si/{MU,KGRID,BANDS}_nonbispinor_2026-05-18/`.

## 2026-05-18: TRUE scalar (nspinor=1) Si non-bispinor μ-sweep + HLO calibration [agent-A]

Mirrored `agent_t_si_bispinor_sweep.md` at the **opposite** extreme — actually
scalar Si (`noncolin=false, lspinorb=false, nspinor=1, nspin=1`) at 4×4×4
25 Ry, 4 GPUs on hbm40g (2×2 mesh), μ ∈ {192, 408, 756, 1176, 1764}
(orbit-pruned counts from requested {192, 384, 768, 1200, 1800}).

**Two latent ns=1 loader bugs surfaced and fixed in-branch** (commits
`8c18925` `unfold_psi` eager path + `dc0b254` `WfnLoader._ensure_phdf5_static`
phdf5 path): both blindly built 2×2 spinor rotation matrices regardless of
the WFN's nspinor, causing silent einsum broadcasting from ns=1 input to
ns=2 output. Pre-existing `KNOWN_SANDBOX_ERRORS.md` 2026-05-18 entry
"scalar nspinor=1 Si WFN.h5 trips kmeans_cli" is now marked FIXED. All 44
loader/unfold tests still pass.

**HLO calibration at ns=1**: `pair_density_slots = 3` and `fft_box_factor_D = 2.0`
both confirmed bit-exact at μ=408 + μ=1176, per-slot bytes match
`_bytes_c128(nk=64, 1, 1, mu_padded, r_chunk=13824, shard=p_xy=4)` exactly.
The invariant 3-slot count now holds across ns=1 (this work), ns=2 (M4),
ns=4 bispinor (M1) — `pair_density_slots` is a structural constant of
`fit_one_rchunk`'s scan-INSIDE-shard_map, NOT an ns-dependent count.

**Planner faithfulness pattern is structurally different from bispinor**:
* μ=192:  HWM_pred=2.81  vs mem_stats peak=8.02 → +185 % UNDER (CUDA-context floor wins)
* μ=384:  HWM_pred=5.99  vs mem_stats peak=8.03 → +34 % under (floor still in play)
* μ=768:  HWM_pred=11.17 vs mem_stats peak=8.38 → −25 % OVER (planner conservative)
* μ=1200: HWM_pred=17.50 vs mem_stats peak=13.04 → −25 % over
* μ=1800: HWM_pred=26.51 vs mem_stats peak=19.60 → −26 % over

vs bispinor's monotone −0.5 % to −10.8 % range. Two new effects identified:
(1) a ~8 GB CUDA-context / JIT-cache / NCCL-buffer-pool floor invisible to
the planner (dominant at small Peak C); (2) the 3-slot prediction is
**conservatively** correct in multi-chunk plans but **over-counts** when
only 1 r-chunk runs (slot 3's lifetime is contained in the OUTPUT slot via
aliasing). Both `pair_density_slots=3` and `fft_box_factor_D=2.0` are still
the right constants — they correctly characterize the structural worst-case.

**Other findings**:
* cusolverMpPotrf returns status=7 INTERNAL_ERROR under BFC + MEM_FRACTION=0.95
  on 2×2 mesh (works on 1×4 in bispinor sweep because `sharded_cholesky`
  path is selected for 1D meshes). Worked around with
  `cusolvermp_charge = off, cusolvermp_lu = off` in cohsex.in. Documented
  in `KNOWN_SANDBOX_ERRORS.md`.
* No OOMs, no crashes, no leaks. r_chunk = 13824 = n_rtot at every μ; the
  Si scalar load is so small the planner always picks a single chunk.

Run dir: `runs/Si/MU_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_a_si_nonbispinor_mu_sweep.md`
Branch: `agent/si-nonbispinor-mu-sweep` on lorrax_A, tip `dc0b254`.

## 2026-05-18: Non-bispinor Si band-count sensitivity of gflat_memory_model planner [agent-C]

Stress-tested the planner's nb-scaling at fixed (kgrid, μ) by sweeping
nb=100 → nb=200 on Si non-bispinor (ns=2 noncolin=true, no SOC; cohsex.in
`bispinor=false`) at two k-grids: 3×3×3 (μ=408 snapped from 384, 1 r-chunk)
and 4×4×4 (μ=816 snapped from 768, 4 r-chunks). 40 Ry ecutwfc (bumped from
25 Ry so nbnd=200 fits the smallest-k sphere on 3×3×3), `cusolvermp_charge=off`
+ `cusolvermp_lu=off` to dodge agent-A's BFC bug, 28 GB budget per device.

**Headline:** planner stays within Agent T's `-0.5% to -10.8%` window at 3
of 4 (kgrid, nb) pairs but **3×3×3 nb=200 breaks out to -13.9% under-prediction**
(HWM_pred 15.70 vs mem_stats peak 18.24 GB/dev). Other 3: 3×3×3 nb=100 -0.5%,
4×4×4 nb=100 -1.2%, 4×4×4 nb=200 -3.2%. The 13.6%-point %-err jump nb=100→nb=200
at 3×3×3 is the largest in any A/B/C sweep so far at non-bispinor scale.

**Per-term nb-scaling:** every planner term scales exactly as predicted —
`centroids_persist` doubles (×2.00 measured ≈ predicted ×2.00 from
`4·c128(nk, ns, μ, nb_total)/p_xy` with `nb_total = nb_left + nb_right =
2·nb_cohsex`); all flat-in-nb terms (P_pair, zeta_out, gflat_acc, L_q)
match within 1%. **Peak C HWM_pred itself moves only +0.07 GB nb=100→200
at 3×3×3** (centroids-only delta), but mem_stats peak grows +2.53 GB —
36× more than the planner predicts.

**Where the missing +2.32 GB lives:** live_arrays at `after_fit_one_rchunk`
grew by only +0.21 GB nb=100→200 at 3×3×3, but `peak − live_total` (the
XLA preallocated-temp budget) grew by **+2.32 GB**. That excess is
unmodeled in-jit transient inside `z_q_from_psi_sm._local`, candidate
shape `c128(nk, band_chunk, ns, r_chunk/p_y)` — the per-rank post-all_gather
slab on `psi_l_X` + `psi_r_X` (×2). Predicted slab diff at 3×3×3
nb=100→200: 2 × 27 × 128 × 2 × 13500 × 16 = 3.0 GB (no aliasing); observed
+2.32 GB is consistent with one slab × 2 with ~25% aliasing discount.

**Proposed planner refinement:** add an `M_all_gather` term to Peak C
persistent base. HLO calibration of `z_q_from_psi_sm._local` at 3×3×3
nb=100 vs nb=200 needed to nail the coefficient (1× vs 2× slabs, and
aliasing). Expected to fix the 3×3×3 nb=200 gap from -13.9% to roughly
-5%, AND explain agent-T's worst-case bispinor μ=768 -10.8% gap (CrI3
production's -8.5% is also probably this all_gather slab not NCCL
overhead alone).

**Cross-r-chunk leak check (4×4×4, 3 r-chunks measured):** zero growth in
live_total across consecutive r-chunks at either nb=100 or nb=200 —
bc-scan correctly aliases its slots, no nb-correlated cross-chunk leak.

**Sandbox infra:** Agent A's `unfold_psi` nspinor=1 fix (commit `8c18925`)
cherry-picked as `d4cb599` on lorrax_C so future truly-scalar runs aren't
blocked (also fixed the nspinor=1 phdf5-loader issue per A's `dc0b254`
landing — A's pair of fixes is now on `origin/agent/si-nonbispinor-mu-sweep`).
Even with the unfold fix, the production loader still pads ns=1 → ns=2 via
`load_wfns.py:psi_G_flat` `ns_pad`, so the planner runs ns=2 for this
sweep too (consistent with agent-A and agent-B).

Run dir: `runs/Si/BANDS_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_c_si_band_sensitivity.md`
Branch: `agent/si-band-sensitivity` on lorrax_C, tip `d4cb599`.

## 2026-05-18: Non-bispinor Si k-grid scaling of gflat_memory_model planner [agent-B]

Stress-tested the planner on scalar (`bispinor=false`, ns=2 non-SOC) Si across
2×2×2 → 6×6×6 k-grids at fixed μ/nk_full ≈ 6 (μ=48, 192, 432, 1348 orbit-
unfolded). Held nb=100 logical bands, 25 Ry, 24³ FFT box. Each kgrid measured
under both `platform_false` (production allocator) and `bfc_pre95` (BFC +
preallocate=true + MEM_FRACTION=0.95) for OOM-relevant `mem_stats` peaks.

**Headline:** 4×4×4 −0.8%, 6×6×6 −6.1% — both inside bispinor's
[−0.5%, −10.8%] window from agent_T. Small kgrids (2×2×2 −96.5%, 3×3×3 −52.9%)
have huge fractional %-err but it's a constant ~5–8 GB CUDA/JAX/cuFFT/NCCL
floor, NOT a multiplicative scaling failure. Δ = peak − HWM_pred is roughly
constant (~5 GB) across kgrids.

**Per-term scaling**: every component within 5% of analytic predicted exponent
(nk^0, nk^1, nk^2, nk^3 classes all match — slopes 0.0, 0.9-1.18, 2.06-2.08,
2.86-2.87 respectively). `sphere_idx_replicated` stays at 1 buffer across all
kgrids — Round-6 canonical-accessor fix holds. `B_CCT_chol` becomes the 2nd-
largest peak at 6×6×6 (17.58 GB = 71% of C=24.82 GB) — at larger μ or with
bispinor cascade it could flip the bottleneck from C to B.

**Sandbox bug found**: `nspinor=1` (true scalar) is blocked by `get_spinor_rotations`
always returning (n_sym, 2, 2). Logged in KNOWN_SANDBOX_ERRORS.md 2026-05-18.
Worked around by using `noncolin=true, lspinorb=false` (nspinor=2 no SOC) —
same per-rank planner formulas exercise, but truly nspinor=1 would require
a fix in `sources/lorrax_B/src/common/symmetry_maps.py:1220` to special-case
the trivial spinor.

Run dir: `runs/Si/KGRID_nonbispinor_2026-05-18/`
Report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_b_si_kgrid_scaling.md`
Branch: `agent/si-kgrid-scaling` on lorrax_B (no LORRAX-source modifications).

## 2026-05-17: HLO calibration of planner constants pair_density_slots / fft_box_factor_D [agent-D]

Production-scale bispinor 80 Ry CrI3 HLO dumps + analysis to empirically
calibrate the two free constants of `gflat_memory_model.py`'s Peak C / Peak D.
JID 53075115 (4 nodes / 16 GPU / 4×4 mesh).

**Result:** `pair_density_slots = 3` and `fft_box_factor_D = 2.0` both confirmed
exactly. M1 (bispinor 4-channel, r=24576, b=32, gflat=360): 3 pair-density slots
× 20.04 GiB each (charge) / 19.83 GiB (transverse) in `fit_one_rchunk`'s 60 GiB
preallocated-temp.  M2: accumulate kernel shows 2 FFT-box slots × 6.03 GiB
(factor_D=2, not 4).  M3 (gflat=1 sanity): all 4 channels complete cleanly,
Peak D drops to 4.35 GiB ≈ planner prediction (< 1% error).  M4
(non-bispinor cross-check): 3 slots × 14.79 GiB (ns=2) matches.

The lorrax_B `agent/bispinor-ibz` working-tree edits to `gflat_memory_model.py`
(`fft_box_factor_D=2.0` + `pair_density_slots_{charge,transverse}=3`) are
empirically validated and safe to commit.  Old `fft_box_factor=4.0` was 2×
over-conservative, leaking ~13.8 GB of phantom Peak D budget at cs=360.

Run dirs (HLO dumps preserved for re-analysis):
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17/`
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_gflat1_2026-05-17/`

Report: `reports/memory_model_refit_2026-05-17/agent_d_hlo_calibration.md`.

## 2026-05-16: CrI3 6×6 **80 Ry** bispinor 16-GPU gate — INCOMPLETE (wall budget) [agent]

Attempted the 80 Ry production-scale Σ^B internal-consistency gate on JID 53057076
(4 nodes / 16 GPU / 4×4 mesh, 2:30 alloc).  Setup complete and on disk:
`runs/CrI3/M_6x6_80Ry_2026-05-07/{0X_lorrax_bispinor_fullbz_16gpu_2026-05-16,
0Y_lorrax_bispinor_ibz_16gpu_2026-05-16}/`.  Centroids regenerated:
charge `centroids_frac_1508.txt` (existing), transverse
`centroids_frac_1504_current.txt` (new, orbit-aware, ~3 min on 2 GPUs).  Both
sets pass orbit-closure under CrI3's 6 spatial sym ops.

**Findings & fix landed**:
1. Auto-planner picked `gflat_chunk_size = 717` at 80 Ry / mesh=16 — cuFFT
   batched-plan scratch allocator fails (`Failed to create cuFFT batched plan
   with scratch allocator`, 12 GiB scratch on top of 12.91 GB FFT box).
   **Worked around**: set `gflat_chunk_size = 360` in cohsex.in → per-iter FFT
   box 6.48 GB/rank, plan creation succeeds.  *Bug class: planner's
   `fft_box_factor=4.0` undercounts cuFFT's actual plan-side scratch at
   large `cs · n_rtot`; should land a correction in
   `gflat_memory_model.py` next session.*

**Why the gate didn't complete**: bispinor ζ-fit per-r-chunk at 80 Ry is
~14 s × 138 chunks ≈ 32 min per channel, × 4 channels (charge + 3 μ_L) =
**~128 min for ζ-fit alone**, plus V_q + Σ^B ≈ ~10–15 min.  Run A total ~140
min vs ~125 min alloc remaining at first ζ-fit start.  Doubling
`band_chunk_size` from 16 → 32 did **not** measurably speed up the inner
chunk (still 13.5 s/chunk; bottleneck is pair-density × cuFFT at
n_rmu=1508 not the bc-loop count).  Run A killed twice, gate output never
written; 30 Ry 51 µeV verdict stands as the strongest evidence to date.

**Next session prerequisites**: same branch `agent/bispinor-ibz` HEAD `d96aa46`,
same run dirs (left intact with cohsex.in + manifest + recon driver), **fresh
4 h allocation** (~3 h Run A, ~2 h Run B, ~10 min recon).  `lxalloc` with
`--time=04:00:00 --constraint="gpu&hbm80g"`.

## 2026-05-16: CrI3 6×6 bispinor IBZ 16-GPU end-to-end gate PASSES (51 µeV / 4×4 mesh) [agent]

Final driver for the bispinor IBZ-cascade end-to-end gate at production scale.
Run A (full-BZ reference, `LORRAX_FORCE_FULL_BZ=1`) and Run B (IBZ cascade) both
completed Σ^B + `v_q_bispinor.h5` write on a 4-node / 16-GPU / 4×4 mesh allocation
(JID 53054263).  Reconstruction via `reconstruct_sigma_b.py` rebuilt ψ once and
diffed Σ^B[k, m, n] between the two `v_q_bispinor.h5` files.

**Verdict: PASS at 0.0512 meV** (gate threshold 1 meV; bit-identical to the 2-GPU
gate's 51 µeV).  `Σ_X scalar` (charge channel) is bit-identical (Δ=0); the 51 µeV
residual lives entirely in Σ^B.  Per-tile trace shifts reproduce the 2-GPU
"Lorentz-mixing-cancels" signature — (μ_L=1,1)↔(1,2) tiles each shift by ~600 meV
in the IBZ cascade leg, with the cross-shifts cancelling in the channel sum.

Wall times: Run A ~356 s to V_q (n_q_solve=36), Run B ~180 s to V_q (n_q_solve=8,
~2× speedup from IBZ cascade), reconstruction ~190 s.

Source: branch `agent/bispinor-ibz` advanced from `9956dff` to `d96aa46` over 5
src/ commits.  The load-bearing one for 16-GPU was **`4930dab`
`fix(v_q_bispinor,reader): mesh-padded sharded tile reads`** — `BispinorVqReader.get_tile`
was reading transverse tiles at the on-disk logical extent (n_rmu_T=298), failing
`_validate_block_divisible` under any mesh with a sharded axis ≥3.  Now rounds μ
extents up to `gx*gy` and passes the padded shape as `shape=` plus the logical
extent as `valid_shape=` — mirrors the write-side `_round_up_to_mesh` at
v_q_tile.py:1116-1118.  Concurrent fixes: `d96aa46` (NamedSharding scope shadow in
gw_jax.py main()), `eb4a1e0` (band_chunk floor at world_size), `2de70eb`/`4b927dc`
(WFN loader band-pad past mnband).

Both runs crashed in *post-Σ* artifact writers (`write_qp_wfn_h5` for Run A,
`write_results` for Run B) — both `nk` vs `nk_irr` indexing bugs in the
bispinor + full-BZ vs IBZ-cascade combination, not exercised by the 2-GPU
gate.  Σ^B and `v_q_bispinor.h5` were emitted before either crash, so the
gate observable is unaffected.  Deferred to a separate writer-fix initiative
before any downstream BSE / WFN_qp consumer is wired up.

Artifacts:
- `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/sigma_b_gate_16gpu_v2_{A,B}.npz`
- `runs/CrI3/.../03_lorrax_bispinor_fullbz_16gpu_2026-05-16/tmp/v_q_bispinor.h5`
- `runs/CrI3/.../04_lorrax_bispinor_ibz_16gpu_2026-05-16/tmp/v_q_bispinor.h5`
- `recon_sigma_b_gate_16gpu_v2_2026-05-16/recon.out`

## 2026-05-16: CrI3 6×6 bispinor IBZ 16-GPU retest v2 — ABORTED (two new bugs at 4×4 mesh) [agent]

Re-attempted Run A on `agent/bispinor-ibz @ 9956dff` (band-pad fixes for
`load_centroids_band_chunked` and `psi_G_store._populate_from_loader` in place) on the
shared 4-node urgent allocation JID 53054263. **Two new bugs surface at 4×4 mesh; both
hit the failure-mode protocol's "STOP, do NOT patch" rule.** No Σ^B produced; cascade leg
(Run B) never launched.

- **Step `.0`** (HEAD as handed off): JAX init + COHSEX header OK, then phdf5 kchunk-union
  reader fails with `HDF5-DIAG: H5Dread failed` and `INTERNAL: phdf5
  read_kchunk_union: H5Dread failed`. Cause: replicated `counts` table doesn't clamp the
  tail rank's `(offset, count)` to the on-disk band extent when `mnband=86` is not
  divisible by `world * bands_per_rank=16 * 6`. SLURM cancelled the step after 41 min.
- **Step `.1`** (after an in-tree "PHDF5 FIX" + "MU FIX" added per-rank clamped counts to
  `src/file_io/wfn_loader.py` and `src/ffi/phdf5/read.py`; not committed): all 16 ranks
  raise `ValueError: empty band range: (86, 86)` in `WfnLoader.load:750`. Call site:
  `psi_G_store._populate_from_loader:225` requests the pad-only sub-block `(86, 86)` (the
  final, entirely-past-mnband band-chunk). The patched `WfnLoader.load` now rejects empty
  windows up-front, overriding the zero-fill path that commit `9956dff` added downstream.
  Followed by segfaults; SLURM cancelled the step.

Sandbox-errors log gained two new entries:
- `LORRAX_NGPU` is per-node in `lxrun`, not total — task specs that set
  `LORRAX_NGPU=16` (total) need `LORRAX_NNODES=4 LORRAX_NGPU=4` instead.
- Shifter env passthrough — `export LORRAX_FORCE_FULL_BZ=1` from the shell does not reach
  the container; the launch must add `--env=LORRAX_FORCE_FULL_BZ=1` to `LORRAX_SHIFTER`.

JID 53054263 still RUNNING at the time of write-up (~1 h 13 min left, 4/4 nodes idle).
No source modified by this session; both new bugs deferred to the orchestrator.

## 2026-05-16: CrI3 6×6 30Ry bispinor IBZ gate FAILS at 16 GPUs — band-axis world-size pad outruns WFN file [agent]

Re-running the prior 2-GPU bispinor IBZ end-to-end gate on the production
16-GPU / 4×4 mesh (`runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/03_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
and `04_lorrax_bispinor_ibz_16gpu_2026-05-16/`, lorrax_B `agent/bispinor-ibz`
@ `82520a1`) hit the **suspected small-`nbnd` band-sharding death mode**
(see `~/.claude/.../project_cri3_small_nbnd_band_sharding_suspect.md`)
on all 16 ranks, before any V_q tile, kernel, or HLO compile.

Mechanism (root cause identified, not patched):

- `common/meta.py:107-109` sets `b_id_4 = round_up(nband_user=84, world_size=16) = 96`.
- `gw/wavefunction_bundle.py:83` exposes `full_range = (b0, b4) = (0, 96)`.
- `gw/gw_init.py:1205-1209` → `common/load_wfns.py:437-439` → `loader.load(bands=(0, 96), ...)`.
- `file_io/wfn_loader.py:678-681` rejects `b_hi=96 > self.nbands=86` (NSCF `nbnd=86`).
- Error: `band range (0, 96) out of [0, 86); use bands_pad_to-style external padding for over-file requests`.
- The `meta.py:100-117` comment promises zero-fill past `b_id_4_user` in
  `load_centroids_band_chunked`; the actual call path doesn't slice or pad
  externally before the loader sees the over-file extent.

Mesh sensitivity (`nband_user=84`, NSCF `nbnd=86`):

| world_size | `round_up(84, ws)` | vs file `mnband=86` |
|------------|-------------------|---------------------|
| 2 (prior gate) | 84 | OK |
| 4 | 84 | OK |
| 8 | 88 | FAILS |
| **16 (this gate)** | **96** | **FAILS** |

So 8 GPUs would already fail; the production 16-GPU mesh fails by 10 slots.

Per task contract: no source patch, no commit, no retry on fewer GPUs.
Allocation 53050082 (`hbm80g`, 4 nodes) released. Full failure analysis +
recommendation in the gate report (returned to the user as text — no
report.md file written per subagent conventions). Run 04 (IBZ-cascade
leg) was skipped because both legs run the same `Meta.from_system` +
`prepare_isdf_and_wavefunctions` code path before either IBZ branch is
taken; the bug would fire identically.

## 2026-05-16: bispinor IBZ Σ^B gate PASS at 51 µeV on CrI3 6×6 30Ry [agent]

End-to-end internal-consistency gate for the new bispinor IBZ cascade
(`lorrax_B agent/bispinor-ibz` @ `882ed4a`, 3-vector Lorentz mixing on
TT tiles). Two paired LORRAX runs on the same CrI3 6×6×1 30 Ry SOC
QE/centroid reference under `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/`
differing only in `cohsex.in: bispinor_use_ibz`:

- `01_lorrax_bispinor_fullbz_ibz_gate_2026-05-16/` — reference (false)
- `02_lorrax_bispinor_ibz_2026-05-16/` — new IBZ cascade (true)

Per-(k, n) Σ^B reconstructed via `reconstruct_sigma_b.py` (calls
LORRAX as library; no source mod) and diffed in `analyze_gate.py`:

- `max |Δ Σ^B[k, n]|` = **0.0507 meV** over (36 k) × (84 sigma bands)
- mean = 0.0027 meV; RMS = 0.0059 meV
- Gate threshold 1 meV → **PASS** by 20×
- Scalar Σ_X (charge channel): **bit-identical** between A and B (Δ = 0)

Per-tile traces of (1,1), (2,2), (1,2), (2,1) each shift by ~600 meV
under in-plane proper rotations with opposite signs that cancel in
the contracted Σ^B — positive evidence the 3-vector Lorentz mixing
`V^{ij} = R^{iα} R^{jβ} V^{αβ}_{IBZ-unfold}` is acting unitarily.

V_q kernel wall: 26.28 s (A) → 6.28 s (B), **4.18× speedup** on the
V_q stage (4.5× IBZ shrink on 36→8 q's for P-3). Total wall is
dominated by ζ-fit on this 6×6 case (~6 min) which is unchanged.

Both runs are blocked at QP output by the pre-existing kin_ion
crash (`KNOWN_SANDBOX_ERRORS.md` 2026-05-14) but Σ_X printing
completes first under `x_only = true`.

Plot: `reports/bispinor_ibz_e2e_gate_2026-05-16/sigma_b_gate_scatter.png`.

## 2026-05-15: CrI3 sym-vs-nosym L-phase + perm-direction fix [agent]

Two-bug fix on `agent/trs-aware-sym-fix` commit `0735c2a`:

1. **Missing per-centroid umklapp phase** `exp(2π i q · (L_μ − L_ν))` in
   `unfold_v_q`. `L_μ = floor(S r_μ + τ)` is now captured by
   `compute_centroid_sym_perm` (which returns `(sym_perm, L_table)`) and
   threaded through `_resolve_ibz_q_list` → `unfold_v_q` in both
   `gw/v_q_g_flat.py` and `gw/compute_vcoul.py`.

2. **Wrong centroid permutation direction** in `unfold_v_q`. The previous
   code used `inv_perm = argsort(sym_perm)` (= π⁻¹). Correct direction is
   `sym_perm` directly (= forward π). For involutive ops (MoS2 σ_h, Si
   cubic) the two coincide — that's why MoS2 + TRS passed at 0.090 meV
   while CrI3 C3/S6 sat at 4 eV.

Unit test: `reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py`
closes V_q to ISDF noise floor (rel 8.73e-6 ≈ 22 eV out of |V|=2.5e6) on
all 36 q's of the CrI3 6×6 30 Ry dump, including non-involutive ops AND
umklapp shifts (kg0 ≠ 0). 13/13 pytest tests in centroid/unfold domain.

Convention reference document:
`reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`. Empirically
verified that `wfn.sym_matrices = U` (forward direct-space sym), NOT K
(reciprocal rotation) as a prior agent claimed; the user's existing DFT
degeneracy tests across multiple systems relied on this convention.

Run: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym_lphase_fix_2026-05-15/`
(pending e2e gate; expected to drop max |ΔΣ_X| from 6 eV → <1 meV).

## 2026-05-14: CrI3 sym-vs-nosym PR3 validation — FAIL (third sym-handling bug) [agent]

Run dir: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `8504994` + `a45f039` + `69ab42c`. Task #30 mirror
for a second inversion-containing system, complementary to MoS₂ (PASS, 0.090 meV)
and Si (FAIL, 160 eV — τ-phase bug).

Pipeline: regenerated nosym NSCF (ntran=1, nrk=36) from existing CrI3 SCF charge
density → 2 LORRAX cohsex runs (x_only=true, do_screened=false, bispinor=false,
bare_coulomb_cutoff=30.0) sharing the existing 300 orbit-closed centroids from
`M_6x6_30Ry_bispinor_2026-05-14`: `run_sym/` (ntran=6, P-3 spatial group with
inversion, IBZ cascade fires n_q_disk=8 → 36 full-BZ unfold) vs `run_nosym/`
(ntran=1, direct full-BZ).

Result: **max |ΔΣ_X(k, n)| = 6022 meV ≈ 6.02 eV** across all 36 k × 84 bands.
Uniform ~5 eV residual at every k-point (no clean k); worst rows at valence-top
d-bands (b=60-61, 56-57, 64-65); systematic mean −2046 meV (sym more negative
than nosym). NOT a PR3 firing — CrI3 has spatial inversion in mtrx ⇒ 0 TRS-fold
k-pts ⇒ PR3 (iσ_y·conj and τ-phase) is a strict no-op for this system, which
this test experimentally confirms. The 6 eV residual exposes a **third, distinct
sym-handling bug**: broken IBZ→full V_q (or ζ) cascade unfold for groups
containing C3 + improper rotations (S6, −I). The MoS₂ pass (E + σ_h only) was
insufficient to detect this; the Si τ-phase bug (non-symmorphic Fd-3m) is a
separate failure mode (CrI3 is symmorphic with τ=0 for all 6 ops).

Triage targets (next session):
1. Bisect against `9e644e9` (pre-Phase-2) on the CrI3 test bed to confirm
   pre-existing — mirroring the Si triage. Likely pre-existing.
2. Suspect: SIGN or CONJUGATE flip missing in V_q (or ζ) unfold when sym op
   has det=−1, OR wrong G-vector mapping under improper rotations / C3.
3. Fix pre-existing nrk=48 vs n_unfold=36 crashes in `qp_wfn.write_qp_wfn_h5`
   (line 137 shape check) and `gw_output.write_results` (line 288 indexer)
   — both fire AFTER sigma_freq_debug.dat is written and so didn't block this
   validation, but they should use `meta.nkpts_unfolded` not `wfn.nkpts`.

Report: `reports/trs_sym_audit_2026-05-14/cri3_sym_vs_nosym_pr3.md`.
Comparison: `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/compare_sigma_x.{py,log}`.
Total cost ~8 GPU-min.

## 2026-05-14: sym-vs-nosym PR3 e2e validation gate — PASS [agent]

Run dir: `runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `8504994` + `a45f039` + `69ab42c`. Task #30 — the
load-bearing end-to-end gate for the Phase-2 sym refactor.

Pipeline: 1 kmeans run on `00_mos2_3x3_cohsex/qe/nscf/WFN.h5` (sym, ntran=2)
→ 399 orbit-closed centroids → 2 LORRAX cohsex runs (`x_only=true`,
`do_screened=false`, `bispinor=false`, `bare_coulomb_cutoff=30.0` explicit)
sharing those centroids: `run_sym/` (sym WFN, exercises PR3 unfold_psi +
iσ_y·conj on TRS k {1, 3, 4, 5}) vs `run_nosym/` (`02_mos2_3x3_nosym/qe/nscf/WFN.h5`,
ntran=1, IBZ cascade trivial).

Result: **max |ΔΣ_X(k, n)| = 0.090 meV** across all 9 k-pts × 56 bands.
Pass gate was ≤ 1 meV; observed residual is 11× below threshold and is
essentially the DFT-eigenvalue ULP-offset (0.069 meV mean) between two
independent SCF runs propagated through the same Σ_X kernel. TRS-group
mean |ΔΣ_X| = 0.028 meV vs non-TRS-group 0.030 meV — indistinguishable,
which is exactly the signature of a correct sym implementation. The PR3
audit (`audit_pr3.md` R3) had shown PR3 shifts Σ_X by ≤95 meV on this same
test bed (bug was firing); this gate confirms PR3's fix produces the
*physically correct* answer (matches direct nosym evaluation to ULP).

Report: `reports/trs_sym_audit_2026-05-14/sym_vs_nosym_pr3_validation.md`.
Comparison: `runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/compare_sigma_x.{py,log}`.
Total cost ~3 GPU-min.  PR1+PR2+PR3 cleared from this gate.

## 2026-05-14: testbed_mos2_3x3_soc — PR3 ψ-side TRS-fix validation baseline [agent]

Run dir: `runs/MoS2/03_mos2_3x3_soc_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `796c043` + a one-line `dft_operators.py` migration
fix (see below). Goal: bring up a **non-inversion SOC** test bed so PR3's
ψ k-unfold iσ_y rotation + TRS-row Gkk τ-phase fix has a non-trivial signal.

Pipeline: QE SCF + NSCF + NSCFq (3×3×1, 30 Ry, noncolin+lspinorb, nbnd=58) →
BGW epsilon + sigma cohsex (`number_bands 56`, `bare_coulomb_cutoff 30.0` explicit) →
LORRAX kmeans (orbit-closed, 399 centroids from 206 reps) → dipole + kin_ion →
LORRAX `gw_jax` cohsex on 2 GPUs (nval=26 divisible).

Symmetry probe (`sym_analysis.log`):

```
ntran = 2  (E + σ_h; no_t_rev=.true. + SOC kills rotations)
len(sym.sym_mats_k) = 4
#k via TRS  = 4  (full-BZ {1, 3, 4, 5})
#q via TRS  = 4  (full-BZ {2, 6, 7, 8})
has_inversion = False  ← suitable PR3 test bed
```

Σ_X finite at every (k, n). Group-mean Σ_X (post-PR3, band 19..30 window):
  - TRS k group     N=48  mean = -17.585 eV
  - non-TRS k group N=60  mean = -17.523 eV

By the time this task was launched, PR3 (`8504994`) had just landed.  To
produce a real pre-PR3 baseline, ran cohsex AGAIN with src/ at `796c043` +
the same dft_operators fix.  PR3 ψ-side TRS-fix diff (`pr3_diff_summary.log`,
band-19..30 × 9-k window):

|       | Δx_bare max\|Δ\| | Δx_bare rms | Δeqp0 max\|Δ\| | Δeqp0 rms |
|-------|-----------------|-------------|----------------|-----------|
| TRS k | 59 mΩ           | 16 mΩ       | 64 mΩ          | 17 mΩ     |
| non-TRS k | 49 mΩ       | 12 mΩ       | 53 mΩ          | 13 mΩ     |

Δx_head and Δcoh_head are bit-equal (scalar head untouched by PR3).
Non-TRS k diffs are non-trivial because the wrong ψ at TRS-folded k
pollutes χ_0(q), hence W(q), hence Σ at every k through q = k − k′.
The 64-mΩ max-Δ_eqp0 sits in the 10-100 meV PR3 prediction band from the
task spec; TRS rows are ~20% larger than non-TRS rows.

**Source-code fix** (commit `a45f039` on `agent/trs-aware-sym-fix`):
`sources/lorrax_B/src/psp/dft_operators.py::generate_gvectors_k` was still
calling the post-P5-removed `sym.get_gvecs_kfull` API; patched to dispatch
through `WfnLoader.gvecs(k="full_bz")`, mirroring the pattern in
`psp/get_DFT_mtxels.py::_gvecs_full_cache`. This unblocks both
`psp.get_dipole_mtxels` AND `gw.kin_ion_io` on this branch (closes a
KNOWN_SANDBOX_ERRORS entry).

## 2026-05-14: testbed_cri3_6x6_30Ry_bispinor — bare Σ_X end-to-end on lorrax_B `agent/trs-aware-sym-fix` [agent]

Run dir: `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/`. Source: lorrax_B
`agent/trs-aware-sym-fix` @ `a00722d` (post-PR2 V_q IBZ→full unfold lift).

Pipeline: QE SCF/NSCF (30 Ry, 6×6×1, noncolin+lspinorb, 86 bands) →
BGW epsilon + sigma (Σ_X reference) → LORRAX kmeans (scalar 300 +
current-density 298) → LORRAX cohsex (`x_only=true`, `bispinor=true`).

Result: finite bare Σ_X printed for the bispinor configuration. Off-diagonal
Σ^B tile traces ~7% of diagonal; spin-doubled degeneracy holds to 4 ULP;
tile hermiticity holds to 4 ULP. Cascade did not fire (bispinor mode
disables IBZ-only ζ writes by design in `gw_init.py:650`; CrI3 also has
`-I` so TRS folds = 0). Downstream QP analysis blocked by a separate
`kin_ion_io` `SymMaps.get_gvecs_kfull` bug (see `KNOWN_SANDBOX_ERRORS.md`).

Important sandbox surface area that this run uncovered (all logged in
`KNOWN_SANDBOX_ERRORS.md` 2026-05-14):
- `centroid.kmeans_isdf` is **not** a CLI module; use `centroid.kmeans_cli`.
- Bispinor V_q requires a second `--density-mode current` kmeans run AND a
  `centroids_file_current = ...` entry in `cohsex.in`; otherwise the
  bispinor branch silently falls back to scalar V_q and then crashes on a
  full-BZ vs IBZ ζ shape mismatch.
- This run config OOMs on 40GB A100 even with `band_chunk_size=2`,
  `r_chunk_size=8192` — `--constraint="gpu&hbm80g"` is required.

Caveat (documented in `README.md`): CrI3 has spatial inversion, so this
test bed validates only the bispinor V_q tile pipeline / cascade
machinery layout, NOT the iσ_y·conj TRS-spinor patch (Agent 1 sites
#5/#6/#7). A non-inversion bispinor system (1H-MoSe₂ + SOC, BiI₃, or
CrI3 + E_perp) will be needed when PR3 lands.

## 2026-05-13: zeta-fit memory model follow-up — 2nd HLO dump + Path D scaffolding on LORRAX_B [agent]

Reports: `reports/zeta_rchunk_memory_model_2026-05-13/{agent_1_hlo_verify,agent_2_structural_fix,hlo_findings}.md`.
Branch: `sources/lorrax_B` at `agent/zeta-bc-scan-shardmap` (commit `cdd0fba`).

Same allocation as the previous entry, follow-up work on the morning
commit `ff5873c` (LORRAX_A).

Two agents re-engaged via the tmux team:
- Agent 1 (HLO verification): independently re-read `module_0408`,
  confirmed `pair_density_slots = 3` (0.01% accuracy), `S_fft ≈ 3`
  (formula matches observed bytes at 0.1%), `psi_Y_full` aliases
  cleanly.  Flagged two errors in `hlo_findings.md` §2/§4: "band_chunk
  is the lever against W_wfn" was wrong (band_chunk is band-axis-
  invariant for this term — only `psig_k_chunk_size` and `(nb_L+nb_R)`
  move it); mesh dimension wasn't explicit.  Both fixed in
  `hlo_findings.md`.
- Agent 2 (structural fix design): evaluated 5 candidate paths
  (`fori_loop`+donation, `scan`+axis-naming, streaming einsum,
  bc-loop-inside-shard_map, `donate_argnums`).  Recommended **Path D**
  — push the bc-loop INSIDE the shard_map body via `lax.scan`.  The
  scan carry is per-rank-local; SPMD has already stopped at the
  shard_map boundary, so the WhileOp/SPMD trap that killed
  `solve_zeta`'s fori_loop attempt (cited in `isdf_fitting.py:1119-1141`)
  does not apply.  Expected: 58 → ~3 FFT-box slots.

2nd HLO dump at `psig_k_chunk_size=1`:
`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13/`.
Planner picked `r_chunk=17 568, band_chunk=16, gflat_chunk_size=320`;
predicted HWM = 47 GB/rank on a 35 GB budget (model now over-
conservative); XLA actually allocated **23.42 GiB/rank** in
`module_0307.jit__kernel.sm_8.0-memory-usage-report.txt`.  Reduction
from the OOMing run: 196 → 22 GiB (8.85× vs the model's 6×
prediction).  Extra factor came from `r_chunk` also dropping ~4×
(W_zeta term collapse).  Run completed the fit_zeta loop end-to-end
with no OOM, confirming `ff5873c` plus `psig_k_chunk_size=1` is a
working empirical config.

Commit `cdd0fba` on LORRAX_B lands the **Path D scaffolding** — two
read-only helpers that are independently testable and unlock the
larger 4c kernel rewrite:
1. `common.wfn_transforms.to_rchunk_inner` (Path D §4b): per-rank-
   local body of `to_rchunk` without the enclosing shard_map.
   Callable from inside another shard_map's body or a `lax.scan`
   body.  Numerical contract verified by three new tests at floating-
   point precision against `to_rchunk` on a 1×1 mesh.
2. `common.psi_G_store.PsiGStore._slice_local_tile_bc` (Path D §4a):
   host-tile slicer that takes a TRACED `bc_idx`, returns a padded
   `(nk, _bpd_max, ns, ngkmax)` array so `io_callback` inside a
   `lax.scan` body sees a static return shape.  Added `_bpd_max`
   field to `__init__`.

15 wfn_transforms tests pass (3 new), 12 psi_g_store + rchunk_gflat
tests pass (no regressions).

Path D 4c-e (rewrite `c_q_from_psi_sm` / `z_q_from_psi_sm` with the
`lax.scan` over bcs inside their shard_map bodies — the load-bearing
kernel rewrite, ~200-250 LOC across `isdf_fitting.py`) deferred to a
focused session.  Implementation sketch + validation plan are at
`reports/zeta_rchunk_memory_model_2026-05-13/agent_2_structural_fix.md`
§4c-e + §5.

Notes recorded for the future implementer:
- The current planner is now **over-conservative** (predicted HWM 47 GB,
  reality 22 GB).  This is a side-effect of the simple "everything
  lives concurrently" model; XLA aliases more aggressively when per-
  slot bytes drop.  Refining the planner to match reality is lower
  priority than Path D, which collapses the slot count entirely.
- The current feasibility-check raise is a sound lower bound
  (`band_fft_pool > budget` ⇒ definitely infeasible) but not tight.
  Folded into Path D's follow-up because Path D removes the need for
  this gate entirely.

## 2026-05-13: zeta-fit r-chunk memory model — 4-agent synthesis + HLO-verified bug fixes on LORRAX_A [agent]

Reports: `reports/zeta_rchunk_memory_model_2026-05-13/{consensus,hlo_findings}.md`
Branch: `sources/lorrax_A` at `agent/zeta-r-chunk-fixes-2026-05-13` (commit `ff5873c`).

Spawned a 4-agent independent study (4-pane tmux + Opus 4.7) of the
GWJAX zeta-fit memory model.  Two rounds: from-scratch v1 drafts
(`agent_{1..4}.md`, ~6k words each) → cross-reading v2 with disagreements
+ open questions (`agent_{1..4}_v2.md`, ~3.5k words each) → orchestrator
synthesis (`consensus.md`, 3.5k words).  Consensus identified two source
bugs everyone agreed on and three HLO-only disputes.

Empirical resolution on a CrI3 6×6 80 Ry HLO dump
(`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/`,
planner-free at the report.md §7 60 GB / `band_chunk=16` config):
- Planner-free pick was `r_chunk=73 328` (16 chunks), HWM-estimated 52 GB,
  not the 12 500 cited in `report.md §7` — the original number was an
  unverified estimate.
- XLA actually requested 200.35 GiB per rank → `RESOURCE_EXHAUSTED` OOM
  at 196.30 GiB on 40 GB A100s.  3.85× model miss attributable entirely
  to the previously-unmodeled band-FFT pool: 58 concurrent live
  `c128[k_chunk, band_chunk, ns, nx, ny, nz]` slots at 3.22 GiB each,
  materialised UNSHARDED on every rank.  XLA cannot alias them because
  the Python-unrolled bc-loop in `c_q_from_psi_sm` / `z_q_from_psi_sm`
  has overlapping lifetimes between iterations.

Commit `ff5873c` lands two fixes on `agent/zeta-r-chunk-fixes-2026-05-13`:

1. `_bytes_centroids_LR` helper — fixes the `centroids_persist` term:
   replace `nk`-typo with `nb_total`; replace `shard=p_xy` with
   per-axis division for the L+R copies that live on disjoint mesh
   axes.  ~4× correction on a balanced 4×4 mesh.  Applied at the
   three sites (lines 184, 316, 350).
2. Add `band_fft_unsharded` term + structural feasibility check in
   `plan_gflat_chunks`.  Total cost is
   `nb_total · S_fft · psig_k_chunk_eff · ns · n_rtot · 16`
   per rank — **band_chunk-independent**; only `psig_k_chunk_size`
   reduces it linearly.  When the pool alone exceeds budget the
   planner raises a structured `ValueError` listing three mitigations
   (lower `psig_k_chunk_size`, narrow `nval/ncond`, or the structural
   `lax.fori_loop` rewrite).  `cfg.memory.psig_k_chunk_size` is now
   threaded through `gw_init.fit_zeta`.

Tests: `tests/test_aot_memory.py` + `tests/test_rchunk_gflat_pair.py`
13 passed / 3 skipped after the fix.  CPU smoke confirmed: planner
correctly refuses the previously-OOMing CrI3 config at
`psig_k_chunk=6` and picks a feasible plan
(`band_chunk=64, r_chunk=21 808`) at `psig_k_chunk=1`.

Follow-ups not in this commit, in priority order:
- Peak A's centroid-load FFT box is likely also unsharded (single
  slot, smaller impact).  Needs Peak A HLO confirmation.
- Wire `common/fft_helpers.query_fft_peak_bytes` per call site to
  replace the global `fft_box_factor=4.0`.
- Structural: convert `c_q_from_psi_sm` / `z_q_from_psi_sm` bc-loop
  to `lax.fori_loop`.  Would alias the `n_bc · S_fft` slots into one
  and recover most of the band-FFT pool's memory cost.

## 2026-05-13: WFN rchunk construction communication profile on LORRAX_D [agent]

Report: `reports/wfn_rchunk_profile_2026-05-13/report.md`

Inspected newest `sources/lorrax_D` WFN loading / `psi_n_XYk(rchunk)` path and
mined the freshest 4-GPU D-usage HLO profile while the new allocation remained
pending.  The source path intends rank-local behavior after
`PsiGStore.fetch_psi_rchunk` pulls each rank's host tile, but the HLO shows
large unwanted all-gathers in the fused per-rchunk kernel: repeated
`506.25 MiB` `all-gather-start` operations attributed to
`common/wfn_transforms.py:109`, gathering local
`c128[4,4,9,24,24,80]` FFT-box shards into full
`c128[16,4,9,24,24,80]` buffers.  `psi_G_store.populate.loader_load` is only
~0.84-0.86 s for five band chunks and `shard_to_host` is ~10 ms, so the
communication problem is not the host tile copy; it is the JAX/SPMD boundary
around G-flat gather / FFT-box materialization.  Recommended next edit: keep
the full `to_rchunk` pipeline inside a single `shard_map` region rather than
only wrapping the FFT.

Follow-up implementation trial on the same branch added an opt-in
`LORRAX_PSIG_RCHUNK_SHARDMAP=1` path:

- `src/common/wfn_transforms.py`: new `to_rchunk_shard_map` keeps G-flat
  gather, local IFFT, r-slice, and Bloch phase inside one `shard_map` region.
- `src/common/psi_G_store.py`: `fetch_psi_rchunk` dispatches to that variant
  when the env var is set.
- Fresh 4-GPU run:
  `runs/MoS2/00_mos2_3x3_cohsex/D_wfn_rchunk_shardmap_2026-05-13`.
  It completed end-to-end; `eqp0.dat` numeric rows match the prior D-usage
  profile exactly (timestamp differs).
- HLO result: the `wfn_transforms.py:109` `506.25 MiB` all-gathers disappear
  from `collectives_details.txt`.  Top remaining collectives are now V_q-side
  `gw/v_q_tile.py:717/718` all-gathers at ~183-188 MiB.
- Targeted validation passed on CPU:
  `24 passed` for `test_wfn_transforms.py`, `test_rchunk_gflat_pair.py`, and
  `test_psi_g_store.py`.  Full CPU-forced suite remains not clean due unrelated
  failures (`gw_jax` regression subprocess OOMed on GPU selection, plus the
  known `make_v_munu_chunked_kernel(... mesh_xy)` API drift in two V_q tests).

## 2026-05-13: GWJAX FFI/JAX boundary profile on LORRAX_A [agent]

Moved the profiling investigation to `sources/lorrax_A` on branch
`agent/ffi-boundary-profile-a`, based on `origin/agent/zeta-ibz-header`.
No LORRAX_A source files were modified.

Report: `reports/ffi_boundary_profile_a_2026-05-13/report.md`

Ran two 4-GPU MoS2 3x3 `gw.gw_jax` profiles with the sandbox profiling stack:

- Baseline A:
  `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_profile_2026-05-13`
  (`path=sharded_cholesky` for charge, JAX/CUDA LU for transverse).
- cuSOLVERMp A:
  `runs/MoS2/00_mos2_3x3_cohsex/A_ffi_boundary_cusolvermp_profile_2026-05-13`
  with `LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=1` and
  `LORRAX_USE_CUSOLVERMP_LU=1`.

Key findings:

- End-to-end `run_module:gw.gw_jax` wall time was essentially unchanged:
  baseline `101.25 s`, cuSOLVERMp `101.40 s`; profiled totals were
  `85.422 s` vs `85.844 s`.
- HLO custom-call counts show only 3 `lorrax_cusolvermp_batched_potrf`,
  3 `lorrax_cusolvermp_batched_potrs`, and 9
  `lorrax_cusolvermp_batched_solve_lu` calls in the full cuSOLVERMp run, so
  direct Python/JAX-to-CustomCall boundary count is not the main wall limiter.
- cuSOLVERMp reduced HLO modules/compile count slightly (`1077 -> 1033`,
  XLA compile `21.5 s -> 16.4 s`) but greatly increased low-level GPU trace
  activity (`12k -> 242k` GPU events, `8 -> 654` compute streams).
- The strongest avoidable JAX overhead is first-run orchestration:
  roughly 600 cache misses in both runs, led by local `_per_rank` jitted
  closures in `src/file_io/_slab_io_ffi.py` read/write paths and repeated small
  primitives in `wfn_loader.py`, `gamma_matrices.py`, and `fft_helpers.py`.
- The largest steady-state levers are still high-level GWJAX collectives and
  data motion: `V_q_compute` at ~38.5 s, zeta fits at ~36-37 s total, repeated
  2.47 GiB all-gathers in `wfn_transforms.py`, and all-reduces in the
  factorization path.
- Follow-up: fast-forwarded `lorrax_A` branch `agent/ffi-boundary-profile-a`
  to the D usage stack (`lorrax_D/agent/cusolvermp-ffi-profile`, commit
  `c21d855`) and reran the same profile in
  `runs/MoS2/00_mos2_3x3_cohsex/A_rebased_D_usage_profile_2026-05-13`.
  Compile/retrace metrics improved (`582/562 -> 478` XLA compiles and
  `629/602 -> 525` cache misses), but wall time regressed to `150.30 s`
  (`133.750 s` profiled) because zeta HDF5 write/close time ballooned.
  The top cache misses still include `_slab_io_ffi.py` `_per_rank` factories
  (`20` read, `17` write), so the first caching pass is incomplete from JAX's
  callable-identity perspective.
- q-loop acceleration probe: prototyped an opt-in CUDA Graph replay path for
  the existing full-mesh `cusolverMpPotrf` q-loop using stable ctx-owned staging
  buffers.  The code built, but `cusolverMpPotrf` failed during stream capture
  with status `7` at `q=0` on all ranks under cuSOLVERMp 0.7.2 / NCCL 2.26.3.
  Baseline potrf for the same shape was `9.523 ms` median.  Removed the failed
  prototype and rebuilt the FFI shared library from the reverted source.

## 2026-05-12: cuSOLVERMp FFI 4-GPU profiling harness + Nsight traces [agent]

Branch `agent/cusolvermp-ffi-profile` on `lorrax_D`.

Added a dedicated 4-GPU benchmark/profiling driver near the cuSOLVERMp FFI:
`sources/lorrax_D/src/ffi/cusolvermp/profile_batched.py`.  The harness runs
under the normal `lxrun`/Shifter/JAX-distributed path, defaults to the
MoS2-3x3-like shape (`nq=9`, `n=640`, `mrhs=640`, `complex128`, `2x2` mesh),
prebuilds donated inputs outside the timed range, and supports
`nsys --capture-range=cudaProfilerApi`.

Report: `reports/cusolvermp_ffi_profile_2026-05-12/report.md`

Code/profile changes:
- Added NVTX step ranges to `batched_potrf_ffi.cc` and `batched_potrs_ffi.cc`
  for cross-stream waits, copies, descriptor setup, buffer-size query,
  workspace ensure, per-q cuSOLVERMp calls, and descriptor teardown.
- Added the CUDA toolkit target include directory to the FFI CMake include
  path so `<nvtx3/nvToolsExt.h>` resolves in the container.
- Captured rank-local Nsight Systems traces under
  `runs/FFI/cusolvermp_batched_profile_2026-05-12/nsys/`.

Findings:
- `potrf` at `nq=9, n=640, c128, 2x2` is ~9.45 ms median; the `nq` sweep
  is nearly linear (`nq=1`: ~1.95 ms, `nq=18`: ~17.86 ms).
- Combined `potrf_potrs` is ~28-29 ms median, with `potrf` ~9 ms and `potrs`
  ~18 ms in the trace.
- Descriptor creation, buffer-size query, workspace ensure, and destroy are
  single-digit microseconds per FFI call on this build; caching them is not
  the first high-impact optimization here.
- Cross-stream waits are also tiny (~5 us median per bridge). The dominant
  overhead is the serial per-q cuSOLVERMp call queue and its internal
  NCCL/cuBLAS/cuSOLVER kernels.
- Quick NCCL env checks (`NCCL_PROTO=Simple`, `NCCL_PROTO=LL128`,
  `NCCL_MAX_NCHANNELS=1`) all regressed this shape, so defaults are currently
  best.
- Verification: FFI C++ rebuild passed, `profile_batched.py` py-compiled,
  and a 4-GPU `potrf_potrs` smoke run passed. Full `uv run python -m pytest -q`
  failed outside the touched FFI/profiling files (`make_v_munu_chunked_kernel`
  test API drift, k-means label tie mismatches, and one CUDA OOM regression
  subprocess); see the report for details.

## 2026-05-12: cuSOLVERMp FFI profiling orientation [agent]

Oriented on `lorrax_D` branch `agent/zeta-ibz-header` for the distributed
linear-algebra FFI overhead investigation.  Read the sandbox skills,
profiling stack, LORRAX_D agent guide, current branch state, and the
cuSOLVERMp/cuBLASMp/SLATE/phdf5 FFI docs and hot paths.

Report: `reports/cusolvermp_ffi_profile_orientation_2026-05-12/report.md`

Notes:
- Active hooks found: `lxalloc`, `lxrun`, `lxshell`, `lxpre`; no `lxattach`
  hook was present in the searched sandbox/source paths.
- `batched_potrf_ffi.cc`, `batched_potrs_ffi.cc`, and cuBLASMp batched GEMM
  all recreate descriptors and re-query workspaces per FFI call; a shared
  descriptor/workspace-size cache on `LorraxCusolverMpCtx` is the clearest
  first optimization after adding NVTX ranges.
- Logged sandbox bookkeeping issues in `KNOWN_SANDBOX_ERRORS.md` for the
  `sources/lorrax` vs `sources/lorrax_D` doc mismatch and missing D-variant
  manifests.

## 2026-05-12: accumulate_rchunk_to_gflat μ-axis chunking [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

`accumulate_rchunk_to_gflat` now chunks the **μ axis (axis 1)**
inside the FFT-batch scan, replacing the previous n_q-axis chunking.
This bounds the per-rank FFT-box transient at CrI3 J_3x3 scale
(n_q=9, n_rmu_padded=1504, FFT grid 45×45×120) without OOM, and
crucially handles the n_q=1 case (Γ-only debug runs) that n_q
chunking could not.

### Why this change

The n_q chunking required `n_batch_chunks | n_q`.  CrI3 J_3x3 has
n_q=9 in the IBZ which is only divisible by {1, 3, 9}; MoS2 has
n_q=9 with the same restriction.  More importantly, single-q
debugging runs (n_q=1) had no way to chunk at all.  The μ axis is
always large (n_rmu_padded = several hundred to a few thousand) and
is already μ-sharded across ranks, so chunking it aligns naturally
with `with_sharding_constraint` decomposing the per-chunk
intermediates across ranks.

### Code changes

- `common/wfn_transforms.py: accumulate_rchunk_to_gflat`:
  - Replaced `_q_chunk` with `_mu_chunk = n_rmu_padded /
    n_batch_chunks`.
  - Scan body now `dynamic_slice_in_dim(..., axis=1)` on rch, with
    `_shard3` / `_shard5` constraints on pad_buf / box / G_box so
    each per-rank chunk is `(n_q, _mu_chunk/p_prod, ...)`.
  - qvec_frac path simplified: qvec is per-q (n_q, 3), broadcasts
    the same way for every μ chunk — no per-chunk slicing.
  - Sphere gather uses the shared `_gather_sphere` helper for both
    one-shot and chunked paths (per-q sphere broadcasts across the
    μ-chunk; no per-chunk sphere slicing needed since axis 0 is
    intact).
  - Divisor check: `n_batch_chunks | n_rmu_padded`.
- `common/isdf_fitting.py`: updated comment + default chunk
  selection (largest divisor of `n_mu_local` ≤ `num_chunks`).
- `tests/test_rchunk_gflat_pair.py`: tightened test (n_rmu = 6,
  divisible by every parametrised chunk count {1, 2, 3}) and
  updated the indivisible-rejection test to match the
  n_rmu_padded check message.

### Numbers (vs the 2026-05-11 baseline, both on the same git tree)

MoS2 3×3 bispinor end-to-end (4 ranks A100, full COHSEX):

| | 2026-05-11 baseline | 2026-05-12 μ-chunked | Δ |
|-|--|--|--|
| Σ^B(μ_L=1,ν_L=1) | -0.242923 eV | -0.242923 eV | 0 |
| Σ^X band 1 k=Γ | -40.0326 eV | -40.0326 eV | 0 |
| eqp0 max abs Δ vs baseline | — | 0 eV | bit-exact |
| Total wall | 47.3 s | 47.95 s | +0.65 s (~1.5%) |
| Per-chunk FFT box | n/a (one-shot) | 9 q × 40 μ × 46080 = 0.27 GB | — |

CrI3 J_3x3 G-flat (8 ranks A100, x-only):

| | 2026-05-11 baseline | 2026-05-12 μ-chunked |
|-|--|--|
| Per-chunk FFT box | n/a (one-shot OOMed at 98 GB) | 9 q × 47 μ × 243000 = 1.64 GB |
| ζ-fit | OOM at 98 GB | succeeds |
| V_q[CC] trace at q=0 | n/a | 120459594.32 |

(Run fails downstream at `kin_ion.h5` — same pre-existing
followup as the 2026-05-11 attempt; not a regression in this
refactor.)

Run dirs:
- `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_shardmap_2026-05-12/`
- `runs/CrI3/D_gflat_cri3_3x3_muchunk_2026-05-12/`

## 2026-05-11: Bispinor V_q orchestrator on G-flat — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The 7-tile bispinor V_q^{μ_L, ν_L} hot loop is now end-to-end on
the G-flat ζ disk format.  All seven unique tiles (CC + 3 TT
diagonal + 3 TT off-diagonal) run through the new per-q +
G-chunked kernel.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_2026-05-11/`
Report:        `reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md`

### Code changes

- `gw/v_q_g_flat.py`: factored a private `_compute_V_q_g_flat_one_tile`
  helper (~250 LOC) that drives one tile end-to-end.  Charge wrapper
  `compute_all_V_q_g_flat` reduced to a ~30-line bare-Coulomb
  v_per_G builder + helper call.  Kernel parametrized over
  `(n_rmu_L, n_rmu_R)` with separate L/R buffers; the same_zeta
  path still aliases L=R inside the jit.
- `gw/v_q_bispinor.py`: added `compute_V_q_bispinor_g_flat_to_h5`
  (~120 LOC).  Loops over `UNIQUE_TILES`, builds per-tile
  `v(q+G)` via new `_make_per_q_v_builder_for_tile` (CC = bare
  Coulomb; TT = bare · `(δ_ij − K̂_i K̂_j)`), calls the shared
  helper, streams each tile to HDF5.  Reuses the existing
  `tile_dataset_name`, `UNIQUE_TILES`, `HERMITIAN_PAIRS`,
  `BispinorVqReader` (output format is unchanged).
- `gw/gw_init.py`: bispinor dispatch reads the charge ζ's
  `isdf_header.zeta_layout` and routes to the new orchestrator
  on G-flat (opens 4 `ZetaReader` handles).  Legacy r-space path
  preserved as fallback.  Also: copy `sys_dim` onto `meta_curr`
  (dataclasses.replace strips dynamic attrs — caught by the
  bispinor shakedown).

### Numbers (vs the legacy bispinor smoke A_bispinor_smoke_2026-05-08)

ζ disk-shrink (per file, all in `tmp/`):

| File              | Legacy r-space | G-flat new | Ratio |
|-------------------|----------------|------------|-------|
| `zeta_q.h5`       | 4.0 GB         | 177 MB     | 23×   |
| `zeta_q_mu1.h5`   | 2.6 GB         | 181 MB     | 14×   |
| `zeta_q_mu2.h5`   | 4.2 GB         | 181 MB     | 23×   |
| `zeta_q_mu3.h5`   | 4.2 GB         | 181 MB     | 23×   |
| **Total ζ**       | **15.0 GB**    | **720 MB** | **~21×** |
| `v_q_bispinor.h5` | 446 MB         | 424 MB     | 1.05× |

`v_q_bispinor.h5` size is unchanged by design — V_q has
(μ × μ) axes, no G-axis.

V_q wall: 4.2 s for all 7 tiles on 4× A100 (extrapolated ~6×
faster than the legacy μ × ν tile driver from the charge-only
shakedown; total bispinor pipeline 47.3 s).

### Numerics

Bare Σ_X print at k=Γ matches the legacy r-space baseline to
**0.01 eV** band-by-band (-40.0326 new vs -40.0277 legacy at
band 1; matching delta of ~5 meV across all sampled bands).  The
residual is per-q sphere ⊂ shared sphere drop-out of cutoff-edge
G's, as designed.

Bispinor unit tests in `tests/test_compute_V_q_bispinor_g_flat.py`
(committed in `ac735cc`):
* 7 tiles agree with a per-q einsum reference V^{μ_L, ν_L}[μ, ν]
  = Σ_G conj(ζ_L) · v_q^{μ_L, ν_L} · ζ_R to 1e-10;
* CC tile from the bispinor orchestrator is bit-identical to the
  charge-only orchestrator on the same ζ_C file (confirms genuine
  code path sharing, not just structural duplication).

### Diagnostic (3-way Bare Σ_X check)

Ran a third comparison to disentangle the V_q rewrite from
code-drift between May 8 and today: legacy r-space writer +
legacy μ × ν V_q driver, on the SAME git rev as the G-flat run.

```
                                            Bare Σ_X k=0, band 1
G-flat new (today)                            -40.0326
Diagnostic — legacy path, same git rev        -40.0325     ← <100 μeV
May 8 baseline (A_bispinor_smoke)             -40.0572     ← 25 meV drift
```

So the V_q rewrite is **bit-equivalent** to the current legacy
code (1e-5 relative on every sampled band).  The 25 meV vs May 8
is intervening fixes to the legacy bispinor path (`_make_K_cart`
qvec_frac convention, IBZ unfold, Bloch-phase unification), not
my rewrite.  The new G-flat code never had those bugs because it
builds K_cart from per-q components with already-divided
fractional q.

xx/yy symmetry sanity (`||V_TT_11 − V_TT_22|| / ||sum||`):
May 8 baseline 0.97 (broken), today's legacy 0.50 (fixed), today's
G-flat 0.50 (matches).

### Notes

- Each new kernel compile emits ~8 `Involuntary full
  rematerialization` SPMD warnings (disk read at `P(None, ('x','y'),
  None)` reshards to `P(('x','y'), None)` inside the jit; XLA does
  full rematerialization instead of an all-to-all).  Non-fatal;
  ~20 MB per q per rank lost to the copy, dwarfed by the kernel
  time.  Followup: have the loader expose a directly-sharded read.

- `eqp0_bisp.dat` (full bispinor Σ^B reading the TT tiles) was
  not emitted by either the baseline or this run for this config —
  appears to need an output flag we haven't enabled.  Separate
  followup.

## 2026-05-11: G-flat ζ + new V_q orchestrator — end-to-end MoS2 3×3 [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

End-to-end shakedown on Perlmutter (1 node × 4× A100, 4-rank).
Writer ran in G-flat mode (`LORRAX_WRITE_G_FLAT_ZETA=1`), V_q via
the new `gw.v_q_g_flat.compute_all_V_q_g_flat` orchestrator, Σ
through the existing path.  No code changes to the kernel since
yesterday's swap commit — only orchestration patches caught in
shakedown.

Run directory: `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_xonly_2026-05-11/`
Report:        `reports/gflat_e2e_mos2_3x3_2026-05-11/report.md`

### Numbers (vs r-space baseline 02_lorrax_xonly)

| Quantity                 | r-space   | G-flat     | Ratio |
|--------------------------|-----------|------------|-------|
| `zeta_q.h5` size         | 2.3 GB    | **101 MB** | **23×** |
| Total wallclock          | 17.2 s    | **11.4 s** | 1.5× |
| `zeta_fit.close_io`      | 3.8 s     | 0.1 s      | **~38×** |
| `V_q_compute`            | 4.4 s     | **0.7 s**  | **6.3×** |
| Σ stage                  | 3.1 s     | 2.9 s      | 1.07× |

sigma_diag agreement: **5 decimals vs r-space baseline** at every
k, band sampled (per-q sphere is a strict subset of the legacy
shared sphere; the few cutoff-edge G's drop out by design since
`v(q+G) → 0` past `zeta_cutoff_ry`).

### Shakedown fixes (commit 6ebfc3e)

- `compute_all_V_q` dispatcher: async prefetch default OFF (env
  `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=0`).  The worker-thread G-flat
  read deadlocks against the PHDF5 FFI collective in production
  (NCCL kernel collectives interleave with the MPI read collective
  via the GIL in ways that hang).  Sync loop is already 6.3× faster
  than the legacy driver — async is a future opt-in.
- `v_q_g_flat.compute_all_V_q_g_flat`: caller in `gw_init.py`
  passes `ZetaReader`, not `ZetaLoader` (the unit tests use the
  loader).  Orchestrator now detects which API is on hand and
  dispatches accordingly.
- Per-q progress print on the sync path (`read=…s, kernel=…s`):
  one line per q so a stuck JIT compile or NCCL hang is visible
  in `tail -F run.log`.
- `gw_init.py` ζ-peek diagnostic: reads `'zeta_q_G'` on G-flat
  files (was hard-coded to `'zeta_q'`).
- `compare_bgw_gwjax.py` (sandbox top-level): replaced the stale
  `common.wfnreader.WFNReader` import with raw h5py k-list read.

### Followups (unchanged)

- BGW agreement: 0.5 eV gap at band 19 for MoS2 3×3 x-only is
  **pre-existing in the r-space baseline** — not introduced by
  this rewrite.  Worth a separate dig.
- Async prefetch re-enable (NCCL ↔ MPI interleave).
- Bispinor 7-tile orchestrator
  (`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`).

## 2026-05-11: V_q driver swap — new G-flat orchestrator [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

The G-flat V_q hot loop is now end-to-end: ζ̃ read off disk in WFN.h5
per-q sphere layout → v(q+G) built at the per-q Miller components →
G-chunked contract → dynamic_update_slice into (V_acc, g0_acc) →
IBZ-to-full unfold.

- `gw/v_q_g_flat.py` (NEW, ~280 LOC) — `compute_all_V_q_g_flat`.
  Replaces the legacy ``compute_V_q_tile`` / ``_choose_v_q_chunks``
  pipeline for the G-flat-on-disk case.  μ × ν tiling, the chooser,
  the in-V_q FFT, and the shared-sphere conversion all collapse:
  per q, one read + one ``compute_v_q_per_q_g_chunked`` call.  Async
  prefetch (single-step) overlaps the next q's read with the current
  q's contract (borrowed from `v_q_tile`).
- `compute_vcoul.compute_all_V_q` now dispatches on
  ``zeta_io.zeta_layout``: G-flat → new orchestrator; r-space →
  legacy `v_q_tile` path (kept as fallback).
- Tests in ``tests/test_compute_all_V_q_g_flat.py``: synthesised
  G-flat ζ file → orchestrator output bit-matches a one-shot
  einsum reference; async vs sync identical; r-space loader is
  rejected with a clear error.

### Followups

- Larger profile (Si 4×4×4 / MoS2 3×3×1) with the new path enabled
  to validate the disk-shrinkage + I/O-overlap wins claimed for
  the writer.
- Bispinor V^{μ_L, ν_L} 7-tile driver
  (`gw/v_q_bispinor.compute_V_q_bispinor_to_h5`) still uses the
  legacy r-space path; swap follows the same pattern (1 q at a time,
  G-chunked, per-tile signed v).

## 2026-05-11: per-q G-chunked V_q kernel + ζ-cutoff separate from V_q cutoff [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

### Two cutoffs, separate plumbing

`cfg.head.zeta_cutoff` is now an independent knob from
`cfg.head.bare_coulomb_cutoff`.  Both default to `ecutwfc` and cap at
`ecutrho`; `bare_coulomb > zeta` is a hard error (V_q would need ζ̃
values the writer never stored).  The on-disk per-q sphere is built
at `zeta_cutoff`; V_q's `sqrt_v(q+G)` mask uses
`bare_coulomb_cutoff`.

- `gw_config.HeadConfig.zeta_cutoff` (new field).
- `gw_init.fit_zeta`: shared `_resolve_cutoff` helper validates ≤
  ecutrho, raises on `bare > zeta`.
- `isdf_fitting.fit_zeta_to_h5(zeta_cutoff_ry=)` builds the per-q
  sphere at that cutoff and writes it to
  `isdf_header/zeta_cutoff_ry` (renamed from
  `bare_coulomb_cutoff_ry`).
- `ZetaReader` / `ZetaLoader.zeta_cutoff_ry` surfaces it.

### Per-q, G-chunked V_q kernel

New `compute_vcoul.compute_v_q_per_q_g_chunked(zeta_q_L, zeta_q_R,
v_q, g_chunk=...)` evaluates

    V_q[μ,ν] = Σ_G  conj(ζ̃_μ(G)) · v(q+G) · ζ̃_ν(G)

at a single q with the G-axis chunked into `g_chunk` slices.  Each
chunk is a GEMM-shape einsum `'mG,nG->mn'` on
`(n_rmu, g_chunk)` blocks — contiguous G access, no FFT, no
shared-sphere conversion.  Accumulator is donated so repeated calls
(e.g. one per q) sum in place under jit.

The companion `compute_v_q_per_G(q_irr_frac, gvec_components, ...)`
builds `v(q+G)` at the writer's per-q Miller list (matches the
legacy kernel's full-FFT-grid `get_sqrt_v_and_phase` output at the
sphere positions for both 2D slab and 3D bulk — tested).

This is the kernel the rewritten V_q driver will call once swapped
over; the existing `compute_V_q_tile` driver in `v_q_tile.py`
remains in place for the current production hot path.

### Tests

- `tests/test_v_q_per_q_g_chunked.py` (NEW, 9 tests):
  - kernel matches one-shot einsum (3 g_chunk sizes);
  - bispinor off-diagonal (L ≠ R, signed/complex v);
  - pad-slot invariance (ζ̃ = 0 at j ≥ ngk[q] ⇒ zero contribution
    regardless of v(G) there);
  - accumulator donation across multiple kernel calls;
  - alignment-error path (ngkmax not divisible by g_chunk);
  - `compute_v_q_per_G` ≡ legacy `get_sqrt_v_and_phase` at the
    sphere positions, for `sys_dim ∈ {2, 3}`.

### Followups

- Swap `compute_V_q_tile` / `_choose_v_q_chunks` over to the new
  per-q kernel.  The chooser shrinks to G-chunk + memory model
  (q-batching gone in this scope; comment marks the seam for a
  future opt-in).
- Sigma readers that consume V_q[μν] are unchanged — V_q's μ × ν
  output shape is identical to the legacy kernel's.

## 2026-05-11: G-flat ζ on-disk with WFN.h5-style per-q sphere padding [agent]

Branch `agent/zeta-ibz-header` on `lorrax_D`.

Writer now produces ``zeta_q_G(n_q, n_rmu, ngkmax)`` instead of
``zeta_q(n_q, n_rtot, n_rmu)`` when ``LORRAX_WRITE_G_FLAT_ZETA=1`` is
set, with per-q WFN.h5-style sphere components stored alongside.

- **`sources/lorrax_D/src/common/coulomb_sphere.py` (NEW)**
  - `compute_bare_coulomb_sphere_idx(...)` — shared single sphere
    used by V_q kernel.  Extracted from inline code in
    `compute_vcoul.py:246-263` so the writer and consumer share one
    source of truth.
  - `compute_per_q_bare_coulomb_components(...)` — per-q sphere
    `{G : |q+G|² ≤ cutoff}` for every IBZ q, padded uniformly to
    `ngkmax = max_q ngk[q]` with sentinel Miller index
    `(-nx/2, -ny/2, -nz/2)`.  Returns `sphere_idx_padded`,
    `gvec_components_padded`, `ngk_per_q`, `ngkmax`.
  - Fixed `_q_max_cart` bug: enumerates the actual BGW-wrapped
    q-list instead of using the `±0.5/kgrid` half-BZ corners
    (under-bound for the real q-list — even kgrid leaves q=K/2 at
    `q_frac = 1/2` outside the Wigner-Seitz cell).
- **`sources/lorrax_D/src/gw/compute_vcoul.py`**
  - Inline sphere construction at lines 246-263 replaced by a call
    to `compute_bare_coulomb_sphere_idx`.
- **`sources/lorrax_D/src/common/wfn_transforms.py`**
  - `accumulate_rchunk_to_gflat` accepts a 2-D per-q
    `sphere_idx (n_q, ngkmax)` in addition to the legacy 1-D shared
    sphere.  Uses `jnp.take_along_axis(mode='promise_in_bounds')`
    to dodge the XLA x64+shard_map verifier bug.
- **`sources/lorrax_D/src/file_io/isdf_header.py`**
  - New fields: `gvec_components (n_q, 3, ngkmax)`, `ngk_per_q (n_q,)`,
    `bare_coulomb_cutoff_ry`.  Required by `IsdfHeader.build` when
    `zeta_layout == 'G_flat'`; legacy r-space files read with these
    fields set to `None`.
- **`sources/lorrax_D/src/common/isdf_fitting.py`**
  - `fit_zeta_to_h5(..., vcoul_cutoff_ry=...)` accepts the bare
    cutoff; builds the per-q sphere, allocates
    `gflat_acc(n_q, n_rmu_padded, ngkmax)`, gathers per-q after each
    chunk's FFT, masks pad slots to zero post-loop, and persists
    components + ngk + cutoff in the isdf_header.
- **`sources/lorrax_D/src/gw/gw_init.py`**
  - Plumbs `vcoul_cutoff_ry` into both `fit_zeta_to_h5` call sites
    (scalar charge + bispinor transverse μ_L=1,2,3).
- **`sources/lorrax_D/src/file_io/zeta_loader.py` / `zeta_reader.py`**
  - Expose `gvec_components`, `ngk_per_q`, `bare_coulomb_cutoff_ry`,
    `ngkmax_zeta`.
  - Loader: G-flat-on-disk reads `zeta_q_G` directly via the new
    `_read_g_flat_disk` helper.  `layout='r_space'` raises
    `NotImplementedError` against a G-flat file (would need IFFT).
  - Reader: G-flat path raises `NotImplementedError` for the
    "narrow to shared sphere" sub-case (per-q → shared scatter not
    yet wired into the V_q hot loop); raw slab returns work.
- **Disk-size win** (`n_G_sph / n_rtot`, smaller is better):
  - MoS2 3×3×1, cutoff=30 Ry: **11.5%** of r-space (~8.7× shrinkage).
  - Si 4×4×4, cutoff=30 Ry: **16.9%** (~5.9× shrinkage).
  - Si 4×4×4, cutoff=120 Ry (=ecutrho): 94.4% (near full FFT box at
    the rho cutoff — expected).
- **Tests**
  - `tests/test_per_q_sphere.py` (NEW, 6 tests): helper correctness
    vs direct `(q+G)` enumeration, shared-sphere ⊇ per-q-sphere
    invariant, per-q accumulate matches reference FFT+gather,
    header round-trip + validation errors.
  - `tests/test_zeta_loader.py`: bumped 1 test's `IsdfHeader.build`
    call to supply the new required G-flat fields.
- **Validation**
  - 33/33 new + existing G-flat tests pass.
  - Full non-GPU pytest sweep: 181 passed, 20 skipped.  Pre-existing
    `test_kmeans_sharded` failures unchanged (independent of this
    branch).  GPU regression needs a CUDA job allocation (login-node
    cuSolver init fails — same as before).
- **Followups**
  - Wire the per-q → shared-sphere scatter into the V_q wrapper so
    the kernel can consume the new G-flat on-disk format.  Until
    then, the kernel keeps using r-space ζ files.

## 2026-05-11: chunk-capable local FFT helpers + slab-only phase helpers [agent]

Branch `agent/fft-batch-chunks` on `lorrax_A`, rebased onto `origin/main`
`92cbd83`.

- **`sources/lorrax_A/src/common/fft_helpers.py`**
  - added `apply_local_fft(...)`, a reusable device-local FFT helper with
    optional `fft_batch_chunks=` batching over all non-transform axes
  - threaded `fft_batch_chunks=` through
    `make_sharded_{f,if}ftn_3d`, `make_flat_k_fft`, and
    `query_fft_peak_bytes`
  - default remains `fft_batch_chunks=1`, so current production callers keep
    today’s one-shot FFT behavior unless a future refactor opts in
- **`sources/lorrax_A/src/common/wfn_transforms.py`**
  - added generic flat-r helpers:
    `extract_flat_rchunk`, `embed_flat_rchunk`,
    `apply_bloch_phase_flat_points`, `apply_bloch_phase_flat_rchunk`
  - `to_rmu(..., kvecs_frac=...)` now phases only the gathered centroid
    points instead of the whole FFT box
  - `to_rchunk(..., kvecs_frac=...)` now slices the flat-r slab first and
    applies the Bloch phase only on that retained slab
  - `to_rbox` / `to_rmu` / `to_rchunk` now also accept `fft_batch_chunks=`
    for future opt-in use
- **`sources/lorrax_A/src/file_io/zeta_reader.py`** and
  **`sources/lorrax_A/src/file_io/zeta_loader.py`**
  - threaded `fft_batch_chunks=` into the `G_flat` zeta read path so the
    upcoming `rchunk <-> G_flat` zeta/V refactor can reuse the same helper
    without reopening reader internals
- **Tests**
  - `tests/test_fft_helpers.py`: new chunked-helper coverage and
    chunk-aware `query_fft_peak_bytes` coverage
  - `tests/test_wfn_transforms.py`: new phased `to_rmu`,
    phased/chunked `to_rchunk`, and flat-r helper coverage
- **Validation**
  - `uv run python -m pytest -q tests/test_fft_helpers.py` → `5 passed`
  - `uv run python -m pytest -q tests/test_wfn_transforms.py` → `16 passed`
  - `uv run python -m pytest -q` → `182 passed, 20 skipped, 4 failed`
  - remaining failures are unchanged from `main`:
    - `tests/test_gw_jax_regression.py::test_gw_jax_matches_reference`
      (`write_qp_wfn_h5` shape mismatch)
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[fcc-avec1]`
    - `tests/test_kmeans_sharded.py::test_refactored_matches_naive[skew-avec2]`
    - `tests/test_kmeans_sharded.py::test_pbc_distance_scan_matches_naive_fcc`
- **Report**
  - `reports/fft_helper_unification_2026-05-11/report.md`

## 2026-04-21: analytic chunk chooser + γ-calibrated AOT memory model [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Closes Phase 6 of the
AOT memory-model initiative — the chooser now predicts runtime peak
bytes to ≤1% error after γ calibration.

- **`src/gw/aot_memory_model/chooser.py`**: new
  `choose_chunks_analytic(sys, mesh, budget)` — regroups the 7 memory
  primitives into 4 scaling classes via a `PRIMITIVE_CLASSES` dict on
  the kernel (`const`, `cr`, `bc`, `crbc`).  The feasibility bound
  ``peak ≤ M`` is linear in chunk_r at fixed bc, so
  ``chunk_r_max(bc) = (M − α₀ − α_bc·bc) / (α_cr + α_crbc·bc)`` is a
  closed-form inversion.  Chooser 1-D searches over bc candidates, no
  2-D grid.  Adds an optional `fft_launch_overhead_flops` knob for
  calibrating the "small-bc performance hit" post-hoc.
- **`src/common/isdf_fitting.py`**: replaced the
  `jax.devices()[0].memory_stats()` peak tracker (returns `None` on
  this CUDA PJRT) with a single `nvidia-smi` sample at the end of the
  r-chunk loop.  Per-chunk sampling inside the Shifter container was
  observed to hang on some Perlmutter node types.
- **`src/gw/gw_init.py`**: prints `γ = runtime_peak / aot_predicted`
  at the end of `fit_zeta` whenever both numbers are available.  Also
  corrected the `aot_sys.n_b` passed to the predictors: use the union
  range (`nb_full`) not `nb_L + nb_R`; the cost primitive's factor of
  2 handles the L+R sum.
- **`fit_one_rchunk__current__fit.json`**: records **γ=0.510**
  calibrated at MoS2 3×3 nosym (runtime nvidia-smi = 3.06 GB vs AOT
  worst-case = 6.00 GB).  `Fit.gamma` is applied by both
  `predict_peak` and the analytic chooser's `_group_alpha`, so
  chooser-predicted peaks now match runtime to within measurement
  noise.
- **Validation** — with `memory_per_device_gb=4` and
  `use_aot_chunk_chooser=true` at MoS2 3×3:

  ```
  AOT chooser: chunk_r=46080 band_chunk=80 (1×1 jits,
      peak=3.06 GB / 3.88 GB = 79%, total=7.3 GF)
  ```

  Chooser-predicted 3.06 GB matches the earlier runtime nvidia-smi
  measurement (3.06 GB).  Budget is genuinely hit.  Bit-identical
  eqp0 vs baseline `c8fc139fb22d2653d585874fe19c72a7`.
- **Follow-ups**: (1) widen the bc DoE axis — current 11-sample fit
  collapses bc-sensitivity into zero β.  (2) γ calibration at Si 4×4×4
  60Ry — may need mesh-scaling γ rather than a global scalar.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — Phase 6 section
  with α decomposition, γ measurement, budget-hit validation table.

## 2026-04-21: phdf5 on-demand G-space during ISDF fit [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.  Follow-on to same-day
"jit the r-chunk body" work.

- **`src/common/isdf_fitting.py`**: new `use_phdf5_gspace: bool`
  parameter to `fit_zeta_chunked_to_h5`.  When True, the driver skips
  the device-resident G-space cache (`load_gspace_for_bands`) and
  instead calls `PhdfWfnReader.coeffs_gspace(band_range)` fresh per
  r-chunk per band-chunk.  The tuple is `del`'d right after the
  `fit_one_rchunk` jit returns — nothing persists between r-chunks.
- **`src/gw/gw_config.py` + `gw_init.py`**: `use_phdf5_gspace` surfaces
  as a `cohsex.in` flag and threads into `fit_zeta`.
- **Duck-type**: `PhdfWfnReader.coeffs_gspace` already returns
  `(n_k, nb_pad, n_s, nx, ny, nz)` with
  `P(None, ('x','y'), None, None, None, None)`, matching the cached
  path's shape/sharding contract exactly.  No FFI-reader signature
  changes were needed; the driver-side factory is four lines.
- **Validation** (MoS2 3×3, `use_phdf5_gspace=true`):
  - single r-chunk + `use_ffi_io=true`:
    md5 `c8fc139fb22d2653d585874fe19c72a7` ✓
  - multi-chunk (5×10000 + remainder 6080) + `use_ffi_io=false`:
    same md5 ✓
  - Multi-chunk + `use_ffi_io=true` fails with concurrent HDF5 MPI-IO
    errors in the async zeta_q writer.  Pre-existing interaction —
    PhdfWfnReader + SlabIO-FFI race on MPI-IO state on the same ranks.
    When both flags are needed, use `use_ffi_io=false`.
- **Memory win**: zero persistent GPU footprint for the per-band-chunk
  G-space cache between r-chunks.  ~265 MB per rank saved at MoS2 3×3
  (small); multi-GB at Si 10×10×10 1000+ bands, where it pushes the
  pre-rchunk CCT/cholesky stages back under budget.
- **Timing**: +0.2 s total at MoS2 3×3 multi-chunk (4.3 s vs 4.1 s).
  Negligible under phdf5; would be slow under legacy h5py (keep flag
  opt-in).
- **AOT model**: per-r-chunk peak unchanged — `psi_bc_G_tuple` is
  still a jit input, so `argument_size_in_bytes` is identical.  Phase
  1b benefits show up in the *between-rchunk* GPU residency, which the
  AOT kernel doesn't currently measure.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new Phase 1b
  section with validation matrix and the FFI-writer conflict note.

## 2026-04-21: jit the r-chunk body + AOT-model fit_one_rchunk [agent C]

Branch `agent-C/aot-memory-model` on `lorrax_C`.

- **`src/common/isdf_fitting.py`**: new
  `_make_fit_one_rchunk_kernel` factory + `fit_one_rchunk` entry point.
  The full per-r-chunk body (FFT+reshard per band-chunk, streamed
  spin-traced pair-density accumulate, ZCT, Z→col reshard, Cholesky
  solve) is now one jitted kernel.  `fit_zeta_chunked_to_h5` calls it
  once per r-chunk.  Two compile variants per run (full + remainder).
- **`src/common/load_wfns.py`**: `get_sharded_wfns_rchunk_slice`
  signature refactored `(r_start, r_end)` → `(r_start, r_chunk_size)`
  so `r_start` can be a tracer inside an outer jit.  Callers in
  `iter_psi_rchunk_bandwise` updated.
- **`src/gw/aot_memory_model/kernels/fit_one_rchunk.py`**: new
  composite AOT kernel mirroring the production factory.  Captures the
  driver-level memory peak including coexisting buffers that per-stage
  kernels can't see.  Primitives: `Pacc`, `PrBc`, `psiBc`, `psiBcY`,
  `psi_cent`, `L_q`, `psiG_total`.
- **`src/gw/aot_memory_model/core.py`**: `SysDims` gains an optional
  `fft_grid` field + `fft_shape` property for kernels that need both
  k-grid and real-space FFT box.
- **`src/gw/aot_memory_model/presets.py`**: `points_fit_one_rchunk`
  for `mos2_3x3` and `si444_60Ry`.
- **`src/gw/gw_init.py`**: logs the AOT-predicted driver peak
  alongside the existing per-stage heuristic — sanity-check-only, does
  not override `chunk_r` yet.
- **Validation**: MoS2 3×3 COHSEX single-chunk (46080 pts) and
  multi-chunk (r_chunk_size=10000, 5 chunks + remainder 6080) both
  produce `md5sum eqp0.dat == c8fc139fb22d2653d585874fe19c72a7` matching
  the reshard-fix baseline.
- **NNLS fit** (11 DoE points, residual RMS 0.23 GB on ~3 GB peaks):
  β[PrBc]=1.03, β[L_q]=5.02, β[psiG_total]=1.65.  Saved at
  `src/gw/aot_memory_model/artifacts/fit_one_rchunk__current__{fit,samples}.json`.
- **Report**: [reports/aot_memory_model_poc_2026-04-20/report.md](
  reports/aot_memory_model_poc_2026-04-20/report.md) — new "Update
  2026-04-21" section with primitives table, fit coefficients, next
  steps.
- **Next**: (1) host-resident `cached_gspace` via phdf5-like duck type
  — attacks the `psiG_total=1.65` primitive directly; (2) switch
  `compute_optimal_chunks` to use AOT prediction for the
  pair+zct+reshard+solve sub-loop; (3) γ-calibrate at Si 4×4×4 60 Ry.

## 2026-04-20: phdf5 FFI — independent writes by default, Cray MPICH now viable [agent A]

Branch `agent-A/independent-writes-default` on `lorrax_A`.

- **`src/ffi/phdf5/cpp/ctx.h`**: split `use_collective` into
  `use_collective_read` (default `true`) and `use_collective_write`
  (default `false`).  `coll_metadata` now defaults to `false`.
- **`src/ffi/phdf5/cpp/context.cc`**: new env-var surface.
  `LORRAX_PHDF5_INDEPENDENT=1` still forces reads independent too (power
  user override).  New `LORRAX_PHDF5_COLLECTIVE_WRITES=1` to opt writes
  back into collective (do NOT set on Cray).  New `LORRAX_PHDF5_COLL_META=1`
  to re-enable collective metadata.
- **`src/ffi/phdf5/cpp/write_ffi.cc` + `read_ffi.cc`**: dxpl selection
  now uses the per-direction flag.
- **Why**: the Cray MPICH collective write driver
  (`ad_cray_write_coll.c:669`) OOMs at ≥ 1 GB/rank regardless of
  `cb_*`, `stripe_*`, `alloc_time`, or `cray_cb_write_lock_mode` knobs.
  The fix that prior investigation missed was the combination of
  `H5FD_MPIO_INDEPENDENT` writes AND non-collective metadata ops —
  both are needed to fully bypass the buggy driver.  Independent
  writes are neutral on OpenMPI at our measured sizes; collective
  reads are preserved (ROMIO two-phase is optimal on both stacks).
- **Regression data** (1 node / 4 GPUs, post-fix defaults):

  | workload | OpenMPI | Cray MPICH |
  |---|---|---|
  | MoS2 3×3 `phdf5_multi_offset_test` | PASS | PASS |
  | MoS2 3×3 `phdf5_profile` (45 MB WFN) | 18.1 ms (was 18.3) | **17.9 ms** |
  | MoS2 3×3 `phdf5_profile --centroids` (gw_jax load) | 26.2 ms | 26.4 ms (parity) |
  | n=16384 C128 `phdf5_read_bench` (4.29 GB) | 3.04 GB/s | **3.79 GB/s** (was CRASH) |

  Cray now works at all scales and beats OpenMPI at large scale; at
  MoS2 3×3 scale the two stacks are within noise.  Unification around
  Cray for cross-cluster portability is viable.
- **Docs**: `src/ffi/phdf5/ARCHITECTURE.md` env-var table and
  `src/ffi/PORTING.md` Option B write-up refreshed.

## 2026-04-20: flat-k FFT helper — one wrapper for kx/ky/kz across the GW pipeline [agent C]

Branch merged to `main` as commit `c9bd801`.

- **`src/common/fft_helpers.py`** — new `make_flat_k_fft` /
  `make_flat_k_ifftn` / `make_flat_k_fftn`.  Callers hand it flat-k
  `(nk, *trail)` arrays, the helper does
  `reshape → with_sharding_constraint → custom-partitioned 3-D FFT →
  reshape back`.  `kgrid` and the 3-D PartitionSpec are closure state;
  the 3-D form never appears in caller code.
- **Call sites wired through**:
  - `gw/w_isdf.py` chi0 minimax — three FFT closures collapsed to
    helper calls (`Gv_ifftn`, `Gc_fftn`, `chi_fftn_local`).
  - `gw/ppm_sigma.py` `_sigma_kij_kernel` — `_fft_flat_G` /
    `_fft_flat_V` closures replaced.
  - `gw/gw_jax.py` — `_make_fft_pair` factory removed in favor of
    direct helper calls for G and V.
  - `common/isdf_fitting.py` — `CCT_LR`, `CCT_LR_spin_matrix`,
    `ZCT_LR`, `ZCT_LR_spin_matrix` all refactored to take flat-k
    input end-to-end.  The pre-ZCT `reshape → with_sharding_constraint`
    in the r-chunk loop was deleted (ZCT now takes flat-k directly).
    `donate_argnums` re-enabled on CCT_LR (0, 1), ZCT_LR `_left_ifft_conj`
    (0) and `_right_ifft_mul_fft` (0, 1), and on the spin-matrix
    variants — a handful of one-shot trace-time XLA aliasing warnings
    remain (rank-3 → rank-5 intermediate), but there are no per-call
    donation failures.
- **Validation** (`runs/Si/C_flatk_si10/`, Si 10×10×10 mem12, 4 GPUs):
  - `eqp0.dat` **byte-identical** to `C_stream_si10_transposed`
    baseline (0-byte diff).
  - Total runtime 304.7 s → **275.9 s** (9.5 % faster).  Savings
    concentrated in `zeta_fit.chunk_loop` (30 s → 25.5 s) from
    pair-density donation; `close_io` unchanged (65.9 s → 63.2 s);
    Σ computation unchanged.
- **Why it matters**: single point where the 3-D FFT happens, so the
  NUFFT substitution the user has in mind is a one-file change with
  no call-site churn.

## 2026-04-18 (midday): scissor-shift for out-of-grid bands + Si 4×4×4 pseudobands end-to-end [agent A]

Branch `agent-A/scissor-shift-sc-gw` on `lorrax_A`
(commits `dfc880c`, `9b0e666`).  Full write-up in
`reports/scissor_shift_2026-04-18/report.md`.

- **`src/gw/scissor.py`** — `ScissorFit` dataclass, `fit_scissor` (numpy
  OLS, separate valence / conduction lines), `extrapolate_delta_e`, and
  `add_diag_to_H_kmn` (shard_map-based diagonal add onto a
  `P(None,'x','y')`-sharded Hamiltonian, ready for the future SC loop).
  Smoke-tested 4-GPU: fit recovers synthetic slopes to <6e-6, sharded
  diagonal add bit-identical to numpy (maxabs 0.0), P(None,'x','y')
  output sharding preserved, divisibility check raises cleanly.
- **`src/gw/gw_jax.py`** — G0W0 PPM post-processing now honors the
  `sigma_at_dft_extrapolate` config knob: out-of-grid bands get the
  fitted affine QP correction instead of the static-COHSEX fallback.
  Fixed two adjacent bugs while wiring:
  - `E_qp_ev * ryd2ev` unit double-count in the original `in_grid`
    mask — every state looked out-of-grid, so the diagonal Sigma
    fixed-point was silently discarded for all bands.
  - in-grid test must use `E_DFT`, not `eigvalsh(H_qp)`, because
    pseudobands' non-unit norms scale `<n|H|n>` by the pseudoband
    weight and produce garbage eigenvalues for compressed states.
- **First end-to-end test** — `runs/Si/A_06_si_4x4x4_scissor/`.
  Si 4×4×4 nosym, BGW-convention pseudobands via
  `psp.run_nscf --pseudobands` (50 windows × 2 pseudobands = 8 prot +
  98 pseudo = 106 bands).  Σ(ω) grid ±5 eV so all pseudobands
  (onset ~+10 eV) are out-of-grid.  GN-PPM G0W0 + scissor on 4×A100:
  run wall 30 s, 668/6784 in-grid, valence fit
  α=-0.44, β=-6.24 eV (RMSE 1.07 eV), conduction fit
  α=-0.64, β=-0.61 eV (RMSE 2.37 eV).  E_QP vs E_DFT is a single
  smooth line across the full 0→330 eV bandrange with no jump at the
  in-grid / out-of-grid boundary — continuity goal met.  Magnitudes
  over-correct at the high-E tail (E_QP ≈ 0.36·E_DFT → highest
  pseudoband lands at ~120 eV vs DFT 330 eV); expected for a line fit
  over 10 eV extrapolated to 300 eV, worth revisiting with a softer
  A + B/E tail or a damped law later.
- **Known issue documented**: `psp.get_dipole_mtxels` crashes on a
  pseudobands WFN (`vnl_velocity_matrix` hits a `None` `dZ`).  Worked
  around in this run by routing the q→0 head through the BGW
  `eps0mat.h5` from `runs/Si/02_si_4x4x4_nosym/01_bgw_gnppm/` with
  `wcoul0_source = epshead` in `cohsex.in`.

Validation: `uv run python -m pytest -q` on `lorrax_A` → 13 passed,
1 pre-existing reshard failure (unchanged from prior state).

## 2026-04-18 (overnight): sigma_ppm cleanup + compile-cache trims + zeta_fit probe [agent C]

Branch `agent/C-sigma-ppm-cleanup` on `lorrax_C`. Full write-up in
`reports/session_2026-04-18_async_probe/report.md`.

MoS2 3×3 / 4-GPU run_module wall: **47.3 s → 34.7 s (−27 %)**, eqp0.dat
bit-identical at every commit (16 substantive + 1 TEMP profiling).

- **Reduce-scatter in `_sigma_kij_kernel`**: `projection_kernel.project_ri`
  tail replaced by a shard_map'd local einsum + `psum_scatter × 2` (m on x,
  n on y).  σ^τ now emerges `(m_X, n_Y)`-sharded; every downstream ω-kernel
  multiply + accumulate is rank-local.  HLO diff shows 4× `all-reduce
  c128[9,2,2,320,80]` flipping to `reduce-scatter c128[2,9,40,2,320]` per
  τ step, same byte volume but output is now sharded.
- **σ^τ as a (re, im) tuple** from the shard_map — removes the
  `sigma_ri[0]/[1]` indexing pjits and the `is_fully_addressable` assert
  that a multi-process tuple-unpack of a sharded (2, …) stacked array
  would trigger.
- **New `_ReduceScatterGpuAccumulator`** is the default buffered path;
  Σ_c(ω, k, m, n) is held sharded on GPU so it's n_b²/p² per rank instead
  of replicated.  `_BufferedGpuAccumulator` deleted (was redundant).
- **lax.scan τ-loop infrastructure** landed as `_get_sigma_tau_scan_kernel`
  + `_ReduceScatterGpuAccumulator.run_window_scan`, **off by default** —
  regresses at MoS2 3×3 scale (fewer overlap opps with big fused module +
  per-window compile).  Reconsider at padded-τ or larger mesh.
- **Physics visibility pass**: module docstring states the quadrature
  formula directly; `_iter_branches` NamedTuple with comments deriving
  kernel_sign / scale flips; `_run_sigma_branch` reads like a physics
  outline; `_combine_coeff_with_sigma_tau` documents the re/im split and
  drops the dead "real" branch; `_convolve_sigma_branch_kij` →
  `_run_sigma_branch`; 'channel' scrubbed from factory names.
- **Dead-param / dead-class purge**: `omega_sign_flip` (always +1), the
  unused `_BufferedGpuAccumulator`, the one-line wrapper
  `_accumulate_tau_into_window`.  −203 / +102 lines in one commit.
- **Compile-cache trims — numpy for tiny host-side helpers**:
  `get_enk_bandrange`, `fft_integer_axes`, `exp_ikr_fftbox`, `_build_Gij`,
  `_build_occ`.  Each had emitted ~8–16 standalone pjits at trace time
  for pure host bookkeeping.  TRACING CACHE MISS 313 → 269 (−44).
  `wavefunction_setup` section **1.79 s → 0.18 s** (the old `jnp.zeros_like`
  + `.at[].set` on sharded input had a non-trivial runtime tied to
  cross-device scatter).
- **zeta_fit chunk loop**: dropped the per-chunk `sync_global_devices`
  (the allgather is itself a collective; one rendezvous at the end is
  enough).  Investigated async-allgather paths and confirmed JAX has no
  async `process_allgather`-to-host API; the 1.95 s first-collective
  NCCL setup is the floor without pre-warming or the phdf5 FFI path.

Future work documented in-tree (heavy comments at each extension point):
  τ batching, m-chunking, `_CollectiveFlushSlabIoAccumulator` (FFI SlabIO
  collective-write variant for multi-process streamed output).  `zeta_fit`
  remains the dominant cost bucket (47.6 % of total) and is the natural
  next target.

## 2026-04-17 (pm): k-means ISDF — parallelism refactor + 4-GPU sharding prototype [agent B]

Branch `agent/kmeans-sharded` on `lorrax_B`. Full write-up in
`reports/kmeans_sharded_2026-04-17/report.md`.

- **Refactored `centroid/kmeans_isdf.kmeans_update_step`** to eliminate the
  double (P, K, 3) tensor materialization: segment-sum over labels replaces
  the one-hot-mask weighted mean; a `lax.scan` over K-chunks replaces the full
  pairwise distance tensor (peak (P, `k_block`, 3) instead of (P, K, 3)).
  PBC minimal image and metric tensor behavior are byte-compatible with the
  old implementation (new regression test covers orthorhombic / FCC / skew
  cells and the cross-boundary minimum-image case).
- **Added `make_sharded_kmeans_update`** — `shard_map`-based parallel Lloyd
  step. P sharded on mesh axis `'x'`, centroids replicated, one `lax.psum`
  per iteration on the (K, 3) / (K,) accumulators. Verified bit-identical
  single-GPU vs 4-GPU on Si 4×4×4 (matching md5 on `centroids_frac_128.txt`),
  same 71-step trajectory.
- **Fixed latent `alat`-vs-`Å` mislabel in `main()`.** BGW WFN.h5 stores
  `avec` in alat units and `alat` in Bohr; the old code treated `|avec row|`
  as Å, which silently inverted a ~2× grid upsample into a ~0.6× downsample.
  `main()` now converts to Å once via `wfn.alat * BOHR_TO_ANG`; the kmeans
  function docstring states distances inherit the caller's avec units.
- **Multi-process bootstrap.** Added the standard `_maybe_init_jax_distributed`
  to the module so `srun -n N>1` works (matches `psp/run_nscf.py`, `gw/gw_jax.py`).
  Prototype uses the simpler single-process-4-GPU path.
- **New tests** (`tests/test_kmeans_sharded.py`): 5 cases, all pass. Full
  suite: 18 pass, 1 pre-existing failure in `test_reshard_all_to_all.py`
  unrelated to this branch.
- **Sandbox doc hardening.** `skills/execute_workflow/SKILL.md` now says
  explicitly: never export a `SLURM_JOBID` you did not allocate yourself; the
  interactive-allocation section documents the background `salloc` +
  `-J lorrax_X_agent` naming pattern. Matching memory pointer at
  `memory/feedback_never_share_allocation.md`.
- **Run**: `runs/Si_B/00_si_4x4x4/` — fresh 4×4×4 Si sym-reduced QE (8 IBZ
  k-pts, 24³ FFT, 16 bands) → 48³ kmeans grid. Three sub-dirs hold the
  baseline, refactored-single-GPU, and sharded-4-GPU centroid outputs for
  the equivalence check.

## 2026-04-17: Three parallel LORRAX checkouts (A/B/C) for concurrent agents

Consolidated the previous per-sandbox LORRAX clones into three sibling
checkouts at `$HOME/software/lorrax_{A,B,C}`, symlinked into the sandbox
as `sources/lorrax_{A,B,C}`. Each agent session claims one letter and
touches only its own checkout. Shared Shifter stage trees remain at
`/pscratch/sd/j/jackm/lorrax_{nvhpc,phdf5_openmpi}` (read-only in the
container), so the three variants share bind-mounted deps but build
their own `src/ffi/common/cpp/build/liblorrax_ffi.so`.

- `config/perlmutter/install.sh`, modulefile template: new
  `LORRAX_MODULE_NAME` variable lets each checkout install its own
  modulefile (`lorrax_A`, `lorrax_B`, `lorrax_C`). `family("lorrax")`
  makes variants mutually exclusive in a single shell; across shells
  they are fully independent. Landed on `main` (LORRAX feature branch
  `agent/multi-checkout`, fast-forwarded).
- Sandbox `AGENTS.md`: new "Which agent are you?" section at the top,
  revised source-code table, non-negotiable rule #7 ("only edit your
  assigned checkout"). `execute_workflow`, `checkpoint`,
  `profiling_stack` skills updated to say `sources/lorrax_X` /
  `module load lorrax_X`.
- Deleted stale sandboxes: `lorrax_sandbox_fresh`,
  `lorrax_sandbox_profiling`, and their backing clones
  `$HOME/software/lorrax_{bse,profile_ppm}`.
- `pyproject.toml`: dropped the sandbox-level `lorrax` editable
  dependency; the path no longer resolves to a single variant. Host
  Python that imports LORRAX should run inside Shifter via `lxrun`, or
  `uv run` from inside a specific `sources/lorrax_X`.

## 2026-04-17: WFNReader full-zone symmetry wrappers

- Audited the raw wavefunction reader usage after the nonsymmorphic-phase work.
  In active `src/`, raw `get_cnk()` / `get_cnk_batch()` are only consumed by
  `SymMaps.get_cnk_fullzone*`; there was no active path pairing unfolded
  `get_gvecs_kfull()` output with raw irreducible-zone coefficients.
- Clarified the API in both `src/common/wfnreader.py` and
  `src/file_io/wfnreader.py`: raw `get_cnk*` / `get_gvec_nk` remain explicit
  irreducible-zone readers, and new `get_gvecs_kfull`,
  `get_cnk_fullzone`, and `get_cnk_fullzone_batch` wrappers now route full-BZ
  access through `SymMaps` so the non-symmorphic `tau` phase is applied in the
  safe path by construction.
- Switched active consumers in `src/common/load_wfns.py`,
  `src/bandstructure/htransform.py`, and
  `src/centroid/get_charge_density.py` to the new WFNReader full-zone
  wrappers.
- Verified on Si `4x4x4` symmetry-vs-nosym WFNs that for all `44` k-points
  unfolded with nonzero `tau`, `get_gvecs_kfull()` matches the nosym G-list as
  a set, confirming `tau` does not act on the integer G-list itself.
- Added wrapper regression coverage in
  `tests/test_symmetry_maps_nonsymmorphic.py`.
- Validation: `uv run python -m pytest -q tests/test_symmetry_maps_nonsymmorphic.py`
  passed (`4 passed`), and full `uv run python -m pytest -q` passed
  (`15 passed, 1 warning`).

## 2026-04-16 (pm): cuSOLVERMp FFI unblocked — NCCL-backed cal_comm_create works

Follow-up to the earlier "cuSOLVERMp WIP" entry.  The SIGFPE in
`cusolverMpSyevd_bufferSize` was a communicator-plumbing bug:
NVIDIA's sample passes `ncclComm_t` directly to a `cal_comm_t`-typed
API, which works under `MPI_Init` but *not* in a JAX-only C++ process
(C implicit pointer-conversion quietly becomes a bug in C++).  The
documented non-MPI CAL path — `cal_comm_create` with user
allgather/req_test/req_free callbacks — routes through NCCL cleanly.

### Result (job 51659364, nid001033, 1 node × 4×A100)

| Path | n   | type        | max \|evals − ref\| |
|------|-----|-------------|---------------------|
| cuSOLVERMp (multi-proc, NCCL)   | 128 | F64  sym    | 9.1e-13 |
| cuSOLVERMp (multi-proc, NCCL)   | 128 | C128 Herm   | 5.7e-13 |

Both on a 2×2 process grid with `NamedSharding(P('x','y'))`.

### Source changes (branch `agent/ffi-cusolvermp`, commits 22ed74a, 7c716a7)

- `src/ffi/cusolvermp/cpp/ctx.h`: add `CalNcclShim` (NCCL comm + stream
  + persistent device scratch buffer), add `cal_comm_t` field on Ctx.
- `src/ffi/cusolvermp/cpp/context.cc`: three static callbacks
  (`cal_nccl_allgather` = H2D→`ncclAllGather`→D2H→stream-sync,
  `cal_nccl_req_test/free` trivial since we're synchronous).  Replace
  `reinterpret_cast<cal_comm_t>(ncclComm)` with a real
  `cal_comm_create(params, &ctx->cal_comm)`.  Teardown extended.
- `src/common/cusolvermp_eigh_test.py`: `gather_to_numpy` now uses
  `multihost_utils.process_allgather(x, tiled=True)` so each rank
  can verify the full logical array in multi-process mode.
- `src/ffi/AGENTS.md`: mark cuSOLVERMp status as working, document the
  three required env vars and why.

### Required runtime env

```
CUSOLVERMP_FORCE_NCCL=1              # route libcal's runtime collectives via NCCL
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5   # leave headroom for cuSOLVERMp workspace
XLA_PYTHON_CLIENT_PREALLOCATE=false  # allocate on demand, not up front
```

Without the first flag, libcal's internal reduce goes through UCC and
trips `Failed to parse ib device list` in the container.  Without the
memory settings, `cudaMalloc(scratch)` inside UCC fails because JAX's
modulefile default reserves 95% of VRAM up front.

### Why this is the real scaffold for ELPA

The multi-process NCCL bootstrap is the hard part; everything else
(`XLA_FFI_DEFINE_HANDLER_SYMBOL` in C++, `jax.ffi.ffi_call` wrapped in
`shard_map` in Python, ctypes-loaded .so, bind-mounted host libs, JAX
KV-store unique-id broadcast) transfers to ELPA 1-for-1.  ELPA takes
an MPI communicator instead of NCCL; that swaps `ncclGetUniqueId` /
`ncclCommInitRank` for their MPI equivalents but does not change the
control flow.

### Report

`reports/ffi_cusolvermp_nccl_2026-04-16/report.md`.

## 2026-04-16: JAX FFI scaffolding — cuSOLVERMg eigh working on 4 GPUs; cuSOLVERMp WIP

New directory `sources/lorrax/src/ffi/` with pluggable scaffolding for
calling compiled parallel-LA libraries from JAX via the XLA FFI.  No
pybind/nanobind; the `.so` is plain C ABI loaded with `ctypes.CDLL` and
its XLA handlers wrapped via `jax.ffi.pycapsule` — the pattern from
NVIDIA's JAX FFI tutorial.

### Working — `ffi.cusolvermg` (single-process, multi-GPU)

- `src/ffi/cusolvermg/cpp/eigh_mg_ffi.cc`: XLA FFI handler that owns a
  lazy `cusolverMgHandle_t` + pairwise peer access, scatters the
  device-0 input into cuSOLVERMg's column-tile layout via
  `cudaMemcpyPeerAsync`, runs `cusolverMgSyevd` across all visible GPUs,
  and gathers `Q` back to device 0.
- `src/ffi/cusolvermg/eigh.py`: `eigh_mg(A, tile_size=32, max_gpus=0)`.
- `src/common/cusolvermg_eigh_test.py`: 4-GPU Python test.

Validation on 1 node × 4×A100 (job 51656242, nid001164):

| n    | tile | max \|evals − ref\| | wall (post-warmup) |
|------|------|---------------------|--------------------|
| 128  | 32   | 9.1e-13             | 57 ms              |
| 2048 | 256  | 2.2e-11             | 509 ms             |

Eigenvector residuals `‖A q_i − λ_i q_i‖∞` ≈ 7e-14 (F64).

### WIP — `ffi.cusolvermp` (multi-process, multi-GPU/multi-node)

Everything builds, links, and runs up to the solve.  NCCL bootstrap
works via `jax.distributed.global_state.client` KV-store broadcast of
a 128-byte `ncclUniqueId` (note: `multihost_utils.broadcast_one_to_all`
silently promotes `uint8 → uint64` under `jax_enable_x64=True` and
scrambles it — workaround is documented in the code).  `cusolverMpCreate`,
`CreateDeviceGrid`, `CreateMatrixDesc` all succeed.  `cusolverMpSyevd_bufferSize`
then SIGFPEs (integer divide-by-zero) at a constant offset inside
`libcusolverMp.so`.

Most likely cause: NVIDIA's `mp_syevd.c` sample passes `ncclComm_t`
directly to an API typed `cal_comm_t` — the C implicit pointer
conversion plus an MPI-initialised libcal recognises the wrap; our
JAX-only process never calls `MPI_Init` so libcal's NCCL-detection path
never arms, `cal_comm_get_size` returns 0, and bufferSize divides.
Preserved as branch `agent/ffi-cusolvermp`; follow-ups
documented in `src/ffi/AGENTS.md` and the report.

### Build + runtime environment

- Container: `nvcr.io/nvidia/jax:25.04-py3` (CUDA 12.9, JAX 0.5.3.dev,
  `libcusolver*`, `libcusolverMg*` in-container).
- NVHPC (for cuSOLVERMp ONLY): `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/`
  staged to `/pscratch/sd/j/jackm/lorrax_nvhpc` and bind-mounted into
  Shifter at `/lorrax_nvhpc` — the Mg path needs nothing outside the
  container.
- Build: `src/ffi/common/cpp/build.sh` via
  `src/ffi/common/cpp/run_shifter.sh`.
- `LORRAX_NTASKS` env added to `run_shifter.sh` so single-process
  multi-GPU runs (1 task × N GPUs) are as easy as multi-process
  (N tasks × 1 GPU each).

### Report

`reports/ffi_cusolvermg_2026-04-16/report.md`.

### Regression

`uv run python -m pytest -q` → 12 passed, 1 OOM failure (GPU contention
with the interactive 4-GPU alloc; not a regression).

## 2026-04-16: JAX profiling stack — skill, helpers, k-parallel run_nscf

New sandbox-level `skills/profiling_stack/` and `scripts/profiling/` that
turn an unfamiliar LORRAX module into a ranked punch-list of bottlenecks
in one command. Four categories covered: memory, compute time, sharding,
compilation.

### Deliverables
- `scripts/profiling/pf.py` — helper library (`setup_env`, `trace_profile`,
  `region`, `annotate`, `snapshot_memory`, `aot_report`, `attach_compile_log`).
  Handles jax.distributed bootstrap, JAX_ENABLE_X64 latching, and the
  per-rank perfetto-trace race that broke multi-process runs.
- `scripts/profiling/run_profiled.py` — one-shot launcher wrapping
  `python -m <module>` with the whole env (XLA_FLAGS dump, JAX_LOG_COMPILES,
  IR dump, xprof trace, pprof snapshot).
- `scripts/profiling/analyze_hlo_dump.py` — XLA dump → ranked
  `hlo_summary.{md,json}` (Memory, Compute + custom calls, Sharding
  collectives, Rematerialization warnings, Retrace groups).
- `scripts/profiling/analyze_compile_log.py` — JAX compile log → ranked
  `compile_summary.{md,json}` (wall-clock totals, cache misses by source
  location, persistent-cache misses).
- `skills/profiling_stack/` — SKILL.md (entry point) + four category docs
  (memory / compute_time / sharding / compilation) + aot_reports.md +
  cookbook.md. All docs lead with "read the ranked summaries first, drill
  into source second" — per-function inspection is the secondary tool.

### LORRAX code change — branch `agent/run-nscf-kpar` (`4617f6e`)
- `src/psp/run_nscf.py`: module-level `_maybe_init_jax_distributed()`
  (same pattern as `gw.gw_jax`); Davidson k-loop strides over
  `jax.process_index()`; `process_allgather` of evals + packed coeffs;
  only rank 0 writes WFN.h5.

### Validation — Si 2×2×2 / 60 Ry / 12 bands
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/00_davidson_only/`:
  1 GPU, Davidson 7.91 s (1 rank), evals[0]=-0.418717 Ry.
- `runs/Si_pseudobands/00_si_2x2x2_60Ry/20_profile_run_nscf/02_davidson_4gpu/`:
  4 GPU k-parallel, Davidson 6.99 s (4 ranks). **WFN.h5 bit-identical to
  1-GPU** (eigenvalue maxabs diff 0.0, coefs maxabs diff 0.0).
- Analyzer on 4-GPU run surfaces 4 collectives (all-gather-start on
  f64[1,8,12] evals + c128[1,8,12,2,2120] coeffs, 31 MiB each) — the
  expected multihost_utils payloads.
- `uv run python -m pytest -q` → 14 passed when login-node GPU not saturated.

### Report
`reports/profiling_stack_2026-04-16/report.md` — deliverables, validation,
top-3 bottlenecks found from the very first profile (memory in
`jit__apply_H_sparse`, 33 % of wallclock spent in XLA compile, 163 cache
misses localised to `solvers/davidson.py` + `psp/vnl_ops.py`).

### Next steps
- A communication-heavy smoke test (multi-GPU `gw.gw_jax`) would exercise
  the Sharding + Rematerialization view at scale — `run_nscf` is
  embarrassingly k-parallel so only holds single-digit MiB collectives.
  Waiting on direction for the next target module.
- Collapse the `jit_multiply` x58 / `jit_broadcast_in_dim` x45 retrace
  groups by wrapping the Davidson k-loop body in one outer jit (or
  `lax.scan`).

## 2026-04-16: Symmetric Si 2x2x2 failure traced to SymMaps index conflation

- Reproduced the current symmetry-path failure directly from
  `runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/WFN.h5`:
  `SymMaps(WFNReader(...))` raises
  `IndexError: index 8 is out of bounds for axis 0 with size 8`.
- Root cause is in `sources/lorrax/src/common/symmetry_maps.py`:
  `create_kpoint_symmetry_map()` stores **symmetry-operation indices** in
  `kpoint_map`, but `kpoint_map_irrbz_ids()` later treats those values as
  **full-/irreducible-k indices** and indexes `full_kpts[idx]`.
- For the Si `2x2x2` WFN this is fatal because `nk_full=8` but the stored
  symmetry ids include `8` and `12`; the symmetric `4x4x4` path only
  appears to survive because its mistaken symmetry ids remain `< 64`.
- Compared against BerkeleyGW `Sigma/genwf_mpi.f90` and
  `Common/find_kpt_match.f90`, which keep irreducible-k index and symmetry
  index as separate state. This is the active bug; time reversal is only a
  secondary latent concern for future TR+nonsymmorphic cases.
- Fixed on source branch `agent/symmetry-maps-fix`:
  `create_kpoint_symmetry_map()` now stores irreducible-k ids rather than
  symmetry ids, and `kpoint_map_irrbz_ids()` now validates that direct map
  instead of reinterpreting it as a full-grid index.
- Added `src/common/symmetry_test.py`, a debug checker that validates both
  atomic-position invariance under the stored spatial symmetries and full-grid
  k-point unfolding from the irreducible wedge.
- Validation:
  `uv run python -m pytest -q` → `14 passed, 1 warning`;
  `uv run python -m common.symmetry_test .../Si_pseudobands/.../WFN.h5`
  → `48/48` symmetries and `8/8` k-points valid;
  `uv run python -m common.symmetry_test .../Si/05_si_4x4x4_sym/.../WFN.h5`
  → `48/48` symmetries and `64/64` k-points valid.

## 2026-04-15: Bare Σ_X invariance analysis — ISDF quality confirmed OK

### Bare exchange is nearly invariant (17 meV shift, BGW: 0 meV)
- Added bare Σ_X diagnostic print to gw_jax.py
- Ran 4 COHSEX calculations with the diagnostic: baseline (400c, 2000c), V1 PB, V2 PB
- Result: bare X shifts only 17-20 meV with pseudobands
- Centroids don't affect bare X (400c vs 2000c identical)
- ISDF quality for exchange is acceptable

### Decomposed comparison vs BGW (using CH' = exact static, per BGW sigma_hp.log)
- LORRAX absolute X differs from BGW by 5.5 eV — nk convention (8 vs 4 k-points)
- PB screening shifts: LORRAX ΔCH ≈ -1.4 to -1.7 eV, BGW ΔCH' ≈ -1.1 to -1.8 eV — within 20%
- Baseline CH offset (LORRAX -6.77 vs BGW -8.46) is k-grid dependent: 1.7 eV at 2×2×2, 0.6 eV at 4×4×4
- No evidence of COHSEX implementation regression from recent refactors

## 2026-04-15: Pseudobands v2 (Gauss-quadrature energies) — implemented, tested, V1 still wins

Branch `agent/nscf-clean-scaffold` (+6 commits).

### New module: `solvers/pseudobands_v2.py`
- **Shifted CJ boundaries** (δ = π/2M) for quadratic POU: Σw_j² ≈ 1 ± 0.04
- **Gauss quadrature** from windowed DOS moments (Stieltjes/Jacobi algorithm)
  gives per-band energies and weights. Numerically fragile for large n_eff;
  falls back to Ritz eigenvalues + uniform weight.
- **Davidson windows**: no-matvec Galerkin from stored eigenvalues
- **n_min = k floor** prevents pathologically narrow windows
- **Window placement** with automatic n_min enforcement
- Wired into `run_nscf.py` via `pb_version = 2` in nscf.in

### COHSEX comparison (Si 2×2×2, VBM)

| Method | sigTOT (eV) | Δ from 40-band |
|:--|:--:|:--:|
| Baseline 40-band | -12.824 | — |
| **V1 hybrid PB** | **-14.145** | **-1.32** |
| V2 Gauss PB | -14.428 | -1.60 |
| V2 Ritz energies | -14.419 | -1.60 |
| BGW reference | — | -1.18 |

**V1 remains the better scheme** (-1.32 vs -1.60 excess). The v2 shifted
boundaries and different window placement create 0.3 eV more over-screening.
Energy assignment (Gauss vs Ritz) has negligible effect (< 10 meV).

### Key findings
- Dominant error: ISDF quality degradation with pseudobands (89 meV sigSX shift)
- Energy assignment is NOT the bottleneck — Gauss vs Ritz ≈ same result
- The v2 infrastructure is complete and working, but the shifted boundaries
  need further investigation to understand why they increase over-screening
- `dos_cjwindows.py` diagnostic plots CJ window indicators on the full spectrum

### Test directories (runs/Si_pseudobands/00_si_2x2x2_60Ry/)
```
11_lorrax_pb_v2_k4_40win/    — v2 k=4, 41 windows (192 bands)
12_lorrax_pb_v2_k6_60win/    — v2 k=6, 59 windows (382 bands)
13_lorrax_cohsex_v2/          — COHSEX with v2 Gauss energies
14_lorrax_pb_v2_ritz_energies/ — v2 with Ritz energies
15_lorrax_cohsex_v2_ritz/     — COHSEX with v2 Ritz energies
```

## 2026-04-15: Hybrid stochastic/CJ-Ritz pseudobands — cross-window fix

Branch `agent/nscf-clean-scaffold` (+1 commit on top of prior work).

### Architecture change
- **Hybrid pseudobands**: three construction modes per window:
  - **Stochastic**: random-phase sums of exact eigenstates (for windows
    where CJ filter can't resolve — near conduction edge).
  - **CJ-Ritz**: Chebyshev-filtered Galerkin-Ritz (high-energy windows).
  - **CJ-0**: zero-weight placeholder (spectral gaps, CJ produces garbage).
- Det bands split into "protected" (below window start, included as-is)
  and "available" (consumed by stochastic construction). Extends Davidson
  deeper (nbnd=60) to provide exact eigenstates for transition zone.

### Bug fixes
- **Window start below det max**: E_cross was 1.31 Ry but det bands
  went to 2.23 Ry. First 3-4 windows were in the det manifold — after
  deflation, CJ produced noise. Now: stochastic for those windows.
- **Zero-norm NaN**: WFNReader clamped zero norms to 1e-30, ISDF divided
  by it → 10^30 → NaN in all zeta. Fixed: clamp to 1.0 (no-op division).
- **n_protected consistency**: fixed band count across k-points by passing
  n_protected from k=0 to subsequent k-points.

### Results (Si 2×2×2, 60 Ry)
- COHSEX pseudobands shift: **-1.32 eV** (was -1.77 eV broken, BGW ref -1.18 eV)
- Excess over BGW: **0.14 eV** (was 0.59 eV — 76% reduction)
- No more NaN output, no cross-window leakage

### Next
- Investigate remaining 0.14 eV excess (ISDF quality with pseudobands)
- Test with more centroids (5000+) to separate ISDF error from PB error
- Consider global QR for CJ windows to further reduce cross-window overlap

## 2026-04-14: NSCF refactor — clean scaffold, 2D Coulomb fix, module reorganization

Branch `agent/nscf-clean-scaffold` (14 commits).

### Bug fix
- **MoS2 2D Coulomb truncation**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False`
  for V_H Poisson solve. QE's `assume_isolated='2D'` now auto-detected from XML and
  applied to both V_loc and V_H. MoS2: 594 mRy → 0.013 mRy offset. Si unchanged.

### Module reorganization
- **`src/solvers/davidson.py`**: generic eigensolver (BSE-ready). `nspinor→n_channels`,
  `n_tgt→n_eig`, `nG→dim`. `psp/davidson.py` → shim.
- **`psp/pseudos.py`**: `load_pseudopotentials`, `symbol_to_Z`, `AtomPP` extracted from
  `get_DFT_mtxels.py` (-300 lines from the kitchen sink).
- **`psp/gvec_utils.py`**: `build_master_gvec_list`, `select_gvecs_for_k`, `compute_ngkmax`,
  `reorder_to_qe` consolidated.
- **`psp/radial/`**: `radial_jax.py`, `solid_harmonics.py`, `build_projectors_qe.py`.
- **`psp/upf/`**: `load_upf.py`, `normalize.py`, `upf_model_2_0_1/`.
- **`file_io/`**: `qe_save_reader.py` + `wfn_writer.py` joined `WFNReader` et al.
- **`dft_operators.py`**: now owns `poisson_potential_from_rhoG`, `generate_gvectors_k`,
  `build_G_cart` (moved from `get_DFT_mtxels` and `charge_density`).
- **Deleted**: `kpar.py`, `get_dipole_mtxels_chunked.py`, debug functions (~750 lines).
- **Archived**: `charge_density.py` (85% dead SCF code).
- **`get_DFT_mtxels.py`**: 1281 → 974 lines.

All three entry points (`run_nscf`, `get_DFT_mtxels`, `get_dipole_mtxels`) and GW drivers
now import shared routines from canonical locations. Validated: Si 0.001 mRy, MoS2 0.013 mRy.

## 2026-04-14: NSCF driver, WFN.h5 writer, k-parallel, MoS2 validation

### New modules
- **`psp/run_nscf.py`**: Full NSCF driver (QE .save → Davidson → WFN.h5)
- **`psp/kpar.py`**: K-point parallel diag via 2D mesh ('k', 'g')  
- **`compare_wfn.py`** (sandbox): Permanent WFN.h5 comparison tool

### WFN.h5 accuracy
- **Si 4×4×4**: 33/37 fields EXACT, eigenvalues 0.0009 mRy MAE, timing competitive with QE
- **MoS2 3×3×1**: 36/37 fields EXACT (all structural, G-vectors byte-identical after QE convention matching). Eigenvalues: 2.7 mRy MAE at Gamma, 1.0 mRy at other k-points.

### Bug fixes
- **bvec.T transpose bugs**: bdot, adot, atom_crys, G_cart — all hidden by cubic Si, exposed by hexagonal MoS2. Fixed in qe_save_reader.py, wfn_writer.py, ionic_gspace.py, charge_density.py.
- **QE G-vector ordering**: Matched exactly via `(round(|G|²×1e8), g1, g2, g3)` lexicographic sort
- **nosym symmetry convention**: ntran=1, identity only, zero-padded to 48
- **scipy_erf**: Replaces jax.scipy.erf in table construction (avoids Shifter PTX crash)

### MoS2 NSCF eigenvalue discrepancy — FIXED
**Root cause**: `compute_V_H_and_V_xc` hardcoded `truncation_2d=False` for V_H Poisson solve.
QE's MoS2 input uses `assume_isolated='2D'`, applying 2D Coulomb truncation to both V_H and
V_loc. LORRAX applied it to V_loc but not V_H, causing 594 mRy offset.

**Fix** (branch `agent/nscf-2d-truncation`): Added `truncation_2d` kwarg to
`compute_V_H_and_V_xc` and threaded from `run_nscf.py`. After fix: **0.013 mRy offset,
0.002 mRy MAE-no-offset** across all 9 k-points. Si unchanged at 0.001 mRy.

## 2026-04-13: Unified ionic G-space pipeline — 195s → 31s (setup: 177s → 5s)

Three changes on branch `agent/rho-core-table-interpolate`:

1. **Unified `build_ionic_and_core`** (new `psp/ionic_gspace.py`):
   - V_loc(r) and ρ_core(r) built in one pass via shared `lax.scan` primitives
   - `species_structure_factors` + `accumulate_species_on_G` — jittable, scannable
   - Cold: 2.37s. Warm: 0.01s. Previously V_loc=1.5s + rho_core=155s.

2. **SciPy CPU table construction** (`radial_jax._spherical_hankel_table_np`):
   - Replaced JIT-compiled `spherical_hankel_table_jax` for one-time setup
   - l=1 table build: 20.27s → 0.24s (84× faster, no JIT overhead)
   - JAX version kept for gradient computations

3. **VNL table reduction** (`vnl_ops.build_vnl_setup` n_q: 50000 → 4000):
   - Linear interpolation accurate to <1e-6 Ry at dq~0.001
   - vnl_setup: 21.5s → 2.6s

Full pipeline Si 4×4×4 nosym 64 k-points: **195s → 31s** total (26s is per-k JIT).
Setup (V_loc+NLCC+VNL): **177s → 5.0s**. Eigenvalues ≤0.0001 mRy.
Branch: `agent/rho-core-table-interpolate`, commits `8e50cbc`..`3c95c63`.
- **Next**: wire `build_ionic_and_core` into `test_dft_hamiltonian.py` callers,
  consider further per-k JIT reduction, merge to main.

## 2026-04-13: Active PSP callers migrated onto unified JAX VNL path

- Switched the remaining active preprocessing callers off the old
  `projector_pipeline` execution backend:
  `psp.get_dipole_mtxels`, `psp.get_dipole_mtxels_chunked`,
  `psp.get_DFT_mtxels`, and `gw.kin_ion_io_chunked` now build one
  `vnl_ops.build_vnl_setup(...)` and use per-k
  `build_vnl_kdata_from_kvec(...)` plus dense JAX contractions for `V_NL`.
- Added canonical sparse-G helpers to `psp.dft_operators` so the active caller
  scripts share one gather / `V_NL` matrix-element path rather than
  reimplementing host-side extraction logic.
- Preserved the custom JAX radial/spline/Bessel handling in one place:
  the migration still flows through `psp.radial_jax` and `psp.vnl_ops` for
  uniform-table interpolation, derivative tables, and stable spherical-Bessel
  behaviour.
- Archived the old CPU-heavy compatibility modules under `src/psp/archive/`:
  `build_projectors.py` and `projector_pipeline.py`.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 19.27s`
  and real sandbox smokes both completed on local GPU:
  `gw.kin_ion_io_chunked` wrote `/tmp/kin_ion_migrated_smoke.h5`
  with shape `(64, 8, 8)` in `38.769 s`, and
  `psp.get_dipole_mtxels_chunked` wrote `dipole.h5`
  with shape `(3, 64, 60, 60)` from a temp staging directory.
- Revalidated both migrated preprocessors in the documented Perlmutter
  interactive-node Shifter environment on job `51487668` so profiling stays
  comparable to earlier sandbox runs:
  `gw.kin_ion_io_chunked` completed with `Total recorded: 17.793 s`
  and `real 30.31`, while
  `psp.get_dipole_mtxels_chunked --vnl-mode analytic` completed with
  `real 49.57`.

## 2026-04-12: Unified JAX radial backend for PSP setup path

- Added a shared source backend for radial transforms:
  [src/psp/radial_jax.py](/global/u2/j/jackm/software/lorrax/src/psp/radial_jax.py:1).
  This now owns the common spherical-Bessel kernels, uniform radial tables,
  interpolation, and radial integration weights used to form `V_NL`, `V_loc`,
  and NLCC/core charge.
- Switched the active production builders away from the old SciPy spline path:
  `vnl_ops.build_vnl_setup(...)`,
  `build_projectors_qe.build_local_ionic_potential_on_G_total(...)`, and
  `charge_density.build_core_density(...)` now all use the shared JAX/table
  backend.
- Simplified the autodiff `V_NL` channel extraction path in
  `dft_operators.py` so it consumes the same uniform tables rather than SciPy
  spline internals.
- Removed a duplicate spherical-Bessel implementation from
  `projector_pipeline.py` by importing the shared backend instead.
- Validation:
  `uv run python -m pytest -q` → `13 passed, 1 warning in 15.24s`
  and the canonical Si DFT-H reproducer still passes with
  `Max MAE: 0.0001 mRy = 0.00 meV`.
- Measured canonical launcher wall time after the refactor:
  `/usr/bin/time -p ./launch_test_dft_hamiltonian.sh` →
  `real 25.67`, `user 0.05`, `sys 0.04`.
- Followed up with a terminology cleanup in the active path so plan/bundle
  fields now prefer `radial_tables` over `splines`, reducing conceptual drift
  after the backend swap.
- Added report:
  [reports/jax_unified_psp_radial_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/jax_unified_psp_radial_2026-04-12/report.md)

## 2026-04-12: Standalone psp DFT-H validation now documented and runnable

- Fast-forwarded `sources/lorrax` again from `f7bc2e2` to `273a7d8`, picking up
  the new upstream reproducer `src/psp/tests/test_dft_hamiltonian.py` and the
  expanded `src/psp/dev_status.md`.
- Logged a new sandbox mismatch in `KNOWN_SANDBOX_ERRORS.md`: the local
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` helper scripts still pointed at
  deleted `psp` setup helpers, so they were no longer a valid entrypoint.
- Added a sandbox-side canonical entrypoint for the standalone DFT path:
  [runs/Si/04_si_4x4x4_davidson/00_davidson/README.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/README.md)
  and
  [runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh](/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson/launch_test_dft_hamiltonian.sh),
  both using this sandbox's real paths and the Shifter environment that
  includes `$SANDBOX/sources` for `jax_xc_local`.
- First launcher run exposed a real upstream test bug: `test_dft_hamiltonian.py`
  passed `CrystalData` into `vnl_ops.build_vnl_setup(...)`, but the current
  implementation needs the `WFNReader` for its k-dependent G-vector scan.
  Patched locally on source branch `agent/test-dft-hamiltonian-fix`.
- Re-ran the canonical test on interactive job `51470500` and obtained:
  `Max MAE: 0.0000 mRy = 0.00 meV`
  and
  `PASS: all k-points match QE to < 0.01 mRy`
  for all 8 Si `4x4x4` IBZ k-points.
- Added report:
  [reports/dft_hamiltonian_validation_2026-04-12/report.md](/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/dft_hamiltonian_validation_2026-04-12/report.md)

## 2026-04-12: Si 4x4x4 no-sym COHSEX output-format rerun

- Created `runs/Si/02_si_4x4x4_nosym/16_lorrax_cohsex_rerun_4gpu_repeat/` as a fresh clone of variant `15` and reran GWJAX on interactive job `51470500` (1 node / 4 GPUs) so the updated logging/output-writing behavior would land in a new `gw.out` without overwriting prior outputs.
- Run completed end to end in `26.661 s`; artifacts written successfully: `gw.out`, `eqp0.dat`, `qp_wfn_rotations.h5`, and `tmp/isdf_tensors_480.h5`.
- The new `gw.out` differs materially from variant `15`: no initial `srun` step line, denser chunked-ISDF setup summary, progress-bar style zeta/V_q status lines, a new `STATIC HEAD TERMS` block, and inline XLA rematerialization warnings captured in the file.
- `eqp0.dat` from variant `16` is not byte-identical to variant `15`, so this should be treated as more than a cosmetic logging-only rerun.

## 2026-04-12: Housekeeping sync

- Fast-forwarded `sources/lorrax` on local `main` from `b0b02f9` to `f7bc2e2` to match `origin/main`.
- Logged a sandbox inconsistency in `KNOWN_SANDBOX_ERRORS.md`: the newest report directory (`reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/`) does not contain the documented `report.md`.
- Added sandbox-local `jax_xc_local` wiring for the standalone `psp` DFT path:
  `sources/jax_xc_local -> /global/u2/j/jackm/software/jax_xc_local_lorrax_sandbox`
  and `sources/jax_xc -> /global/u2/j/jackm/software/jax_xc`.
  Verified `jax_xc_local.pbe` and `psp.dft_operators.compute_V_H_and_V_xc` import and execute under the documented Shifter flow when `PYTHONPATH` includes `$SANDBOX/sources`.
- Pulled the current Si Davidson/NSCF test drivers from `../lorrax_sandbox_fresh` into
  `runs/Si/04_si_4x4x4_davidson/00_davidson/` and updated `run_direct_diag_v2.py`
  to the current `origin/main` `psp` API (`setup_H_k`, `build_matrix_k`, `vnl_ops.build_vnl_setup`).
- First live Perlmutter/Shifter validation now works end-to-end for the direct-diag rung.
  `run_direct_diag_v2.py` reaches all 8 IBZ k-points and reports:
  diagonalized occupied-band MAE `94.890 mRy`, offset `-94.890 mRy`, MAE-no-offset
  `19.943 mRy`, max `153.478 mRy`. Nontrivial k-points show `H` non-Hermitian warnings
  (`~1e-4` to `4.5e-4`) and pathological Rayleigh quotients, which is the clearest
  current testing signal before Davidson wall-time work.

## 2026-04-12: Major code clarity refactor

Session focused on making gw_jax.py main() read like a physics outline.

**Screening pipeline surfaced at top level:**
- `compute_chi0(wfns, quad, meta, mesh_xy)` and `solve_w(V_q, chi0_q, meta, mesh_xy)` now visible in main() for both COHSEX and PPM paths
- `build_static_quadrature` / `build_imag_quadrature` are clean one-liners for quadrature setup
- `fit_gn_ppm(W_q, Wiwp_q, V_q, omega_p, mesh_xy)` extracted from monolithic PPM builder

**ppm_sigma.py (-347 lines):**
- PPM arrays stored as flat-q (nq,μ,μ) — eliminated transpose round-trip
- Fixed _mu_nu_sharding (was 5D for dead k-last layout)
- Fixed _build_single_sigma_window missing mask_B args (would crash on kernel_sign=-1)
- Stripped all profiling boilerplate; replaced verbose prints with per-window summary
- _convolve_sigma_branch_kij takes wfns bundle (28→22 params)

**gw_jax.py (-267 lines from ISDF move, +gw_output.py):**
- ISDF pipeline moved to gw_init.py (fixes circular import), split: fit_zeta + compute_V_q
- Output formatting extracted to gw_output.py (GWResults dataclass + write_results)
- V_q/W_q naming used consistently everywhere (no more bare V/W aliases)
- solve_w_from_chi_q_jax → solve_w; print0= → print_fn= standardized

**w_isdf.py:**
- Fixed chi0 accumulator sharding for non-divisible k-grids: P(None,'x','y')
- Fixed Dyson solve padding order (pad before reshard)
- Both verified on 4×A100 with MoS2 3×3 (nk=9)

All changes GPU-regression-tested (MoS2 3×3 COHSEX, 4×A100-40GB, bit-identical).
COHSEX chi0_W timing dropped from 2.7s→1.7s (old path computed unnecessary PPM head terms).

## 2026-04-09: GWJAX pipeline refactor status

Primary initiative: remove non-jitted stages, eliminate incorrect host/replicated
materializations, and make the active no-symmetry GWJAX pipeline safe on multi-GPU
Si `4x4x4` and `10x10x10`.


## Current status

What is now in good shape:
- head corrections for sigma_{X,static SX-X/CH, GN-PPM cor}
- active multi-GPU minimax screening path
- active GN-PPM fit path
- active dynamic sigma path
- post-PPM tail safety on `10x10x10`
- one process per GPU execution

What still looks worth improving:
- `compute_sigma_c_ppm_omega_grid` dominates runtime on large grids
- post-PPM fixed-point / QSGW work is safer now, but not yet distributed over
  band tiles on the `XY` mesh. This is a significant issue.
- likely next architectural step is a band-sharded `sigma_mnk.h5` / post-PPM
  path over `(omega, k, m_X, n_Y)`

## Known environment notes

- For multi-GPU GWJAX on Perlmutter, use Shifter, not `uv run`.
- *Keep one MPI rank per GPU. Do not ever run one mpi rank per node with 4 GPUs or so forth.*
