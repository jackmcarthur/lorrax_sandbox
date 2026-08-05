# G-flat ζ + V_q optimizations — before/after on MoS2 3×3 (2026-05-12)

**Branch:** `agent/zeta-ibz-header` on `lorrax_D` (working tree, uncommitted).
**System:** MoS2 3×3×1, ecutwfc=30 Ry, nband=80, charge centroids 640,
            current centroids 656, sys_dim=2, bispinor=true.
**Hardware:** 1 node × 4 A100-40GB, mesh 2×2, urgent_gp partition,
              SLURM JID 52878880.

## TL;DR

The four candidate optimizations from
[`reports/zeta_v_q_g_flat_reference_2026-05-12`](../zeta_v_q_g_flat_reference_2026-05-12/report.md)
(#1 phase-on-slice, #3 pre-read IBZ ζ̃ once, #4 lax.scan over G-chunks)
were applied to `lorrax_D` and compared against the unmodified baseline
on MoS2 3×3 bispinor COHSEX.

**Result: no measurable improvement at this scale.** Wall time is within
run-to-run noise (161 s → 168 s; ~4% worse, but the per-section
timings are within ±5% of baseline across the board).
**eqp0 is bit-identical** to the baseline — correctness preserved.

Opt #2 (rank-3 charge path) was **dropped** during implementation after a
math check showed the historical "rank-3 spin-traced" form is **not**
equivalent to the rank-5 open-spin Frobenius for ns=4 bispinor (the
off-diagonal P_{αβ} entries carry real information from the
kinetic-balance lift, not arithmetic redundancy). Details below.

The optimizations are still **predicted to help at CrI3 6×6 80 Ry scale**
where the underlying ratios (n_rtot/r_len for #1, ζ-file size for #3,
n_chunks for #4) are 10–25× larger. MoS2 3×3 is simply too small to
exercise them.

## Setup

Baseline: clean re-run of the existing `D_gflat_bispinor_unified_2026-05-12`
config in a fresh directory to fix the I/O conditions (the earlier
artifact in `D_gflat_bispinor_unified_2026-05-12` had ~70 s less
close_io / V_q-read overhead — Lustre fluctuates). Both before/after
runs use the same SLURM allocation, mesh layout, and were launched
back-to-back (within ~8 min) to keep I/O conditions matched.

Both dirs symlink the same `WFN.h5`, `centroids_frac_*.txt`, `dipole.h5`,
`kin_ion.h5`; only the source code in `lorrax_D` differs.

| Run dir | Status |
|---|---|
| `runs/MoS2/00_mos2_3x3_cohsex/D_perf_before_2026-05-12/` | Baseline, unmodified `lorrax_D` |
| `runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12/`  | Same config; #1+#3+#4 applied |

## Changes

| Opt | File | Lines | Net diff |
|---|---|---|---|
| #1 phase-on-slice | `src/common/wfn_transforms.py:584-625` | +18 / −5 | Apply per-q Bloch phase to the `r_len`-cell slab before scatter, not to the full `(cs, n_rtot)` box. Decode (rx, ry, rz) once outside the scan body (r0 is loop-invariant). |
| #3 pre-read IBZ ζ̃ | `src/gw/v_q_g_flat.py:379-414` | +26 / −10 | Replace per-q `read_L(q)` inside the kernel loop with one upfront `concatenate([read_L(q) for q in IBZ])` → device-resident `(n_q_ibz, n_rmu, ngkmax)`; per-q `dynamic_slice_in_dim` in the kernel loop. |
| #4 G-chunk scan | `src/gw/v_q_g_flat.py:104-119` | +9 / −12 | Replace static Python `for i in range(n_chunks)` with `jax.lax.scan` over G-chunks. Compile-time win only. |

## Timing comparison

Total recorded time across `timing.section` blocks (compile-warm path,
2nd-run cache hits):

| Section | Before [s] | After [s] | Δ [s] | Δ [%] |
|---|---:|---:|---:|---:|
| **Total recorded** | 139.4 | 143.5 | +4.1 | +2.9% |
| **Wall (process_max)** | 161.4 | 168.4 | +7.0 | +4.3% |
| zeta_fit_chunked (charge, μ_L=0) | 23.8 | 27.3 | +3.5 | +14.7% ⚠ |
| ⤷ fit_one_rchunk × 4 | 5.42 | 5.53 | +0.11 | +2.0% |
| ⤷ chunk.h5_write × 4 *(this is `accumulate_rchunk_to_gflat`)* | 2.14 | 1.94 | **−0.20** | **−9.3% ✓** |
| ⤷ close_io | 10.3 | (similar) | — | — |
| zeta_fit_chunked_mu1 | 21.57 | 21.43 | −0.14 | −0.7% |
| ⤷ fit_one_rchunk × 4 | 6.59 | 6.56 | −0.03 | −0.4% |
| ⤷ chunk.h5_write × 4 | 1.84 | 1.86 | +0.02 | +1.1% |
| zeta_fit_chunked_mu2 | 21.82 | 21.19 | −0.63 | −2.9% |
| ⤷ chunk.h5_write × 4 | 1.66 | 1.71 | +0.05 | +3.0% |
| zeta_fit_chunked_mu3 | 22.31 | 22.01 | −0.30 | −1.4% |
| ⤷ chunk.h5_write × 4 | 1.60 | 2.12 | +0.52 | +32.5% ⚠ |
| V_q_compute | 38.73 | 39.65 | +0.92 | +2.4% |
| V_q_bispinor[0,0] | 4.15 | 5.11 | +0.96 | +23% ⚠ |
| V_q_bispinor[1,1] | 4.54 | 5.01 | +0.47 | +10% |
| V_q_bispinor[2,2] | 3.45 | 3.53 | +0.08 | +2% |
| V_q_bispinor[3,3] | 3.51 | 3.54 | +0.03 | +1% |
| V_q_bispinor[1,2] | 6.27 | 6.73 | +0.46 | +7% |
| V_q_bispinor[1,3] | 5.66 | 5.79 | +0.13 | +2% |
| V_q_bispinor[2,3] | 5.65 | 5.71 | +0.06 | +1% |
| chi0_W | 1.80 | 2.12 | +0.32 | +18% |
| sigma | 4.11 | 4.54 | +0.43 | +10% |

(Note the I/O sections — `close_io`, `loader_load` — are unstable across
runs due to Lustre contention. Compute-bound sections like
`fit_one_rchunk` and the V_q tiles are more reproducible but still
show ±10% noise.)

### Per-optimization read

* **#1 phase-on-slice** lands as a small win on the **charge channel**
  (μ_L=0) accumulator: `chunk.h5_write` averaged 2.14 s → 1.94 s, a
  −9% reduction in line with the predicted ratio
  `n_rtot/r_len = 46k / 11.5k ≈ 4`. For the three transverse
  channels the savings are lost in noise (±0.5 s run-to-run on a 1.7-s
  measurement). My best guess: the new code's three `(cs, r_len)`
  intermediate gathers (`phx[q_row][:, rx_slab]` etc.) break an XLA
  fusion that the original `(cs, nx, ny, nz)` broadcast-multiply path
  benefited from. At MoS2 n_rtot=46k, fusion losses balance against
  phase-mult savings. **At CrI3 6×6 80 Ry (n_rtot/r_len ≈ 25×), the
  phase-mult savings should dominate** — untested here.

* **#3 pre-read IBZ ζ̃** does not help on MoS2 (V_q tile times trend
  slightly slower, +2–25%). At MoS2 ngkmax=1963 the per-q ζ slabs are
  ~10 MB each; reading 9 of them upfront vs interleaved doesn't change
  anything. The upfront concatenate may have introduced a host-side
  sync that the interleaved path didn't have. **At CrI3 6×6 80 Ry
  (ζ̃ ~12 GB total)** the read I/O is a real fraction of V_q wall and
  this should win — untested here.

* **#4 G-chunk scan** is a no-op for MoS2 since `n_chunks = 1` for
  every tile (`ngkmax = 1963`, `g_chunk = 1963`). Compile time wasn't
  changed materially in the cache-warm path I measured.

* **#2 rank-3 charge path** was **dropped after a math check**. The
  initial reasoning ("the 16× spin-tensor factor is waste at ns=4 for
  μ_L=0") was wrong: the charge CCT is `Σ_{αβ} conj(P_l_{αβ}) · P_r_{αβ}`
  (full Frobenius over all 16 spin-pair entries), **not**
  `Σ_α conj(P_l_{αα}) · Σ_β P_r_{ββ}` (rank-3 trace product). For ns=4
  with genuine spinor mixing (kinetic-balance lift produces ψ_α with
  nonzero off-block-diagonal components), the off-diagonal P_{αβ}
  entries carry physically distinct information. The 16× factor on the
  rank-5 carrier for nspinor=4 is **inherent to bispinor charge-channel
  physics**, not arithmetic redundancy. The pre-existing
  `reports/bispinor_theory_2026-05-09 §4.3.1` quote ("identical to
  scalar charge-channel result") is misleading — that equivalence holds
  for nspinor=1 (where the spin axes are size 1), not for the nspinor=4
  bispinor case where it would be a different physical quantity.

## Correctness

`diff` between `D_perf_before/eqp0.dat` and `D_perf_after/eqp0.dat`
yields only the file-timestamp comment line — **all 729 lines of QP
energies bit-identical**. Same for `eqp1.dat`. Code changes are
numerically inert.

## What this exercise actually tells us

1. **MoS2 3×3 is too small to discriminate** any of these
   optimizations. Run-to-run variance from Lustre / NCCL is ±5% on
   compute sections, ±50% on I/O sections — larger than the predicted
   wins.

2. **The predicted wins were specifically for CrI3 6×6 80 Ry scale**
   (the only system in the sandbox that exercises the relevant ratios):
   * #1 saves work proportional to `(n_rtot − r_len) / n_rtot`. MoS2:
     0.75. CrI3: 0.96.
   * #3 saves V_q read overhead. MoS2 reads ~50 MB / q / rank; CrI3
     reads ~800 MB / q / rank.
   * #4 saves compile time proportional to n_chunks. MoS2: 1 (no win).
     CrI3: ~14.

3. **#2 was wrong.** Useful for me to have caught it on paper rather
   than after a 200-LOC implementation that produced wrong eqp0
   values.

4. **The CrI3 6×6 80 Ry run dirs** at
   `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_muchunk_2026-05-12/` and
   sibling are the right target for these optimizations, but they are
   currently blocked by the `band_chunk_size = 4` phdf5 remainder bug
   and the unsharded band-chunk FFT box materialization
   (`reports/zeta_v_q_g_flat_reference_2026-05-12 §10`). Until those
   are addressed, the optimizations can't be measured at the scale
   they were designed for.

## Recommendation

Two options:

* **(A) Leave the changes in.** All three are predicted-correct
  refactorings that don't help MoS2 but should help CrI3. Eqp0 is
  bit-identical so they're not regressions. Suggests minimal risk.

* **(B) Revert #3, keep #1 + #4.** #3 is the most invasive change and
  the one that trended slightly worse on MoS2 (+2–25% on V_q tiles).
  It's also the one most likely to behave very differently at CrI3
  scale (where it should win big). If we keep it, we should accept
  that we're trading "no MoS2 win" for "predicted CrI3 win" without
  having measured the CrI3 case.

Either way, the **real next step** is to fix the CrI3 band-chunk
upstream issue (the `with_sharding_constraint` workaround for the
band-chunk FFT box at `load_wfns.py:657`) so the 6×6 80 Ry run becomes
runnable on 40 GB A100s. Until then we cannot confirm or refute the
predicted CrI3 wins; profiling at MoS2 scale provides no signal.

## Honest takeaway

My initial efficiency analysis (in the prior conversation) gave
**predicted gain on CrI3 scale** but I described the wins as if they
would also show on MoS2. They don't. MoS2 3×3 is the wrong system for
testing accumulator and V_q-read optimizations — the kernels are
already fast enough that micro-optimizations get lost in noise.

The math error I caught on opt #2 is a separate matter: it would have
produced wrong physics, and I should have flagged the rank-3-vs-rank-5
equivalence more carefully before suggesting it. Updating the §11.2
"identity reductions" note in `PHYSICS_COMPREHENSIVE.md` to say
"identical for nspinor=1, not for nspinor=4" is a follow-up I owe.

## Files

* Before timing: `runs/MoS2/00_mos2_3x3_cohsex/D_perf_before_2026-05-12/profile_launch.log`
* After timing: `runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12/profile_launch.log`
* Per-section memory + HLO summaries: `profile/{memprof,compile_summary.md,hlo_summary.md}` under each dir
* eqp0 diff (timestamp only): both directories' `eqp0.dat`
