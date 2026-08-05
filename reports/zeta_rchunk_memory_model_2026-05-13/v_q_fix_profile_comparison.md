# V_q kernel fix profile comparison — MoS2 3×3

Date: 2026-05-14. Author: profiler agent. Material: MoS2 3×3, COHSEX
(`runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex_round8_baseline_2026-05-14`
and `*_postfix_*`). Mesh: 4× A100-SXM4-40GB (2×2). Source: `sources/lorrax_B`
branch `agent/zeta-bc-scan-shardmap`.

## 1. Summary

(filled after Phase 3.)

## 2. Baseline (Phase 1) — `c796420` HEAD

Run directory:
`runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex_round8_baseline_2026-05-14`.

Two runs were performed: run 1 hit a `dipole.h5 not found` head-correction
crash *after* V_q completed (V_q timings still valid); run 2 with dipole.h5
symlinked completed end-to-end. Numbers below are from run 2 unless noted.

### 2.1 V_q stage diagnostics

| Metric                                | Value |
|---------------------------------------|-------|
| Centroid orbit closure                | **FAILED** (588/1280 sym maps land outside centroid table) |
| IBZ vs full-BZ fallback (Issue 0)     | **full-BZ** — all 9 q-points iterated |
| `n_q_ibz` reported                    | 9 (= full 3×3×1 BZ) |
| `ngkmax`                              | 1963 (1 G-chunk per q) |
| `n_rmu_L = n_rmu_R`                   | 640 → 640 (no pad) |
| Same-ζ optimization                   | yes (charge channel) |
| Pre-read all 9 IBZ ζ̃ slabs           | 0.20 s |
| q=0 kernel (compile-dominated)        | 0.69 s |
| q=1…8 kernel each                     | 0.00–0.02 s (sum ≈ 0.06 s) |
| Σ V_q kernel wall (per-q only)        | ≈ 0.75 s |
| V_q_compute (gw_jax timing line)      | **1.176 s** |
| `Involuntary full rematerialization` warnings (run 2, JAX cache warm) | 6 |
| `Involuntary full rematerialization` warnings (run 1, cold) | 8 |
| All remats source                     | `v_q_g_flat.py:95` (the `zeta_L_3d[0]` / `zeta_R_3d[0]` combined slice+reshard — Issue 1) |
| Remat target shape                    | `c128[1,640,1963]` ≈ 19.2 MiB/dev |
| HLO `with_sharding_constraint` count (post-XLA-opt, per-q kernel module) | 5 |
| Per-q kernel temp peak (HLO mem report) | 38.34 MiB/dev |
| `V_q=0` trace (numerical fingerprint) | 841273035.5346 |

### 2.2 End-to-end timing (run 2, JAX cache warm)

```
gw_jax.load_centroid_wfns        2.906 s  (18.4 %)
gw_jax.zeta_fit_chunked          8.501 s  (53.8 %)
gw_jax.V_q_compute               1.176 s  ( 7.4 %)
gw_jax.wavefunction_setup        0.033 s
gw_jax.chi0_W                    1.618 s  (10.2 %)
gw_jax.sigma                     1.558 s  ( 9.9 %)
Total recorded                  15.792 s
```

### 2.3 Σ_X bit-identity reference

`Bare Σ_X diagonal (eV), k=0`:

```
-40.0279  -40.0279  -33.8689  -33.8689  -33.3545  -33.3545  -33.4930  -33.4930
```

(printed by `gw_jax.py:330`; degenerate-set averaged via
`average_within_degenerate_sets`; the doubly-degenerate pattern is the
expected bispinor-like spin pairing for charge mode here.)

### 2.4 Observations

- **Issue 0 fires at MoS2 3×3** (contrary to the `v_q_fix_plan` "bonus"
  question — closure also fails for the 640-centroid table here, so the
  full-BZ fallback is exercised). The fallback is ~9 q vs whatever an
  orbit-closed IBZ would yield; at the C3v + mirror MoS2 3×3 we'd expect
  IBZ ≈ 3 q (one Γ + two general). Speedup multiplier from Issue 0 alone
  at MoS2 scale: ~3×.
- **Issue 1 fires at MoS2 3×3** — 6–8 remat warnings at `v_q_g_flat.py:95`.
  Each remat is 19.2 MiB/dev (tiny here) but at CrI3 6×6 80 Ry the same
  slot is 14.4 GB/dev, where it dominates compile time.
- **V_q_compute = 1.18 s** is small compared to fit_zeta = 8.5 s; the
  MoS2 baseline profile is **not load-balanced toward V_q**. The fix
  payoff will be visible as a 2–4× reduction of an already-cheap stage,
  not the dramatic minutes→seconds payoff seen at CrI3 scale. Wall-time
  speedup factor on MoS2 will undersell the CrI3 payoff.

## 3. Post-fix profile (Phase 3) — `<commit>` HEAD

(filled after orchestrator commits land.)

## 4. Bit-identity verification

(filled after Phase 3; expectation per `v_q_fix_plan` §Test plan: bit-equal,
NOT ULP-class drift.)

## 5. Per-stage attribution

(filled after Phase 3.)

## 6. Surprises

(filled after Phase 3.)

## 7. Recommendation for CrI3 re-validation

(filled after Phase 3.)
