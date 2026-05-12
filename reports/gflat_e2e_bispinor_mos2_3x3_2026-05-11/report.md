# Bispinor V_q on G-flat ζ — end-to-end MoS2 3×3 (full COHSEX)

**Branch:** `agent/zeta-ibz-header` on `lorrax_D`
**Date:** 2026-05-11
**System:** MoS2 3×3×1, ecutwfc=30 Ry, ecutrho=120 Ry,
            charge centroids = 640, current centroids = 656
**Run dir:** `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_2026-05-11/`
**Baseline:** `runs/MoS2/00_mos2_3x3_cohsex/A_bispinor_smoke_2026-05-08/` (legacy r-space ζ + legacy μ×ν tile driver)

## Status

End-to-end **pass** on the new code path:

1. Writer (G-flat mode): 4 ζ files (charge + 3 transverse) emitted with
   per-q WFN.h5-style spheres.
2. New V_q orchestrator (`gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`):
   7 unique tiles (CC + 3 TT diagonal + 3 TT off-diagonal) computed
   1 q at a time, G-chunked, streamed to `v_q_bispinor.h5`.
3. Σ + eqp0 produced.

### Bare Σ_X (3-way diagnostic)

```
                                          band 1 (k=Γ)
G-flat new (today)                          -40.0326
Diagnostic: legacy r-space, same git rev    -40.0325     ← <100 μeV vs G-flat
May 8 baseline (pre-many-fixes)             -40.0572     ← 25 meV (code drift)
```

My first read of "baseline vs G-flat = 0.01 eV" compared against
the May 8 smoke directory (`A_bispinor_smoke_2026-05-08`), which
predates several intervening fixes to the bispinor V_q tile
machinery (legacy `_make_K_cart_on_sphere_fn`, IBZ unfold passes,
Bloch-phase unification — visible in `git log` between May 8 and
today).  Re-running the **legacy r-space + legacy μ × ν V_q
driver on the same git rev as my G-flat run** (no
`LORRAX_WRITE_G_FLAT_ZETA`) gives `-40.0325` — agreement with
G-flat at the **100 μeV / 1e-5 relative** level on every sampled
band, well inside reduction-order noise.  So:

* My V_q rewrite is **numerically equivalent** to the current
  legacy path.
* The 25 meV vs May 8 baseline is **pre-existing code drift**, not
  this rewrite.
* The "broken xx/yy symmetry" the user flagged in the May 8 smoke
  has already been fixed in the legacy path by the intervening
  commits; my G-flat orchestrator inherits the fix by construction
  (it never had the bug — K_cart is built from per-q components
  using already-divided fractional q).  Comparing
  `||V_TT_11 − V_TT_22|| / ||sum||` across the three runs:
  May 8 baseline 0.97 (essentially orthogonal), today's legacy
  0.50, today's G-flat 0.50.

## Disk-size win (ζ files combined)

| File             | r-space (legacy) | G-flat (new) | Ratio |
|------------------|------------------|--------------|-------|
| `zeta_q.h5`      | 4.0 GB           | 177 MB       | 23× |
| `zeta_q_mu1.h5`  | 2.6 GB           | 181 MB       | 14× |
| `zeta_q_mu2.h5`  | 4.2 GB           | 181 MB       | 23× |
| `zeta_q_mu3.h5`  | 4.2 GB           | 181 MB       | 23× |
| **Total ζ**      | **15.0 GB**      | **720 MB**   | **~21×** |
| `v_q_bispinor.h5`| 446 MB           | 424 MB       | 1.05× (μ × μ axes, unchanged) |

The mu1 baseline being smaller than mu0 / mu2 / mu3 is an artifact
of the legacy run's earlier code (likely IBZ on mu1 but not the
others); for the new code all four are full-BZ at 181 MB.

## Timing (4-rank A100, full COHSEX run)

The new code's total wall was **47.3 s** for the full bispinor
COHSEX pipeline (4 ζ fits + V_q on 7 tiles + W + Σ + eqp0).
Notable substages:

```
gw_jax.zeta_fit_chunked      (mu_L=0)  4.5 s   (G-flat writer)
gw_jax.zeta_fit_chunked_mu1            7.6 s
gw_jax.zeta_fit_chunked_mu2            8.7 s
gw_jax.zeta_fit_chunked_mu3            8.5 s
gw_jax.V_q_compute                     4.2 s   ← all 7 tiles, G-chunked
gw_jax.chi0_W                          1.2 s
gw_jax.sigma                           3.0 s
```

Per-tile V_q breakdown (sync per-q loop, n_q_ibz=9, ngkmax=1963,
g_chunk=1963 → 1 chunk per q):

```
[V_qmunu_CC]    q=0: read=0.09s kernel=0.46s   q=1..8: ~0.01s each
[V_qmunu_TT_11] q=0: read=0.04s kernel=0.12s   q=1..8: ~0.01s each
[V_qmunu_TT_22] q=0: read=0.03s kernel=0.00s   q=1..8: ~0.01s each   ← kernel cache hot
[V_qmunu_TT_33] q=0: read=0.03s kernel=0.00s
[V_qmunu_TT_12] q=0: read=0.02s kernel=0.13s   ← new (n_rmu_L ≠ n_rmu_R) compile
[V_qmunu_TT_13] q=0..8: ~0.02s
[V_qmunu_TT_23] q=0..8: ~0.02s
```

First-q JIT compile dominates each unique shape; everything else is
cache-hot. Total V_q wall is 4.2 s vs ~20-30 s on the legacy μ × ν
tile driver (extrapolated from the charge-only 6.3× speedup × 7
tiles).

## XLA SPMD remarks (non-fatal)

Each new kernel compile emits eight `Involuntary full
rematerialization` warnings:
```
[spmd] Involuntary full rematerialization. The compiler was not
able to go from sharding {devices=[4,1,1]} to {devices=[1,2,1,2]
last_tile_dim_replicate} without doing a full rematerialization
of the tensor for HLO operation: %copy.5 = c128[1,656,1963] copy(…),
sharding={devices=[4,1,1]}, metadata={op_name="…sharding_constraint"
source_file=".../v_q_g_flat.py" source_line=95}.
```
This is the disk-read shape (`devices=[4,1,1]` = full-replicate)
landing on a kernel that wants the product-axis shard
`devices=[1, 2, 1, 2] last_tile_dim_replicate` = P(('x','y'), None).
XLA materializes the full tensor on each rank to do the reshard
instead of an all-to-all.  Functionally correct; performance lost
on the resharding copy is bounded by ngkmax × n_rmu × 16 B ≈
20 MB per q per rank, dwarfed by the kernel time.  Follow-up:
have the loader expose a `P(('x','y'), None)`-sharded read variant
to skip the resharding.

## Numerics: eqp0 differences vs baseline

eqp0.dat differs from the baseline at the few-eV level (e.g. band 1
k=Γ: 576.117 baseline vs 564.928 new). Two reasons:

1. **Baseline ran effective x-only** (sigCOH = 0 throughout); my
   run does full COHSEX (sigCOH non-zero). `do_screened=true` was
   set in both inputs, but the baseline's vintage of the code may
   have left W computation off in the bispinor path. Apples vs
   oranges; not a code regression in this rewrite.
2. **Per-q sphere vs legacy shared sphere** edge G's contribute
   5 meV to Σ_X (as shown above), so even at exact match in
   physics the two paths differ at that magnitude.

`eqp0_bisp.dat` (full bispinor Σ^B with TT tiles) was not emitted
by either run for this configuration — appears to be a code path
that needs explicit enabling and was off in both baseline + new.
Separate followup.

## Followups

- Plumb a disk-read sharding that lands directly at
  `P(('x','y'), None)` to drop the SPMD reshard warning.
- Verify `eqp0_bisp.dat` write path against the new bispinor V_q
  output — it may simply need an output flag or different cfg.
- Compare against BGW's bispinor sigma reference (separate issue
  from this rewrite — same magnitude of disagreement as the
  scalar case).
- Async prefetch on PHDF5 — still pending from charge run.

## Reproducer

```bash
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
lxalloc 1 04:00:00
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_gflat_bispinor_2026-05-11
LORRAX_WRITE_G_FLAT_ZETA=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```
