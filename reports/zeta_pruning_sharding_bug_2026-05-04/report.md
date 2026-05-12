# ζ-pruning is sharding-dependent — Bare Σ_X varies by ~3 orders of magnitude with device count

**Status:** open
**Branch where reproduced:** `agent-D/head-wing-fix` (also reproduces on `agent-D/hl-ppm` after cherry-picking 5 commits from main).
**Owner:** unassigned

## TL;DR

The same LORRAX `cohsex.in`, same WFN, same `centroids_frac_*.txt`, same git HEAD, run on **different device counts** (1 node × 4 GPU vs 2 nodes × 8 GPU) produces **different `tmp/zeta_q.h5` files** and therefore different bare exchange. On Si 4×4×4 nosym this swing is **137 meV → 0.05 meV** (2700×). The downstream non-determinism is large enough to be the dominant residual in any GW comparison against BGW.

The user-confirmed working Si recipe (`reports/cohsex_si_444_gamma_agreement_2026-05-02/report.md`, MAE 0.12 meV) only delivers sub-meV agreement on the 2-node configuration. The 1-node configuration silently regresses to ~100 meV, which is misattributed to ISDF-rank or k-grid issues on every system that can't be run on 2 nodes.

For **MoS2 3×3×1**, the 2-node sharding throws at startup (`nky=3, nkz=1` can't be split across an 8-device mesh of `(x:2, y:4)`), so MoS2 is stuck on the worse 1-node ζ — explaining why MoS2 bare-X plateaus at 12–15 meV regardless of centroid count, k-means flavor, or canonical-recipe overlay flags.

## Reproduction

Two existing run dirs differ only in `LORRAX_NNODES`:

```
runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg/      # 1 node / 4 GPU  (May 3 03:39)
runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg_v2/   # 2 nodes / 8 GPU (May 4 00:20)
```

`cohsex.in` is **byte-identical** between v1 and v2 (md5 confirmed). `centroids_frac_1440.txt` is byte-identical. `bgw_vcoul.dat` is byte-identical. WFN and `kih.dat` are symlinked to the same files.

The eqp0.dat differs:

```
v1 k=0 n=0:  sigSX = -17.4143 eV    Δ vs BGW X (-17.1316) = +283 meV
v2 k=0 n=0:  sigSX = -17.1315 eV    Δ vs BGW X            = -0.04 meV
```

Bare-X MAE all-k (1024 (k,n) pairs):
- **v1 (1 node):** 137.26 meV — max\|Δ\| 282.7 meV
- **v2 (2 nodes):** 0.0496 meV — max\|Δ\| 0.21 meV

The static head correction printed by `format_static_head_diagnostics` is **identical** in both runs (same vhead/whead overrides):
```
Σ^X head (occ)     = -1.911133e-01 Ry
Σ^SX head (occ)    = -8.696709e-03 Ry
Σ^(SX-X) head(occ) = +1.824166e-01 Ry
```
So the head is not the source of the swing — the body bare X itself differs by ~280 meV per band.

## What changes between v1 and v2 (besides just the sharding)

`md5sum` over the run dirs:
- `centroids_frac_1440.txt`: **same**
- `bgw_vcoul.dat`: **same**
- `cohsex.in`: **same**
- `eqp0.dat`: **differ**
- `tmp/zeta_q.h5`: **differ** ← root cause
- `tmp/isdf_tensors_1440.h5`: differ (downstream of zeta)
- `tmp/isdf_tensors.rank{0..3}.{x,y}.h5`: differ (downstream)

The ζ tensor itself is sharding-dependent. Same centroids in, different ζ out.

Run banner from gw.out:
```
v1: [lxrun] JID 52389983 · 1 nodes · ... · 1/1 free   (4 GPUs)
v2: [lxrun] JID 52418020 · 2 nodes · ... · 2/2 free   (8 GPUs)
```

The user can confirm v1 was launched with `LORRAX_NNODES=1`, v2 with `LORRAX_NNODES=2`. No other env diff.

## Suspected mechanism

The ζ fit in `src/gw/gw_init.py` (driven from `compute_V_q` → `compute_zeta` path) uses pivoted Cholesky to prune the over-sampled centroid set down to `N_c`. Pivoted-Cholesky pivot selection is sensitive to floating-point ordering of partial sums. With JAX sharding splitting the relevant tensor across `nx · ny` devices, the local-then-global reduction order of `argmax(diag)` and the subsequent rank-1 update changes when device count changes.

If two centroid candidates have nearly-degenerate pivot scores, sharding can flip which one survives. In a 1440-pivot pruning over a thousand-plus-dim space, this can cascade and produce a noticeably different basis — which then gives a different ζ matrix on the same input.

The Si v1/v2 difference is much larger (~280 meV per band) than I would have guessed from "two near-degenerate pivots flipping." The pivoted-Cholesky path may have a bug where some part of the diag/score isn't deterministically reduced (e.g., a `lax.psum` over a non-replicated axis), so the sharding doesn't just permute results — it produces a different pivot sequence entirely.

## MoS2 is the canary that can't run on 2 nodes

MoS2 3×3×1 nosym, attempted with `LORRAX_NNODES=2` (8 GPUs) on `agent-D/head-wing-fix`:

```
ValueError: One of device_put args was given the sharding of
  NamedSharding(mesh=Mesh('x': 2, 'y': 4), spec=PartitionSpec('y',)),
  which implies that the global size of its dimension 0 should be
  divisible by 4, but it is equal to 2 (full shape: (2,))
```

The fail is in the `(x:2, y:4)` mesh layout for an 8-device pool. With nkx=3, nky=3, nkz=1, no axis is divisible by 4, so any flat-q sharding with a y=4 split breaks.

So **MoS2 cannot use the 2-node "good" ζ-fit path** until either (a) the sharding fallback handles small/odd k-grid axes, or (b) the pivoted-Cholesky prune is made sharding-independent.

The downstream consequence: MoS2 bare-X plateau at 12–15 meV on 1 node (D_lorrax_xonly_1600 confirms this with use_bgw_vcoul=true and the canonical recipe). All MoS2 GW comparisons (COHSEX 47 meV all-k, GN-PPM 1.3 eV, HL-PPM 115 meV) inherit at least the bare-X portion of this error.

## What we need from a fix

1. **Pivoted-Cholesky pruning must be sharding-independent.** Same centroids in → same pivots out, regardless of device mesh. The right way is probably to do the prune on a fully-replicated array (all-gather first if necessary, since the full `(n_rmu × n_rmu)` Gram is small), or to use a deterministic tie-break that doesn't depend on local-reduction order.

2. **The `(x:2, y:4)` mesh fallback for small k-grids.** When any of `(nkx, nky, nkz)` is too small, the runtime should fall back to a mesh where the σ-window can fit (e.g., flatten to `(nq=9)` and shard `(8,)` with 1 q on the remainder, or replicate over `y` and only split `x`). The current behavior — throwing in `device_put` — silently locks 2D systems out of the 2-node code path.

3. **Diagnostic: a deterministic-reproducibility test.** A pytest that runs ζ on the same inputs across `(1 GPU, 4 GPU, 8 GPU)` and asserts ζ matches to 1e-12. Right now we only catch this by accident when comparing bare X values across run dirs.

## Where to look in the source

- `src/gw/gw_init.py:554` `compute_V_q` — driver for ζ computation (passes `bgw_v_grid_fn`, calls into zeta build).
- `src/centroid/kmeans_isdf.py` — pivoted-Cholesky prune (oversample → N_c). The `select` step is what's shard-dependent.
- `src/gw/compute_vcoul.py:compute_all_V_q_sharded` — uses ζ from disk; this is downstream.
- `src/file_io/*` — `zeta_q.h5` writer; ζ is the artifact whose md5 differs across device counts.

## Diagnostic data attached

- `runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg/eqp0.dat` (v1, 137 meV MAE)
- `runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg_v2/eqp0.dat` (v2, 0.05 meV MAE)
- `runs/MoS2/02_mos2_3x3_nosym/D_lorrax_xonly_1600/` — 1-node MoS2 (12.25 meV plateau)
- `runs/MoS2/02_mos2_3x3_nosym/07_lorrax_xonly_bgwhead/` — older 1-node MoS2 with 640c (9.05 meV; lower N_c happens to land on a luckier pivot set)

The Si v1 vs v2 dirs are the cleanest reproducer — same code, same input, only `LORRAX_NNODES` differs.

## Why this wasn't caught earlier

The Si canonical recipe was always run on 2 nodes (the dev's default), so the 137 meV degradation on 1 node never surfaced. Si is also one of the few systems whose k-grid (4×4×4 = 64 q-points) shards cleanly onto an 8-device mesh, so we never hit the MoS2-style ValueError. Other systems silently fall back to 1 node (or refuse to launch at 2 nodes) and accept bare-X residuals at the 10s-of-meV level.

## Side note: `bare_coulomb_cutoff = ecutwfc` default

Commit `cbda1f6` (May 1) made `bare_coulomb_cutoff` default to `ecutwfc` (was `4·ecutwfc`). Recipes that explicitly set the cutoff (Si and MoS2 canonical) are unaffected. But anyone relying on defaults pre-May-1 will see additional shifts on top of the sharding bug. Mention in the fix because future debugging may collide with this.
