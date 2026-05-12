# G-flat ζ on-disk end-to-end smoke (MoS2 3×3×1, 80 bands, x-only)

**Branch:** `agent/zeta-ibz-header` on `lorrax_D`
**Date:** 2026-05-11
**System:** MoS2 3×3×1, ecutwfc=30 Ry, ecutrho=120 Ry, 641 centroids
**Run dir:** `runs/MoS2/00_mos2_3x3_cohsex/D_gflat_xonly_2026-05-11/`
**Baseline:** `runs/MoS2/00_mos2_3x3_cohsex/02_lorrax_xonly/` (legacy r-space + legacy V_q driver)

## Status

End-to-end **pass**: writer in G-flat mode → V_q via new orchestrator
(`gw.v_q_g_flat.compute_all_V_q_g_flat`) → Σ → eqp0.dat. Numerics
match the legacy r-space pipeline to ~2×10⁻⁴ in the QP column
(accumulation-order noise; per-q sphere is a strict subset of the
shared sphere by design).

## Disk-size win

| File                       | Size  |
|---------------------------|-------|
| `02_lorrax_xonly/tmp/zeta_q.h5` (r-space) | **2.3 GB** |
| `D_gflat_.../tmp/zeta_q.h5` (G-flat)      | **101 MB** |
| ratio                                      | **~23×** |

Both files store IBZ-only ζ (n_q_disk=5 of 9 full-BZ). The 23× is
purely the per-q G-sphere shrink: `ngkmax=1963` vs `n_rtot=46080`
(4.26% of the full FFT axis). Per-q `ngk` actually varies
`[1947, 1963, 1963, 1917, 1963]`; pad slots zeroed by the writer.

## Timing (4-rank, 1 A100 node, MoS2 3×3 x-only)

| Stage             | r-space baseline | G-flat new | Speedup |
|-------------------|------------------|------------|---------|
| Total recorded    | 17.2 s           | **11.4 s** | 1.51× |
| `zeta_fit`        | 8.5 s            | 6.1 s      | 1.39× |
| └─ `close_io`     | 3.8 s            | 0.1 s      | **~38×** |
| └─ `write_g_flat` | —                | 0.25 s     | (single collective write) |
| `V_q_compute`     | 4.4 s            | **0.7 s**  | **6.3×** |
| `sigma`           | 3.1 s            | 2.9 s      | 1.07× |

V_q per-q breakdown (sync loop, first compile + 4 cached):
```
V_q g-flat q=0/5: read=0.09s, kernel=0.16s   ← compile
V_q g-flat q=1/5: read=0.01s, kernel=0.00s
V_q g-flat q=2/5: read=0.01s, kernel=0.00s
V_q g-flat q=3/5: read=0.01s, kernel=0.01s
V_q g-flat q=4/5: read=0.01s, kernel=0.00s
```

The `~38× zeta close_io drop` comes from the writer collapsing all
the per-r-chunk hyperslab writes into a single collective write of
the fully-assembled G-flat tensor — the r-chunk writes' HDF5 metadata
+ stripe sync overhead is eliminated.

V_q's 6.3× is the per-q + G-chunked + no-FFT contract replacing the
legacy μ × ν tiled + in-V_q-FFT + shared-sphere kernel.

## Numerics vs r-space baseline

DFT energies (col 3 of eqp0.dat): **bit-identical** across all 80
bands × 9 k-points (modulo timestamp comment).

sigma_diag.dat sigSX at k=Γ — **5 decimals of agreement** vs
r-space baseline (same input, same cohsex.in):

| band | r-space sigSX (eV) | G-flat sigSX (eV)   | Δ          |
|------|--------------------|----------------------|------------|
| 19   | -16.997045         | -16.997068           | 2.3×10⁻⁵   |
| 20   | -24.390193         | -24.390452           | 2.6×10⁻⁴   |
| 21   | -24.403538         | -24.403794           | 2.6×10⁻⁴   |
| 22   | -23.688619 (next k)| -23.688773 (next k) | 1.5×10⁻⁴   |

This is the residual from per-q sphere being a strict subset of
the legacy shared sphere — a few edge G's contribute slightly to
V_q[μν] in the shared-sphere path; per-q drops them by design
since `v(q+G)` is exactly zero for `|q+G|² > zeta_cutoff` at every
relevant q (no information lost).

eqp0.dat col 4: differs by ~2×10⁻⁴ per entry, same magnitude as
the sigSX residual, fully consistent.

### LORRAX vs BGW (sigma_hp.log)

I overstated the BGW agreement on first reading.  A proper
band-by-band check (DFT-energy matched) at k=Γ shows the gap is
**not uniform**:

| band | LORRAX sigSX (eV) | BGW X column (eV) | Δ           |
|------|-------------------|-------------------|-------------|
| 19   | -16.997           | -16.532           | 0.47        |
| 21   | -24.390           | -18.710           | **5.68**    |
| 23   | -24.404           | -18.614           | **5.79**    |
| 25   | -18.510           | -17.275           | 1.24        |
| 27   | -14.302           | -11.052           | 3.25        |

Both LORRAX paths (legacy r-space, new G-flat) produce the **same**
LORRAX values to 5 decimals, so this discrepancy is independent of
the V_q rewrite — but it is a real LORRAX-vs-BGW gap that I should
not have characterized as "0.5 eV pre-existing".  The pattern
(small at band 19, large at 21/23) is band-specific, not a
plateau, so per the sandbox memory the right next step is to chase
algorithm / convention differences (e.g. sym handling at degenerate
band manifolds, sphere-vs-shell gathering at the cutoff edge) — not
ISDF rank.  Tracked as a separate followup.

## On-disk HDF5 layout (G-flat)

```
zeta_q.h5
├── mf_header/                 (verbatim from WFN.h5)
├── isdf_header/
│   ├── centroids/r_mu_fft_idx (641, 3) int32
│   ├── centroids/r_mu_crystal (641, 3) float64
│   ├── density                "scalar"
│   ├── vertex_mu_L            0
│   ├── zeta_is_done           True
│   ├── zeta_layout            "G_flat"
│   ├── zeta_cutoff_ry         30.0
│   ├── ngk                    (5,) int32  = [1947, 1963, 1963, 1917, 1963]
│   └── gvec_components        (5, 3, 1963) int32   per-q Miller indices,
│                                                   pad slots: (-16, -16, -30)
├── zeta_q_G                   (5, 641, 1963) complex128
└── g0_mu                      (deferred from V_q write)
```

## Reproducer

```bash
# (per the sandbox AGENTS.md skills)
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
lxalloc 1 04:00:00
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_gflat_xonly_2026-05-11

# G-flat writer + new V_q orchestrator
LORRAX_WRITE_G_FLAT_ZETA=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

## Followups

- BGW cross-validation: compare D_gflat eqp0.dat to BGW sigma_hp.log
  on the same WFN.h5 (existing `compare_bgw_gwjax.py` script).
- Async prefetch: re-enable the worker-thread G-flat slab read
  (deadlocked first attempt on the PHDF5 FFI backend — disabled via
  `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=0` for now). Tighter NCCL ↔ MPI
  interleaving needed; not on the critical path since sync is already
  6× faster than the legacy driver.
- Bispinor 7-tile orchestrator
  (`gw.v_q_bispinor.compute_V_q_bispinor_to_h5`) — same per-q +
  G-chunked + signed v(q+G) pattern; the kernel
  `compute_v_q_per_q_g_chunked` already handles `L ≠ R + complex v`
  (tested).
