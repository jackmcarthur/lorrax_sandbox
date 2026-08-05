#!/usr/bin/env python3
"""CrI3 6x6 80Ry GN-PPM — planner OOM-vs-nband prediction (mu = 10*nband).

Drives the REAL gw.gflat_memory_model.plan_gflat_chunks (current main) for
non-bispinor and bispinor on the 4x4 GPU mesh, budget 70 GB/dev (= cohsex.in
memory_per_device_gb; ~10 GB/dev reserved for NCCL pool + CUDA context on an
80 GB A100), GPU pair_density_slots=3.

Two V_q configs per mode:
  - "production"  : use_ibz_T=False, n_q_ibz=None  -> Peak E at full BZ (=36).
                    Exactly what gw_init.py:606 passes; CONSERVATIVE (the planner
                    over-predicts Peak E by ~36/8 because the runtime V_q cascade
                    is IBZ-only).
  - "ibz-aware"   : use_ibz_T=True,  n_q_ibz=8     -> Peak E at IBZ (=8).
                    Reflects the runtime IBZ cascade (orbit-closed centroids).
The true OOM nband is bracketed between the two; the live sweep pins it.
"""
from types import SimpleNamespace
from gw.gflat_memory_model import plan_gflat_chunks

NGKMAX = 59990          # 80 Ry CrI3 6x6 (production WFN; test_planner_refit_2026-05-17)
N_RTOT = 1_125_000
FFT    = (75, 75, 200)
NK     = 36             # full BZ; planner n_q_disk = nk_tot
NQ_IBZ = 8              # IBZ q-count (P-3, 6 spatial ops, has inversion)
MESH   = SimpleNamespace(shape={'x': 4, 'y': 4})
BUDGET = 70.0

def roundup(x, m):
    return ((x + m - 1) // m) * m

def meta_for(mu):
    return SimpleNamespace(nk_tot=NK, nspinor=2, n_rmu=mu, n_rmu_padded=mu,
                           n_rtot=N_RTOT, ngkmax=NGKMAX, fft_grid=FFT)

def plan_for(nband, bispinor, use_ibz_T):
    mu = roundup(10 * nband, 16)
    return plan_gflat_chunks(
        meta=meta_for(mu), mesh_xy=MESH, nb_total=nband,
        ngkmax=NGKMAX, n_q_disk=NK, budget_gb=BUDGET,
        target_utilization=0.80, fft_box_factor=4.0,
        # let slots resolve from backend (GPU) exactly as production does —
        # bispinor transverse channels give a steeper Peak C slope than charge
        is_bispinor=bispinor, max_chunks=64,
        use_ibz_T=use_ibz_T, n_q_ibz=(NQ_IBZ if use_ibz_T else None))

NBANDS = [80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 798]

print(f"# CrI3 6x6 80Ry GN-PPM planner prediction | mesh 4x4 | budget {BUDGET:.0f} GB/dev | slots=3 | mu=10*nband")
for bispinor in (False, True):
    for use_ibz_T in (False, True):
        vq = "ibz-aware (n_q_ibz=8)" if use_ibz_T else "production/conservative (fullBZ)"
        print("\n" + "=" * 96)
        print(f"  {'BISPINOR' if bispinor else 'NON-BISPINOR'}   |   V_q = {vq}")
        print("=" * 96)
        print(f"{'nband':>6} {'mu':>6} {'r_chunk':>9} {'nrc':>4} {'bc':>4} {'HWM_GB':>8} {'fit<=70':>8}  bottleneck")
        maxfit = None
        for nb in NBANDS:
            try:
                p = plan_for(nb, bispinor, use_ibz_T)
                hwm = p.hwm_bytes / 1e9
                fit = hwm <= BUDGET
                if fit:
                    maxfit = nb
                print(f"{nb:>6} {roundup(10*nb,16):>6} {p.r_chunk:>9} {p.n_r_chunks:>4} "
                      f"{p.band_chunk:>4} {hwm:>8.2f} {('Y' if fit else 'N'):>8}  {p.bottleneck}")
            except Exception as e:
                print(f"{nb:>6} {roundup(10*nb,16):>6}  PLAN-FAIL: {type(e).__name__}: {e}")
        print(f"  --> max nband fitting <= {BUDGET:.0f} GB/dev: {maxfit}")

# Per-peak breakdown at a representative near-cliff point (production-conservative)
for bispinor, nb in ((False, 500), (True, 300)):
    print("\n" + "#" * 96)
    print(f"# PER-PEAK BREAKDOWN  {'BISPINOR' if bispinor else 'NON-BISPINOR'}  "
          f"nband={nb} mu={roundup(10*nb,16)}  (V_q=production/fullBZ conservative)")
    print("#" * 96)
    print(plan_for(nb, bispinor, False).format())
