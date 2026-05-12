"""Quantify v1 vs v2 zeta_q divergence per q-point.

Hypothesis testing layout:
  - If diff is uniform ~1e-15·|zeta|, it's pure fp noise from kernel choice.
  - If diff is concentrated in a few amplified modes, it's Cholesky instability
    on near-singular C_q — the regime where a 1e-14 ridge could matter.
  - If diff is structurally different (e.g. different sign/magnitude in
    coherent blocks), the algorithm itself is sharding-dependent.
"""
import os, sys, h5py, numpy as np

V1 = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg/tmp/zeta_q.h5"
V2 = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg_v2/tmp/zeta_q.h5"

f1 = h5py.File(V1, "r")
f2 = h5py.File(V2, "r")
z1 = f1["zeta_q"]   # (nq=64, n_rtot=13824, n_rmu=1440) complex128
z2 = f2["zeta_q"]
g01 = f1["g0_mu"][:]
g02 = f2["g0_mu"][:]
print(f"zeta shape: {z1.shape}, dtype: {z1.dtype}")
print(f"g0 shape:   {g01.shape}\n")

# --- Per-q diff scan (read one q at a time to keep memory bounded) ---
nq = z1.shape[0]
header = f"{'q':>3s} {'|z1|_F':>11s} {'|z1-z2|_F':>11s} {'rel_F':>10s} {'|d|_max':>10s} {'rel_max':>10s} {'med_rel':>10s} {'p99_rel':>10s}"
print(header)
print("-" * len(header))

# Track q-points with the largest relative diff for follow-up.
worst_q = []
for q in range(nq):
    a = z1[q, :, :]
    b = z2[q, :, :]
    d = a - b
    nrm_a = np.linalg.norm(a)
    nrm_d = np.linalg.norm(d)
    rel_F = nrm_d / max(nrm_a, 1e-30)
    abs_d = np.abs(d)
    abs_a = np.abs(a)
    max_d = abs_d.max()
    rel_max = max_d / max(abs_a.max(), 1e-30)
    # Element-wise relative diff (avoid div by tiny |a|)
    floor = abs_a.max() * 1e-15
    pw_rel = abs_d / np.maximum(abs_a, floor)
    med_rel = float(np.median(pw_rel))
    p99_rel = float(np.percentile(pw_rel, 99))
    print(f"{q:3d} {nrm_a:11.3e} {nrm_d:11.3e} {rel_F:10.3e} "
          f"{max_d:10.3e} {rel_max:10.3e} {med_rel:10.3e} {p99_rel:10.3e}")
    worst_q.append((rel_F, q))

worst_q.sort(reverse=True)
print(f"\nTop-5 worst q (by Frobenius-relative diff):")
for r, q in worst_q[:5]:
    print(f"  q={q:2d}  rel={r:.3e}")

print(f"\ng0_mu Frobenius-relative diff: "
      f"{np.linalg.norm(g01 - g02) / max(np.linalg.norm(g01), 1e-30):.3e}")

# Quick distributional sanity: histogram of element-wise rel diff at q=0.
print("\nq=0 element-wise diff histogram:")
a = z1[0]; b = z2[0]; d = a - b
abs_a = np.abs(a); abs_d = np.abs(d)
floor = abs_a.max() * 1e-15
pw_rel = abs_d / np.maximum(abs_a, floor)
edges = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2]
for lo, hi in zip(edges[:-1], edges[1:]):
    n = int(((pw_rel >= lo) & (pw_rel < hi)).sum())
    pct = 100.0 * n / pw_rel.size
    print(f"  [{lo:.0e}, {hi:.0e}): {n:11d}  ({pct:5.2f}%)")
print(f"  >= {edges[-1]:.0e}: {int((pw_rel >= edges[-1]).sum())}")

f1.close(); f2.close()
