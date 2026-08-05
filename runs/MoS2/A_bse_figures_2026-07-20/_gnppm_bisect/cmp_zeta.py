import sys, numpy as np, h5py
W = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect/work"
def load(d):
    with h5py.File(f"{W}/{d}/tmp/zeta_q.h5", "r") as f:
        keys = [k for k in f.keys() if getattr(f[k], 'ndim', 0) >= 2]
        return {k: np.asarray(f[k]) for k in keys}
def frob(a, b):
    return float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(a.ravel()), 1e-300))
runs = ["fix_1600_1x1", "fix_1600_2x2", "base_1600_full_2x2", "cur_1600_full"]
data = {r: load(r) for r in runs}
print("datasets:", list(data[runs[0]].keys()))
pairs = [
    ("fix_1x1", "fix_2x2", "fix_1600_1x1", "fix_1600_2x2"),       # mesh-invariance of FIX
    ("fix_2x2", "cusolv_2x2", "fix_1600_2x2", "base_1600_full_2x2"),
    ("fix_2x2", "cusolv_4x4", "fix_1600_2x2", "cur_1600_full"),
    ("cusolv_2x2", "cusolv_4x4", "base_1600_full_2x2", "cur_1600_full"),  # the ORIGINAL drift
    ("fix_1x1", "cusolv_4x4", "fix_1600_1x1", "cur_1600_full"),
]
for kA, kB, dA, dB in pairs:
    print(f"\n== {kA}  vs  {kB} ==")
    for ds in data[dA]:
        if ds in data[dB] and data[dA][ds].shape == data[dB][ds].shape:
            print(f"   {ds:12s} shape={data[dA][ds].shape}  frob-rel={frob(data[dA][ds], data[dB][ds]):.4e}")
