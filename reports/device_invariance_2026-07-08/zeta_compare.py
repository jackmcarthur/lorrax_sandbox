"""Compare transverse/charge zeta files across the three bispinor runs.

A = head_4g   (cusolvermp 2x2, transverse n_rmu=668 unpadded)
B = head_16g  (cusolvermp 4x4, transverse padded 668->672)
C = head_16g_luoff (legacy jnp LU at 16 GPU, padded 668->672)

If cuSolverMp-4x4 LU corrupts: B far from A and C; A ~ C.
If mu-padding corrupts:        B ~ C; both far from A.
Run single-rank (host h5py reads only).
"""
import h5py
import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/Z_memplanner_validation_2026-07-06/B_bispinor"
RUNS = {"A_4g": "head_4g", "B_16g": "head_16g", "C_16g_luoff": "head_16g_luoff"}
FILES = ["zeta_q.h5", "zeta_q_mu1.h5", "zeta_q_mu2.h5", "zeta_q_mu3.h5"]


def load(run, fname):
    path = f"{BASE}/{run}/tmp/{fname}"
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        # take the largest dataset
        dsets = [(k, f[k].size) for k in keys if isinstance(f[k], h5py.Dataset)]
        name = max(dsets, key=lambda t: t[1])[0]
        return name, f[name][...]


for fname in FILES:
    arrs = {}
    for tag, run in RUNS.items():
        try:
            name, a = load(run, fname)
            arrs[tag] = np.asarray(a)
        except Exception as e:
            print(f"{fname} {tag}: LOAD FAIL {e}")
    if len(arrs) < 2:
        continue
    shapes = {t: a.shape for t, a in arrs.items()}
    print(f"\n== {fname} shapes={shapes} dataset={name}")
    tags = list(arrs)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            x, y = arrs[tags[i]], arrs[tags[j]]
            n = min(x.shape[-1], y.shape[-1])  # clip pad on last axis if any
            m = min(x.shape[-2], y.shape[-2])
            xs = x[..., :m, :n]
            ys = y[..., :m, :n]
            num = np.linalg.norm(xs - ys)
            den = np.linalg.norm(xs)
            print(f"  {tags[i]} vs {tags[j]}: rel ||diff|| = {num/den:.3e}  "
                  f"(max abs diff {np.max(np.abs(xs-ys)):.3e})")
