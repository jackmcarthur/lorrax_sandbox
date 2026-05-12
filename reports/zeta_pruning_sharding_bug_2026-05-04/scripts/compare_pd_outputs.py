"""Compare pair-density / CCT / L_q across mesh shapes (memory-frugal)."""
import numpy as np, os, sys

D = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_pruning_sharding_bug_2026-05-04/scripts/pd_out"
configs = ["1x1", "2x2", "2x4", "4x4"]

def stream_diff(a_path, b_path, chunk_q=4):
    """‖a-b‖_F, |a-b|_max, ‖a‖_F via memmap (avoid 2GB peak)."""
    a = np.load(a_path, mmap_mode='r')
    b = np.load(b_path, mmap_mode='r')
    # arrays are (nq, n, n) = (64, 1440, 1440) for L_q/C_q, (64, 1440, 1440) for P_k
    nq = a.shape[0]
    sum_d2 = 0.0
    sum_a2 = 0.0
    max_d = 0.0
    max_a = 0.0
    for i in range(0, nq, chunk_q):
        ach = a[i:i+chunk_q].astype(np.complex128)
        bch = b[i:i+chunk_q].astype(np.complex128)
        d = ach - bch
        sum_d2 += float(np.sum(np.abs(d) ** 2))
        sum_a2 += float(np.sum(np.abs(ach) ** 2))
        m = float(np.abs(d).max())
        if m > max_d: max_d = m
        m = float(np.abs(ach).max())
        if m > max_a: max_a = m
    return np.sqrt(sum_d2), np.sqrt(sum_a2), max_d, max_a

for kind in ["P_k", "C_q", "L_q"]:
    print(f"\n=== {kind} ===", flush=True)
    paths = {c: f"{D}/{kind}_mesh{c}.npy" for c in configs if os.path.exists(f"{D}/{kind}_mesh{c}.npy")}
    if not paths:
        print("  no files"); continue
    ref_label = "1x1" if "1x1" in paths else list(paths)[0]
    ref_path = paths[ref_label]
    print(f"  reference = {ref_label}", flush=True)
    print(f"  {'config':>8s} {'‖Δ‖_F':>13s} {'rel':>11s} {'|Δ|_max':>13s} {'rel_max':>11s}", flush=True)
    for c, p in paths.items():
        d_F, a_F, max_d, max_a = stream_diff(ref_path, p)
        rel = d_F / max(a_F, 1e-30)
        rel_max = max_d / max(max_a, 1e-30)
        print(f"  {c:>8s}  {d_F:13.3e}  {rel:11.3e}  {max_d:13.3e}  {rel_max:11.3e}", flush=True)
