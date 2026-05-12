"""Where in (R, μ)-space is the v1-v2 diff actually concentrated?

Three diagnostics on q=0:
  1. SVD of ζ(R, μ) — does ζ have a near-null space (small singular values)?
  2. Diff vs |ζ| scatter — are big rel-diffs at small |ζ| (boring) or big |ζ|?
  3. Per-μ column norm of (ζ_v1 - ζ_v2) — is the diff concentrated in a few
     μ columns (which would be the near-null modes of C_q)?
"""
import h5py, numpy as np

V1 = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg/tmp/zeta_q.h5"
V2 = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/D_lorrax_xonly_overlay_1440c_noavg_v2/tmp/zeta_q.h5"

f1 = h5py.File(V1, "r"); f2 = h5py.File(V2, "r")
print("Loading q=0 slabs ...")
a = f1["zeta_q"][0]   # (n_rtot=13824, n_rmu=1440) complex128
b = f2["zeta_q"][0]
d = a - b
print(f"a shape {a.shape} dtype {a.dtype}")

# --- 1. SVD of ζ ---
print("\nSVD of zeta_v1 at q=0 (this is slow, ~1 min) ...")
sv = np.linalg.svd(a, compute_uv=False)
print(f"top-5  σ: {sv[:5]}")
print(f"mid-5  σ: {sv[len(sv)//2-2:len(sv)//2+3]}")
print(f"tail-5 σ: {sv[-5:]}")
print(f"σ_max / σ_min = {sv[0] / max(sv[-1], 1e-300):.3e}")
print(f"#σ < 1e-6 σ_max: {int((sv < 1e-6 * sv[0]).sum())} / {len(sv)}")
print(f"#σ < 1e-10 σ_max: {int((sv < 1e-10 * sv[0]).sum())} / {len(sv)}")
print(f"#σ < 1e-14 σ_max: {int((sv < 1e-14 * sv[0]).sum())} / {len(sv)}")

# --- 2. Where is the diff (in |ζ| space)? ---
abs_a = np.abs(a).astype(np.float64)
abs_d = np.abs(d).astype(np.float64)

print("\nDiff magnitude distribution by |ζ| bin:")
print(f"  {'|ζ| bin':>20s} {'count':>11s} {'med |Δ|':>11s} {'p99 |Δ|':>11s} {'med rel':>11s}")
zmax = abs_a.max()
edges = [0, zmax*1e-12, zmax*1e-9, zmax*1e-6, zmax*1e-3, zmax*1e-1, zmax*2]
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (abs_a >= lo) & (abs_a < hi)
    n = int(mask.sum())
    if n == 0:
        continue
    md = float(np.median(abs_d[mask]))
    pd = float(np.percentile(abs_d[mask], 99))
    rel = abs_d[mask] / np.maximum(abs_a[mask], zmax * 1e-15)
    mr = float(np.median(rel))
    print(f"  [{lo:.2e}, {hi:.2e}) {n:11d} {md:11.3e} {pd:11.3e} {mr:11.3e}")

# --- 3. Per-μ column norm of diff ---
print("\nPer-μ column-norm of diff (top 10 columns):")
col_d = np.linalg.norm(d, axis=0)            # (n_rmu,)
col_a = np.linalg.norm(a, axis=0)            # (n_rmu,)
col_rel = col_d / np.maximum(col_a, 1e-30)
order = np.argsort(-col_rel)[:10]
for mu in order:
    print(f"  μ={mu:5d}  |a_μ|={col_a[mu]:.3e}  |d_μ|={col_d[mu]:.3e}  rel={col_rel[mu]:.3e}")

# --- 4. Per-μ column-norm distribution ---
print(f"\nPer-μ column |a_μ| distribution: "
      f"min={col_a.min():.3e}, median={np.median(col_a):.3e}, max={col_a.max():.3e}")
print(f"#μ with |a_μ| < 1e-10·max: {int((col_a < 1e-10 * col_a.max()).sum())} / {len(col_a)}")
print(f"#μ with |a_μ| < 1e-6·max:  {int((col_a < 1e-6 * col_a.max()).sum())} / {len(col_a)}")

# --- 5. Project diff onto SVD basis of a ---
# This is the cleanest test: if d lives in the small-singular-value subspace,
# C_q is the culprit. If d is uniform across modes, it's just GEMM noise.
print("\nProjecting d onto right-singular vectors of a ...")
# Use just top-k via truncated SVD (full SVD gives 1440 modes, projection cheap)
U, S, Vh = np.linalg.svd(a, full_matrices=False)
# Project d into the right-singular basis: d_proj[mode] = ||d @ V[:, mode]||
# (in the column / right-singular space)
d_modes = d @ Vh.conj().T          # (n_rtot, n_rmu)
mode_norms = np.linalg.norm(d_modes, axis=0)  # (n_rmu,)
# Group modes by σ magnitude
print(f"  {'σ_k bin (rel σ_max)':>22s} {'#modes':>8s} {'sum |d|² in bin':>18s} {'frac of |d|²':>14s}")
total_d_sq = float(np.sum(mode_norms ** 2))
edges_s = [(1, 1e-2), (1e-2, 1e-4), (1e-4, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12), (1e-12, 1e-15), (1e-15, 0)]
sigma_rel = S / S.max()
for lo, hi in edges_s:
    mask = (sigma_rel <= lo) & (sigma_rel > hi)
    n = int(mask.sum())
    if n == 0:
        print(f"  ({hi:.0e}, {lo:.0e}]  {n:8d}  {'(empty)':>18s}  {'-':>14s}")
        continue
    s = float(np.sum(mode_norms[mask] ** 2))
    print(f"  ({hi:.0e}, {lo:.0e}]  {n:8d}  {s:18.3e}  {s/total_d_sq:14.3e}")
print(f"total |d|^2 = {total_d_sq:.3e}; sqrt = {np.sqrt(total_d_sq):.3e}")
print(f"|a|_F = {np.linalg.norm(a):.3e}")

f1.close(); f2.close()
