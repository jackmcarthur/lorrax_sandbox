"""Combined metrics for the 4×4-sub-grid → 8×8 g_μ upscale."""
import os; os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys; sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
import numpy as np, h5py
from psp.finite_q_head_interp import fourier_interpolate_coarse_to_fine

HERE="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"
with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5","r") as f:
    g_8 = f["g0_mu"][...]                          # (8,8,1,μ)
g_4 = g_8[::2, ::2, :, :]
g_recon = fourier_interpolate_coarse_to_fine(g_4, (8,8,1))
err   = g_recon - g_8

# Mask out Q=0 (g0 not meaningful at q=0 in the same convention).
mask_Q0 = np.ones((8,8,1), dtype=bool); mask_Q0[0,0,0] = False
# Mask coincident points (every-other) — they're machine-precision by construction.
ix, iy = np.indices((8,8))
mask_non = mask_Q0[..., 0] & ~((ix % 2 == 0) & (iy % 2 == 0))   # (8,8)
mask_non = mask_non[..., None]                                    # (8,8,1)

# Restrict to non-coincident, non-Q=0 elements.
g_truth_nz = g_8[mask_non[..., 0]]                                # (48, μ)
err_nz     = err[mask_non[..., 0]]


# ─── (1) Norm ratio per Q (size only) ──────────────────────────────
print("=== (1) per-Q norm ratio ‖g_recon‖/‖g_truth‖ (non-coincident, Q ≠ 0) ===")
nrm_t = np.linalg.norm(g_8, axis=-1)        # (8,8,1)
nrm_r = np.linalg.norm(g_recon, axis=-1)
ratio = (nrm_r / np.maximum(nrm_t, 1e-30))[mask_non]
print(f"  median {np.median(ratio):.4f},  p10 {np.quantile(ratio, 0.1):.3f},  "
      f"p90 {np.quantile(ratio, 0.9):.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}")


# ─── (2) per-Q cosine (already done; recap) ─────────────────────────
print("\n=== (2) per-Q cosine similarity (direction only) ===")
def cos_q(ix,iy):
    a = g_recon[ix,iy,0]; b = g_8[ix,iy,0]
    return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
coses = np.array([abs(cos_q(ix, iy)) for ix in range(8) for iy in range(8)
                   if mask_non[ix, iy, 0]])
print(f"  |cos|: median {np.median(coses):.4f}, mean {coses.mean():.4f}, "
      f"p10 {np.quantile(coses, 0.1):.3f}, p90 {np.quantile(coses, 0.9):.3f}")


# ─── (3) Bucketed rel L2: where does the 41 % error come from? ──────
print("\n=== (3) Magnitude-bucketed rel L2 ===")
# Element-by-element on all (Q, μ) values among non-coincident Q's:
flat_t = g_truth_nz.flatten()    # (48 * μ,) complex
flat_e = err_nz.flatten()
mag = np.abs(flat_t)
# Sort by |b|.
order = np.argsort(mag)[::-1]    # descending
flat_t_sorted = flat_t[order]
flat_e_sorted = flat_e[order]
mag_sorted    = mag[order]
N = mag.size

# Bucket boundaries by element-count percentile.
buckets = [
    ("top  1 %", 0,                int(0.01 * N)),
    ("top 10 %", 0,                int(0.10 * N)),
    ("10–25 %",  int(0.10 * N),    int(0.25 * N)),
    ("25–50 %",  int(0.25 * N),    int(0.50 * N)),
    ("50–75 %",  int(0.50 * N),    int(0.75 * N)),
    ("bot 25 %", int(0.75 * N),    N),
    ("bot 10 %", int(0.90 * N),    N),
    ("bot  1 %", int(0.99 * N),    N),
]
total_t_norm2 = (mag ** 2).sum()
total_err_norm2 = (np.abs(flat_e) ** 2).sum()
print(f"{'bucket':>10} {'count':>7}  {'|b| range':>22}  {'rel L2 within':>15}  "
      f"{'frac of ‖truth‖²':>18}  {'frac of ‖err‖²':>16}")
for label, lo, hi in buckets:
    bk_t = flat_t_sorted[lo:hi]
    bk_e = flat_e_sorted[lo:hi]
    bk_mag = mag_sorted[lo:hi]
    rel_in_bk = np.linalg.norm(bk_e) / max(np.linalg.norm(bk_t), 1e-30)
    frac_truth = (np.abs(bk_t)**2).sum() / total_t_norm2
    frac_err   = (np.abs(bk_e)**2).sum() / total_err_norm2
    print(f"{label:>10} {len(bk_t):>7}  [{bk_mag.min():>7.2e}, {bk_mag.max():>7.2e}]  "
          f"{rel_in_bk:>15.3e}  {frac_truth:>18.4f}  {frac_err:>16.4f}")

print(f"\n  total rel ‖err‖_F / ‖truth‖_F = {np.sqrt(total_err_norm2/total_t_norm2):.3e}")


# ─── (4) Per-element rel err distribution ─────────────────────────
print("\n=== (4) per-element rel-err |a−b|/|b| distribution (non-coinc, |b|>1e-3 to avoid noise) ===")
mask_finite_b = mag > 1e-3
per_elem_rel = np.abs(flat_e[mask_finite_b]) / mag[mask_finite_b]
print(f"  count {len(per_elem_rel)}: median {np.median(per_elem_rel):.3e}, "
      f"p10 {np.quantile(per_elem_rel, 0.1):.3e}, "
      f"p90 {np.quantile(per_elem_rel, 0.9):.3e}, "
      f"max {per_elem_rel.max():.3e}")
print(f"  fraction with |a−b|/|b| < 0.10: {(per_elem_rel < 0.10).mean():.3f}")
print(f"  fraction with |a−b|/|b| < 0.25: {(per_elem_rel < 0.25).mean():.3f}")
print(f"  fraction with |a−b|/|b| < 0.50: {(per_elem_rel < 0.50).mean():.3f}")
