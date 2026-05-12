"""Cosine similarity of g_recon vs g_truth at every fine Q (4×4 sub-grid → 8×8 case).

Three views:
  1. per-Q cosine: 640-vector at each Q (shape-as-function-of-μ).
  2. per-μ cosine: 64-vector across Q (shape-as-function-of-Q for each centroid).
  3. element-magnitude distribution: how skewed is |g_μ(Q)|?
"""
import os; os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys; sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
import numpy as np
from psp.finite_q_head_interp import fourier_interpolate_coarse_to_fine

import h5py
HERE="/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"
with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5","r") as f:
    g_8 = f["g0_mu"][...]                            # (8,8,1,μ)
g_4 = g_8[::2, ::2, :, :]
g_recon = fourier_interpolate_coarse_to_fine(g_4, (8,8,1))

def cos(a, b):
    """Complex cosine similarity Re(a·b†) / (‖a‖ ‖b‖)."""
    inner = np.vdot(a.flatten(), b.flatten())
    return inner / (np.linalg.norm(a) * np.linalg.norm(b))


print("=== element-magnitude distribution ===")
mag_8 = np.abs(g_8).flatten()
print(f"  |g_truth|: min {mag_8.min():.3e}, p1 {np.quantile(mag_8,0.01):.3e}, "
      f"median {np.median(mag_8):.3e}, p99 {np.quantile(mag_8,0.99):.3e}, max {mag_8.max():.3e}")
# how is the L2 power distributed across elements?
sq = mag_8 ** 2
sq_sorted = np.sort(sq)[::-1]
cum = np.cumsum(sq_sorted) / sq.sum()
for frac in [0.5, 0.8, 0.9, 0.99]:
    idx = np.searchsorted(cum, frac) + 1
    print(f"  top {idx}/{sq.size} elements ({100*idx/sq.size:.2f}%) carry {frac*100:.0f}% of L2 power")


print("\n=== per-Q cosine similarity (shape-vs-μ at each fine Q) ===")
print(f"{'iq':>3} {'qfrac':>22}  {'on coarse?':>10}  {'|cos|':>9}  "
      f"{'arg(cos)/π':>10}  {'rel ‖·‖_F':>11}")
qfrac_8 = np.array([[i/8, j/8, 0.0] for i in range(8) for j in range(8)]).reshape(8,8,3)
cosines = np.zeros((8,8), dtype=complex); rel_F = np.zeros((8,8))
for ix in range(8):
    for iy in range(8):
        a = g_recon[ix,iy,0]; b = g_8[ix,iy,0]
        c = cos(a, b)
        cosines[ix,iy] = c
        rel_F[ix,iy] = np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)
        if (ix == 0 and iy == 0): continue
        coinc = "  yes" if (ix%2==0 and iy%2==0) else " no "
        if (ix*8+iy) % 4 == 0 or coinc == "  yes":
            qf = qfrac_8[ix,iy]
            print(f"{ix*8+iy:>3} ({qf[0]:5.3f},{qf[1]:5.3f},{qf[2]:5.3f})  {coinc}  "
                  f"{abs(c):>9.4f}  {np.angle(c)/np.pi:>10.5f}  {rel_F[ix,iy]:>11.3e}")

# split coincident vs non-coincident.
mask_coinc = np.zeros((8,8), bool)
mask_coinc[::2, ::2] = True
mask_coinc[0,0] = False                              # exclude Q=0
mask_non = (~mask_coinc) & (np.indices((8,8)).sum(0) > 0)   # non-coinc, non-Q=0
print()
print(f"  coincident (15 pts):     |cos| min {np.abs(cosines[mask_coinc]).min():.6f},  "
      f"mean {np.abs(cosines[mask_coinc]).mean():.6f}")
print(f"  non-coincident (48 pts): |cos| min {np.abs(cosines[mask_non]).min():.4f},  "
      f"mean {np.abs(cosines[mask_non]).mean():.4f},  "
      f"median {np.median(np.abs(cosines[mask_non])):.4f}")


print("\n=== per-μ cosine similarity (shape-vs-Q for each centroid) ===")
# For each μ, take the 8×8 = 64 values of g_truth[:,:,0,μ] vs g_recon[:,:,0,μ].
cos_per_mu = np.array([cos(g_recon[:,:,0,m], g_8[:,:,0,m]) for m in range(g_8.shape[-1])])
print(f"  |cos| over 640 μ:   min {np.abs(cos_per_mu).min():.4f},  "
      f"mean {np.abs(cos_per_mu).mean():.4f},  "
      f"median {np.median(np.abs(cos_per_mu)):.4f},  "
      f"max {np.abs(cos_per_mu).max():.4f}")
print(f"  fraction of μ with |cos| > 0.99: "
      f"{(np.abs(cos_per_mu) > 0.99).sum()}/{cos_per_mu.size}")
print(f"  fraction of μ with |cos| > 0.95: "
      f"{(np.abs(cos_per_mu) > 0.95).sum()}/{cos_per_mu.size}")
print(f"  fraction of μ with |cos| > 0.90: "
      f"{(np.abs(cos_per_mu) > 0.90).sum()}/{cos_per_mu.size}")


# Per-Q dot-products with the dominant μ contribution and overall norm comparison.
print("\n=== norms — does the upscaler get the OVERALL magnitude right? ===")
norm_recon = np.linalg.norm(g_recon, axis=-1)        # (8,8,1)
norm_truth = np.linalg.norm(g_8,    axis=-1)
ratio = norm_recon / np.maximum(norm_truth, 1e-30)
print(f"  per-Q ‖g_recon‖/‖g_truth‖:  min {ratio[mask_non[..., None]].min():.3f}, "
      f"max {ratio[mask_non[..., None]].max():.3f}, "
      f"median {np.median(ratio[mask_non[..., None]]):.3f}")
