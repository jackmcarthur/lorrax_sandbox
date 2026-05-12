"""8×8 → 12×12 g_μ(q) interpolation via explicit Fourier evaluation.

Unlike 4×4→8×8, the 8 / 12 ratio isn't integer, so spectral-zero-pad
upscaling doesn't apply directly.  Use the bandlimited DFT sum:

    coarse on N_c grid → R-space coefficients via ifftn
    g(q_continuous) = Σ_R c_R · exp(2πi q · R)

evaluated at the 144 fine q-points of the 12×12 grid.

This MIXES two error sources:
  - bandlimit aliasing of g0(q) past 8×8 Nyquist (predicted from
    the 12×12 R-spectrum at |R| > 4: √0.0206 ≈ 14 %).
  - ζ-fit difference between 8×8 and 12×12 LORRAX runs at *coincident*
    physical q's (= the 4×4 sub-grid of points where both grids sample;
    earlier measured ~7 % rel L2).

At the 16 coincident q's (q = k/4), the upscaler returns the 8-grid
sample exactly (round-trip identity, machine precision); the residual
vs 12-grid TRUTH at those points is purely the ζ-fit difference.

Same bucketed-by-magnitude diagnostic as the 4×4-subgrid test.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
import numpy as np
import h5py


def fourier_interp_via_lcm_pad(coarse, target_grid):
    """8 × 8 × 1 → 12 × 12 × 1 by routing through 24 × 24 × 1 (LCM=24).

    Spectral-pad coarse 8×8 → 24×24 (integer ratio 3 — uses the tested
    fourier_interpolate_coarse_to_fine), then sub-sample every other
    point along axis 0 and 1 to land on the 12×12 grid (q = j/12 =
    2j/24 hits indices 0, 2, 4, …, 22 of the 24-grid).
    """
    from psp.finite_q_head_interp import fourier_interpolate_coarse_to_fine
    Nx_c, Ny_c, Nz_c = coarse.shape[:3]
    Nx_f, Ny_f, Nz_f = target_grid
    # LCM trick only works for the (8, 12) case here.
    assert (Nx_c, Nx_f) == (8, 12) and (Ny_c, Ny_f) == (8, 12) and (Nz_c, Nz_f) == (1, 1), \
        "this helper is hard-coded for 8×8 → 12×12"
    padded = fourier_interpolate_coarse_to_fine(coarse, (24, 24, 1))   # (24, 24, 1, *T)
    return padded[::2, ::2, :, ...]                                       # → (12, 12, 1, *T)


HERE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"

with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5", 'r') as f:
    g_8 = f["g0_mu"][...]                              # (8, 8, 1, n_μ)
with h5py.File(f"{HERE}/dense_12x12/tmp/zeta_q.h5", 'r') as f:
    g_12 = f["g0_mu"][...]                             # (12, 12, 1, n_μ)
n_mu = g_8.shape[-1]
print(f"shapes: g_8 {g_8.shape}, g_12 {g_12.shape}")

# Interpolate via LCM padding (8 → 24 → sub-sample to 12).
g_recon = fourier_interp_via_lcm_pad(g_8, (12, 12, 1))                 # (12, 12, 1, n_μ)
err = g_recon - g_12

# Round-trip sanity at coincident q (= 4×4 sub-grid: q = k/4 means index 3k in 12-grid).
# 8-grid samples q = m/8 = 2m/16; 12-grid samples q = n/12. Coincident: m/8 = n/12 ⇔ 3m = 2n.
# Smallest non-trivial: m=2, n=3.  So 8-grid index (2,2) coincides with 12-grid (3,3) at q=(1/4, 1/4).
# Generally q = k/4 hits both: 8-grid index 2k, 12-grid index 3k.  k = 0..3 → 4×4 = 16 coincident.
print("\n=== upscale round-trip check at 16 coincident q's (q = k/4) ===")
roundtrip_err = []
isdf_diff = []
for kx in range(4):
    for ky in range(4):
        v8       = g_8 [2*kx, 2*ky, 0, :]            # truth at q=k/4 from 8-grid run
        v_recon  = g_recon[3*kx, 3*ky, 0, :]         # upscale evaluated at the same q
        v12      = g_12[3*kx, 3*ky, 0, :]            # truth at q=k/4 from 12-grid run
        roundtrip_err.append(np.linalg.norm(v_recon - v8) / max(np.linalg.norm(v8), 1e-30))
        isdf_diff.append   (np.linalg.norm(v8     - v12) / max(np.linalg.norm(v12), 1e-30))
print(f"  rt err (upscale↔coarse-truth):   max {max(roundtrip_err):.2e}  (machine eps if upscaler clean)")
print(f"  ISDF-fit difference 8↔12 at coincident q:  mean {np.mean(isdf_diff):.3f}, max {max(isdf_diff):.3f}")


# ─── Bucketed magnitude analysis on NON-coincident 12-grid q's ─────────────
# 144 - 16 = 128 non-coincident q's; minus the Q=0 origin = 127.
mask = np.ones((12, 12), bool); mask[0, 0] = False
mask_coinc = np.zeros((12, 12), bool)
for kx in range(4):
    for ky in range(4):
        mask_coinc[3*kx, 3*ky] = True
mask_non = mask & ~mask_coinc

print("\n=== Magnitude-bucketed rel L2 (non-coincident 12-grid q's, n=127) ===")
flat_t = g_12[mask_non, 0, :].flatten()
flat_e = err [mask_non, 0, :].flatten()
mag = np.abs(flat_t)
order = np.argsort(mag)[::-1]
flat_t_s = flat_t[order]; flat_e_s = flat_e[order]; mag_s = mag[order]
N = mag.size

buckets = [
    ("top  1 %", 0,                int(0.01 * N)),
    ("top 10 %", 0,                int(0.10 * N)),
    ("10–25 %",  int(0.10 * N),    int(0.25 * N)),
    ("25–50 %",  int(0.25 * N),    int(0.50 * N)),
    ("50–75 %",  int(0.50 * N),    int(0.75 * N)),
    ("bot 25 %", int(0.75 * N),    N),
    ("bot 10 %", int(0.90 * N),    N),
]
total_t2 = (mag**2).sum()
total_e2 = (np.abs(flat_e)**2).sum()

print(f"{'bucket':>10} {'count':>8}  {'|b| range':>22}  {'rel L2 within':>15}  "
      f"{'frac of ‖truth‖²':>18}  {'frac of ‖err‖²':>16}")
for label, lo, hi in buckets:
    rel = np.linalg.norm(flat_e_s[lo:hi]) / max(np.linalg.norm(flat_t_s[lo:hi]), 1e-30)
    ft  = (np.abs(flat_t_s[lo:hi])**2).sum() / total_t2
    fe  = (np.abs(flat_e_s[lo:hi])**2).sum() / total_e2
    print(f"{label:>10} {hi-lo:>8}  [{mag_s[hi-1]:>7.2e}, {mag_s[lo]:>7.2e}]  "
          f"{rel:>15.3e}  {ft:>18.4f}  {fe:>16.4f}")
print(f"\n  total rel ‖err‖_F / ‖truth‖_F (non-coincident 12-grid q) = {np.sqrt(total_e2/total_t2):.3e}")


# ─── Per-Q cosine + norm ratio ─────────────────────────────────────────────
print("\n=== per-Q diagnostics (non-coincident 12-grid q) ===")
def cos_q(a, b):
    return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
coses = []; norm_ratio = []
for ix in range(12):
    for iy in range(12):
        if not mask_non[ix, iy]: continue
        a = g_recon[ix, iy, 0]; b = g_12[ix, iy, 0]
        coses.append(abs(cos_q(a, b)))
        norm_ratio.append(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))
coses = np.array(coses); norm_ratio = np.array(norm_ratio)
print(f"  per-Q ‖recon‖/‖truth‖:  median {np.median(norm_ratio):.4f}, "
      f"p10 {np.quantile(norm_ratio, 0.1):.3f}, p90 {np.quantile(norm_ratio, 0.9):.3f}")
print(f"  per-Q |cos|:            median {np.median(coses):.4f}, "
      f"mean {coses.mean():.4f}, p10 {np.quantile(coses, 0.1):.3f}")


# ─── Per-element rel err ───────────────────────────────────────────────────
mask_finite = mag > 1e-3
per_elem = np.abs(flat_e[mask_finite]) / mag[mask_finite]
print(f"\n=== per-element rel err (|b|>1e-3, n={len(per_elem)}) ===")
print(f"  median {np.median(per_elem):.3e}, p10 {np.quantile(per_elem, 0.1):.3e}, "
      f"p90 {np.quantile(per_elem, 0.9):.3e}")
for thr in [0.05, 0.10, 0.25, 0.50]:
    print(f"  fraction < {thr*100:>4.0f} %: {(per_elem < thr).mean():.3f}")


# ─── Bandlimit prediction for context ──────────────────────────────────────
print("\n=== bandlimit prediction (12×12 R-spectrum past 8×8 Nyquist = |R|>4) ===")
R_grid = np.fft.ifftn(g_12, axes=(0, 1))
fx, fy = np.meshgrid(np.fft.fftfreq(12), np.fft.fftfreq(12), indexing='ij')
fmag = np.round(np.sqrt(fx**2 + fy**2) * 12).astype(int)
pwr = (np.abs(R_grid)**2).sum(axis=tuple(range(2, R_grid.ndim)))
total = pwr.sum()
past = (fmag > 4).sum() if fmag.max() > 4 else 0
frac_past = pwr[fmag > 4].sum() / total
print(f"  fraction of L2 past 8×8 Nyquist:  {frac_past:.4f}")
print(f"  predicted upscale rel L2 (bandlimit only): √{frac_past:.4f} = {np.sqrt(frac_past):.3e}")
