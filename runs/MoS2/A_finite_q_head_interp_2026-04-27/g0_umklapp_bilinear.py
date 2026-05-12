"""8×8 → 12×12 g_μ(q) interpolation with umklapp-aware boundary handling.

The issue with global Fourier interpolation: g_μ(q) := z_{q,μ}(G=0) is the
FT of ζ_μ(r) at the *absolute* reciprocal vector q.  As q sweeps within
[0, 1) frac, this samples FT[ζ_μ] over [0, b_x).  Continuing past q=1
into the next BZ corresponds to FT[ζ_μ] at q + b_x, which equals
z_{q,μ}(G = b_x) by the umklapp identity.  In other words, the smooth
continuation of g_μ(q) across the BZ boundary is **not** g_μ(0) (what
Fourier-periodic interpolation assumes) but z_{q=0,μ}(G = b_x).

Minimal fix:
  1. From the persisted real-space ζ_q,μ(r), compute z_{q,μ}(G) for the
     three extra G-shells we need at the boundary:
        A_x[q_y, μ] = z_{(q_x=0, q_y, 0), μ}(G = b_x)       (8 vals × n_μ)
        A_y[q_x, μ] = z_{(q_x, q_y=0, 0), μ}(G = b_y)
        A_xy[μ]     = z_{(0, 0, 0), μ}(G = b_x + b_y)        (n_μ vals)
     each via a partial FFT (just one (G_x, G_y) bin of the r-FFT).

  2. Bilinear interpolate to the 12×12 grid.  At "interior" cells
     (i_lo + 1 ≤ 7, j_lo + 1 ≤ 7) use plain bilinear in g0.  At the
     11/12 boundary in either x or y, wrap and look up the corner
     value from A_x / A_y / A_xy instead of from the periodic image.

Compare three reconstructions:
  (a) baseline: spectral upscale 8 → 12 (= what we already had, ~25 % rel L2).
  (b) bilinear with periodic wrap (same data, no umklapp shift).
  (c) bilinear with umklapp-aware corner lookup (the proposed fix).
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')

import numpy as np
import h5py

HERE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"

# ── Load ζ_q,μ(r) at the q's we need (boundary edges + Γ) ──────────────────
print("=== loading 8×8 ζ_q,μ(r) at q-edges (q_x=0 row, q_y=0 col, Γ) ===")
with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5", 'r') as f:
    g0_full = f["g0_mu"][...]                                # (8, 8, 1, n_μ) — truth at G=0
    # Flat-q indices: q = (i, j, 0) → flat = i*8 + j.
    # q_x = 0 row: flat = j for j = 0..7.
    zeta_qx0 = f["zeta_q"][0:8, :, :].astype(np.complex128)  # (8, n_FFT, n_μ)
    # q_y = 0 col: flat = i*8 for i = 0..7.
    zeta_qy0 = f["zeta_q"][np.arange(0, 64, 8), :, :].astype(np.complex128)

with h5py.File(f"{HERE}/dense_12x12/tmp/zeta_q.h5", 'r') as f:
    g_12 = f["g0_mu"][...]                                    # (12, 12, 1, n_μ) — truth

with h5py.File(f"{HERE}/fine_8x8/WFN.h5", 'r') as f:
    fft_grid = tuple(int(x) for x in f["mf_header/gspace/FFTgrid"][...])
N_x, N_y, N_z = fft_grid
n_FFT = N_x * N_y * N_z
n_mu = g0_full.shape[-1]
print(f"  fft_grid {fft_grid}, n_μ {n_mu},  zeta_qx0 {zeta_qx0.shape}, zeta_qy0 {zeta_qy0.shape}")


# ── Compute z_{q,μ}(G) at the 3 needed off-Γ G-shells ──────────────────────
# A_x[q_y, μ] = (1/n_FFT) Σ_{r} ζ_{(0, q_y, 0), μ}(r) · e^{-i 2π r_x_idx / N_x}
#             = the (G_x=1, G_y=0, G_z=0) coefficient of the r-FFT.
zeta_qx0_3d = zeta_qx0.reshape(8, N_x, N_y, N_z, n_mu)         # (q_y, r_x, r_y, r_z, μ)
zeta_qy0_3d = zeta_qy0.reshape(8, N_x, N_y, N_z, n_mu)         # (q_x, r_x, r_y, r_z, μ)
phase_x = np.exp(-2j * np.pi * np.arange(N_x) / N_x)            # (N_x,)
phase_y = np.exp(-2j * np.pi * np.arange(N_y) / N_y)            # (N_y,)

print("  computing A_x[q_y, μ]  (r-axis FFT bin (1, 0, 0))...")
# LORRAX convention: g0_mu = Σ_r ζ_q,μ(r) (NO 1/N_FFT factor — verified by direct
# comparison against persisted g0_mu at q=Γ, machine precision).
A_x  = np.einsum('qxyzm,x->qm', zeta_qx0_3d, phase_x)             # (8, n_μ)
print("  computing A_y[q_x, μ]  (r-axis FFT bin (0, 1, 0))...")
A_y  = np.einsum('qxyzm,y->qm', zeta_qy0_3d, phase_y)             # (8, n_μ)
print("  computing A_xy[μ]      (r-axis FFT bin (1, 1, 0))...")
zeta_q00 = zeta_qx0_3d[0]                                          # (N_x, N_y, N_z, n_μ)
A_xy = np.einsum('xyzm,x,y->m', zeta_q00, phase_x, phase_y)
print(f"  A_x {A_x.shape},  A_y {A_y.shape},  A_xy {A_xy.shape}")


# ── Bilinear interpolation, two flavours ───────────────────────────────────
def bilinear_recon(g0, A_x, A_y, A_xy, *, umklapp=True):
    """Bilinear reconstruction of g_μ on the 12×12 fine grid from 8×8.

    If umklapp=True, boundary cells use the umklapp-shifted A_x/A_y/A_xy
    values for corners that fall at i=8 or j=8.  If umklapp=False, they
    use periodic-wrap g0[0, ·] / g0[·, 0] (= the Fourier-periodic image).
    """
    g_out = np.zeros((12, 12, 1, n_mu), dtype=np.complex128)
    for qx_f in range(12):
        Qx_eight = qx_f / 12 * 8                  # fractional position in 8-grid units
        ix_lo = int(np.floor(Qx_eight))            # 0..7 typically; can be 7 with hi=8
        ax = Qx_eight - ix_lo                      # in [0, 1)
        for qy_f in range(12):
            Qy_eight = qy_f / 12 * 8
            iy_lo = int(np.floor(Qy_eight))
            ay = Qy_eight - iy_lo
            result = np.zeros(n_mu, dtype=np.complex128)
            for dx in (0, 1):
                for dy in (0, 1):
                    ix_c, iy_c = ix_lo + dx, iy_lo + dy
                    w = (ax if dx == 1 else 1 - ax) * (ay if dy == 1 else 1 - ay)
                    if ix_c == 8 and iy_c == 8:
                        val = A_xy if umklapp else g0[0, 0, 0]
                    elif ix_c == 8:
                        val = A_x[iy_c % 8] if umklapp else g0[0, iy_c % 8, 0]
                    elif iy_c == 8:
                        val = A_y[ix_c % 8] if umklapp else g0[ix_c % 8, 0, 0]
                    else:
                        val = g0[ix_c, iy_c, 0]
                    result += w * val
            g_out[qx_f, qy_f, 0, :] = result
    return g_out

g_bilin_naive   = bilinear_recon(g0_full, A_x, A_y, A_xy, umklapp=False)
g_bilin_umklapp = bilinear_recon(g0_full, A_x, A_y, A_xy, umklapp=True)


# ── Spectral upscale baseline (already known result) ──────────────────────
from psp.finite_q_head_interp import fourier_interpolate_coarse_to_fine
g_spec_24 = fourier_interpolate_coarse_to_fine(g0_full, (24, 24, 1))
g_spec_12 = g_spec_24[::2, ::2, :, :]


# ── Metrics ─────────────────────────────────────────────────────────────────
def rel_F(recon, truth, mask):
    diff = (recon - truth)[mask, 0, :]
    tru  = truth[mask, 0, :]
    return np.linalg.norm(diff) / np.linalg.norm(tru)

def per_q_metrics(recon, truth, mask):
    coses, ratios = [], []
    for ix in range(12):
        for iy in range(12):
            if not mask[ix, iy]: continue
            a = recon[ix, iy, 0]; b = truth[ix, iy, 0]
            coses.append(abs(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
            ratios.append(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))
    return np.array(coses), np.array(ratios)

# Three masks: all non-Γ; non-coincident-with-4×4-subgrid only;
# only the "boundary strip" (qx_idx=11 or qy_idx=11).
mask_full = np.ones((12,12), bool); mask_full[0,0] = False
mask_coinc = np.zeros((12,12), bool)
for kx in range(4):
    for ky in range(4):
        mask_coinc[3*kx, 3*ky] = True
mask_non = mask_full & ~mask_coinc

# Boundary strip: qx_idx==11 or qy_idx==11.
mask_bnd = np.zeros((12,12), bool)
mask_bnd[11, :] = True; mask_bnd[:, 11] = True
mask_bnd &= mask_full
# Interior (non-coincident, non-boundary).
mask_int = mask_non & ~mask_bnd

print("\n=== rel ‖err‖_F  (Q ≠ 0; subset masks) ===")
for label, mask in [("all non-Γ (143 pts)",  mask_full),
                    ("non-coincident (127)",  mask_non),
                    ("boundary strip (23)",   mask_bnd),
                    ("interior non-coinc (104)", mask_int)]:
    rs = rel_F(g_spec_12,        g_12, mask)
    rb = rel_F(g_bilin_naive,    g_12, mask)
    ru = rel_F(g_bilin_umklapp,  g_12, mask)
    print(f"  {label:>26}:  spectral {rs:.3e},  bilin-wrap {rb:.3e},  bilin-umklapp {ru:.3e}")

print("\n=== per-Q cosine median  (Q ≠ 0) ===")
for label, mask in [("all non-Γ",          mask_full),
                    ("boundary strip",     mask_bnd),
                    ("interior non-coinc", mask_int)]:
    cs = per_q_metrics(g_spec_12,       g_12, mask)[0]
    cb = per_q_metrics(g_bilin_naive,   g_12, mask)[0]
    cu = per_q_metrics(g_bilin_umklapp, g_12, mask)[0]
    print(f"  {label:>22}:  spectral {np.median(cs):.4f},  "
          f"bilin-wrap {np.median(cb):.4f},  bilin-umklapp {np.median(cu):.4f}")


# ── Per-Q breakdown on the boundary strip (where umklapp should matter most) ─
print("\n=== Per-Q rel ‖err‖_F on the 23 boundary-strip q's ===")
print(f"{'qfrac':>22}  {'spectral':>12}  {'bilin-wrap':>12}  {'bilin-umklapp':>15}")
for ix in range(12):
    for iy in range(12):
        if not mask_bnd[ix, iy]: continue
        Qx, Qy = ix/12, iy/12
        rs_ = np.linalg.norm(g_spec_12[ix,iy,0] - g_12[ix,iy,0]) / max(np.linalg.norm(g_12[ix,iy,0]), 1e-30)
        rb_ = np.linalg.norm(g_bilin_naive[ix,iy,0] - g_12[ix,iy,0]) / max(np.linalg.norm(g_12[ix,iy,0]), 1e-30)
        ru_ = np.linalg.norm(g_bilin_umklapp[ix,iy,0] - g_12[ix,iy,0]) / max(np.linalg.norm(g_12[ix,iy,0]), 1e-30)
        print(f"  ({Qx:5.3f}, {Qy:5.3f})       {rs_:>12.3e}  {rb_:>12.3e}  {ru_:>15.3e}")


# ── Sanity: A_x[q_y=0] should match z_{Γ,μ}(G=b_x), independent of "extra" data ──
# z_{q=0,μ}(G=b_x) computed via the same FFT-bin trick from full ζ at Γ.
print("\n=== sanity: A_x[0] = z_{Γ,μ}(G=b_x); compare amplitude with g0_mu(Γ) ===")
print(f"  ‖A_x[0]‖    = {np.linalg.norm(A_x[0]):.3e}")
print(f"  ‖g0_mu(Γ)‖  = {np.linalg.norm(g0_full[0,0,0]):.3e}    "
      f"(the same number persisted in the file)")
print(f"  ‖A_xy‖     = {np.linalg.norm(A_xy):.3e}")
