"""Pure-interpolation test: 4×4 sub-grid of the 8×8 LORRAX run → 8×8.

This isolates **only** the spectral-interpolation step.  We use the 8×8
LORRAX run's data for both the coarse and the fine sample — the exact
same ζ-fit and exact same V_qmunu / W_qmunu construction defines the
ground truth at every q.  A 4×4 sub-grid (every other q in each dir)
is taken as the "coarse data".  The other 48 fine q's are the held-out
truth that interpolation must reproduce.

Reports rel ‖·‖_F errors of three quantities:

  g0_μ(q)         (the FFT(e^{-iq·r} ζ_μ)(G=0) projector)
  V_qmunu(q)      (full centroid Coulomb)
  W^0_qmunu(q,ω)  (screened W from the GW solve)

Each in two variants:

  naive    : Fourier-upscale the quantity directly.
  split    : split off the rank-1 head channel, upscale the body and g
             separately, reapply the analytic v_head(Q)/V_cell · g·g†
             (and analogously for W via the user's Schur-reconstruction
             with the W_head scalar).

Bandlimit prediction: from the 8×8 R-space spectrum of each quantity,
the fraction of L2 power above |freq|=2/N (= 4×4 Nyquist) tells us
the predicted rel L2 upscale error analytically.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

import numpy as np
import h5py

from psp.finite_q_head_interp import (
    v_head_2d_slab,
    fourier_interpolate_coarse_to_fine,
)

HERE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"
FINE = f"{HERE}/fine_8x8"

NK_F = (8, 8, 1)
NK_C = (4, 4, 1)

# ── Load 8×8 truth ─────────────────────────────────────────────────────────
print("=== loading 8×8 LORRAX truth ===")
with h5py.File(f"{FINE}/tmp/zeta_q.h5", 'r') as f:
    g0_8 = f["g0_mu"][...]                           # (8, 8, 1, n_μ)
with h5py.File(f"{FINE}/tmp/isdf_tensors_640.h5", 'r') as f:
    V_8  = f["V_qmunu"][...]                          # (1,1,1, 8,8,1, n_μ, n_ν)
    W_8  = f["W0_qmunu"][...]
    vhead = complex(f["vhead"][()])
    whead = f["whead"][...]                           # (n_omega,)
with h5py.File(f"{FINE}/WFN.h5", 'r') as f:
    bvec  = f["mf_header/crystal/bvec"][...].astype(np.float64)
    blat  = float(f["mf_header/crystal/blat"][()])
    avec  = f["mf_header/crystal/avec"][...].astype(np.float64)
    alat  = float(f["mf_header/crystal/alat"][()])
    cell_volume = float(f["mf_header/crystal/celvol"][()])
bvec_bohr = bvec * blat
z_c = float(avec[2, 2]) * alat * 0.5

V_8 = V_8[0,0,0]                                    # (8, 8, 1, n_μ, n_ν)
W_8 = W_8[0,0,0]
n_mu = g0_8.shape[-1]
print(f"  shapes: g0 {g0_8.shape},  V_qmunu {V_8.shape},  W0_qmunu {W_8.shape}")
print(f"  cell_volume {cell_volume:.3f},  z_c {z_c:.3f},  vhead {vhead.real:.3f}")


# ── 4×4 sub-grid: every other q in x and y ────────────────────────────────
g0_4 = g0_8[::2, ::2, :, :]                         # (4, 4, 1, n_μ)
V_4  = V_8 [::2, ::2, :, :, :]
W_4  = W_8 [::2, ::2, :, :, :]
print(f"\n  coarse subset: g0 {g0_4.shape}, V_qmunu {V_4.shape}")


# ── q-coordinates ──────────────────────────────────────────────────────────
qfrac_8 = np.array([[i/8, j/8, 0.0] for i in range(8) for j in range(8)]).reshape(8, 8, 3)
qfrac_4 = qfrac_8[::2, ::2, :]


# ── Compute v_head(Q) at all 64 fine Q's (analytic, slab-truncated) ────────
def v_head_per_q(qfrac_grid):
    out = np.zeros(qfrac_grid.shape[:-1], dtype=np.float64)
    for idx in np.ndindex(qfrac_grid.shape[:-1]):
        qf = qfrac_grid[idx]
        if np.linalg.norm(qf) < 1e-12:
            out[idx] = np.nan
        else:
            out[idx] = v_head_2d_slab(bvec_bohr.T @ qf, z_c)
    return out

vh_8 = v_head_per_q(qfrac_8)                         # (8, 8, 1)
vh_4 = v_head_per_q(qfrac_4)                         # (4, 4, 1)


# ── Helper: spectral upscale 4×4 → 8×8 ─────────────────────────────────────
def upscale(coarse):
    """Generic spectral upscale (4, 4, 1, *T) → (8, 8, 1, *T)."""
    return fourier_interpolate_coarse_to_fine(coarse, NK_F)


# ── Helper: rel L2 err per fine q vs truth ────────────────────────────────
def err_grid(recon, truth, mask_origin=True):
    """Returns mean / max rel ‖·‖_F per fine q (skips Q=0 if mask_origin)."""
    diffs = []
    for ix in range(8):
        for iy in range(8):
            if mask_origin and ix == 0 and iy == 0:
                continue
            num = np.linalg.norm(recon[ix, iy, 0] - truth[ix, iy, 0])
            den = max(np.linalg.norm(truth[ix, iy, 0]), 1e-30)
            diffs.append(num / den)
    return np.array(diffs)


# ── Bandlimit prediction from 8×8 R-spectrum ───────────────────────────────
def predict_upscale_err(truth_grid):
    """L2 fraction of |truth|² past 4×4 Nyquist (= |freq|/N > 2)."""
    R = np.fft.ifftn(truth_grid, axes=(0, 1))
    fx, fy = np.meshgrid(np.fft.fftfreq(8), np.fft.fftfreq(8), indexing='ij')
    fmag_int = np.round(np.sqrt(fx*fx + fy*fy) * 8).astype(int)
    pwr = (np.abs(R)**2).sum(axis=tuple(range(2, R.ndim)))   # collapse trailing
    total = pwr.sum()
    past = (fmag_int > 2)
    return float(pwr[past].sum() / total)


# ─────────────────────────────────────────────────────────────────────────
# Test 1: g0_μ(Q) interpolation
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 1: g0_μ(Q) — coarse 4×4 sub-grid → fine 8×8 ===")
g0_recon = upscale(g0_4)                             # (8, 8, 1, n_μ)
errs_g = err_grid(g0_recon, g0_8)
frac_past_g = predict_upscale_err(g0_8)
print(f"  rel ‖g_recon − g_truth‖_F (Q ≠ 0):  mean {errs_g.mean():.3e}, max {errs_g.max():.3e}")
print(f"  bandlimit prediction √(power past 4×4 Nyquist) = √{frac_past_g:.4f} = {np.sqrt(frac_past_g):.3e}")


# ─────────────────────────────────────────────────────────────────────────
# Test 2: V_qmunu(Q) — naive vs split
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 2: V_qmunu(Q) — naive vs head-split ===")

# Naive Fourier upscale of V_qmunu directly.
V_naive_recon = upscale(V_4)                         # (8, 8, 1, n_μ, n_μ)
errs_V_naive = err_grid(V_naive_recon, V_8)
print(f"  naive Fourier upscale:")
print(f"    rel err (Q ≠ 0):  mean {errs_V_naive.mean():.3e}, max {errs_V_naive.max():.3e}")

# Split: V_body = V − v_head/V_cell · g·g†.
V_body_8 = V_8.copy()
for ix in range(8):
    for iy in range(8):
        if np.isnan(vh_8[ix, iy, 0]):
            continue
        g = g0_8[ix, iy, 0]
        V_body_8[ix, iy, 0] -= (vh_8[ix, iy, 0] / cell_volume) * np.einsum(
            'm,n->mn', np.conj(g), g)

V_body_4 = V_body_8[::2, ::2, :, :, :]                # (4, 4, 1, n_μ, n_μ)
V_body_recon = upscale(V_body_4)

# Reconstruct V at every fine Q with analytic v_head + interpolated g.
V_split_recon = np.zeros_like(V_8)
for ix in range(8):
    for iy in range(8):
        if np.isnan(vh_8[ix, iy, 0]):
            V_split_recon[ix, iy, 0] = V_body_recon[ix, iy, 0]
            continue
        g = g0_recon[ix, iy, 0]
        V_split_recon[ix, iy, 0] = (V_body_recon[ix, iy, 0]
            + (vh_8[ix, iy, 0] / cell_volume) * np.einsum('m,n->mn', np.conj(g), g))

errs_V_split = err_grid(V_split_recon, V_8)
print(f"  head-split + Fourier upscale (V_body, g) + analytic v_head:")
print(f"    rel err (Q ≠ 0):  mean {errs_V_split.mean():.3e}, max {errs_V_split.max():.3e}")

# Body-only error (how well does V_body itself interpolate?)
errs_V_body = err_grid(V_body_recon, V_body_8)
frac_past_V_body = predict_upscale_err(V_body_8)
print(f"  V_body alone interpolation:")
print(f"    rel err (Q ≠ 0):  mean {errs_V_body.mean():.3e}, max {errs_V_body.max():.3e}")
print(f"    bandlimit prediction √{frac_past_V_body:.4f} = {np.sqrt(frac_past_V_body):.3e}")

frac_past_V = predict_upscale_err(V_8)
print(f"  V_qmunu (no split) bandlimit prediction √{frac_past_V:.4f} = {np.sqrt(frac_past_V):.3e}")


# ─────────────────────────────────────────────────────────────────────────
# Test 3: W^0_qmunu(Q,0) — naive vs Sherman-Morrison head split
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Test 3: W^0_qmunu(Q, ω=0) — naive vs head-split ===")

# Naive.
W_naive_recon = upscale(W_4)
errs_W_naive = err_grid(W_naive_recon, W_8)
print(f"  naive Fourier upscale:")
print(f"    rel err (Q ≠ 0):  mean {errs_W_naive.mean():.3e}, max {errs_W_naive.max():.3e}")

# Split: W_body = W − w_head/V_cell · g·g†.  whead is the static (ω=0) value.
wh = complex(whead[0])
W_body_8 = W_8.copy()
for ix in range(8):
    for iy in range(8):
        if np.isnan(vh_8[ix, iy, 0]):
            # at q=0 the persisted W_qmunu has G=0 zeroed; W_body == W there.
            continue
        g = g0_8[ix, iy, 0]
        # For W there's no analytic singular form pre-known at finite Q;
        # use the persisted whead at q=0 as a constant scalar multiplier.
        # This is "subtract the same rank-1 head as at q=0".
        W_body_8[ix, iy, 0] -= (wh / cell_volume) * np.einsum(
            'm,n->mn', np.conj(g), g)

W_body_4 = W_body_8[::2, ::2, :, :, :]
W_body_recon = upscale(W_body_4)

# Reconstruct W_qmunu by adding the same constant-scalar rank-1 head back.
W_split_recon = np.zeros_like(W_8)
for ix in range(8):
    for iy in range(8):
        if np.isnan(vh_8[ix, iy, 0]):
            W_split_recon[ix, iy, 0] = W_body_recon[ix, iy, 0]
            continue
        g = g0_recon[ix, iy, 0]
        W_split_recon[ix, iy, 0] = (W_body_recon[ix, iy, 0]
            + (wh / cell_volume) * np.einsum('m,n->mn', np.conj(g), g))

errs_W_split = err_grid(W_split_recon, W_8)
print(f"  head-split (rank-1 with constant whead) + Fourier upscale (W_body, g):")
print(f"    rel err (Q ≠ 0):  mean {errs_W_split.mean():.3e}, max {errs_W_split.max():.3e}")

errs_W_body = err_grid(W_body_recon, W_body_8)
frac_past_W_body = predict_upscale_err(W_body_8)
print(f"  W_body alone interpolation:")
print(f"    rel err (Q ≠ 0):  mean {errs_W_body.mean():.3e}, max {errs_W_body.max():.3e}")
print(f"    bandlimit prediction √{frac_past_W_body:.4f} = {np.sqrt(frac_past_W_body):.3e}")
frac_past_W = predict_upscale_err(W_8)
print(f"  W_qmunu (no split) bandlimit prediction √{frac_past_W:.4f} = {np.sqrt(frac_past_W):.3e}")


# ─────────────────────────────────────────────────────────────────────────
# Per-Q breakdown for V and W
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Per-Q breakdown ===")
print(f"{'iq':>3} {'qfrac':>22}  {'on coarse?':>10}  "
      f"{'V naive':>9}  {'V split':>9}  {'W naive':>9}  {'W split':>9}")
ix_lin = 0
for ix in range(8):
    for iy in range(8):
        if ix == 0 and iy == 0:
            print(f"  0 (0.000,0.000,0.000)        Q=0 SKIP")
            continue
        qf = qfrac_8[ix, iy]
        coinc = (ix % 2 == 0 and iy % 2 == 0)
        if not (ix in (0,4) or iy in (0,4) or (ix == iy and ix in (1,3,5,7))) and ix < 4:
            continue   # trim output
ix_print = 0
for ix in range(8):
    for iy in range(8):
        if ix == 0 and iy == 0:
            continue
        qf = qfrac_8[ix, iy]
        coinc = "  yes" if (ix % 2 == 0 and iy % 2 == 0) else " no "
        # find this Q in errs arrays (they skip the (0,0) origin only).
        # err arrays are flat; index = ix*8 + iy − 1 if ix*8+iy > 0 else N/A.
        flat = ix*8 + iy - 1
        if flat < 0:
            continue
        # only print every other Q to keep output compact.
        if ix_print % 4 != 0:
            ix_print += 1; continue
        print(f"{ix*8+iy:>3} ({qf[0]:5.3f},{qf[1]:5.3f},{qf[2]:5.3f})  {coinc}  "
              f"{errs_V_naive[flat]:>9.3e}  {errs_V_split[flat]:>9.3e}  "
              f"{errs_W_naive[flat]:>9.3e}  {errs_W_split[flat]:>9.3e}")
        ix_print += 1


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
print("\n=== Summary (all errors are rel ‖·‖_F, Q ≠ 0, 63 fine pts) ===")
print(f"  g0_μ           upscale 4×4-sub of 8×8 → 8×8:  mean {errs_g.mean():.3e},  max {errs_g.max():.3e}")
print(f"     bandlimit prediction = {np.sqrt(frac_past_g):.3e}")
print()
print(f"  V_qmunu naive   :  mean {errs_V_naive.mean():.3e},  max {errs_V_naive.max():.3e}")
print(f"     bandlimit prediction (no split) = {np.sqrt(frac_past_V):.3e}")
print(f"  V_qmunu split   :  mean {errs_V_split.mean():.3e},  max {errs_V_split.max():.3e}")
print(f"  V_body alone    :  mean {errs_V_body.mean():.3e},  max {errs_V_body.max():.3e}")
print(f"     bandlimit prediction (V_body) = {np.sqrt(frac_past_V_body):.3e}")
print()
print(f"  W_qmunu naive   :  mean {errs_W_naive.mean():.3e},  max {errs_W_naive.max():.3e}")
print(f"     bandlimit prediction (no split) = {np.sqrt(frac_past_W):.3e}")
print(f"  W_qmunu split   :  mean {errs_W_split.mean():.3e},  max {errs_W_split.max():.3e}")
print(f"  W_body alone    :  mean {errs_W_body.mean():.3e},  max {errs_W_body.max():.3e}")
print(f"     bandlimit prediction (W_body) = {np.sqrt(frac_past_W_body):.3e}")

np.savez(f"{HERE}/v_q_subgrid_results.npz",
         g0_recon=g0_recon, g0_truth=g0_8, errs_g=errs_g,
         V_recon_naive=V_naive_recon, V_recon_split=V_split_recon, V_truth=V_8,
         V_body_recon=V_body_recon, V_body_truth=V_body_8,
         errs_V_naive=errs_V_naive, errs_V_split=errs_V_split, errs_V_body=errs_V_body,
         W_recon_naive=W_naive_recon, W_recon_split=W_split_recon, W_truth=W_8,
         W_body_recon=W_body_recon, W_body_truth=W_body_8,
         errs_W_naive=errs_W_naive, errs_W_split=errs_W_split, errs_W_body=errs_W_body,
         frac_past_g=frac_past_g, frac_past_V=frac_past_V, frac_past_V_body=frac_past_V_body,
         frac_past_W=frac_past_W, frac_past_W_body=frac_past_W_body)
print(f"\nSaved {HERE}/v_q_subgrid_results.npz")
