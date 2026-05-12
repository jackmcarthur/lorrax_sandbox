"""Within-8×8 disambiguation: 4×4 sub-grid → 8×8 with periodic-wrap vs umklapp.

Single ζ-fit (the 8×8 LORRAX run) defines BOTH the coarse 4×4 sub-sample
AND the fine 8×8 truth.  No inter-run ζ-fit drift.  The only difference
between this test and what gives `0.04` machine-precision-coincident-points
in the spectral upscale is whether we use periodic wrap or umklapp at the
4×4-cell-boundary.

  - periodic-wrap (= bilinear with g0_8 [0, ·] used for the q_x=1=0 corner):
       g_corner = g0_8[0, j_y] (= the periodic image)
  - umklapp (= bilinear with z_{q=0,μ}(G=b_x) used for the q_x=1 corner):
       g_corner = A_x[j_y] = z_{(0, j_y/8, 0), μ}(G=(1,0,0))

If umklapp wins here, the BZ-boundary smooth continuation is genuine
extended-zone physics and my 8→12 result was muddied by inter-run ζ-fit
drift (at coincident-q's between 8×8 and 12×12 runs, ‖g_8 − g_12‖/‖g_12‖ ≈ 7%).
If periodic-wrap still wins, it really is the right interpretation
for this LORRAX dataset.

Uses correct LORRAX normalization: g0_mu = Σ_r ζ_q,μ(r) (no 1/N_FFT).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
import numpy as np
import h5py

HERE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"

# ── Load 8×8 truth + ζ at q=Γ-strip-points needed for umklapp neighbours ───
print("=== loading 8×8 truth ===")
with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5", 'r') as f:
    g_8 = f["g0_mu"][...]                                  # (8, 8, 1, n_μ) — TRUTH
    # Need ζ at the 4×4-sub-grid boundary q's:
    # The 4×4 sub-grid is q = (i/4, j/4) for i, j ∈ {0..3}.  In 8-grid index
    # that's (2i, 2j).  The cell that wraps in x has corners at 8-grid-x = 6
    # and 8-grid-x = 8 (= wraps to 0).  The wrapped corner needs z at q_x=0
    # for various q_y values.  In the 4×4 sub-grid context, the boundary
    # column needs umklapp values for q on the 4×4 sub-grid's q_y values
    # (= 8-grid q_y indices 0, 2, 4, 6) — but since we're sub-sampling the
    # 8-grid, "q_y values" of the 4×4 sub-grid points are 0/4 = 0, 1/4, 2/4,
    # 3/4 — equivalently 8-grid indices 0, 2, 4, 6.
    #
    # To do bilinear at any 8-grid fine point that lies in a wrap cell of the
    # 4×4 sub-grid, we need umklapp values at every 4×4 sub-grid q_y, i.e.
    # 8-grid index 0, 2, 4, 6.  Fortunately the umklapp identity says
    # z_{q + b_x, μ}(G=0) = z_{q, μ}(G=b_x), so we need z_{q, μ}(G=b_x) for
    # the four q's q = (q_x=0, q_y_sub=0/4, 1/4, 2/4, 3/4) = (0, q_y) for
    # 8-grid q_y indices 0, 2, 4, 6.  Read those 4 q's from zeta_q.
    zeta_qx0 = f["zeta_q"][[0, 2, 4, 6], :, :].astype(np.complex128)  # (4, n_FFT, n_μ)
    # And q_y=0 strip for y-wrap: q = (q_x_sub=0/4, 1/4, 2/4, 3/4, q_y=0) → 8-grid (0, 0), (2, 0), (4, 0), (6, 0).
    # Flat index = q_x_idx*8 + q_y_idx → for q_y_idx=0: indices 0, 16, 32, 48.
    zeta_qy0 = f["zeta_q"][[0, 16, 32, 48], :, :].astype(np.complex128)

with h5py.File(f"{HERE}/fine_8x8/WFN.h5", 'r') as f:
    fft_grid = tuple(int(x) for x in f["mf_header/gspace/FFTgrid"][...])
N_x, N_y, N_z = fft_grid
n_mu = g_8.shape[-1]

# ── Compute z_{q,μ}(G=...) using LORRAX convention:
#     z_q(G) = Σ_r ζ_q(r) · e^{-iq·r} · e^{-iG·r}     (NO 1/N_FFT factor)
# The previous version omitted the e^{-iq·r} factor for non-Γ boundary q's,
# which silently put the umklapp neighbours in the wrong gauge — that was
# the cause of bilin-umklapp losing to bilin-wrap on the boundary strip.
# Now include phase_q(r) for each q before the extra G-shift phase.
zeta_qx0_3d = zeta_qx0.reshape(4, N_x, N_y, N_z, n_mu)            # (q_y_sub, x, y, z, μ)
zeta_qy0_3d = zeta_qy0.reshape(4, N_x, N_y, N_z, n_mu)
phase_x = np.exp(-2j * np.pi * np.arange(N_x) / N_x)              # e^{-iG_x·r}
phase_y = np.exp(-2j * np.pi * np.arange(N_y) / N_y)              # e^{-iG_y·r}

# q-phases for the four sub-grid boundary q's, using LORRAX's wrapped-q
# convention (compute_vcoul.py L959-963):  q_wrapped = q if q ≤ N/2 else q - N.
# For nk=8 and 4×4 sub-grid (q_idx_8 ∈ {0, 2, 4, 6}) the wrapped 8-grid
# integer q is {0, 2, 4, -2}, so the y-fractional q's are {0, 1/4, 1/2, -1/4}
# (and likewise for x).  My previous version used 3/4 instead of -1/4 at the
# j=3 entry, putting the umklapp neighbour in the wrong gauge.
qy_wrapped = np.array([0, 2, 4, -2], dtype=np.float64) / 8.0       # (4,)  wrapped q_y for zeta_qx0[j]
qx_wrapped = np.array([0, 2, 4, -2], dtype=np.float64) / 8.0       # (4,)  wrapped q_x for zeta_qy0[j]
phase_q_y_for_qx0 = np.exp(-2j * np.pi * qy_wrapped[:, None] *
                           (np.arange(N_y)[None, :] / N_y))         # (4, N_y)
phase_q_x_for_qy0 = np.exp(-2j * np.pi * qx_wrapped[:, None] *
                           (np.arange(N_x)[None, :] / N_x))         # (4, N_x)

print("=== computing umklapp neighbour values (with full q+G phase) ===")
A_x  = np.einsum('qxyzm,x,qy->qm', zeta_qx0_3d, phase_x, phase_q_y_for_qx0)   # (4, n_μ)
A_y  = np.einsum('qxyzm,y,qx->qm', zeta_qy0_3d, phase_y, phase_q_x_for_qy0)   # (4, n_μ)
A_xy = np.einsum('xyzm,x,y->m',
                 zeta_qx0_3d[0], phase_x, phase_y)                  # (n_μ,) — z_{Γ,μ}(G=b_x+b_y), q=Γ so no q-phase

print(f"  shapes: A_x {A_x.shape}, A_y {A_y.shape}, A_xy {A_xy.shape}")

# ── Direct sanity: compute g0 at the four (0, j/4, 0) sub-grid q's via full FFT
#     formula, then compare to the persisted g0_mu values.  If the q-phase fix
#     is correct, this should match at machine precision.
print("\n=== sanity: reproduce persisted g0(q) from ζ_q at the same 4 q's ===")
g0_mine = np.einsum('qxyzm,qy->qm', zeta_qx0_3d, phase_q_y_for_qx0)  # (4, n_μ), G=0
for j in range(4):
    qy_8 = 2 * j  # 8-grid index of q = (0, j/4, 0) is (0, 2j, 0)
    rel = np.linalg.norm(g0_mine[j] - g_8[0, qy_8, 0]) / max(np.linalg.norm(g_8[0, qy_8, 0]), 1e-30)
    print(f"  q = (0, {j}/4, 0): rel ‖my g0 − persisted‖ = {rel:.3e}")
print(f"  ‖g_8(Γ)‖    = {np.linalg.norm(g_8[0,0,0]):.4e}")
print(f"  ‖A_x[0]‖    = {np.linalg.norm(A_x[0]):.4e}   "
      f"(= z_Γ(G=b_x) ≈ FT[ζ] at extended-zone Q = b_x)")
print(f"  ‖g_8(7/8,0)‖= {np.linalg.norm(g_8[7,0,0]):.4e}   "
      f"(= the 'inside-BZ' boundary sample)")


# ── 4×4 sub-grid (every other 8-grid point) ────────────────────────────────
g_4 = g_8[::2, ::2, :, :]                                             # (4, 4, 1, n_μ)


# ── Bilinear: 4×4 → 8×8, two boundary conventions ─────────────────────────
def bilinear_4to8(g4, A_x, A_y, A_xy, *, umklapp):
    """Bilinear interpolate the 4×4 sub-grid to the 8×8 fine grid.

    For 8-grid fine index (ix_f, iy_f) in 0..7:
      * ix_lo_4 = floor((ix_f / 8) * 4) = floor(ix_f / 2)  (in 4-grid units, 0..3)
      * a_x = (ix_f / 8) * 4 - ix_lo_4                       (fractional in [0, 1))

    For ix_f even: a_x = 0; result is just g4[ix_lo_4, iy_lo_4, ...] etc.
    For ix_f odd:  a_x = 0.5; equal mixture of low and high corners.

    Wrap or umklapp at ix_hi_4 == 4 (likewise iy).
    """
    g_out = np.zeros((8, 8, 1, n_mu), dtype=np.complex128)
    for ix_f in range(8):
        ix_real = ix_f / 8 * 4
        ix_lo = int(np.floor(ix_real))
        ax = ix_real - ix_lo
        for iy_f in range(8):
            iy_real = iy_f / 8 * 4
            iy_lo = int(np.floor(iy_real))
            ay = iy_real - iy_lo
            result = np.zeros(n_mu, dtype=np.complex128)
            for dx in (0, 1):
                for dy in (0, 1):
                    ix_c, iy_c = ix_lo + dx, iy_lo + dy
                    w = (ax if dx == 1 else 1 - ax) * (ay if dy == 1 else 1 - ay)
                    if ix_c == 4 and iy_c == 4:
                        val = A_xy if umklapp else g4[0, 0, 0]
                    elif ix_c == 4:
                        val = A_x[iy_c % 4] if umklapp else g4[0, iy_c % 4, 0]
                    elif iy_c == 4:
                        val = A_y[ix_c % 4] if umklapp else g4[ix_c % 4, 0, 0]
                    else:
                        val = g4[ix_c, iy_c, 0]
                    result += w * val
            g_out[ix_f, iy_f, 0, :] = result
    return g_out

g_recon_wrap     = bilinear_4to8(g_4, A_x, A_y, A_xy, umklapp=False)
g_recon_umklapp  = bilinear_4to8(g_4, A_x, A_y, A_xy, umklapp=True)


# ── Spectral upscale (for comparison) ─────────────────────────────────────
from psp.finite_q_head_interp import fourier_interpolate_coarse_to_fine
g_recon_spec = fourier_interpolate_coarse_to_fine(g_4, (8, 8, 1))


# ── Metrics: aggregate + boundary strip + per-q breakdown ──────────────────
mask_full = np.ones((8, 8), bool); mask_full[0, 0] = False
# Coincident with 4×4 sub-grid → machine precision by construction.
mask_coinc = np.zeros((8, 8), bool); mask_coinc[::2, ::2] = True; mask_coinc[0,0] = False
mask_non = mask_full & ~mask_coinc
# "Boundary strip" of the 4×4 sub-grid in 8-grid: ix_f ∈ {7} or iy_f ∈ {7} or
# (ix_f ∈ {6, 7} and iy_f ∈ {6, 7})  — the cells whose corners include ix_c=4
# or iy_c=4.  In 8-grid: ix_f ∈ {7} or iy_f ∈ {7} (= cells [3, 4]/4 in 4-grid).
mask_bnd = np.zeros((8, 8), bool); mask_bnd[7, :] = True; mask_bnd[:, 7] = True
mask_bnd &= mask_full
mask_int = mask_non & ~mask_bnd

def rel_F(recon, truth, mask):
    diff = (recon - truth)[mask, 0, :]
    tru = truth[mask, 0, :]
    return np.linalg.norm(diff) / np.linalg.norm(tru)

def per_q_cos_med(recon, truth, mask):
    coses = []
    for ix in range(8):
        for iy in range(8):
            if not mask[ix, iy]: continue
            a = recon[ix, iy, 0]; b = truth[ix, iy, 0]
            coses.append(abs(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return np.median(coses)

print("\n=== rel ‖err‖_F  (Q ≠ 0; subset masks) ===")
for label, mask in [("all non-Γ (63 pts)",       mask_full),
                    ("non-coincident (48)",       mask_non),
                    ("4×4-cell boundary (15)",    mask_bnd),
                    ("interior non-coinc (33)",   mask_int)]:
    rs = rel_F(g_recon_spec,    g_8, mask)
    rb = rel_F(g_recon_wrap,    g_8, mask)
    ru = rel_F(g_recon_umklapp, g_8, mask)
    print(f"  {label:>26}:  spectral {rs:.3e},  bilin-wrap {rb:.3e},  bilin-umklapp {ru:.3e}")

print("\n=== per-Q cosine median  (Q ≠ 0) ===")
for label, mask in [("all non-Γ",                mask_full),
                    ("4×4-cell boundary",        mask_bnd),
                    ("interior non-coinc",       mask_int)]:
    cs = per_q_cos_med(g_recon_spec,    g_8, mask)
    cb = per_q_cos_med(g_recon_wrap,    g_8, mask)
    cu = per_q_cos_med(g_recon_umklapp, g_8, mask)
    print(f"  {label:>22}:  spectral {cs:.4f},  bilin-wrap {cb:.4f},  bilin-umklapp {cu:.4f}")


# Per-Q on the boundary strip — where the only difference between methods lives.
print("\n=== Per-Q rel ‖err‖_F on the 15 boundary q's ===")
print(f"  {'qfrac':>22}  {'spec':>11}  {'bilin-wrap':>12}  {'bilin-umklapp':>15}")
for ix in range(8):
    for iy in range(8):
        if not mask_bnd[ix, iy]: continue
        Qx, Qy = ix/8, iy/8
        rs_ = np.linalg.norm(g_recon_spec[ix,iy,0]    - g_8[ix,iy,0]) / max(np.linalg.norm(g_8[ix,iy,0]), 1e-30)
        rb_ = np.linalg.norm(g_recon_wrap[ix,iy,0]    - g_8[ix,iy,0]) / max(np.linalg.norm(g_8[ix,iy,0]), 1e-30)
        ru_ = np.linalg.norm(g_recon_umklapp[ix,iy,0] - g_8[ix,iy,0]) / max(np.linalg.norm(g_8[ix,iy,0]), 1e-30)
        print(f"  ({Qx:5.3f}, {Qy:5.3f})        {rs_:>11.3e}  {rb_:>12.3e}  {ru_:>15.3e}")
