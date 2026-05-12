"""Analyse which sym ops respect band-degenerate subspaces at k=(0, 1/2, 1/2)
in Si Fd-3m. Two independent implementations of the action of (R, tau) on
Bloch wavefunctions, both of which should give the same overlap matrix:

* Real-space:  psi'(r) = D(R) . psi(R(r-tau)) * e^{-i (k+G_uk).tau}
* G-space   :  follows SymMaps.get_cnk_fullzone_batch exactly --
               c'(S G - kg0) = U_spinor . c(G) * e^{-i (S G).tau}
               then iFFT to compare in real space on the same grid.

If the two paths disagree, that's diagnostic of a bug in either my
real-space formula or SymMaps's G-space path. If they agree but BROKEN
ops still appear, the issue is conceptual (the sub-block structure is
genuine).

Tests run on bands 1-4 (one BGW 4-fold) and on the full 16-band block.
"""
import numpy as np, sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src')

from file_io import WFNReader
from common.symmetry_maps import SymMaps


def fft_g_to_r(cG_full, fft_grid, gvecs):
    """Insert (ngk,) G-coeffs on the FFT grid, iFFT to real space.
    cG_full: (n_band, n_spin, ngk); gvecs: (ngk, 3) int -> shape (nb, ns, Nx, Ny, Nz)."""
    Nx, Ny, Nz = fft_grid
    nb, ns, ngk = cG_full.shape
    box = np.zeros((nb, ns, Nx, Ny, Nz), dtype=np.complex128)
    gx = gvecs[:, 0] % Nx
    gy = gvecs[:, 1] % Ny
    gz = gvecs[:, 2] % Nz
    box[:, :, gx, gy, gz] = cG_full
    return np.fft.ifftn(box, axes=(2, 3, 4)) * (Nx * Ny * Nz)


def apply_sym_realspace(psi_r, k_frac, tau_s, R_s, R_k_s, D_s, fft_grid):
    """Real-space recipe.

    psi'(r) = D(R) . psi(M(r - tau))    where M = sym_matrices = R_s.
    psi_r in this codepath stores u(r) (the periodic part), not full
    e^{ikr} u(r), since fft_g_to_r FFT-d the c(G) array directly.

    For an op in the small group with M^T k = k + g_uk, the rotated
    function written as a Bloch state at the SAME k has a periodic part

        u_rot(r) = e^{+i 2pi g_uk . r} . [const phase] . u(M(r - tau))

    The constant phase is absorbed into the per-band normalization. The
    e^{+i 2pi g_uk . r} factor on the grid IS load-bearing: it re-Blochs
    the rotated function from k+g_uk back to k so that overlaps with
    psi_orig (which lives at k) compare apples-to-apples.
    """
    Nx, Ny, Nz = fft_grid
    ix, iy, iz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz),
                             indexing='ij')
    r_int = np.stack([ix, iy, iz], axis=-1)
    tau_grid = np.round(tau_s * np.array([Nx, Ny, Nz])).astype(int)
    r_shift = r_int - tau_grid
    r_rot = np.einsum('xyzi,ji->xyzj', r_shift, R_s)   # = (r-tau) @ R.T
    r_rot[..., 0] %= Nx; r_rot[..., 1] %= Ny; r_rot[..., 2] %= Nz
    psi_rot = psi_r[:, :, r_rot[..., 0], r_rot[..., 1], r_rot[..., 2]]
    psi_rot = np.einsum('uv,bvxyz->buxyz', D_s, psi_rot)

    # Re-Bloch from k+g_uk back to k. g_uk = M^T k - k = R_k k - k.
    g_uk = np.round(R_k_s @ k_frac - k_frac).astype(int)
    if np.any(g_uk != 0):
        xs = np.arange(Nx) / Nx
        ys = np.arange(Ny) / Ny
        zs = np.arange(Nz) / Nz
        ph = (np.exp(2j * np.pi * g_uk[0] * xs)[:, None, None]
              * np.exp(2j * np.pi * g_uk[1] * ys)[None, :, None]
              * np.exp(2j * np.pi * g_uk[2] * zs)[None, None, :])
        psi_rot = psi_rot * ph[None, None, :, :, :]
    return psi_rot


def apply_sym_gspace(cnk, gvecs, k_frac, tau_s, M_inv_s, D_s, fft_grid):
    """G-space recipe -- BGW Common/gmap.f90 convention.

    BGW stores ``mtrx = inv(R_crystallographic)`` (symmetries.f90:189) and
    applies ``inv(mtrx) = R_crystallographic`` to G-vectors (gmap.f90:125,
    149). For Si fcc primitive, M is NOT orthogonal in fractional coords,
    so M^T != M^{-1} -- we need the actual matrix inverse, not transpose.

    Pass ``M_inv_s = inv(sym_matrices[s]) = R_crystallographic`` (= Rinv_grid).

      G_target = M_inv . G_source + (M_inv k - k)   # = R_cryst k - k
      c_new(G_target) = D . c_old(G_source) * e^{-i (M_inv G_source) . tau}
    """
    cnk = np.asarray(cnk, dtype=np.complex128)
    gvecs = np.asarray(gvecs, dtype=np.int32)
    M_inv_s = np.asarray(M_inv_s, dtype=np.int32)
    g_rot = np.einsum('ij,gj->gi', M_inv_s, gvecs)
    kg0_BGW = np.rint(k_frac - M_inv_s @ k_frac).astype(np.int32)
    g_new = g_rot - kg0_BGW[None, :]
    # Per-G phase from non-symmorphic translation.
    phase_arg = g_rot.astype(np.float64) @ tau_s
    phase = np.exp(-2j * np.pi * phase_arg)            # tau already in 2π units
    cnk_phased = cnk * phase[None, None, :]
    cnk_rot = np.einsum('uv,bvg->bug', D_s, cnk_phased)
    psi_r = fft_g_to_r(cnk_rot, fft_grid, g_new)
    return psi_r


def normalize(psi_r):
    norms = np.sqrt(np.einsum('bsxyz,bsxyz->b', np.conj(psi_r), psi_r).real)
    return psi_r / norms[:, None, None, None, None]


def overlap(psi_a, psi_b):
    return np.einsum('asxyz,bsxyz->ab', np.conj(psi_a), psi_b)


def small_group_ops(R_k_all, k_frac, n_sym):
    """Indices s in [0, n_sym) such that R_k @ k = k mod G."""
    out = []
    for s in range(n_sym):
        d = R_k_all[s] @ k_frac - k_frac
        if np.max(np.abs(d - np.round(d))) < 1e-6:
            out.append(s)
    return out


def analyze_block(wfn, sym, kid, k_frac, band_lo, band_hi, label,
                  R_grid, R_inv, R_k, tau, D, fft_grid):
    """Compute defect for both real- and G-space paths over the small group."""
    N = band_hi - band_lo
    band_idx = list(range(band_lo, band_hi))
    cnk = wfn.get_cnk_batch(kid, band_idx)
    gvecs = np.asarray(wfn.get_gvec_nk(kid))
    psi_orig_r = normalize(fft_g_to_r(cnk, fft_grid, gvecs))

    # Small-group test uses sym_mats_k for k-action regardless (that part
    # is unambiguous: k transforms by the matrix that fixes irreducible-zone
    # k_full = S k_irred mod G).
    sg = small_group_ops(R_k, k_frac, len(R_k))

    print(f"\n=== {label}: bands {band_lo+1}..{band_hi} (N={N}), "
          f"|small group of k_X| = {len(sg)} ===")
    print(f"{'s':>3} {'tau':>20} "
          f"{'||U_R||_F^2':>12} {'||U_G||_F^2':>12} "
          f"{'||U_R-U_G||':>12}")
    print("-" * 95)
    for s in sg:
        symmorphic = bool(np.all(np.abs(tau[s] - np.round(tau[s])) < 1e-8))
        # Real-space
        psi_R = apply_sym_realspace(
            psi_orig_r, k_frac, tau[s], R_grid[s], R_k[s], D[s], fft_grid,
        )
        # G-space (SymMaps convention: rotate G by sym_mats_k = M^T)
        psi_G = apply_sym_gspace(
            cnk, gvecs, k_frac, tau[s], R_k[s], D[s], fft_grid,
        )
        psi_R = psi_R / np.sqrt(np.einsum('bsxyz,bsxyz->b',
                                          np.conj(psi_R), psi_R).real)[:, None, None, None, None]
        psi_G = psi_G / np.sqrt(np.einsum('bsxyz,bsxyz->b',
                                          np.conj(psi_G), psi_G).real)[:, None, None, None, None]

        # U^(s)_ab = <psi_a | S psi_b> on the degenerate subspace.
        # If S keeps the subspace closed, U is unitary -> ||U||_F^2 = N.
        # Otherwise S leaks weight into bands outside this 4D block.
        U_R = overlap(psi_orig_r, psi_R)        # <psi_orig | psi_R>
        U_G = overlap(psi_orig_r, psi_G)
        norm_R_sq = float(np.linalg.norm(U_R, 'fro')**2)
        norm_G_sq = float(np.linalg.norm(U_G, 'fro')**2)

        # If both U^R and U^G are valid representations of the same group
        # element on the same subspace, they are equal: same operator,
        # same input basis, same overlap with the same output basis.
        # Any difference is a real bug in one method (or a different basis
        # convention for the SAME irrep, which would still give equal U
        # since psi_orig is the source for both).
        delta_RG = float(np.linalg.norm(U_R - U_G, 'fro'))

        print(f"{s:>3} {str(tau[s].round(4)):>20} "
              f"{norm_R_sq:>12.4f} {norm_G_sq:>12.4f} "
              f"{delta_RG:>12.3e}")


def main():
    wfn = WFNReader('WFN.h5')
    sym = SymMaps(wfn)

    # --- Find k = (0, 1/2, 1/2) in the IBZ list ---
    kpts = wfn.kpoints.astype(np.float64)
    target = np.array([0.0, 0.5, 0.5])
    diff = kpts - target[None, :]
    diff -= np.round(diff)
    kid = int(np.argmin(np.linalg.norm(diff, axis=1)))
    k_frac = kpts[kid].copy()
    print(f"Target k = (0, 1/2, 1/2) -> WFN IBZ index {kid}, k_frac = {k_frac}")

    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    n_sym = int(wfn.ntran)
    R_grid = np.asarray(sym.R_grid[:n_sym], dtype=np.int32)
    R_inv  = np.asarray(sym.Rinv_grid[:n_sym], dtype=np.int32)
    R_k    = np.asarray(sym.sym_mats_k[:n_sym], dtype=np.int32)
    tau    = np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)
    sym_cart = sym.syms_crystal_to_cartesian(wfn)
    D = sym.get_spinor_rotations(wfn, sym_cart)

    # 4-band block (one BGW 4-fold) and full 16-band block.
    analyze_block(wfn, sym, kid, k_frac,  0,  4, "Block 1 (4-fold lowest)",
                  R_grid, R_inv, R_k, tau, D, fft_grid)
    analyze_block(wfn, sym, kid, k_frac,  0, 16, "Block 2 (4 stacked 4-folds)",
                  R_grid, R_inv, R_k, tau, D, fft_grid)


if __name__ == '__main__':
    main()
