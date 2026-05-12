"""Test [Sigma_COHSEX, U^(s)] on the lowest 4-fold DFT-degenerate subspace
at k = (0, 1/2, 1/2) in Si Fd-3m.

Strategy
--------
Use the LORRAX cohsex run's `qp_wfn_rotations.h5` to recover

    H_QP = U_mnk @ diag(E_qp) @ U_mnk^dagger   (in DFT band basis at k_full)

For the lowest 4 DFT-degenerate bands at k_X (which BGW reports as a 4-fold
at -4.231 eV, but LORRAX splits 2+2), project H_QP to the 4x4 subspace.

Build U^(s)_ab = <psi_orig_a | (S_s psi_orig)_b> using the G-space recipe
(verified to match real-space + SymMaps unfold) for each small-group op.

Compute the commutator [H_QP_4x4, U^(s)] for each op. A sym-respecting
self-energy must give a zero commutator on every degenerate subspace.

Run
---
   lxrun python3 -u sigma_commutator_test.py
"""
import numpy as np, sys, h5py
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src')

from file_io import WFNReader
from common.symmetry_maps import SymMaps


def fft_g_to_r(cG_full, fft_grid, gvecs):
    Nx, Ny, Nz = fft_grid
    nb, ns, ngk = cG_full.shape
    box = np.zeros((nb, ns, Nx, Ny, Nz), dtype=np.complex128)
    box[:, :, gvecs[:, 0] % Nx, gvecs[:, 1] % Ny, gvecs[:, 2] % Nz] = cG_full
    return np.fft.ifftn(box, axes=(2, 3, 4)) * (Nx * Ny * Nz)


def apply_sym_gspace(cnk, gvecs, k_frac, tau_s, sym_krep_s, D_s, fft_grid):
    """Mirror of SymMaps.get_cnk_fullzone_batch unfold + iFFT to real space."""
    cnk = np.asarray(cnk, dtype=np.complex128)
    gvecs = np.asarray(gvecs, dtype=np.int32)
    sym_krep_s = np.asarray(sym_krep_s, dtype=np.int32)
    g_rot = np.einsum('ij,gj->gi', sym_krep_s, gvecs)
    kg0_BGW = np.rint(k_frac - sym_krep_s @ k_frac).astype(np.int32)
    g_new = g_rot - kg0_BGW[None, :]
    phase = np.exp(-2j * np.pi * (g_rot.astype(np.float64) @ tau_s))
    cnk_phased = cnk * phase[None, None, :]
    cnk_rot = np.einsum('uv,bvg->bug', D_s, cnk_phased)
    return fft_g_to_r(cnk_rot, fft_grid, g_new)


def main():
    wfn = WFNReader('WFN.h5')
    sym = SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # Locate IBZ k_X = (0, 1/2, 1/2)
    target = np.array([0.0, 0.5, 0.5])
    diff = wfn.kpoints.astype(np.float64) - target[None, :]
    diff -= np.round(diff)
    kid = int(np.argmin(np.linalg.norm(diff, axis=1)))
    k_frac = wfn.kpoints[kid].copy()
    print(f"IBZ k_X index = {kid},  k = {k_frac}")

    # Sym ops + spinor reps
    n_sym = int(wfn.ntran)
    R_k = np.asarray(sym.sym_mats_k[:n_sym], dtype=np.int32)
    tau = np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)
    sym_cart = sym.syms_crystal_to_cartesian(wfn)
    D = sym.get_spinor_rotations(wfn, sym_cart)

    # Lowest 4 DFT-degenerate bands at k_X (BGW: bands 1-4, all -4.231 eV at this k)
    band_lo, band_hi = 0, 4
    cnk = wfn.get_cnk_batch(kid, list(range(band_lo, band_hi)))
    gvecs = np.asarray(wfn.get_gvec_nk(kid))
    psi_orig_r = fft_g_to_r(cnk, fft_grid, gvecs)
    norms = np.sqrt(np.einsum('bsxyz,bsxyz->b', np.conj(psi_orig_r), psi_orig_r).real)
    psi_orig_r = psi_orig_r / norms[:, None, None, None, None]

    # ------------------------------------------------------------------
    # Recover H_QP at this k from qp_wfn_rotations.h5
    # ------------------------------------------------------------------
    with h5py.File('qp_wfn_rotations.h5', 'r') as f:
        U_mnk = f['U_mnk'][...]                  # (nk_full, nb, nb)
        E_qp_ry = f['E_qp_nk_rydberg'][...]      # (nk_full, nb)
        kirr_to_kfull = f['kirr_to_kfull'][...]
        kpoints_crys = f['kpoints_crys'][...]

    kfull = int(kirr_to_kfull[kid])
    print(f"  full-zone k_full index = {kfull},  kpoint = {kpoints_crys[kfull]}")

    nb = U_mnk.shape[1]
    U_kf = U_mnk[kfull]                          # (nb, nb): QP-eigvec columns
    E_kf = E_qp_ry[kfull]                        # (nb,)

    # H_QP in DFT band basis: H_QP_mn = sum_p U[p, m]^* * E_qp[p] * U[p, n]
    # (LORRAX convention: U_mnk[k, m, n] = <DFT_m | QP_n>, so QP_p = sum_m U[m, p] DFT_m,
    #  and H_QP in DFT basis = U @ diag(E_qp) @ U^dagger.)
    H_QP_DFT = U_kf @ np.diag(E_kf) @ U_kf.conj().T   # (nb, nb), Ry

    # Slice to the 4-fold degenerate subspace
    H4 = H_QP_DFT[band_lo:band_hi, band_lo:band_hi]  # (4, 4)
    print(f"\nH_QP[k_X] in DFT basis, lowest 4-fold subspace (Ry):")
    print(np.real(H4))
    eigs = np.linalg.eigvalsh((H4 + H4.conj().T) / 2)
    print(f"\nEigenvalues of H_QP_4x4 (eV): "
          + ", ".join(f"{e * 13.6057:.4f}" for e in eigs))

    # ------------------------------------------------------------------
    # Build U^(s) for each small-group op at k_X via G-space recipe
    # ------------------------------------------------------------------
    sg = []
    for s in range(n_sym):
        d = R_k[s] @ k_frac - k_frac
        if np.max(np.abs(d - np.round(d))) < 1e-6:
            sg.append(s)
    print(f"\n|small group of k_X| = {len(sg)}")

    print(f"\n{'s':>3} {'tau':>20} {'||[H4,U]||_F (Ry)':>20} {'||[H4,U]||_F (eV)':>20}")
    print('-' * 80)

    for s in sg:
        psi_G = apply_sym_gspace(cnk, gvecs, k_frac, tau[s], R_k[s], D[s], fft_grid)
        psi_G = psi_G / np.sqrt(np.einsum('bsxyz,bsxyz->b',
                                          np.conj(psi_G), psi_G).real)[:, None, None, None, None]
        # U^(s)_ab = <psi_a | S psi_b>  (4x4)
        U_s = np.einsum('asxyz,bsxyz->ab', np.conj(psi_orig_r), psi_G)
        # Commutator [H4, U_s]
        comm = H4 @ U_s - U_s @ H4
        norm_ry = float(np.linalg.norm(comm, 'fro'))
        norm_ev = norm_ry * 13.6057
        print(f"{s:>3} {str(tau[s].round(4)):>20} {norm_ry:>20.6e} {norm_ev:>20.6e}")


if __name__ == '__main__':
    main()
