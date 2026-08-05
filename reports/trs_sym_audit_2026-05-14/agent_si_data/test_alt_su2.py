"""Try an alternative U_spinor convention and see if (b) gauge-residual vs nosym closes.

We rebuild R_cart in BGW convention (R_cart = bvec @ mtrx @ bvecinv, where
bvec.shape=(3,3) with rows=b_i) using LORRAX's bvec (which we suspect is also
rows=b_i since b1, b2, b3 are columns of bvec in this WFN — wait, we found
avec.T @ bvec = I, so bvec[:, i] = b_i in cartesian; equivalently bvec.T[i, :] = b_i).

We try the four likely orderings:
  (1) LORRAX current: B_T_inv @ sym_mats_k @ B_T   where B_T = bvec
  (2) BGW direct:     bvec @ mtrx @ bvecinv
  (3) flipped:        bvecinv @ mtrx @ bvec
  (4) flipped:        bvec.T @ mtrx @ bvec.T_inv

For each, build U_spinor[s] then redo the hand-rolled ψ unfold on the same
(k_full, sym_idx) pairs from test 1. Pick proper_nonsymm case k_f=4 s=1 which
showed gauge_resid = 7.1e-2 in test 1.
"""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader
from common.symmetry_maps import SymMaps


def get_su2_from_R_cart(R_cart):
    """BGW's quaternion-based SU(2) construction (= LORRAX's get_spinor_rotations
    pure-numpy port)."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    R = R_cart.copy()
    if np.linalg.det(R) < 0:
        R = -R
    Q = np.zeros((4, 4))
    Q[0, 0] = R[0, 0] + R[1, 1] + R[2, 2]
    Q[0, 1] = Q[1, 0] = R[1, 2] - R[2, 1]
    Q[0, 2] = Q[2, 0] = R[2, 0] - R[0, 2]
    Q[0, 3] = Q[3, 0] = R[0, 1] - R[1, 0]
    Q[1, 1] = R[0, 0] - R[1, 1] - R[2, 2]
    Q[1, 2] = Q[2, 1] = R[0, 1] + R[1, 0]
    Q[1, 3] = Q[3, 1] = R[0, 2] + R[2, 0]
    Q[2, 2] = -R[0, 0] + R[1, 1] - R[2, 2]
    Q[2, 3] = Q[3, 2] = R[1, 2] + R[2, 1]
    Q[3, 3] = -R[0, 0] - R[1, 1] + R[2, 2]
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    q = eigenvectors[:, np.argmax(eigenvalues)]
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    theta = 2 * np.arccos(np.clip(q0, -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - q0**2))
    if sin_half < 1e-8:
        theta = 0.0
        n = np.array([1.0, 0.0, 0.0])
    elif np.isclose(theta, np.pi):
        axis = np.array([q1, q2, q3])
        n = axis / np.linalg.norm(axis)
    else:
        n = np.array([q1, q2, q3]) / sin_half
        n = n / np.linalg.norm(n)
    cos_h = np.cos(theta / 2)
    sin_h = np.sin(theta / 2)
    spinor = cos_h * np.eye(2, dtype=complex)
    spinor -= 1j * sin_h * (n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z)
    return spinor


def g_to_box_flat(gvecs, fft_grid):
    Nx, Ny, Nz = [int(x) for x in fft_grid]
    g = np.asarray(gvecs, dtype=np.int64)
    return ((g[..., 0] % Nx) * Ny + (g[..., 1] % Ny)) * Nz + (g[..., 2] % Nz)


def scatter_to_box(psi, gvecs_for_psi, ngk_valid, fft_grid):
    nb, ns, _ = psi.shape
    Nx, Ny, Nz = [int(x) for x in fft_grid]
    Nbox = Nx * Ny * Nz
    box = np.zeros((nb, ns, Nbox), dtype=np.complex128)
    bidx = g_to_box_flat(gvecs_for_psi[:ngk_valid], fft_grid)
    box[:, :, bidx] = psi[:, :, :ngk_valid]
    return box


def compare_groups(box_x, box_y, e_kirr, tol=5e-6):
    nb = box_x.shape[0]
    groups = []
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and abs(e_kirr[j] - e_kirr[i]) < tol:
            j += 1
        groups.append((i, j)); i = j
    max_unit = 0.0; max_gauge = 0.0
    for (lo, hi) in groups:
        ng = hi - lo
        X = box_x[lo:hi].reshape(ng, -1)
        Y = box_y[lo:hi].reshape(ng, -1)
        Nx_ = np.linalg.norm(X, axis=1)
        Ny_ = np.linalg.norm(Y, axis=1)
        Xn = X / np.where(Nx_[:, None] == 0, 1, Nx_[:, None])
        Yn = Y / np.where(Ny_[:, None] == 0, 1, Ny_[:, None])
        U = Xn @ np.conj(Yn).T
        max_unit = max(max_unit, float(np.linalg.norm(U @ np.conj(U.T) - np.eye(ng))))
        max_gauge = max(max_gauge, float(np.linalg.norm(Xn - U @ Yn)))
    return max_unit, max_gauge


def main():
    sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
    nos = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5")
    s = sym._ensure_sym()
    bvec = np.asarray(sym.bvec)
    bvecinv = np.linalg.inv(bvec)
    mtrx = np.asarray(sym.sym_matrices)         # (ntran,3,3) int — BGW convention
    sym_mats_k = np.asarray(s.sym_mats_k[:sym.ntran])  # = mtrx.T

    # Four candidate R_cart conventions
    R_cands = {
        "A_lorrax_current (Btinv·mTrxT·Bt)": np.einsum('ij,njk,kl->nil', bvecinv, sym_mats_k, bvec),
        "B_bvec·mtrx·bvecinv":             np.einsum('ij,njk,kl->nil', bvec, mtrx, bvecinv),
        "C_bvecinv·mtrx·bvec":             np.einsum('ij,njk,kl->nil', bvecinv, mtrx, bvec),
        "D_bvec·mtrxT·bvecinv":            np.einsum('ij,njk,kl->nil', bvec, sym_mats_k, bvecinv),
        "E_bvecT·mtrx·bvecTinv":           np.einsum('ij,njk,kl->nil', bvec.T, mtrx, np.linalg.inv(bvec.T)),
    }
    fft_grid = np.asarray(sym.fft_grid, dtype=np.int64)
    ntran = sym.ntran

    # Pick the most non-symmorphic-heavy test case from test 1
    test_cases = [
        ("proper_nonsymm k_f=4  s=1", 4, 1, 1, [0, 0, 0]),     # nontrivial proper_nonsymm
        ("proper_nonsymm k_f=8  s=1", 8, 2, 1, [0, 1, 0]),
        ("proper_nonsymm k_f=15 s=2", 15, 3, 2, [0, 1, 1]),
        ("proper_nonsymm k_f=37 s=5", 37, 5, 2, [0, 0, 0]),
        ("proper_symm    k_f=11 s=5", 11, 4, 5, [0, 0, 1]),
    ]

    # Match nosym k
    def find_nos_k(kf):
        diff = nos.kpoints - s.unfolded_kpts[kf]
        diff_w = diff - np.round(diff)
        return int(np.argmin(np.max(np.abs(diff_w), axis=1)))

    b_lo, b_hi = 0, 16
    print(f"{'convention':40s}  {'k_full':10s}  {'unit_dev':>12s}  {'gauge_resid':>12s}")
    for name, R_cart_all in R_cands.items():
        # Build U_spinor for all ntran ops
        U_spinor = np.zeros((ntran, 2, 2), dtype=complex)
        for i in range(ntran):
            U_spinor[i] = get_su2_from_R_cart(R_cart_all[i])
        # apply hand-rolled unfold for each test case
        for tag, kf, kirr, s_idx, _ in test_cases:
            sym_krep = np.asarray(s.sym_mats_k[s_idx], dtype=np.int64)
            ngk = int(sym.ngk[kirr])
            start = int(sym._kpt_starts[kirr])
            raw = sym._coeffs_raw[b_lo:b_hi, :, start:start+ngk, :]
            psi_kbar = raw[..., 0] + 1j*raw[..., 1]
            g_kbar = sym._gvecs_raw[start:start+ngk]
            g_rot_raw = np.einsum('ij,kj->ki', sym_krep, g_kbar)
            tau = np.asarray(sym.translations[s_idx])
            phase = np.exp(-1j * (g_rot_raw.astype(float) @ tau))
            U_s = U_spinor[s_idx]
            psi_b = np.einsum('ab,nbk->nak', U_s, psi_kbar) * phase[None, None, :]
            k_full = s.unfolded_kpts[kf]
            kg0 = np.rint(k_full - sym_krep @ sym.kpoints[kirr]).astype(np.int64)
            g_full = g_rot_raw - kg0[None, :]
            kn = find_nos_k(kf)
            ngk_c = int(nos.ngk[kn])
            start = int(nos._kpt_starts[kn])
            raw = nos._coeffs_raw[b_lo:b_hi, :, start:start+ngk_c, :]
            psi_c = raw[..., 0] + 1j*raw[..., 1]
            g_c = nos._gvecs_raw[start:start+ngk_c]
            box_b = scatter_to_box(psi_b, g_full, ngk, fft_grid)
            box_c = scatter_to_box(psi_c, g_c, ngk_c, fft_grid)
            e_kirr = np.asarray(sym.energies[0, kirr, b_lo:b_hi])
            u, gr = compare_groups(box_b, box_c, e_kirr)
            print(f"{name:40s}  {tag:30s}  {u:12.3e}  {gr:12.3e}")
        print()


if __name__ == "__main__":
    main()
