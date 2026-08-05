"""STEP 3: the transverse current is COVARIANT (step 2 proved it). Does the
EXACT C_q (CCT) contraction the zeta-fit performs preserve that covariance,
or does the 4x4 gamma-tilde placement break it (esp. sigma_y / gamma2)?

We replicate the code's contraction byte-for-byte:
  pair_density einsum 'kmna,knbr->kabmr':  P[a,b](mu,col) = sum_n psi*_{n,a}(mu) psi_{n,b}(col)
  (a = conjugated/bra index, b = ket index)
  gamma_double_contract(conj(P_l), P_r, gamma_L=gam_i, gamma_R=gam_i):
     C(mu,col) = sum_{ab a'b'} conj(P_l[a,b]) gam_i[a,a'] gam_i[b,b'] P_r[a',b']
  with gam_i[a,a'] = phase[a] delta_{a', perm[a]}  (monomial gamma-tilde)

We build it at q=0 conceptually using the SAME ps i at one centroid mu (the
diagonal C(mu,mu) is the quantity that must rotate covariantly). We then check
whether C^i(child, mu) relates to C^j(parent, alpha(mu)) by the R-rotation that
covariance demands, OR whether the gamma placement pins it to the lab frame.

KEY 4x4 algebra check (the heart):
  Under symmetry the 4-spinor psi rotates by  Lam(S) = diag(U, U)  (lift commutes).
  Covariance of  ψ† γ̃^i ψ  requires  Lam† γ̃^i Lam = sum_j R[j,i] γ̃^j.
  We check this directly for the code's gamma1,gamma2,gamma3 and U_spinor.
  A sigma_y/gamma2 sign error shows up as gamma2 NOT transforming, or transforming
  with a flipped R-row.
"""
import os; os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')
np.set_printoptions(precision=4, suppress=True, linewidth=170)

WFN = 'runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
CENTS = 'runs/CrI3/C_cri3_ibz_active_2026-06-16/centroids_frac_102_current.txt'

with h5py.File(WFN, 'r') as f:
    c, k, s = f['/mf_header/crystal'], f['/mf_header/kpoints'], f['/mf_header/symmetry']
    wfn = SimpleNamespace(
        avec=c['avec'][()], atom_crys=c['apos'][()], atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()], translations=s['tnp'][()], ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()], kpoints=k['rk'][()], nkpts=int(k['nrk'][()]),
        shift=k['shift'][()])
    fft_grid = f['/mf_header/gspace/FFTgrid'][()]
from common.symmetry_maps import SymMaps
sym = SymMaps(wfn)
U = np.asarray(sym.U_spinor)
Rp = np.asarray(sym.R_proper)
sym_idx_k = np.asarray(sym.sym_idx_k)
irr_idx_k = np.asarray(sym.irr_idx_k)
unfolded = np.asarray(sym.unfolded_kpts)
ntran = wfn.ntran
nx, ny, nz = [int(v) for v in fft_grid]

# ---- the EXACT gamma-tilde matrices the code uses (gamma_matrices.py) ----
from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3, gamma_perm_phase
gam = [np.asarray(g) for g in (gamma0, gamma1, gamma2, gamma3)]   # 4x4
print("=== gamma1 (code) ===\n", gam[1].real)
print("=== gamma2 (code) ===\n", gam[2])
print("=== gamma3 (code) ===\n", gam[3].real)

# ---- 4x4 covariance algebra check: Lam = diag(U,U) ----
sig = [None,
       np.array([[0, 1], [1, 0]], complex),
       np.array([[0, -1j], [1j, 0]]),
       np.array([[1, 0], [0, -1]], complex)]
print("\n############ 4x4 GAMMA COVARIANCE: Lam^dag gamma^i Lam =? sum_j R[j,i] gamma^j ############")
for op in (1, 2):
    Us = U[op]
    Lam = np.zeros((4, 4), complex)
    Lam[:2, :2] = Us; Lam[2:, 2:] = Us           # diag(U,U)
    R = Rp[op]
    print(f"\n--- C3 op {op}, R_proper=\n{R}")
    for i in (1, 2, 3):
        lhs = Lam.conj().T @ gam[i] @ Lam
        rhs = sum(R[j - 1, i - 1] * gam[j] for j in (1, 2, 3))
        err = np.abs(lhs - rhs).max()
        # also test the WRONG transform that a transpose/conj bug gives: Lam^T g Lam*
        wrong = (Lam.T @ gam[i] @ np.conj(Lam))
        err_wrong = np.abs(wrong - rhs).max()
        tag = 'COVARIANT' if err < 1e-9 else 'BROKEN'
        print(f"   i={i}: |Lam^dag g^i Lam - sum_j R[j,i] g^j| = {err:.2e}  [{tag}]"
              f"    (|Lam^T g^i Lam* - R-pred|={err_wrong:.2e})")

# ============================================================================
# Now the ACTUAL CCT contraction, at q=0, on the loaded 4-spinor psi.
# C^i(mu) = sum_{m n} |M^i_{mn}(mu)|^2-like object that the fit produces.
# We compute the per-centroid DIAGONAL C^i(mu,mu) exactly as gamma_double_contract,
# then test covariance.
# ============================================================================
from file_io.wfn_loader import WfnLoader
from file_io.centroids import load_centroids
from centroid.orbit_syms import compute_centroid_sym_perm

_, cents_idx, n_rmu = load_centroids(CENTS, tuple(int(v) for v in fft_grid))
cents_idx = np.asarray(cents_idx, np.int64)
sym_perm, _ = compute_centroid_sym_perm(
    cents_idx, wfn.sym_matrices, wfn.translations, fft_grid,
    validate=False, extend_trs=True)

NB = 24
loader = WfnLoader(WFN, backend='eager')
nb = min(NB, int(loader.nbands))
# load 4-spinor psi (bispinor lift) at full BZ — production path
psi4 = np.asarray(loader.load(bands=(0, nb), k='full_bz', sharding=None, bispinor=True))
print(f"\npsi4 (4-spinor) shape={psi4.shape}")    # (nk,nb,4,ngkmax)
gvecs_full = np.asarray(loader.gvecs(k='full_bz'))
ngk_valid = np.asarray(loader.ngk_valid(k='full_bz'))


def sample4(ik):
    gv = gvecs_full[ik][: int(ngk_valid[ik])]
    gi = (gv % np.array([nx, ny, nz]))
    out = np.zeros((nb, 4, n_rmu), complex)
    box = np.zeros((nx, ny, nz), complex)
    for b in range(nb):
        for sp in range(4):
            box[:] = 0.0
            box[gi[:, 0], gi[:, 1], gi[:, 2]] = psi4[ik, b, sp, : int(ngk_valid[ik])]
            r = np.fft.ifftn(box, norm='ortho')
            out[b, sp, :] = r[cents_idx[:, 0], cents_idx[:, 1], cents_idx[:, 2]]
    return out


def perm_phase(mu_L):
    p, ph = gamma_perm_phase(mu_L)
    return np.asarray(p), np.asarray(ph)


def cct_diag(psi_rmu, mu_L):
    """Replicate gamma_double_contract diagonal C^{mu_L}(mu,mu) at q=0.
    P[a,b](mu) = sum_n psi*_{n,a}(mu) psi_{n,b}(mu)   (a=bra/conj, b=ket)
    C(mu) = sum_{ab a'b'} conj(P[a,b]) gam[a,a'] gam[b,b'] P[a',b']
          (gamma on both legs; this is what the fit's CCT diagonal is)
    """
    perm, phase = perm_phase(mu_L)
    # P[a,b,mu] = sum_n conj(psi[n,a,mu]) psi[n,b,mu]
    P = np.einsum('anm,bnm->abm', np.conj(psi_rmu).transpose(1, 0, 2),
                  psi_rmu.transpose(1, 0, 2))
    # actually do it cleanly: psi_rmu is (nb,4,n_rmu)
    P = np.einsum('nam,nbm->abm', np.conj(psi_rmu), psi_rmu)   # (4,4,n_rmu)
    # gamma_double_contract: C = sum_{ab} conj(P_l[a,b]) phase[a] phase[b] P_r[perm[a],perm[b]]
    Pl = P
    Pr = P
    C = np.zeros(n_rmu, complex)
    for a in range(4):
        for b in range(4):
            C += np.conj(Pl[a, b]) * phase[a] * phase[b] * Pr[perm[a], perm[b]]
    return C   # (n_rmu,)  -- C^{mu_L}(mu,mu)


for op in (1, 2):
    cand = np.where(sym_idx_k == op)[0]
    child = None
    for cc in cand:
        if np.linalg.norm(wfn.kpoints[irr_idx_k[cc]]) > 1e-6:
            child = int(cc); break
    if child is None:
        child = int(cand[0])
    kbar = irr_idx_k[child]
    pc = np.where((irr_idx_k == kbar) & (sym_idx_k == 0))[0]
    parent = int(pc[0]) if len(pc) else int(np.where(irr_idx_k == kbar)[0][0])
    print("\n" + "=" * 100)
    print(f"OP {op}: child(full)={child} parent(full)={parent} IBZ={kbar}")
    psc = sample4(child)
    psp = sample4(parent)
    alpha = sym_perm[op]

    # C^{mu_L} for mu_L=1,2,3 at child and parent
    Cc = {m: cct_diag(psc, m) for m in (1, 2, 3)}
    Cp = {m: cct_diag(psp, m) for m in (1, 2, 3)}

    # The diagonal C^i(mu,mu) is a *quadratic* in the current; it should
    # transform like  C^i(child,mu) = sum_{jk} R[j,i]R[k,i] C^{(jk)}(parent,alpha)
    # i.e. NOT a simple vector. The cleaner covariant object is the
    # band-summed current rho^i (step2, already proven covariant).
    # Here we directly compare the per-channel diagonal magnitude AND
    # whether child C^i equals parent C at the permuted centroid (lab) or
    # the R-mixed combination (covariant).
    dens = np.abs(Cp[3]).real
    pick = np.argsort(-dens)[:6]
    print("  Per-channel diagonal C^{mu_L}(mu,mu): is child == parent(alpha) [lab] "
          "or R-mixed [covariant]?")
    for m in (1, 2, 3):
        cc_m = Cc[m][pick].real
        cp_perm = Cp[m][alpha][pick].real
        # covariant prediction for the DIAGONAL quadratic:
        # C^i = sum_jk R[j,i]R[k,i] C^{jk}; we approximate by building the
        # full cross-channel C^{jk} too. For a quick lab-vs-cov verdict we
        # compare child vs parent-permuted directly here and report ratio.
        print(f"   mu_L={m}: child C diag (pick) = {cc_m}")
        print(f"            parent(alpha) C diag = {cp_perm}")
        e = np.abs(Cc[m][alpha[pick]] - Cp[m][alpha[pick]])  # placeholder
    # Build full cross-channel C^{ij}(mu,mu) to test the quadratic covariance law.
    def cct_cross(psi_rmu, mi, mj):
        pi, phi = perm_phase(mi); pj, phj = perm_phase(mj)
        P = np.einsum('nam,nbm->abm', np.conj(psi_rmu), psi_rmu)
        C = np.zeros(n_rmu, complex)
        for a in range(4):
            for b in range(4):
                # left gamma = mi, right gamma = mj
                C += np.conj(P[a, b]) * phi[a] * phj[b] * P[pi[a], pj[b]]
        return C
    print("\n  QUADRATIC covariance law for diagonal C: "
          "C^i_child(mu) =? sum_{j,k} R[j,i] R[k,i] C^{jk}_parent(alpha)")
    R = Rp[op]
    for i in (1, 2, 3):
        pred = np.zeros(n_rmu, complex)
        for j in (1, 2, 3):
            for kk in (1, 2, 3):
                pred += R[j - 1, i - 1] * R[kk - 1, i - 1] * cct_cross(psp, j, kk)[alpha]
        meas = Cc[i]
        rel = np.linalg.norm((meas - pred).real) / (np.linalg.norm(meas.real) + 1e-30)
        rel_lab = np.linalg.norm((meas - Cp[i][alpha]).real) / (np.linalg.norm(meas.real) + 1e-30)
        verdict = ('COVARIANT' if rel < 1e-6 else
                   ('LAB-FRAME' if rel_lab < 1e-6 else 'NEITHER'))
        print(f"   i={i}:  rel|child - R-quad parent| = {rel:.3e}   "
              f"rel|child - parent(perm)| = {rel_lab:.3e}   -> {verdict}")
