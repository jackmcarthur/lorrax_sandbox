"""STEP 1 & 2: does the loaded transverse psi carry U_spinor, and is the
band-summed transverse current rho^i covariant?

We load the 2-spinor psi at full-BZ via the EXACT production unfold
(WfnLoader.load(k='full_bz', bispinor=...) -> eager _eager_build -> unfold_psi,
then _apply_bispinor_lift). We then IFFT psi(G) to the FFT box (same scatter
gflat_to_rmu uses: box[g_index]=psi(G)) and sample at the transverse centroids.

For a C3-child k and its IBZ parent (op s, centroid source-perm alpha(mu)=sym_perm[s,mu]):
  STEP1  n_{ss'}(k,mu) = sum_b psi[k,b,s,mu] conj(psi[k,b,s',mu])   (2x2 spin density)
         covariant  => n(child,mu) = U[s] n(parent, alpha(mu)) U[s]^dag
         lab-frame  => n(child,mu) = n(parent, alpha(mu))
  STEP2  rho^i(k,mu) = sum_b sum_ss' conj(psi[k,b,s,mu]) sig^i_{ss'} psi[k,b,s',mu]
         covariant  => rho^i(child,mu) = sum_j R_proper[s][j,i] rho^j(parent, alpha(mu))
"""
import os; os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')
np.set_printoptions(precision=4, suppress=True, linewidth=160)

WFN = 'runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
CENTS = 'runs/CrI3/C_cri3_ibz_active_2026-06-16/centroids_frac_102_current.txt'

# ---- build a lightweight wfn namespace + SymMaps (host) ----
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
U = np.asarray(sym.U_spinor)           # (ntran,2,2)
Rp = np.asarray(sym.R_proper)          # (2*ntran,3,3) (spatial+TRS same spatial)
sym_idx_k = np.asarray(sym.sym_idx_k)  # (nk_full,)
irr_idx_k = np.asarray(sym.irr_idx_k)  # (nk_full,)  -> IBZ index in wfn.kpoints
unfolded = np.asarray(sym.unfolded_kpts)
nk_full = int(sym.nk_tot)
ntran = wfn.ntran
nx, ny, nz = [int(v) for v in fft_grid]
print(f"nk_full={nk_full} ntran={ntran} fft_grid={nx,ny,nz}")

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], complex)
sig = [sx, sy, sz]

# ---- load centroids (FFT-grid integer indices) ----
from file_io.centroids import load_centroids
_, cents_idx, n_rmu = load_centroids(CENTS, tuple(int(v) for v in fft_grid))
cents_idx = np.asarray(cents_idx, np.int64)   # (n_rmu, 3)
print(f"n_rmu(transverse centroids)={n_rmu}")

# ---- centroid source-permutation alpha(mu)=sym_perm[s,mu] (extend_trs) ----
from centroid.orbit_syms import compute_centroid_sym_perm
sym_perm, L_table = compute_centroid_sym_perm(
    cents_idx, wfn.sym_matrices, wfn.translations, fft_grid,
    validate=False, extend_trs=True)
print(f"sym_perm shape={sym_perm.shape}")

# ---- load 2-spinor psi at full BZ via PRODUCTION unfold (eager) ----
from file_io.wfn_loader import WfnLoader
NB = 24   # band-sum window (occupied + some); band-sum is gauge invariant
loader = WfnLoader(WFN, backend='eager')
print(f"loader.nbands={loader.nbands} nspinor={loader.nspinor}")
nb = min(NB, int(loader.nbands))

# Load WITHOUT bispinor lift first: 2-spinor that carries U_spinor.
psi2 = np.asarray(loader.load(bands=(0, nb), k='full_bz', sharding=None, bispinor=False))
print(f"psi2 (2-spinor full-BZ) shape={psi2.shape}")   # (nk_full, nb, 2, ngkmax)
gvecs_full = np.asarray(loader.gvecs(k='full_bz'))      # (nk_full, ngkmax, 3)
ngk_valid = np.asarray(loader.ngk_valid(k='full_bz'))   # (nk_full,)
ngkmax = psi2.shape[3]


def sample_at_centroids(ik):
    """IFFT psi2[ik] to box, sample at centroid indices. Returns (nb,2,n_rmu)."""
    gv = gvecs_full[ik][: int(ngk_valid[ik])]            # (ngk,3)
    gi = (gv % np.array([nx, ny, nz]))                  # wrap into box
    out = np.zeros((nb, 2, n_rmu), complex)
    box = np.zeros((nx, ny, nz), complex)
    for b in range(nb):
        for sp in range(2):
            box[:] = 0.0
            coeff = psi2[ik, b, sp, : int(ngk_valid[ik])]
            box[gi[:, 0], gi[:, 1], gi[:, 2]] = coeff
            r = np.fft.ifftn(box, norm='ortho')
            out[b, sp, :] = r[cents_idx[:, 0], cents_idx[:, 1], cents_idx[:, 2]]
    return out


def spin_density(psi_rmu):  # (nb,2,n_rmu) -> n_{ss'}(mu) (n_rmu,2,2)
    # n_{ss'} = sum_b psi[b,s] conj(psi[b,s'])
    return np.einsum('bsm,btm->mst', psi_rmu, np.conj(psi_rmu))


def transverse_current(psi_rmu):  # -> rho^i(mu) (3, n_rmu)
    n = spin_density(psi_rmu)     # (n_rmu,2,2)
    rho = np.zeros((3, n_rmu), complex)
    for i in range(3):
        rho[i] = np.einsum('mst,ts->m', n, sig[i])   # tr(n sig^i) = sum_ss' n_ss' sig^i_{ts}
    return rho


# ---- find a C3-about-z child/parent pair ----
# pick op s in [1,2] (order-3 C3) and a full-BZ k whose sym_idx_k==s
def find_pair(op):
    cand = np.where(sym_idx_k == op)[0]
    # avoid k near Gamma (numerically trivial); pick one with nonzero parent k
    for child in cand:
        kbar = irr_idx_k[child]
        if np.linalg.norm(wfn.kpoints[kbar]) > 1e-6:
            return int(child), int(kbar)
    return int(cand[0]), int(irr_idx_k[cand[0]])

# parent in full-BZ index: the full-BZ k that is the identity image of kbar
def parent_full_idx(kbar):
    # full-BZ k mapping to this IBZ with identity op (sym_idx==0 ideally)
    cands = np.where(irr_idx_k == kbar)[0]
    for c0 in cands:
        if sym_idx_k[c0] == 0:
            return int(c0)
    return int(cands[0])

for op in (1, 2):
    child, kbar = find_pair(op)
    parent = parent_full_idx(kbar)
    print("\n" + "=" * 90)
    print(f"OP s={op}  child(full)={child} k={unfolded[child]}  parent(full)={parent} k={unfolded[parent]}  IBZ={kbar} k={wfn.kpoints[kbar]}")
    print(f"   sym_idx_k[child]={sym_idx_k[child]} sym_idx_k[parent]={sym_idx_k[parent]}  R_proper[op]=\n{Rp[op]}")
    Us = U[op]

    pc = sample_at_centroids(child)
    pp = sample_at_centroids(parent)

    nc = spin_density(pc)   # (n_rmu,2,2)
    npar = spin_density(pp)
    rc = transverse_current(pc)   # (3,n_rmu)
    rpar = transverse_current(pp)

    alpha = sym_perm[op]    # source centroid of target mu
    # We compare child centroid mu to parent centroid alpha(mu).

    # ---- STEP 1: n(child,mu) vs U n(parent,alpha) U^dag vs n(parent,alpha) ----
    # pick a few representative centroids with appreciable density
    dens = np.einsum('mss->m', npar).real
    pick = np.argsort(-dens)[:5]
    print("\n  STEP1: spin-density 2x2 at a few high-density centroids")
    for mu in pick:
        a = int(alpha[mu])
        n_child = nc[mu]
        n_par_a = npar[a]
        n_U = Us @ n_par_a @ Us.conj().T
        e_lab = np.abs(n_child - n_par_a).max()
        e_U = np.abs(n_child - n_U).max()
        print(f"   mu={mu:4d} alpha={a:4d}  |n_child - n_par|={e_lab:.3e}   |n_child - U n_par U^dag|={e_U:.3e}"
              f"  -> {'U-APPLIED' if e_U < e_lab*0.3 else ('LAB-FRAME' if e_lab < e_U*0.3 else 'AMBIGUOUS')}")
        if mu == pick[0]:
            print(f"       n_child=\n{n_child}\n       U n_par U^dag=\n{n_U}\n       n_par(alpha)=\n{n_par_a}")

    # ---- STEP 2: rho^i(child,mu) vs sum_j R[j,i] rho^j(parent,alpha) ----
    print("\n  STEP2: transverse current rho^i (i=x,y,z) at high-density centroids")
    print("         compare rho(child,mu) vs R^T-rotated rho(parent,alpha) and vs raw rho(parent,alpha)")
    for mu in pick[:3]:
        a = int(alpha[mu])
        rho_child = rc[:, mu]                        # (3,)
        rho_par = rpar[:, a]                          # (3,)
        rho_rot = np.einsum('ji,j->i', Rp[op], rho_par)   # sum_j R[j,i] rho^j(parent)
        print(f"   mu={mu:4d} alpha={a:4d}")
        print(f"     rho(child)      = {rho_child.real}")
        print(f"     R^T rho(parent) = {rho_rot.real}   (covariant prediction)")
        print(f"     rho(parent)     = {rho_par.real}   (lab-frame/no-rotate)")
        e_cov = np.abs(rho_child - rho_rot).max()
        e_lab = np.abs(rho_child - rho_par).max()
        print(f"     |child - R^T parent|={e_cov:.3e}   |child - parent|={e_lab:.3e}"
              f"  -> {'COVARIANT' if e_cov < e_lab*0.3 else ('LAB-FRAME' if e_lab < e_cov*0.3 else 'AMBIGUOUS')}")

    # aggregate over all centroids (norm of the residual fields)
    rho_child_all = rc                                          # (3,n_rmu)
    rho_par_all = rpar[:, alpha]                                # gather parent at alpha
    rho_rot_all = np.einsum('ji,jm->im', Rp[op], rho_par_all)
    num_cov = np.linalg.norm((rho_child_all - rho_rot_all).real, axis=1)
    num_lab = np.linalg.norm((rho_child_all - rho_par_all).real, axis=1)
    denom = np.linalg.norm(rho_child_all.real, axis=1) + 1e-30
    print("\n  STEP2 aggregate (over all centroids), per channel i=x,y,z:")
    print(f"     rel |child - R^T parent| = {num_cov/denom}")
    print(f"     rel |child - parent|     = {num_lab/denom}")
