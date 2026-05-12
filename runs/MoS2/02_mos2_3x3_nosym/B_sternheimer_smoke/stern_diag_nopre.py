"""CG without preconditioner, test 2 bands in isolation"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.h_dft import make_apply_H
from psp.run_sternheimer import build_sternheimer_source, make_density_perturbation, _psi_box_to_G_sphere
from solvers.cg_posdef import cg_posdef
from solvers.projectors import make_P_val, make_Q_kminq

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

iq, ik_full = 1, 1
ik_kminq = int(sym.kq_map[ik_full, iq])
H_kminq = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_kminq]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H_kminq = make_apply_H(H_kminq)
Gkminq_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32))
psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
P_val_kminq = make_P_val(U_val_kminq_G)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
b = build_sternheimer_source(psi_k, Gkminq_int, make_density_perturbation(wfn.fft_grid), Q_kminq)

# TRY ONLY 2 BANDS: v=0 (works) and v=8 (blows up)
v_test = jnp.array([0, 8])
b2 = b[v_test]
eps_vk2 = eps_vk[v_test]
alpha_pv = 8.688
def apply_A(x):
    return (apply_H_kminq(x) - eps_vk2[:, None, None].astype(x.dtype) * x
            + alpha_pv * P_val_kminq(x))

# No precond
x, info = cg_posdef(apply_A, -b2, tol=1e-6, max_iter=200)
print("No precond, 200 iters:")
print(f"  ||x|| per v = {np.asarray(jnp.sqrt(jnp.sum(jnp.abs(x)**2, axis=(1,2))))}")
print(f"  res = {np.asarray(info.res_norms)}")

# Let me also try JUST band 8 alone
b1 = b[jnp.array([8])]
eps_vk1 = eps_vk[jnp.array([8])]
def apply_A1(x):
    return (apply_H_kminq(x) - eps_vk1[:, None, None].astype(x.dtype) * x
            + alpha_pv * P_val_kminq(x))
x1, info1 = cg_posdef(apply_A1, -b1, tol=1e-6, max_iter=200)
print(f"\nBand 8 alone, 200 iters:  ||x||={float(jnp.sqrt(jnp.sum(jnp.abs(x1)**2))):.2e}  res={np.asarray(info1.res_norms)}")

# How about using numpy to solve it directly, small enough?
# Actually too big (ngk=1947) but let me try Krylov subspace from scipy
from scipy.sparse.linalg import LinearOperator, cg as spcg
nG = b1.shape[-1]
dim = 2 * nG   # nspinor=2
def A_np(xflat):
    x = jnp.asarray(xflat.reshape(1, 2, nG), dtype=jnp.complex128)
    y = apply_A1(x)
    return np.asarray(y).ravel()
A_op = LinearOperator((dim, dim), matvec=A_np, dtype=np.complex128)
b1_np = -np.asarray(b1).ravel()
x_sp, info_sp = spcg(A_op, b1_np, rtol=1e-6, maxiter=200)
print(f"scipy cg band 8: info={info_sp} ||x||={np.linalg.norm(x_sp):.2e}")
