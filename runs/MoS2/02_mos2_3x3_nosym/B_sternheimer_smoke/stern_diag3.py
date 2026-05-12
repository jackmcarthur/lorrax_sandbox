"""Trace NaN location in MINRES."""
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
from solvers.minres import minres
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, make_tpa_preconditioner

wfn = WFNReader('WFN.h5')
sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

iq = 0        # q=0 case for diagnostic
ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq])
kvec_kminq = np.asarray(sym.unfolded_kpts[ik_kminq])
H_kminq = setup_H_k_from_kvec(kvec_kminq, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H_kminq = make_apply_H(H_kminq)
Gkminq = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32))

psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full,  n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
Gk = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_full), dtype=np.int32))

def gather(box, G):
    nx, ny, nz = box.shape[-3:]
    ix = jnp.mod(G[:,0], nx); iy = jnp.mod(G[:,1], ny); iz = jnp.mod(G[:,2], nz)
    return box[..., ix, iy, iz]

U_k = gather(psi_k, Gk)
U_p = gather(psi_p, Gkminq)
Q_kminq = make_Q_kminq(U_p)
eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
print(f'eps_vk[:5] = {np.asarray(eps_vk[:5])}')
print(f'eps_vk has NaN: {bool(jnp.any(jnp.isnan(eps_vk)))}')

# Build source with V_pert = 1
V_pert_real = jnp.ones(wfn.fft_grid, dtype=jnp.complex128)
u_r = jnp.fft.ifftn(psi_k, axes=(-3,-2,-1), norm='ortho')
Vu_r = V_pert_real[None, None, :, :, :] * u_r
Vu_box = jnp.fft.fftn(Vu_r, axes=(-3,-2,-1), norm='ortho')
Vu_G = gather(Vu_box, Gkminq)
b = Q_kminq(Vu_G)
print(f'||b|| per v: min={float(jnp.min(jnp.sqrt(jnp.sum(jnp.abs(b)**2,axis=(1,2))))):.3e} max={float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(b)**2,axis=(1,2))))):.3e}')
print(f'b has NaN: {bool(jnp.any(jnp.isnan(b)))}')

# TPA precond
H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
K_bar_sq = compute_per_band_kinetic(U_k, H_k.T_diag)
print(f'K_bar_sq range: [{float(jnp.min(K_bar_sq)):.3e}, {float(jnp.max(K_bar_sq)):.3e}]')

precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)

def apply_A(x):
    return Q_kminq(apply_H_kminq(x) - eps_vk[:, None, None].astype(x.dtype) * x)

delta_u, info = minres(apply_A, -b, precond=precond, project=Q_kminq, tol=1e-6, max_iter=50)
print(f'info.res_norms (first 5) = {np.asarray(info.res_norms[:5])}')
print(f'info.converged (any NaN?) = {bool(jnp.any(jnp.isnan(info.res_norms)))}')
print(f'δu has NaN: {bool(jnp.any(jnp.isnan(delta_u)))}')
print(f'||δu|| per v: min={float(jnp.min(jnp.sqrt(jnp.sum(jnp.abs(delta_u)**2,axis=(1,2))))):.3e} max={float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(delta_u)**2,axis=(1,2))))):.3e}')
