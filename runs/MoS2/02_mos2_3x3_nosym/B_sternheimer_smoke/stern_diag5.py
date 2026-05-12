"""Replicate driver's ordering precisely: H_kminq first, then H_k."""
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
from psp.run_sternheimer import (build_sternheimer_source, accumulate_chi_density,
    make_density_perturbation, _psi_box_to_G_sphere)
from solvers.minres import minres
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, make_tpa_preconditioner

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
psi_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(sym.nk_tot)], axis=0)

iq = 0; ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq])
kvec_kminq = np.asarray(sym.unfolded_kpts[ik_kminq])
Gk_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_full), dtype=np.int32))
Gkminq_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32))

V_pert_real = make_density_perturbation(wfn.fft_grid)

# Exactly the driver's pattern (H_kminq first, then H_k):
H_kminq = setup_H_k_from_kvec(kvec_kminq, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H_kminq = make_apply_H(H_kminq)

U_val_k_box = psi_full[ik_full, :n_occ]
U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int)
U_val_kminq_box = psi_full[ik_kminq, :n_occ]
U_val_kminq_G = _psi_box_to_G_sphere(U_val_kminq_box, Gkminq_int)
Q_kminq = make_Q_kminq(U_val_kminq_G)

eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
b = build_sternheimer_source(U_val_k_box, Gkminq_int, V_pert_real, Q_kminq)
print(f'||b|| max = {float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(b)**2, axis=(1,2))))):.3e}')

H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
print(f'K_bar_sq NaN? {bool(jnp.any(jnp.isnan(K_bar_sq)))}')
print(f'H_k.T_diag[0:4] = {np.asarray(H_k.T_diag[:4])}')
print(f'H_kminq.T_diag[0:4] = {np.asarray(H_kminq.T_diag[:4])}')
print(f'H_k.T_diag == H_kminq.T_diag ? {bool(jnp.allclose(H_k.T_diag, H_kminq.T_diag))}')
precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)
def apply_A(x, aH=apply_H_kminq, Q=Q_kminq, ep=eps_vk):
    return Q(aH(x) - ep[:, None, None].astype(x.dtype) * x)
delta_u, info = minres(apply_A, -b, precond=precond, project=Q_kminq, tol=1e-6, max_iter=100)
print(f'\ndelta_u NaN? {bool(jnp.any(jnp.isnan(delta_u)))}')
print(f'|du| max = {float(jnp.max(jnp.abs(delta_u))):.3e}')
print(f'info.res_norms first 3: {np.asarray(info.res_norms[:3])}')
