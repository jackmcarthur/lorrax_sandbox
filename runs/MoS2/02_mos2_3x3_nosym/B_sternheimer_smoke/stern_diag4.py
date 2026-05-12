"""Where does NaN appear in chi accumulation at q=0?"""
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
from psp.run_sternheimer import (
    build_sternheimer_source, accumulate_chi_density,
    project_density_to_Gsphere, make_density_perturbation, build_Gprime_list,
    _psi_box_to_G_sphere,
)
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
psi_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(sym.nk_tot)], axis=0)
print(f"psi_full NaN? {bool(jnp.any(jnp.isnan(psi_full)))}")

iq = 0
qvec = np.asarray(wfn.kpoints[iq])
V_pert_real = make_density_perturbation(wfn.fft_grid)
Gprime = build_Gprime_list(qvec, wfn, 16)
delta_n_r = jnp.zeros(wfn.fft_grid, dtype=jnp.complex128)
gvecs_full = [np.asarray(sym.get_gvecs_kfull(wfn, ik), dtype=np.int32) for ik in range(sym.nk_tot)]

for ik_full in range(sym.nk_tot):
    ik_kminq = int(sym.kq_map[ik_full, iq])
    kvec_kminq = np.asarray(sym.unfolded_kpts[ik_kminq])
    H_kminq = setup_H_k_from_kvec(kvec_kminq, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    apply_H_kminq = make_apply_H(H_kminq)
    Gkminq_int = jnp.asarray(gvecs_full[ik_kminq])

    U_val_k_box = psi_full[ik_full, :n_occ]
    U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, jnp.asarray(gvecs_full[ik_full]))
    U_val_kminq_box = psi_full[ik_kminq, :n_occ]
    U_val_kminq_G = _psi_box_to_G_sphere(U_val_kminq_box, Gkminq_int)
    Q_kminq = make_Q_kminq(U_val_kminq_G)

    eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])

    b = build_sternheimer_source(U_val_k_box, Gkminq_int, V_pert_real, Q_kminq)
    H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)
    def apply_A(x, aH=apply_H_kminq, Q=Q_kminq, ep=eps_vk):
        return Q(aH(x) - ep[:, None, None].astype(x.dtype) * x)
    delta_u, info = minres(apply_A, -b, precond=precond, project=Q_kminq, tol=1e-6, max_iter=50)
    delta_n_contrib = accumulate_chi_density(U_val_k_box, delta_u, Gkminq_int, wfn.fft_grid)

    # Diagnostics per k
    b_nan = bool(jnp.any(jnp.isnan(b)))
    du_nan = bool(jnp.any(jnp.isnan(delta_u)))
    dn_nan = bool(jnp.any(jnp.isnan(delta_n_contrib)))
    bnorm = float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(b)**2, axis=(1,2)))))
    dnorm = float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(delta_u)**2, axis=(1,2)))))
    res = float(jnp.max(info.res_norms))
    print(f'ik={ik_full}: b_nan={b_nan} du_nan={du_nan} dn_nan={dn_nan}  '
          f'||b||={bnorm:.2e} ||du||={dnorm:.2e} res={res:.2e}')
    delta_n_r = delta_n_r + delta_n_contrib

print(f'\nFinal delta_n_r NaN? {bool(jnp.any(jnp.isnan(delta_n_r)))}  |max|={float(jnp.max(jnp.abs(delta_n_r))):.3e}')
chi_col = project_density_to_Gsphere(delta_n_r, jnp.asarray(Gprime))
print(f'chi_col NaN? {bool(jnp.any(jnp.isnan(chi_col)))} first few: {np.asarray(chi_col[:5])}')
