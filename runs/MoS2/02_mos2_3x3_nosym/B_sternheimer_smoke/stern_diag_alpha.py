"""Does bumping alpha_pv fix the CG divergence?"""
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
from solvers.sternheimer_precond import compute_per_band_kinetic, make_tpa_preconditioner

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
Gk_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_full), dtype=np.int32))

psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_k_G = _psi_box_to_G_sphere(psi_k, Gk_int)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
P_val_kminq = make_P_val(U_val_kminq_G)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
b = build_sternheimer_source(psi_k, Gkminq_int, make_density_perturbation(wfn.fft_grid), Q_kminq)

H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)

# Confirm A is PD by computing <x, Ax>/|x|^2 on random vectors per v
rng = np.random.default_rng(0)
x_rand = jnp.asarray(rng.standard_normal(b.shape) + 1j*rng.standard_normal(b.shape))

for alpha_pv in [8.688, 20.0, 50.0, 100.0, 500.0]:
    def apply_A(x, ap=alpha_pv):
        return (apply_H_kminq(x) - eps_vk[:, None, None].astype(x.dtype) * x
                + ap * P_val_kminq(x))
    # Rayleigh quotient on random vector, per band
    Ax = apply_A(x_rand)
    xAx = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(x_rand), Ax))
    xx  = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(x_rand), x_rand))
    RQ = xAx / xx
    print(f"alpha={alpha_pv:6.2f}  Rayleigh min={float(jnp.min(RQ)):.3f}  max={float(jnp.max(RQ)):.3f}  (sign of Rayleigh ≈ sign of spectrum extremes)")
    x, info = cg_posdef(apply_A, -b, precond=precond, tol=1e-6, max_iter=50)
    xmax = float(jnp.max(jnp.abs(x)))
    bad = np.sum(~np.isfinite(np.asarray(jnp.sqrt(jnp.sum(jnp.abs(x)**2, axis=(1,2))))))
    print(f"            |x|max={xmax:.2e}  bad bands={bad}/{b.shape[0]}")
