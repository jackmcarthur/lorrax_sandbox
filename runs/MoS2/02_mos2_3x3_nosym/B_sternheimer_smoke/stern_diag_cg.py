"""Diagnose CG on the real MoS2 Sternheimer op."""
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

iq = 1; ik_full = 1  # the failing case
ik_kminq = int(sym.kq_map[ik_full, iq])
kvec_kminq = np.asarray(sym.unfolded_kpts[ik_kminq])
print(f"iq={iq} ik_full={ik_full} ik_kminq={ik_kminq} kvec_kminq={kvec_kminq}")
H_kminq = setup_H_k_from_kvec(kvec_kminq, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H_kminq = make_apply_H(H_kminq)
Gkminq_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32))
Gk_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_full), dtype=np.int32))

psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_k_box = psi_k
U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
P_val_kminq = make_P_val(U_val_kminq_G)
Q_kminq = make_Q_kminq(U_val_kminq_G)

eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
print(f"eps_vk range: [{float(jnp.min(eps_vk)):.3f}, {float(jnp.max(eps_vk)):.3f}] Ry")
# Build source
b = build_sternheimer_source(U_val_k_box, Gkminq_int, make_density_perturbation(wfn.fft_grid), Q_kminq)
print(f"||b|| per v:   max={float(jnp.max(jnp.sqrt(jnp.sum(jnp.abs(b)**2, axis=(1,2))))):.3e}  min={float(jnp.min(jnp.sqrt(jnp.sum(jnp.abs(b)**2, axis=(1,2))))):.3e}")
print(f"<U_p, b>       max={float(jnp.max(jnp.abs(jnp.einsum('nsG,vsG->vn', jnp.conj(U_val_kminq_G), b)))):.2e}")

H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
print(f"K_bar_sq: min={float(jnp.min(K_bar_sq)):.3f}  max={float(jnp.max(K_bar_sq)):.3f}")
precond = make_tpa_preconditioner(H_kminq.T_diag, K_bar_sq)

# α_pv = 2 (E_max - E_min) of occupied
alpha_pv = float(2.0 * (np.max(np.asarray(eps_vk)) - np.min(np.asarray(eps_vk))))
print(f"α_pv = {alpha_pv:.3f} Ry")

def apply_A(x):
    return (apply_H_kminq(x) - eps_vk[:, None, None].astype(x.dtype) * x
            + alpha_pv * P_val_kminq(x))

# Try progressively longer CG with TPA
for m in [1, 2, 3, 5, 10, 20, 50]:
    x, info = cg_posdef(apply_A, -b, precond=precond, tol=1e-6, max_iter=m)
    xmax = float(jnp.max(jnp.abs(x)))
    xnorm = np.asarray(jnp.sqrt(jnp.sum(jnp.abs(x)**2, axis=(1,2))))
    print(f"  m={m:3d}: |x|max={xmax:.2e}  ||x||_v[:5]={xnorm[:5]}  res[:5]={np.asarray(info.res_norms[:5])}")
