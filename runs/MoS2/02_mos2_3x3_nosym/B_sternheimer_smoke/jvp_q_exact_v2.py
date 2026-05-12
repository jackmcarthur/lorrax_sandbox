"""End-to-end ∂δu/∂q with exact VNL, using the public
``psp.run_sternheimer.build_sternheimer_op_at_kvec_traced`` helper.

Also exercises nested jvp for ∂²δu/∂q² to validate composition.
"""
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
from psp.run_sternheimer import (build_sternheimer_source, make_density_perturbation,
                                  _psi_box_to_G_sphere,
                                  build_sternheimer_op_at_kvec_traced)
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag
from solvers.sternheimer_solve import sternheimer_solve

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

q_base = np.asarray([1.0/3, 1.0/3, 0.0])
iq_base = 4; ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq_base])
kvec_k = np.asarray(sym.unfolded_kpts[ik_full])
kvec_kminq_base = np.asarray(sym.unfolded_kpts[ik_kminq])

H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
H_kminq = setup_H_k_from_kvec(kvec_kminq_base, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int_j = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
Gkminq_int = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)
Gkminq_np = np.asarray(Gkminq_int)

psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_k_G = _psi_box_to_G_sphere(psi_k, Gk_int_j)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)
alpha_pv = 2.0 * (np.max(np.asarray(eps_vk)) - np.min(np.asarray(eps_vk)))
V_pert = make_density_perturbation(wfn.fft_grid)
b = build_sternheimer_source(psi_k, Gkminq_int, V_pert, Q_kminq)

bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
kvec_kminq_base_j = jnp.asarray(kvec_kminq_base, dtype=jnp.float64)
q_base_j = jnp.asarray(q_base, dtype=jnp.float64)
alpha_pv_j = jnp.asarray(alpha_pv, dtype=jnp.float64)

def solve_delta_u(qvec):
    kvec_p = kvec_kminq_base_j - (qvec - q_base_j)
    op = build_sternheimer_op_at_kvec_traced(
        kvec_p, Gkminq_np, vnl_setup,
        V_scf=H_kminq.V_scf, mask=H_kminq.mask,
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        fft_grid=H_kminq.fft_grid,
        bdot=bdot, vnl_E_super=H_kminq.vnl_E,
        U_val_kminq_G=U_val_kminq_G, eps_v=eps_vk,
        alpha_pv_sc=alpha_pv_j, precond_diag=precond_diag,
    )
    return sternheimer_solve(op, b, tol=1e-10, max_iter=200)

dq_x = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

# 1st jvp
_, x_dot = jax.jvp(solve_delta_u, (q_base_j,), (dq_x,)); x_dot.block_until_ready()
x_dot_norm = float(jnp.sqrt(jnp.sum(jnp.abs(x_dot)**2)))
print(f"||∂δu/∂q_x||_JVP    = {x_dot_norm:.6e}")

# FD cross-check
h = 1e-5
xp = solve_delta_u(q_base_j + h*dq_x); xm = solve_delta_u(q_base_j - h*dq_x)
x_dot_fd = (xp - xm) / (2*h)
re = float(jnp.sqrt(jnp.sum(jnp.abs(x_dot - x_dot_fd)**2))) / float(jnp.sqrt(jnp.sum(jnp.abs(x_dot_fd)**2)))
print(f"||∂δu/∂q_x||_FD     = {float(jnp.sqrt(jnp.sum(jnp.abs(x_dot_fd)**2))):.6e}")
print(f"  rel_err(1st jvp vs FD) = {re:.3e}")

# 2nd jvp — nested  ∂²δu/∂q_x²
def first_deriv(q):
    _, xd = jax.jvp(solve_delta_u, (q,), (dq_x,))
    return xd
_, x_ddot = jax.jvp(first_deriv, (q_base_j,), (dq_x,)); x_ddot.block_until_ready()
x_ddot_norm = float(jnp.sqrt(jnp.sum(jnp.abs(x_ddot)**2)))
print(f"\n||∂²δu/∂q_x²||_nested = {x_ddot_norm:.6e}")

# FD of 1st-deriv
xdp = first_deriv(q_base_j + h*dq_x); xdm = first_deriv(q_base_j - h*dq_x)
x_ddot_fd = (xdp - xdm) / (2*h)
re2 = float(jnp.sqrt(jnp.sum(jnp.abs(x_ddot - x_ddot_fd)**2))) / float(jnp.sqrt(jnp.sum(jnp.abs(x_ddot_fd)**2)))
print(f"||∂²δu/∂q_x²||_FD-of-1st = {float(jnp.sqrt(jnp.sum(jnp.abs(x_ddot_fd)**2))):.6e}")
print(f"  rel_err(nested vs FD-of-1st) = {re2:.3e}")
