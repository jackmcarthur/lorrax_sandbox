"""End-to-end ∂δu/∂q via jax.jvp, with EXACT VNL derivative threaded through.

Previous version ``jvp_q_test.py`` used the 'freeze V_NL' approximation
(only T_diag differentiated wrt q).  This upgrade recomputes both T_diag
and the VNL Z matrix from a traced kvec using the already-JAX-friendly
``_build_vnl_kdata_core`` — gets the exact ∂H/∂q tangent.

Also exercises the user-pointed-out identity  ∂V_NL/∂q = −∂V_NL/∂k
when k-q is the Bloch momentum of the H we're solving at.
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
                                  make_umklapp_phase, _psi_box_to_G_sphere)
from psp.vnl_ops import _build_vnl_kdata_core, build_vnl_setup
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag
from solvers.sternheimer_solve import SternheimerOp, sternheimer_solve

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

# Base q = (1/3, 1/3), ik_full=0, non-wrapping
q_base = np.asarray([1.0/3, 1.0/3, 0.0])
iq_base = 4; ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq_base])
kvec_k = np.asarray(sym.unfolded_kpts[ik_full])
kvec_kminq_base = np.asarray(sym.unfolded_kpts[ik_kminq])

# H at base q
H_k     = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
H_kminq = setup_H_k_from_kvec(kvec_kminq_base, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int_j     = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
Gkminq_int   = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)
Gkminq_np    = np.asarray(Gkminq_int)

psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_k_G     = _psi_box_to_G_sphere(psi_k, Gk_int_j)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk  = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)
alpha_pv = 2.0 * (np.max(np.asarray(eps_vk)) - np.min(np.asarray(eps_vk)))
V_pert_real = make_density_perturbation(wfn.fft_grid)                # q_base has no wrap → phase_wrap = 1
b = build_sternheimer_source(psi_k, Gkminq_int, V_pert_real, Q_kminq)

bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
Gk_float = Gkminq_int.astype(jnp.float64)
kvec_kminq_base_j = jnp.asarray(kvec_kminq_base, dtype=jnp.float64)
q_base_j = jnp.asarray(q_base, dtype=jnp.float64)
E_super_frozen = H_kminq.vnl_E                                        # ε_super doesn't depend on k

def rebuild_op_at_q(qvec):
    """Full-JAX rebuilder: T_diag AND vnl_Z recomputed from traced kvec_kminq(q)."""
    kvec_p = kvec_kminq_base_j - (qvec - q_base_j)                    # kvec(k-q)
    # T_diag = |k+G|² with bdot metric (crystal coords → Ry kinetic).
    kG = Gk_float + kvec_p[None, :]
    T_diag = jnp.einsum('gi,ij,gj->g', kG, bdot, kG)
    # Exact VNL Z via the jax-friendly kernel.  Pass Gk as a static numpy array
    # (integers don't need derivatives); kvec_p is the traced tangent-carrier.
    kdata = _build_vnl_kdata_core(kvec_p, Gkminq_np, vnl_setup, compute_dZ=False)
    return SternheimerOp(
        T_diag=T_diag, V_scf=H_kminq.V_scf,
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        vnl_Z=kdata.Z, vnl_E=E_super_frozen, mask=H_kminq.mask,
        U_val=U_val_kminq_G, eps_v=eps_vk,
        alpha_pv=jnp.asarray(alpha_pv, dtype=jnp.float64),
        precond_diag=precond_diag, fft_grid=H_kminq.fft_grid,
    )

def solve_delta_u(qvec):
    op = rebuild_op_at_q(qvec)
    return sternheimer_solve(op, b, tol=1e-10, max_iter=200)

# Primal
x_primal = solve_delta_u(q_base_j); x_primal.block_until_ready()
print(f"||δu(q_base)||  = {float(jnp.sqrt(jnp.sum(jnp.abs(x_primal)**2))):.6e}")

# JVP along q_x
dq_x = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
_, x_dot = jax.jvp(solve_delta_u, (q_base_j,), (dq_x,))
x_dot.block_until_ready()
x_dot_norm = float(jnp.sqrt(jnp.sum(jnp.abs(x_dot)**2)))
print(f"||∂δu/∂q_x via JVP (exact VNL)|| = {x_dot_norm:.6e}")

# FD (full-path)
h = 1e-5
x_plus  = solve_delta_u(q_base_j + h*dq_x)
x_minus = solve_delta_u(q_base_j - h*dq_x)
x_dot_fd = (x_plus - x_minus) / (2*h)
x_dot_fd_norm = float(jnp.sqrt(jnp.sum(jnp.abs(x_dot_fd)**2)))
print(f"||∂δu/∂q_x via FD||             = {x_dot_fd_norm:.6e}")

diff = x_dot - x_dot_fd
rel_err = float(jnp.sqrt(jnp.sum(jnp.abs(diff)**2))) / x_dot_fd_norm
print(f"  rel_err(JVP vs FD) = {rel_err:.3e}")
print(f"  max|diff|          = {float(jnp.max(jnp.abs(diff))):.3e}")
