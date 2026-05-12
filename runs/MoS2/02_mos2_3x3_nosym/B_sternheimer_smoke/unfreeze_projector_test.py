"""Test: unfreezing the projector changes ∂χ/∂q (frozen-vs-unfrozen), and
FD matches the unfrozen jvp tangent better than the frozen one."""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import time
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.run_sternheimer import (
    _psi_box_to_G_sphere, build_Gprime_list,
    build_sternheimer_source, chi_col_contrib_at_kvec_traced,
    compute_kp_tangent_at_kvec,
    make_density_perturbation, make_umklapp_phase,
)
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
nk_full = int(sym.nk_tot); nspinor = int(wfn.nspinor)
psi_box_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(nk_full)], axis=0)
en_full = jnp.asarray(wfn.energies[0, np.asarray(sym.irk_to_k_map), :n_occ])

# ── Fix q_base at (1/3, 1/3)  (K-orbit, well-defined test) ──
iq_base = 4
q_pos = np.asarray(wfn.kpoints[iq_base])
q_base_signed = q_pos - np.round(q_pos)

V_cell = float(wfn.cell_volume); N_grid = int(np.prod(wfn.fft_grid))
prefactor = jnp.asarray((2.0 * 1 * np.sqrt(N_grid)) / (V_cell * nk_full), dtype=jnp.float64)
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
en_occ = np.asarray(en_full)
alpha_pv_j = jnp.asarray(2.0 * (en_occ.max() - en_occ.min()), dtype=jnp.float64)

# Build per-k fixtures
per_k = []
for ik in range(nk_full):
    ik_kminq = int(sym.kq_map[ik, iq_base])
    kvec_k = np.asarray(sym.unfolded_kpts[ik])
    kvec_kminq_wrap = np.asarray(sym.unfolded_kpts[ik_kminq])
    H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    H_kminq = setup_H_k_from_kvec(kvec_kminq_wrap, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    Gkminq_int = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)
    U_val_k_box = psi_box_full[ik, :n_occ]
    U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int)
    U_val_kminq_G = _psi_box_to_G_sphere(psi_box_full[ik_kminq, :n_occ], Gkminq_int)
    G_wrap = np.rint((kvec_k - q_base_signed) - kvec_kminq_wrap).astype(np.int32)
    phase_wrap = make_umklapp_phase(G_wrap, wfn.fft_grid, sign=+1)
    phase_unwrap = make_umklapp_phase(G_wrap, wfn.fft_grid, sign=-1)
    V_pert = make_density_perturbation(wfn.fft_grid) * phase_wrap
    Q_kminq = make_Q_kminq(U_val_kminq_G)
    b = build_sternheimer_source(U_val_k_box, Gkminq_int, V_pert, Q_kminq)
    eps_vk = en_full[ik, :n_occ]
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)

    per_k.append(dict(
        kvec_kminq_wrap_j=jnp.asarray(kvec_kminq_wrap, dtype=jnp.float64),
        Gkminq_int_np=np.asarray(Gkminq_int),
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        V_scf=H_kminq.V_scf, mask=H_kminq.mask, vnl_E_super=H_kminq.vnl_E,
        U_val_kminq_G=U_val_kminq_G, eps_v=eps_vk,
        precond_diag=precond_diag,
        U_val_k_box=U_val_k_box, b=b, phase_unwrap=phase_unwrap,
        eps_v_at_kvec_p=en_full[ik_kminq, :n_occ],   # eigenvalues AT kvec_p
    ))

ng_out = 4
Gprime_int = build_Gprime_list(q_base_signed, wfn, ng_out)
Gprime_j = jnp.asarray(Gprime_int)
q_base_j = jnp.asarray(q_base_signed, dtype=jnp.float64)

# ── Precompute k·p tangent of U_val at each k_wrap ──
print("Precomputing k·p tangents of U_val (3 batched solves per k_wrap)...")
t0 = time.perf_counter()
for pk in per_k:
    # Note: for the k·p response we need eigenvalues AT kvec_p, which may need
    # H_kminq's TPA precond — we reuse the same precond since the preconditioner
    # is diagonal in G and the perturbation is the same physical operator.
    pk['U_val_grad_kp'] = compute_kp_tangent_at_kvec(
        kvec_p_base_np=np.asarray(pk['kvec_kminq_wrap_j']),
        Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
        V_scf=pk['V_scf'], mask=pk['mask'],
        Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
        U_val_G=pk['U_val_kminq_G'],
        eps_v_at_kvec_p=pk['eps_v_at_kvec_p'],
        alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
        tol=1e-10, max_iter=200,
    )
    pk['U_val_grad_kp'].block_until_ready()
print(f"k·p precompute time = {time.perf_counter()-t0:.1f}s "
      f"  (for {nk_full} k-wraps × 3 directions = {3*nk_full} batched solves)")

def make_chi_total(use_unfrozen_projector):
    def chi_col_total(qvec):
        chi = jnp.zeros((ng_out,), dtype=jnp.complex128)
        for pk in per_k:
            kvec_p = pk['kvec_kminq_wrap_j'] - (qvec - q_base_j)
            U_val_grad = pk['U_val_grad_kp'] if use_unfrozen_projector else None
            kvec_p_base = pk['kvec_kminq_wrap_j'] if use_unfrozen_projector else None
            chi += chi_col_contrib_at_kvec_traced(
                kvec_p_traced=kvec_p,
                Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
                V_scf=pk['V_scf'], mask=pk['mask'],
                Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
                fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
                U_val_kminq_G=pk['U_val_kminq_G'], eps_v=pk['eps_v'],
                alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
                U_val_k_box=pk['U_val_k_box'], b=pk['b'],
                Gprime_int=Gprime_j, phase_unwrap=pk['phase_unwrap'],
                prefactor=prefactor,
                tol=1e-10, max_iter=200,
                U_val_grad_kp=U_val_grad,
                kvec_p_base_for_grad=kvec_p_base,
            )
        return chi
    return chi_col_total

chi_frozen   = make_chi_total(False)
chi_unfrozen = make_chi_total(True)

# Primal (must match between frozen/unfrozen — they differ only in TANGENT)
c_f = chi_frozen(q_base_j); c_f.block_until_ready()
c_u = chi_unfrozen(q_base_j); c_u.block_until_ready()
print(f"\nPrimal χ at q=(1/3,1/3)  frozen:  {complex(c_f[0]):+.6e}")
print(f"                         unfrozen: {complex(c_u[0]):+.6e}")
print(f"  (should match — projector affects only the derivative)")

# JVP at q_x
dq_x = jnp.asarray([1., 0., 0.], dtype=jnp.float64)
for label, chi_fn in [('frozen', chi_frozen), ('unfrozen', chi_unfrozen)]:
    _, chi_dot = jax.jvp(chi_fn, (q_base_j,), (dq_x,))
    chi_dot.block_until_ready()
    h = 1e-5
    chi_p = chi_fn(q_base_j + h*dq_x); chi_m = chi_fn(q_base_j - h*dq_x)
    chi_dot_fd = (chi_p - chi_m) / (2*h)
    rel = float(jnp.linalg.norm(chi_dot - chi_dot_fd)) / float(jnp.linalg.norm(chi_dot_fd) + 1e-30)
    dnorm = float(jnp.linalg.norm(chi_dot))
    print(f"\n[{label}]  ||∂χ/∂q_x||_JVP = {dnorm:.6e}")
    print(f"            FD rel_err      = {rel:.3e}")
    print(f"            ∂χ_{{G'=0}}/∂q_x (JVP)  = {complex(chi_dot[0]):+.6e}")
    print(f"            ∂χ_{{G'=0}}/∂q_x (FD)   = {complex(chi_dot_fd[0]):+.6e}")
