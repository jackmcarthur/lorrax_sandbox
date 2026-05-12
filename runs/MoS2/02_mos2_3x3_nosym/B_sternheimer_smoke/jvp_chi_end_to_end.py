"""End-to-end ∂χ_{G'0}(q, 0)/∂q via jax.jvp through the FULL pipeline:
   qvec → δu → δn → FFT → χ_col[G'=0].

Sum of per-k contributions, each computed by
``chi_col_contrib_at_kvec_traced``.  Validated against FD.
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
from psp.run_sternheimer import (
    build_sternheimer_source, make_density_perturbation, make_umklapp_phase,
    build_Gprime_list, _psi_box_to_G_sphere,
    chi_col_contrib_at_kvec_traced,
)
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

nk_full = int(sym.nk_tot)
psi_box_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(nk_full)], axis=0)
en_full = jnp.asarray(wfn.energies[0, np.asarray(sym.irk_to_k_map), :n_occ])
nspinor = int(wfn.nspinor)
spin_factor = 2 if nspinor == 1 else 1

# α_pv (global)
en_occ = np.asarray(en_full)
alpha_pv = float(2.0 * (en_occ.max() - en_occ.min()))
alpha_pv_j = jnp.asarray(alpha_pv, dtype=jnp.float64)

# χ prefactor (Adler-Wiser):
N_grid = int(np.prod(wfn.fft_grid))
V_cell = float(wfn.cell_volume)
prefactor = jnp.asarray(
    (2.0 * spin_factor * np.sqrt(N_grid)) / (V_cell * nk_full), dtype=jnp.float64)
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)

# Per-k scaffolding: cache kvec_kminq_base for each k (at q_base), Gkminq, etc.
iq_base = 4  # signed q = (1/3, 1/3)
q_base = np.asarray(wfn.kpoints[iq_base])
q_base = q_base - np.round(q_base)            # signed representative

def per_k_data(ik_full, q_base_signed):
    kvec_k = np.asarray(sym.unfolded_kpts[ik_full])
    ik_kminq = int(sym.kq_map[ik_full, iq_base])
    kvec_kminq_wrap = np.asarray(sym.unfolded_kpts[ik_kminq])
    H_k     = setup_H_k_from_kvec(kvec_k,             V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    H_kminq = setup_H_k_from_kvec(kvec_kminq_wrap,    V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    Gk_int_j     = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    Gkminq_int_j = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)
    Gkminq_np = np.asarray(Gkminq_int_j)

    U_val_k_box = psi_box_full[ik_full, :n_occ]
    U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int_j)
    U_val_kminq_G = _psi_box_to_G_sphere(psi_box_full[ik_kminq, :n_occ], Gkminq_int_j)

    G_wrap_np = np.rint((kvec_k - q_base_signed) - kvec_kminq_wrap).astype(np.int32)
    phase_wrap   = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=+1)
    phase_unwrap = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=-1)
    V_pert = make_density_perturbation(wfn.fft_grid) * phase_wrap
    Q = make_Q_kminq(U_val_kminq_G)
    b = build_sternheimer_source(U_val_k_box, Gkminq_int_j, V_pert, Q)

    eps_vk = en_full[ik_full, :n_occ]
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)

    return dict(
        Gkminq_int_np=Gkminq_np,
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        V_scf=H_kminq.V_scf, mask=H_kminq.mask,
        vnl_E_super=H_kminq.vnl_E,
        U_val_kminq_G=U_val_kminq_G, eps_v=eps_vk,
        precond_diag=precond_diag,
        U_val_k_box=U_val_k_box,
        b=b,
        phase_unwrap=phase_unwrap,
        kvec_kminq_wrap_j=jnp.asarray(kvec_kminq_wrap, dtype=jnp.float64),
    )

ng_out = 4
Gprime_int = build_Gprime_list(q_base, wfn, ng_out)
Gprime_j = jnp.asarray(Gprime_int)
q_base_j = jnp.asarray(q_base, dtype=jnp.float64)

per_k = [per_k_data(ik, q_base) for ik in range(nk_full)]

def chi_col_total(qvec):
    """qvec → χ_col[G']  (ng_out,)   — sum over all full-BZ k."""
    chi = jnp.zeros((ng_out,), dtype=jnp.complex128)
    for pk in per_k:
        kvec_p = pk['kvec_kminq_wrap_j'] - (qvec - q_base_j)
        chi += chi_col_contrib_at_kvec_traced(
            kvec_p_traced=kvec_p,
            Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
            V_scf=pk['V_scf'], mask=pk['mask'],
            Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
            fft_grid=wfn.fft_grid,
            bdot=bdot, vnl_E_super=pk['vnl_E_super'],
            U_val_kminq_G=pk['U_val_kminq_G'], eps_v=pk['eps_v'],
            alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
            U_val_k_box=pk['U_val_k_box'], b=pk['b'],
            Gprime_int=Gprime_j, phase_unwrap=pk['phase_unwrap'],
            prefactor=prefactor,
            tol=1e-10, max_iter=200, use_schur=False,
        )
    return chi

# Primal
chi0 = chi_col_total(q_base_j); chi0.block_until_ready()
print(f"χ_col(q_base) at G'={np.asarray(Gprime_int[:ng_out]).tolist()}:")
for i in range(ng_out):
    print(f"    G'={Gprime_int[i].tolist()}:  χ = {complex(chi0[i]):+.6e}")

# JVP along q_x and q_y
for direction_name, dq in [('x', [1., 0., 0.]), ('y', [0., 1., 0.])]:
    dq_j = jnp.asarray(dq, dtype=jnp.float64)
    _, chi_dot = jax.jvp(chi_col_total, (q_base_j,), (dq_j,)); chi_dot.block_until_ready()

    # FD
    h = 1e-5
    chi_p = chi_col_total(q_base_j + h*dq_j)
    chi_m = chi_col_total(q_base_j - h*dq_j)
    chi_dot_fd = (chi_p - chi_m) / (2*h)

    rel = float(jnp.linalg.norm(chi_dot - chi_dot_fd)) / float(jnp.linalg.norm(chi_dot_fd) + 1e-30)
    print(f"\n∂χ_col/∂q_{direction_name}:")
    for i in range(ng_out):
        print(f"    G'={Gprime_int[i].tolist()}:  JVP={complex(chi_dot[i]):+.4e}   FD={complex(chi_dot_fd[i]):+.4e}")
    print(f"  rel_err(JVP vs FD) = {rel:.3e}")
