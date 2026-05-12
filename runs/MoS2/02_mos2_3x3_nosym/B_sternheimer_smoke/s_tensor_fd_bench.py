"""Option A sanity:  S-tensor at q=0  via  FD-of-∂χ/∂q_j  at q = ±h·ê_i.

Evaluates only ∂²χ₀₀/∂q_i∂q_j components (i, j ∈ {x, y}), reports
with a sweep over h to find the optimal step.

Cost per Hessian element:  2 first-derivative evaluations (each = 1 primal +
1 tangent CG through all k-points).  At ~1s per evaluation on MoS2 3×3,
the 4 symmetric {xx, yy, xy} components take ~8s.
"""
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
    build_sternheimer_source, build_sternheimer_source_preQ,
    chi_col_contrib_at_kvec_traced,
    compute_kp_tangent_at_kvec, make_density_perturbation, make_umklapp_phase,
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
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
V_cell = float(wfn.cell_volume); N_grid = int(np.prod(wfn.fft_grid))
prefactor = jnp.asarray((2.0 * 1 * np.sqrt(N_grid)) / (V_cell * nk_full), dtype=jnp.float64)
alpha_pv_j = jnp.asarray(2.0 * (np.asarray(en_full).max() - np.asarray(en_full).min()), dtype=jnp.float64)

# q_base = 0
iq_base = 0
q_base_signed = np.zeros(3)
q_base_j = jnp.asarray(q_base_signed, dtype=jnp.float64)

# Per-k scaffolding at iq=0 (signed q=0)
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
    # Pre-Q source:  V_pert(r)·u_{v,k_s}(r) gathered on the (k-q) G-sphere,
    # before Q-projection.  Passing Vu_G (not a frozen b) is required so the
    # source's q-tangent gets rebuilt with a q-dependent Q_{k-q}(q) inside the
    # traced chi-column function — otherwise ∂²χ/∂q² at q=0 comes out zero
    # (at q=0 exactly, Q_k·u_v = 0 identically → frozen-b stays 0 for any q).
    Vu_G = build_sternheimer_source_preQ(U_val_k_box, Gkminq_int, V_pert)
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)

    per_k.append(dict(
        kvec_kminq_wrap_j=jnp.asarray(kvec_kminq_wrap, dtype=jnp.float64),
        Gkminq_int_np=np.asarray(Gkminq_int),
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        V_scf=H_kminq.V_scf, mask=H_kminq.mask, vnl_E_super=H_kminq.vnl_E,
        U_val_kminq_G=U_val_kminq_G, eps_v=en_full[ik, :n_occ],
        precond_diag=precond_diag,
        U_val_k_box=U_val_k_box, Vu_G=Vu_G, phase_unwrap=phase_unwrap,
        eps_v_at_kvec_p=en_full[ik_kminq, :n_occ],
    ))

# Precompute k·p tangents
print("Precomputing k·p tangents of U_val at each k_wrap...")
t0 = time.perf_counter()
for pk in per_k:
    pk['U_val_grad_kp'] = compute_kp_tangent_at_kvec(
        kvec_p_base_np=np.asarray(pk['kvec_kminq_wrap_j']),
        Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
        V_scf=pk['V_scf'], mask=pk['mask'],
        Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
        U_val_G=pk['U_val_kminq_G'],
        eps_v_at_kvec_p=pk['eps_v_at_kvec_p'],
        alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
        tol=1e-12, max_iter=300,
    )
    pk['U_val_grad_kp'].block_until_ready()
print(f"  k·p precompute: {time.perf_counter()-t0:.1f}s")

ng_out = 1  # just G'=0 for the S-tensor
Gprime_int = build_Gprime_list(q_base_signed, wfn, ng_out)
Gprime_j = jnp.asarray(Gprime_int)

def chi_col_total(qvec):
    chi = jnp.zeros((ng_out,), dtype=jnp.complex128)
    for pk in per_k:
        kvec_p = pk['kvec_kminq_wrap_j'] - (qvec - q_base_j)
        chi += chi_col_contrib_at_kvec_traced(
            kvec_p_traced=kvec_p,
            Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
            V_scf=pk['V_scf'], mask=pk['mask'],
            Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
            fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
            U_val_kminq_G=pk['U_val_kminq_G'], eps_v=pk['eps_v'],
            alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
            U_val_k_box=pk['U_val_k_box'],
            b=jnp.zeros_like(pk['Vu_G']),   # unused when Vu_G_preQ is provided
            Vu_G_preQ=pk['Vu_G'],
            Gprime_int=Gprime_j, phase_unwrap=pk['phase_unwrap'],
            prefactor=prefactor,
            tol=1e-12, max_iter=300,
            U_val_grad_kp=pk['U_val_grad_kp'],
            kvec_p_base_for_grad=pk['kvec_kminq_wrap_j'],
        )
    return chi

def first_deriv_along(dqvec, q_eval):
    _, dchi = jax.jvp(chi_col_total, (q_eval,), (dqvec,))
    return dchi

# Check: primal χ(0)=0 and ∂χ/∂q|_0 = 0 by inversion symmetry
c0 = chi_col_total(q_base_j); c0.block_until_ready()
print(f"\nχ(q=0)            = {complex(c0[0]):+.3e}  (expect 0)")
for name, dq in [('q_x', [1,0,0]), ('q_y', [0,1,0])]:
    dq_j = jnp.asarray(dq, dtype=jnp.float64)
    dc = first_deriv_along(dq_j, q_base_j); dc.block_until_ready()
    print(f"∂χ/∂{name}|_0 = {complex(dc[0]):+.3e}  (expect 0)")

# S-tensor component S_{ij} = ∂²χ/∂q_i∂q_j |_0  via central FD of the
# ∂χ/∂q_j first-derivative along i.
print(f"\nS-tensor at q=0 via FD-of-1st-deriv, sweep over h:")
print(f"  {'h':>7} {'S_xx':>12} {'S_yy':>12} {'S_xy':>12} {'|Sxx-Syy|':>12} {'|Sxy|':>12}")
for h in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]:
    ex = jnp.asarray([1.,0.,0.], dtype=jnp.float64)
    ey = jnp.asarray([0.,1.,0.], dtype=jnp.float64)

    # S_xx = d/dq_x (d/dq_x χ) | q=0  via  central FD of  ∂χ/∂q_x  along x.
    dpx = first_deriv_along(ex, q_base_j + h*ex)
    dmx = first_deriv_along(ex, q_base_j - h*ex)
    S_xx = complex((dpx[0] - dmx[0]) / (2*h))

    # S_yy
    dpy = first_deriv_along(ey, q_base_j + h*ey)
    dmy = first_deriv_along(ey, q_base_j - h*ey)
    S_yy = complex((dpy[0] - dmy[0]) / (2*h))

    # S_xy  = d/dq_y (d/dq_x χ) | q=0
    dxpy = first_deriv_along(ex, q_base_j + h*ey)
    dxmy = first_deriv_along(ex, q_base_j - h*ey)
    S_xy = complex((dxpy[0] - dxmy[0]) / (2*h))

    spread = abs(S_xx.real - S_yy.real)
    off    = abs(S_xy.real)
    print(f"  {h:7.0e} {S_xx.real:+12.4e} {S_yy.real:+12.4e} {S_xy.real:+12.4e} "
          f"{spread:12.3e} {off:12.3e}")
