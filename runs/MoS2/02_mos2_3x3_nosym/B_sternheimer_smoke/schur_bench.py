"""Benchmark Schur warm-start vs plain CG on a representative (ik, iq).

We pick iq=4 (K/K'-orbit) which has the slowest-converging CG on this
system.  Vary max_iter and compare residual decay; also report δu equality.
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
from psp.run_sternheimer import (build_sternheimer_source, make_density_perturbation,
                                  make_umklapp_phase, _psi_box_to_G_sphere)
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag
from solvers.sternheimer_solve import SternheimerOp, sternheimer_solve, _apply_A_inline

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec); n_band = int(wfn.nbands)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=n_band-n_occ, nband=n_band, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
psi_box_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_band) for ik in range(sym.nk_tot)], axis=0)
en_full = jnp.asarray(wfn.energies[0, np.asarray(sym.irk_to_k_map), :n_band], dtype=jnp.float64)

# q = signed K
iq = 4
qvec_pos = np.asarray(wfn.kpoints[iq], dtype=np.float64)
qvec = qvec_pos - np.round(qvec_pos)      # signed
ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq])

H_k     = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_full]),  V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
H_kminq = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_kminq]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
Gkminq_int = jnp.stack([H_kminq.Gx, H_kminq.Gy, H_kminq.Gz], axis=-1).astype(jnp.int32)

kvec_k_np = np.asarray(sym.unfolded_kpts[ik_full])
kvec_kminq_wrap_np = np.asarray(sym.unfolded_kpts[ik_kminq])
G_wrap_np = np.rint((kvec_k_np - qvec) - kvec_kminq_wrap_np).astype(np.int32)
phase_wrap   = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=+1)
phase_unwrap = make_umklapp_phase(G_wrap_np, wfn.fft_grid, sign=-1)
V_pert_real = make_density_perturbation(wfn.fft_grid) * phase_wrap

U_val_k_box     = psi_box_full[ik_full,  :n_occ]
U_val_k_G       = _psi_box_to_G_sphere(U_val_k_box, Gk_int)
U_val_kminq_box = psi_box_full[ik_kminq, :n_occ]
U_val_kminq_G   = _psi_box_to_G_sphere(U_val_kminq_box, Gkminq_int)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk  = en_full[ik_full, :n_occ]
b = build_sternheimer_source(U_val_k_box, Gkminq_int, V_pert_real, Q_kminq)

K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
precond_diag = tpa_preconditioner_diag(H_kminq.T_diag, K_bar_sq)
alpha_pv = 2.0 * (np.max(np.asarray(eps_vk)) - np.min(np.asarray(eps_vk)))

def make_op(n_extra):
    if n_extra > 0:
        U_extra_box = psi_box_full[ik_kminq, n_occ:n_occ+n_extra]
        U_extra_G = _psi_box_to_G_sphere(U_extra_box, Gkminq_int)
        eps_extra = en_full[ik_kminq, n_occ:n_occ+n_extra]
    else:
        U_extra_G = None; eps_extra = None
    return SternheimerOp(
        T_diag=H_kminq.T_diag, V_scf=H_kminq.V_scf,
        Gx=H_kminq.Gx, Gy=H_kminq.Gy, Gz=H_kminq.Gz,
        vnl_Z=H_kminq.vnl_Z, vnl_E=H_kminq.vnl_E, mask=H_kminq.mask,
        U_val=U_val_kminq_G, eps_v=eps_vk,
        alpha_pv=jnp.asarray(alpha_pv, dtype=jnp.float64),
        precond_diag=precond_diag,
        fft_grid=H_kminq.fft_grid,
        U_extra=U_extra_G, eps_extra=eps_extra,
    )

# Reference: plain CG w/ many iterations to find the converged answer.
op0 = make_op(0)
x_ref = sternheimer_solve(op0, b, tol=1e-12, max_iter=200, use_schur=False)
x_ref.block_until_ready()
ref_norm = float(jnp.sqrt(jnp.sum(jnp.abs(x_ref)**2)))
print(f"reference ||δu|| = {ref_norm:.6e}  (max_iter=200, tol=1e-12)\n")

def relerr(x):
    d = x - x_ref
    return float(jnp.sqrt(jnp.sum(jnp.abs(d)**2)) / ref_norm)

# Sweep: n_extra ∈ {0, 5, 10, 20, 40}  ×  max_iter ∈ {5, 10, 20, 40, 80}
print(f"{'n_extra':>8} {'max_iter':>8} {'rel_err':>12} {'time_s':>8}")
for n_extra in [0, 5, 10, 20, 40]:
    op = make_op(n_extra)
    for mi in [5, 10, 20, 40, 80]:
        # warm up JIT
        _ = sternheimer_solve(op, b, tol=1e-12, max_iter=mi, use_schur=(n_extra>0)).block_until_ready()
        t0 = time.perf_counter()
        x = sternheimer_solve(op, b, tol=1e-12, max_iter=mi, use_schur=(n_extra>0))
        x.block_until_ready()
        dt = time.perf_counter() - t0
        print(f"{n_extra:>8d} {mi:>8d} {relerr(x):>12.3e} {dt:>8.2f}")
    print()
