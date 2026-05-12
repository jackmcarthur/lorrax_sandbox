"""Exact S-tensor  S_{ij} = ∂²χ_{00}/∂q_i∂q_j|_0  via explicit q=0 formulas.

Option C, clean version.  Bypasses nested jax.jvp (and its dead-band-at-q=0
pathology) by directly writing out the first and second implicit-derivative
equations at q = 0:

    A · δu̇_i   = −grad_i[v]                                         (3 solves)
    A · δü_ij  = +Hess_ij[v]  +  Q[ ∂H/∂k_i·δu̇_j + ∂H/∂k_j·δu̇_i ]   (6 solves)

where grad[v], Hess[v] come from the existing k·p and k·p² solvers.  Then
accumulate χ's G'=0 column contribution from δü_ij via the same machinery as
the main Sternheimer driver.

Cross-check: sum-over-states S-tensor from dipole.h5 via
``common.chi_from_dipole.compute_S_omega(omega=0)``.  Tolerates truncation in
the c-band count (same caveat as ``sos_band_convergence_signed.py``).
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import os
import time
import numpy as np
import jax, jax.numpy as jnp
import h5py

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from common.chi_from_dipole import read_dipole_h5, compute_S_omega
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.run_sternheimer import (
    _psi_box_to_G_sphere, compute_kp_tangent_at_kvec, compute_kp2_tangent_at_kvec,
    compute_s_tensor_contrib_at_q0,
)
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag

# ══════════════════════════════════════════════════════════════════════════
#  System setup (matches other s_tensor_*.py scripts)
# ══════════════════════════════════════════════════════════════════════════

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
nk_full = int(sym.nk_tot); nspinor = int(wfn.nspinor)

psi_box_full = jnp.stack(
    [load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(nk_full)], axis=0)
en_full = jnp.asarray(wfn.energies[0, np.asarray(sym.irk_to_k_map), :n_occ])
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
V_cell = float(wfn.cell_volume); N_grid = int(np.prod(wfn.fft_grid))
alpha_pv_j = jnp.asarray(
    2.0 * (np.asarray(en_full).max() - np.asarray(en_full).min()), dtype=jnp.float64)

# Adler-Wiser prefactor — compute_s_tensor_contrib_at_q0 returns the raw
# k-sphere overlap  Σ_v [<grad_i | δu̇_j> + <grad_j | δu̇_i>]  (no FFT).
# The standard density-response normalisation is  2·spin_factor / (V·N_k).
spin_factor = 2 if nspinor == 1 else 1
prefactor = (2.0 * spin_factor) / (V_cell * nk_full)

iq_base = 0                                      # q = 0 case
q_base_signed = np.zeros(3)

# ══════════════════════════════════════════════════════════════════════════
#  Per-k scaffolding (at q = 0, kvec_kminq_wrap == kvec_k — no umklapp)
# ══════════════════════════════════════════════════════════════════════════

per_k = []
for ik in range(nk_full):
    ik_kminq = int(sym.kq_map[ik, iq_base])
    assert ik_kminq == ik, "At q=0 the k-q map must be identity (no umklapp)."
    kvec_k = np.asarray(sym.unfolded_kpts[ik])
    H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U_val_k_box = psi_box_full[ik, :n_occ]
    U_val_k_G = _psi_box_to_G_sphere(U_val_k_box, Gk_int)
    K_bar_sq = compute_per_band_kinetic(U_val_k_G, H_k.T_diag)
    precond_diag = tpa_preconditioner_diag(H_k.T_diag, K_bar_sq)

    per_k.append(dict(
        kvec_k_j=jnp.asarray(kvec_k, dtype=jnp.float64),
        Gkminq_int_np=np.asarray(Gk_int),
        Gx=H_k.Gx, Gy=H_k.Gy, Gz=H_k.Gz,
        V_scf=H_k.V_scf, mask=H_k.mask, vnl_E_super=H_k.vnl_E,
        U_val_k_G=U_val_k_G,
        eps_v=en_full[ik, :n_occ],
        precond_diag=precond_diag,
        U_val_k_box=U_val_k_box,
    ))

# ══════════════════════════════════════════════════════════════════════════
#  Precompute k·p (3 solves × nk) and k·p² (6 solves × nk)
# ══════════════════════════════════════════════════════════════════════════

print(f"Precomputing ∂U_val/∂kvec_p   ({nk_full} k × 3 solves)...")
t0 = time.perf_counter()
for pk in per_k:
    pk['U_val_grad_kp'] = compute_kp_tangent_at_kvec(
        kvec_p_base_np=np.asarray(pk['kvec_k_j']),
        Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
        V_scf=pk['V_scf'], mask=pk['mask'],
        Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
        U_val_G=pk['U_val_k_G'],
        eps_v_at_kvec_p=pk['eps_v'],
        alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
        tol=1e-12, max_iter=300,
    )
    pk['U_val_grad_kp'].block_until_ready()
print(f"  k·p  precompute: {time.perf_counter()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
#  Explicit S-tensor (3 Sternheimer solves per k + overlaps)
# ══════════════════════════════════════════════════════════════════════════

print(f"\n── Explicit S-tensor at q=0 (no nested jvp, no kp² Hess) ──")
t0 = time.perf_counter()
S_total = np.zeros((3, 3), dtype=np.complex128)
for ik, pk in enumerate(per_k):
    S_k = compute_s_tensor_contrib_at_q0(
        kvec_p_base_np=np.asarray(pk['kvec_k_j']),
        Gkminq_int_np=pk['Gkminq_int_np'], vnl_setup=vnl_setup,
        V_scf=pk['V_scf'], mask=pk['mask'],
        Gx=pk['Gx'], Gy=pk['Gy'], Gz=pk['Gz'],
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=pk['vnl_E_super'],
        U_val_G=pk['U_val_k_G'],
        U_val_grad_kp=pk['U_val_grad_kp'],
        eps_v=pk['eps_v'],
        alpha_pv_sc=alpha_pv_j, precond_diag=pk['precond_diag'],
        tol=1e-12, max_iter=300,
    )
    S_total += np.asarray(S_k)
S_total *= prefactor
print(f"  solve + FFT: {time.perf_counter()-t0:.1f}s")
print(f"\n  S-tensor [Stern, real part]:")
for i, row in enumerate(S_total.real):
    print(f"    {row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}")
print(f"  S-tensor [Stern, imag part]:")
for i, row in enumerate(S_total.imag):
    print(f"    {row[0]:+.3e}  {row[1]:+.3e}  {row[2]:+.3e}")

# ══════════════════════════════════════════════════════════════════════════
#  Reference:  sum-over-states S(ω=0) from autodiff-consistent dipole
#
#  dipole.h5 on disk stores  p − ∂V_NL/∂k  (the BGW sign convention applied
#  in  get_dipole_mtxels.py:284   `vNL_cart = -vNL_cart`).  That sign flip
#  makes dipole.h5 agree with BerkeleyGW outputs but DISagree with what
#  ``jax.jvp`` through ``apply_H_k`` produces — so using dipole.h5 directly
#  in the SoS S-tensor comparison gives Stern/SoS ≈ 1.64 instead of the
#  expected 2.0 (the  ~18%  deficit is  4·Re(p*·v_NL)/|p+v_NL|²  weighted
#  over the low-ΔE c-v pairs that dominate  1/ΔE³ ).
#
#  Rebuild the dipole here  **without**  the sign flip —
#     dipole_signed  =  p  +  ∂V_NL/∂k
#  which is the true velocity operator matrix element in the convention
#  produced by autodiff, then feed it into  compute_S_omega.
# ══════════════════════════════════════════════════════════════════════════

from psp.dft_operators import momentum_matrix_k, generate_gvectors_k
from psp.get_dipole_mtxels import compute_vnl_velocity_cart

print(f"\n── SoS reference: rebuilding sign-corrected dipole = p + ∂V_NL/∂k ──")
nb_cmp = int(wfn.nbands)
dipole_signed = np.zeros((3, nk_full, nb_cmp, nb_cmp), dtype=np.complex128)
deltaE_full = np.zeros((nk_full, nb_cmp, nb_cmp), dtype=np.float64)
for ik in range(nk_full):
    kvec_k = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
    Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
    meta_full = Meta.from_system(wfn, sym, nval=n_occ, ncond=nb_cmp - n_occ,
                                   nband=nb_cmp, n_rmu=0, bispinor=False)
    psi_k_box_full = load_kpoint_fftbox(wfn, sym, meta_full, ik, nb_cmp)
    Gk_int_f = jnp.asarray(Gk_crys, dtype=jnp.int32)
    psi_k_G_full = _psi_box_to_G_sphere(psi_k_box_full, Gk_int_f)

    p_cart = momentum_matrix_k(psi_k_G_full,
                                np.asarray(Gk_crys).astype(np.float64),
                                kvec_k, float(wfn.blat) * np.asarray(wfn.bvec))
    vNL_auto = compute_vnl_velocity_cart(psi_k_box_full, np.asarray(Gk_crys),
                                          kvec_k, vnl_setup)
    dipole_signed[:, ik] = np.asarray(p_cart + vNL_auto)     # +, not −

    e_b = np.asarray(wfn.energies[0, int(sym.irk_to_k_map[ik]), :nb_cmp])
    deltaE_full[ik] = e_b[:, None] - e_b[None, :]

from common.chi_from_dipole import compute_S_omega
nspin = 1
f_nk = jnp.zeros((nk_full, nb_cmp), dtype=jnp.float64)
f_nk = f_nk.at[:, :n_occ].set(1.0)

S_sos = compute_S_omega(
    dipole_cart=jnp.asarray(dipole_signed),
    deltaE=jnp.asarray(deltaE_full),
    f_nk=f_nk,
    cell_volume=V_cell, nk_tot=nk_full,
    nspin=nspin, nspinor=nspinor,
    omegas=jnp.asarray(0.0, dtype=jnp.float64), eta=0.0,
)[0]
S_sos_np = np.asarray(S_sos).real

B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
S_sos_crys = B_cart @ S_sos_np @ B_cart.T
print(f"\n  S-tensor [SoS, cartesian, real]:")
for row in S_sos_np:
    print(f"    {row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}")
print(f"\n  S-tensor [SoS, crystal, real]:   (= B · S_cart · Bᵀ)")
for row in S_sos_crys:
    print(f"    {row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}")

print(f"\n  ratio Stern / SoS_crys (real)  — should be 2.0 (Hessian vs q² coeff):")
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = S_total.real / np.where(np.abs(S_sos_crys) > 1e-6, S_sos_crys, np.nan)
for row in ratio:
    print(f"    {row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}")
