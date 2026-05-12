"""Recompute SoS S-tensor using sign-corrected dipole = p + ∂V_NL/∂k
(autodiff-consistent), NOT dipole.h5's p - ∂V_NL/∂k.

The expected outcome: Stern/SoS ratio → 2.0 exactly (Taylor series Hessian
vs q² coefficient).
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import h5py
import numpy as np
import jax, jax.numpy as jnp
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from common.chi_from_dipole import compute_S_omega
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec, momentum_matrix_k, generate_gvectors_k
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.get_dipole_mtxels import compute_vnl_velocity_cart

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
nb_cmp = min(80, int(wfn.nbands))
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=nb_cmp - n_occ, nband=nb_cmp,
                        n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

V_cell = float(wfn.cell_volume); nk_full = int(sym.nk_tot); nspinor = int(wfn.nspinor)
B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
Binv = np.linalg.inv(B_cart)

# Reconstruct dipole with the CORRECT autodiff sign:  p + v_NL  (no flip).
print(f"Building sign-corrected dipole (p + ∂V_NL/∂k, no BGW sign flip)...")
dipole_signed = np.zeros((3, nk_full, nb_cmp, nb_cmp), dtype=np.complex128)
for ik in range(nk_full):
    kvec_k = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
    Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
    psi_k_box = load_kpoint_fftbox(wfn, sym, meta, ik, nb_cmp)
    Gk_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    psi_k_G = _psi_box_to_G_sphere(psi_k_box, Gk_int)

    p_cart  = momentum_matrix_k(psi_k_G, np.asarray(Gk_crys).astype(np.float64),
                                 kvec_k, B_cart)
    vNL_auto = compute_vnl_velocity_cart(psi_k_box, np.asarray(Gk_crys),
                                          kvec_k, vnl_setup)
    # NOTE: using +vNL_auto  (NOT  −vNL_auto  as in get_dipole_mtxels.py line 284).
    dipole_signed[:, ik] = np.asarray(p_cart + vNL_auto)

# ΔE
deltaE_full = np.zeros((nk_full, nb_cmp, nb_cmp), dtype=np.float64)
for ik in range(nk_full):
    k_red = int(sym.irk_to_k_map[ik])
    e_b = np.asarray(wfn.energies[0, k_red, :nb_cmp], dtype=float)
    deltaE_full[ik] = e_b[:, None] - e_b[None, :]

nspin = 1
f_nk = np.zeros((nk_full, nb_cmp), dtype=np.float64)
f_nk[:, :n_occ] = 1.0

S_sos_signed = compute_S_omega(
    dipole_cart=jnp.asarray(dipole_signed),
    deltaE=jnp.asarray(deltaE_full),
    f_nk=jnp.asarray(f_nk),
    cell_volume=V_cell, nk_tot=nk_full,
    nspin=nspin, nspinor=nspinor,
    omegas=jnp.asarray(0.0, dtype=jnp.float64), eta=0.0,
)[0]
S_sos_signed_np = np.asarray(S_sos_signed).real

S_sos_crys = B_cart @ S_sos_signed_np @ B_cart.T

# Stern values (from s_tensor_exact.py output after factor-2 removal)
S_stern_crys = np.array([
    [-1.079949e+00, -5.399738e-01, -3.428561e-10],
    [-5.399738e-01, -1.079952e+00, +7.249189e-11],
    [-3.428561e-10, +7.249189e-11, -2.319296e-02],
])

print(f"\n── S-tensor comparison (crystal coords) ──")
print(f"Stern [xx, yy, zz] = [{S_stern_crys[0,0]:+.4e}, {S_stern_crys[1,1]:+.4e}, {S_stern_crys[2,2]:+.4e}]")
print(f"Stern [xy, xz, yz] = [{S_stern_crys[0,1]:+.4e}, {S_stern_crys[0,2]:+.4e}, {S_stern_crys[1,2]:+.4e}]")
print()
print(f"SoS (sign-corrected) cart:")
for row in S_sos_signed_np:
    print(f"  {row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}")
print()
print(f"SoS (sign-corrected) crys (= B·S_cart·Bᵀ):")
for row in S_sos_crys:
    print(f"  {row[0]:+.6e}  {row[1]:+.6e}  {row[2]:+.6e}")
print()
print(f"Ratio Stern / SoS_crys (should be 2.0 if Hessian-vs-q²coefficient is the only difference):")
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = S_stern_crys / np.where(np.abs(S_sos_crys) > 1e-6, S_sos_crys, np.nan)
for row in ratio:
    print(f"  {row[0]:+.4f}  {row[1]:+.4f}  {row[2]:+.4f}")
