"""Compare the p-only (kinetic) velocity matrix element — strip V_NL from both
sides.

    Stern:   jvp of T_diag only (no V_NL, no V_loc)
              vs
    compute_p_operator_k:  momentum_matrix_k = 2·⟨m|K_cart|n⟩
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import numpy as np
import jax, jax.numpy as jnp

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.dft_operators import (setup_H_k_from_kvec, momentum_matrix_k,
                                generate_gvectors_k)
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.run_sternheimer import _psi_box_to_G_sphere

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
nb_cmp = min(80, int(wfn.nbands))
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=nb_cmp - n_occ, nband=nb_cmp,
                        n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
Binv = np.linalg.inv(B_cart)
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)

ik = 4
kvec_k = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
kvec_k_j = jnp.asarray(kvec_k, dtype=jnp.float64)
H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)

psi_k_box = load_kpoint_fftbox(wfn, sym, meta, ik, nb_cmp)
psi_k_G = _psi_box_to_G_sphere(psi_k_box, Gk_int)

# ── Path A: compute_p_operator_k's momentum_matrix_k ────────────────────
#    Returns 2·⟨m|K_cart|n⟩ — i.e., the factor-of-2 velocity in Ry units.
p_ref_cart = momentum_matrix_k(
    psi_k_G, np.asarray(Gk_int).astype(np.float64), kvec_k, B_cart)
p_ref_cart = np.asarray(p_ref_cart)   # (3, nb, nb)

# ── Path B: Stern jvp of T_diag only (no V_NL, no V_loc) ────────────────
# T_diag(k) = Σ_ij (G+k)_i · bdot_ij · (G+k)_j = |K_cart|²
def _T_apply_at(k, x):
    Gk_float = jnp.asarray(np.asarray(Gk_int), dtype=jnp.float64)
    kG = Gk_float + k[None, :]
    T_diag = jnp.einsum('gi,ij,gj->g', kG, bdot, kG)
    return T_diag[None, None, :] * x

def dT_dk_apply(k_dir_crys, x):
    _, out = jax.jvp(lambda k: _T_apply_at(k, x), (kvec_k_j,), (k_dir_crys,))
    return out

e_vec = [jnp.zeros(3, dtype=jnp.float64).at[a].set(1.0) for a in range(3)]
p_stern_crys = np.zeros((3, nb_cmp, nb_cmp), dtype=np.complex128)
for a in range(3):
    Tk_dot = dT_dk_apply(e_vec[a], psi_k_G)
    mtx = jnp.einsum('msG,nsG->mn', jnp.conj(psi_k_G), Tk_dot, optimize=True)
    p_stern_crys[a] = np.asarray(mtx)

# Transform crys → cart
p_stern_cart = np.einsum('ia,amn->imn', Binv, p_stern_crys)

# ── Path C: direct summation p_manual + vNL_manual (both should individually
#   match Stern jvp of T + jvp of V_NL).
import h5py
from psp.get_dipole_mtxels import compute_vnl_velocity_cart
vNL_manual_cart = compute_vnl_velocity_cart(
    psi_k_box, np.asarray(Gk_int), kvec_k, vnl_setup)
vNL_manual_cart = -np.asarray(vNL_manual_cart)   # get_dipole_mtxels's sign flip
stern_vnl_cart = np.zeros_like(p_stern_crys)
def _vnl_apply_at(k, x):
    from psp.vnl_ops import _build_vnl_kdata_core, apply_vnl
    kdata = _build_vnl_kdata_core(k, np.asarray(Gk_int), vnl_setup, compute_dZ=False)
    return apply_vnl(x, kdata.Z, kdata.E_super)
for a in range(3):
    _, dvNL_psi = jax.jvp(lambda k: _vnl_apply_at(k, psi_k_G),
                          (kvec_k_j,), (e_vec[a],))
    stern_vnl_cart[a] = np.asarray(
        jnp.einsum('msG,nsG->mn', jnp.conj(psi_k_G), dvNL_psi, optimize=True))
stern_vnl_cart = np.einsum('ia,amn->imn', Binv, stern_vnl_cart)

# Now compare dipole.h5 with reconstructed (p_manual + vNL_manual)
with h5py.File('../00_lorrax_cohsex/dipole.h5', 'r') as h5:
    dip_from_file = np.asarray(h5['dipole_cart'])[:, ik]

recon_manual = p_ref_cart + vNL_manual_cart  # what dipole.h5 *should* be
recon_stern  = p_stern_cart + stern_vnl_cart

print(f"\n── Reconstructed dipole = p + v_NL ──")
print(f"<c=26|p+vNL|v=25>  (direct assembly):")
for a, lbl in enumerate('xy'):
    print(f"  α={lbl}: from_file       = {dip_from_file[a, 26, 25]:+.4e}")
    print(f"  α={lbl}: p_ref + vNL_ref = {recon_manual[a, 26, 25]:+.4e}")
    print(f"  α={lbl}: p_stern + vNL_stern = {recon_stern[a, 26, 25]:+.4e}")

# Ratio stern/file of the full thing
mask = np.abs(dip_from_file) > 1e-3
if mask.any():
    ratio_stern_file = recon_stern[mask] / dip_from_file[mask]
    ratio_manual_file = recon_manual[mask] / dip_from_file[mask]
    print(f"\n  ratio (p_stern + vNL_stern) / file: median = "
          f"{float(np.median(ratio_stern_file.real)):.6f}")
    print(f"  ratio (p_ref    + vNL_ref)   / file: median = "
          f"{float(np.median(ratio_manual_file.real)):.6f}")

print(f"\n── Momentum matrix elements (p-only, V_NL/V_loc stripped) ──")
print(f"ik = {ik}, kvec_crys = {kvec_k}")
print(f"\n<c=26|p|v=25> cart components:")
for a, lbl in enumerate('xyz'):
    s = p_stern_cart[a, 26, 25]
    m = p_ref_cart[a, 26, 25]
    print(f"  α={lbl}: Stern (jvp T) = {s:+.4e}    Manual (momentum_matrix_k) = {m:+.4e}")

# Overall ratio
mask = np.abs(p_ref_cart) > 1e-3
if mask.any():
    ratio = p_stern_cart[mask] / p_ref_cart[mask]
    print(f"\n  ratio stats (|ref|>1e-3):  median = {float(np.median(ratio.real)):.6f},  "
          f"std = {float(np.std(ratio.real)):.3e}")
    print(f"  imag: median = {float(np.median(ratio.imag)):.3e},  "
          f"std = {float(np.std(ratio.imag)):.3e}")
    print(f"  max |stern - manual| / |manual|: {float(np.max(np.abs(p_stern_cart - p_ref_cart)[mask] / np.abs(p_ref_cart[mask]))):.3e}")
