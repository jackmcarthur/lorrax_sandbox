"""Verify the fix: gather at H_k's G order."""
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
from psp.run_sternheimer import _psi_box_to_G_sphere

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

H_Gamma = setup_H_k_from_kvec(np.zeros(3), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H = make_apply_H(H_Gamma)
# Use H_k's G-order, not SymMaps'
G_H = jnp.stack([H_Gamma.Gx, H_Gamma.Gy, H_Gamma.Gz], axis=-1)
psi_g = load_kpoint_fftbox(wfn, sym, meta, 0, n_occ)
U_Gamma = _psi_box_to_G_sphere(psi_g, G_H)

HU = apply_H(U_Gamma)
diag = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(U_Gamma), HU))
eps = np.asarray(wfn.energies[0, sym.irk_to_k_map[0], :n_occ])
print(f"With H_k's G-order:")
print(f"  <u|H|u>[:6]  = {np.asarray(diag[:6])}")
print(f"  eps[:6]      = {eps[:6]}")
print(f"  diff[:6]     = {np.asarray(diag[:6]) - eps[:6]}")
# off-diag
MH = jnp.einsum('msG,vsG->mv', jnp.conj(U_Gamma), HU)
off_diag_max = float(jnp.max(jnp.abs(MH - jnp.diag(jnp.diag(MH)))))
print(f"  Off-diagonal max: {off_diag_max:.3e}")
