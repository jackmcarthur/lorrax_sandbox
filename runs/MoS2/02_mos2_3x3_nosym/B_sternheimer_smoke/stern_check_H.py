"""Sanity: does apply_H_kminq at Γ give back ε_{v,Γ} · u_{v,Γ} for v ≤ 26?"""
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
from solvers.projectors import make_P_val

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

# Apply H at Γ to u_{v,Γ}
H_Gamma = setup_H_k_from_kvec(np.zeros(3), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H = make_apply_H(H_Gamma)
Gg_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, 0), dtype=np.int32))
psi_g = load_kpoint_fftbox(wfn, sym, meta, 0, n_occ)
U_Gamma = _psi_box_to_G_sphere(psi_g, Gg_int)  # (26, 2, ngk)
eps_Gamma = wfn.energies[0, sym.irk_to_k_map[0], :n_occ]
print(f"eps_Gamma[:8] = {np.asarray(eps_Gamma[:8])}")

# Check: H u_v ≈ eps_v u_v ?
HU = apply_H(U_Gamma)
# Diagonal: <u_v | H | u_v>
diag = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(U_Gamma), HU))
print(f"Diagonal <u|H|u>[:8] = {np.asarray(diag[:8])}")
print(f"Diff from eps[:8]    = {np.asarray(diag[:8] - jnp.asarray(eps_Gamma[:8]))}")

# Off-diagonal (u_m | H | u_n) should be ~0 for m != n
MH = jnp.einsum('msG,vsG->mv', jnp.conj(U_Gamma), HU)
off_diag_max = float(jnp.max(jnp.abs(MH - jnp.diag(jnp.diag(MH)))))
print(f"Off-diagonal max: {off_diag_max:.3e}")
