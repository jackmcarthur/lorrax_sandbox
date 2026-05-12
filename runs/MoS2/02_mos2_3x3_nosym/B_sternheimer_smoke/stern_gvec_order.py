"""Does SymMaps.get_gvecs_kfull match setup_H_k_from_kvec's G order?"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.pseudos import load_pseudopotentials
from psp.dft_operators import setup_H_k_from_kvec

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

H_Gamma = setup_H_k_from_kvec(np.zeros(3), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gx_H = np.asarray(H_Gamma.Gx); Gy_H = np.asarray(H_Gamma.Gy); Gz_H = np.asarray(H_Gamma.Gz)
G_H = np.stack([Gx_H, Gy_H, Gz_H], axis=-1)       # (ngk_H, 3)

G_sym = np.asarray(sym.get_gvecs_kfull(wfn, 0))   # from SymMaps
print(f"G_H   shape={G_H.shape}  first 3 = \n{G_H[:3]}")
print(f"G_sym shape={G_sym.shape}  first 3 = \n{G_sym[:3]}")
print(f"Sets equal? {set(map(tuple, G_H.tolist())) == set(map(tuple, G_sym.tolist()))}")
# order match?
print(f"Byte-identical? {np.array_equal(G_H, G_sym)}")
# kinetic diagonal — T[G] = |G|²·alat/2/ecut ... let me just check first few
print(f"T_diag[:5] = {np.asarray(H_Gamma.T_diag[:5])}")
# |G|² check
B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
G_cart_H   = G_H   @ B
G_cart_sym = G_sym @ B
print(f"|G_H|²[:5]:  {np.sum(G_cart_H**2, axis=1)[:5]}")
print(f"|G_sym|²[:5]: {np.sum(G_cart_sym**2, axis=1)[:5]}")
