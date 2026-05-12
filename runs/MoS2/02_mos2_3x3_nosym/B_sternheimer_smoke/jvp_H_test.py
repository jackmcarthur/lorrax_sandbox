"""Does setup_H_k_from_kvec take jax.jvp gracefully over kvec?"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

def get_T_diag(kvec):
    H = setup_H_k_from_kvec(kvec, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    return H.T_diag

kvec0 = jnp.asarray([0.33333333, 0.0, 0.0], dtype=jnp.float64)
dq    = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
try:
    T0, T_dot = jax.jvp(get_T_diag, (kvec0,), (dq,))
    print(f"  jvp(T_diag) wrt kvec → shape {T_dot.shape}, max|T_dot|={float(jnp.max(jnp.abs(T_dot))):.3e}")
    # FD comparison
    h = 1e-4
    T_plus  = get_T_diag(kvec0 + h*dq)
    T_minus = get_T_diag(kvec0 - h*dq)
    T_dot_fd = (T_plus - T_minus) / (2*h)
    rel = float(jnp.max(jnp.abs(T_dot - T_dot_fd))) / float(jnp.max(jnp.abs(T_dot_fd)))
    print(f"  FD check: rel err = {rel:.3e}  ✓")
except Exception as e:
    print(f"  jvp(T_diag) FAILED: {type(e).__name__}: {e}")

# Try vnl_Z (harder — has custom JVPs internally)
def get_vnl_Z(kvec):
    H = setup_H_k_from_kvec(kvec, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
    return H.vnl_Z
try:
    Z0, Z_dot = jax.jvp(get_vnl_Z, (kvec0,), (dq,))
    print(f"  jvp(vnl_Z) wrt kvec → shape {Z_dot.shape}, max|Z_dot|={float(jnp.max(jnp.abs(Z_dot))):.3e}")
    h = 1e-4
    Z_plus  = get_vnl_Z(kvec0 + h*dq)
    Z_minus = get_vnl_Z(kvec0 - h*dq)
    Z_dot_fd = (Z_plus - Z_minus) / (2*h)
    rel = float(jnp.max(jnp.abs(Z_dot - Z_dot_fd))) / float(jnp.max(jnp.abs(Z_dot_fd)))
    print(f"  FD check vnl_Z: rel err = {rel:.3e}")
except Exception as e:
    print(f"  jvp(vnl_Z) FAILED: {type(e).__name__}: {e}")
