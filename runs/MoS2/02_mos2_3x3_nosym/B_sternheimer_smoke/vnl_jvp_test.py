"""Does _build_vnl_kdata_core support jax.jvp through kvec?"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.vnl_ops import _build_vnl_kdata_core, build_vnl_setup

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
vnl_setup = build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                             q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)

# integer G at some fixed sphere — use k=Gamma sphere for simplicity
from psp.dft_operators import build_T_diag_from_kvec
T0, Gx, Gy, Gz = build_T_diag_from_kvec(np.asarray([0., 0., 0.]), wfn)
Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1).astype(np.int32)

def get_Z(kvec):
    kdata = _build_vnl_kdata_core(kvec, Gk_int, vnl_setup, compute_dZ=False)
    return kdata.Z

k0 = jnp.asarray([0.1, 0.2, 0.0], dtype=jnp.float64)
dk = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

try:
    Z0, Z_dot = jax.jvp(get_Z, (k0,), (dk,))
    print(f"jax.jvp through _build_vnl_kdata_core: WORKS")
    print(f"  Z shape = {Z0.shape}")
    print(f"  max|Z_dot| = {float(jnp.max(jnp.abs(Z_dot))):.3e}")
    # FD cross-check
    h = 1e-5
    Zp = get_Z(k0 + h*dk); Zm = get_Z(k0 - h*dk)
    Z_dot_fd = (Zp - Zm) / (2*h)
    rel = float(jnp.max(jnp.abs(Z_dot - Z_dot_fd))) / float(jnp.max(jnp.abs(Z_dot_fd)))
    print(f"  FD rel err = {rel:.3e}")
except Exception as e:
    print(f"jax.jvp FAILED: {type(e).__name__}: {e}")
