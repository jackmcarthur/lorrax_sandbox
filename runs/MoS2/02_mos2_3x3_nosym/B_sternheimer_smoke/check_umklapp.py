"""Check whether k-q umklapp wraparound is being handled right in source construction."""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import symmetry_maps
from file_io import WFNReader

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)

# For each iq, what ik_kminq does k=0 map to, and what's the difference?
for iq in range(9):
    q = np.asarray(wfn.kpoints[iq])
    ik_kminq_from_0 = int(sym.kq_map[0, iq])
    kmq_from_map = np.asarray(sym.unfolded_kpts[ik_kminq_from_0])
    # Naive k-q (may go outside [0,1))
    kmq_naive = - q
    # Does the map point to naive - int (i.e. wrap modulo 1)?
    wrap = kmq_naive - kmq_from_map
    print(f"iq={iq}  q={q}   k=0-q naive={kmq_naive}  from map={kmq_from_map}  wrap={np.round(wrap, 3)}")
