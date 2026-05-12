import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src')
import numpy as np
import jax.numpy as jnp
from file_io import WFNReader
from file_io.epsreader import EPSReader
from common import symmetry_maps
from common.meta import Meta
from gw.vcoul import compute_q0_averages

w = WFNReader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5')
sym = symmetry_maps.SymMaps(w)
m = Meta()
m.kgrid = (3,3,1)
m.fft_grid = tuple(int(x) for x in w.fft_grid)
m.cell_volume = float(w.cell_volume)
m.bdot = np.asarray(w.bdot, dtype=np.float64)
m.bvec = np.asarray(w.bvec, dtype=np.float64)
m.sys_dim = 2
m.bispinor = False
m.nk_tot = sym.nk_tot

eps = EPSReader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/eps0mat.h5')
print(f'epshead = {complex(eps.epshead).real:.8f} + {complex(eps.epshead).imag:.2e}j')

vc0, w0 = compute_q0_averages(w, jnp.asarray(eps.epshead, dtype=jnp.complex128), m, S_cart=None)
print(f'vhead       = {float(complex(vc0).real):.8f}  a.u.')
print(f'whead_0freq = {float(complex(w0).real):.8f}  a.u.')
