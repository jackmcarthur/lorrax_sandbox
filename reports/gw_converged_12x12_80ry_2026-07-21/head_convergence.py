"""Control for the reduced-band dipole.h5 used by the q=0 Coulomb head.

psp.get_dipole_mtxels cannot run at the production nband=326 on this reference
(262 GB single allocation -- see KNOWN_SANDBOX_ERRORS.md), so dipole.h5 is
built with a reduced band window.  dipole.h5 feeds ONLY gw.head_correction's
's_tensor' wcoul0 source, so the window is a HEAD-convergence parameter.  This
script measures that convergence directly: it reproduces exactly what
``resolve_head_sample`` -> ``from_s_tensor`` does (read_dipole_h5 ->
compute_S_omega -> compute_q0_averages) for each dipole file given, and prints
vc0 and wcoul0 so the truncation can be read as a number rather than assumed.

usage: python3 head_convergence.py <cohsex.in> <wfn.h5> <dipole1.h5> [<dipole2.h5> ...]
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "LORRAX_SRC",
    "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gw_converged/src"))

from runtime import set_default_env
set_default_env()

import numpy as np
import jax.numpy as jnp

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
from gw.gw_config import read_lorrax_input
from common.chi_from_dipole import read_dipole_h5, compute_S_omega
from gw.vcoul import compute_q0_averages

INP, WFN = sys.argv[1], sys.argv[2]
DIPOLES = sys.argv[3:]

params = read_lorrax_input(INP)
nval = int(params["nval"])
ncond = int(params["ncond"])
nband = int(params.get("nband", nval + ncond))

wfn = WFNReader(WFN)
sym = symmetry_maps.SymMaps(wfn)
meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, False)
# gw_jax sets this on the Meta instance after construction (gw_jax.py:214);
# Meta.from_system does not, and the 2D Coulomb kernel dispatch needs it.
meta.sys_dim = int(params.get("sys_dim", 3))
print(f"WFN {WFN}\n  nk_tot={sym.nk_tot} nelec={wfn.nelec} "
      f"nspin={wfn.nspin} nspinor={wfn.nspinor} vol={wfn.cell_volume:.4f}")
print(f"  sys_dim={params.get('sys_dim')} "
      f"bare_coulomb_cutoff={params.get('bare_coulomb_cutoff')}")
print()
print(f"{'dipole file':>28} {'nbands':>7} {'vc0 (Ry)':>14} "
      f"{'Re wcoul0 (Ry)':>16} {'Im wcoul0':>12}")
print("-" * 82)

ref = None
for dp in DIPOLES:
    dipole_cart, deltaE = read_dipole_h5(dp)
    nb = int(dipole_cart.shape[2])
    nelec = int(wfn.nelec)
    occ = np.zeros((int(sym.nk_tot), nb), dtype=float)
    occ[:, :max(0, min(nelec, nb))] = 1.0
    S = compute_S_omega(
        dipole_cart, deltaE, jnp.asarray(occ, dtype=jnp.float64),
        float(wfn.cell_volume), int(sym.nk_tot), int(wfn.nspin),
        int(wfn.nspinor), jnp.asarray([0.0 + 0.0j], dtype=jnp.complex128),
        eta=0.0)[0]
    vc0, wcoul0 = compute_q0_averages(
        wfn, jnp.asarray(0.0, dtype=jnp.float64), meta, S_cart=S)
    vc0, wcoul0 = complex(vc0), complex(wcoul0)
    print(f"{os.path.basename(dp):>28} {nb:7d} {vc0.real:14.8f} "
          f"{wcoul0.real:16.8f} {wcoul0.imag:12.3e}")
    if ref is None:
        ref = wcoul0
    else:
        d = wcoul0.real - ref.real
        print(f"{'':>28} {'':>7} {'':>14} "
              f"  delta vs first = {d:+.8f} Ry = {d*13.6056980659*1e3:+.3f} meV"
              f"  ({100*abs(d/ref.real):.3f} %)")
