"""Targeted probe: time + count compiles for ``build_dft_potentials`` alone.

The user observed 65+ small-op XLA compiles during setup which seemed
wildly excessive — should be 'a couple per Hamiltonian term + a few per
projector + NLCC/PBE'.  This script isolates just the DFT-potentials
build under PF instrumentation so we can see exactly which call sites
trace.  Run with PF_OUT=probe_dft to dump compile.log + analyse.
"""
from __future__ import annotations
import os, sys, time

sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling')

import pf
if 'PF_OUT' in os.environ:
    pf.setup_env(os.environ['PF_OUT'])
    pf.attach_compile_log(os.path.join(os.environ['PF_OUT'], 'compile.log'))

from runtime import set_default_env; set_default_env()
from common.jax_compile_cache import ensure_jax_compile_cache
ensure_jax_compile_cache()

from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn

t0 = time.perf_counter()
wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
print(f"  pre-setup wall = {time.perf_counter()-t0:.2f}s", flush=True)

t0 = time.perf_counter()
with pf.region("build_dft_potentials"):
    V_scf, V_loc, vnl_setup = build_dft_potentials(
        wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
    # Block on actual array materialization
    V_scf.block_until_ready()
print(f"  build_dft_potentials wall = {time.perf_counter()-t0:.2f}s", flush=True)
