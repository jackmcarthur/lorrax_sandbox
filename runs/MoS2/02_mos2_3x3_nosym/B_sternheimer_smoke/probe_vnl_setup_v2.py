"""Time everything in build_vnl_setup, including the parts after build_all_tables."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env; set_default_env()
import numpy as np
import jax.numpy as jnp
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.species import extract_species, build_atom_species_map
from psp.radial_tables import build_all_tables
from psp.radial.build_projectors_qe import build_E_blocks_full

wfn = WFNReader('WFN.h5')
pseudos = load_pseudopotentials('.')
nspinor = int(wfn.nspinor)
ecut = float(wfn.ecutwfc)
q_max = float(np.sqrt(ecut)) * 1.01

t0 = time.perf_counter()
species_list = extract_species(pseudos, nspinor=nspinor)
print(f"  [{time.perf_counter()-t0:6.3f}s]  extract_species")

t0 = time.perf_counter()
tables = build_all_tables(species_list, q_max, 4000)
print(f"  [{time.perf_counter()-t0:6.3f}s]  build_all_tables")

t0 = time.perf_counter()
species_natoms, species_tau, _ = build_atom_species_map(wfn, species_list)
print(f"  [{time.perf_counter()-t0:6.3f}s]  build_atom_species_map")

t0 = time.perf_counter()
for sp in species_list:
    _ = build_E_blocks_full(pseudos[sp.element])
print(f"  [{time.perf_counter()-t0:6.3f}s]  build_E_blocks_full × n_species (cold)")

# Run it a second time to test if it's compute-cached anywhere
t0 = time.perf_counter()
for sp in species_list:
    _ = build_E_blocks_full(pseudos[sp.element])
print(f"  [{time.perf_counter()-t0:6.3f}s]  build_E_blocks_full × n_species (warm)")

# Try with a Python compile-cache style check on stage 2 (jnp asarray of big tables)
n_q = 4000
q_grid = tables["q"]
G_rows_dummy = [tables["proj_tables"][i][p] for i in range(len(species_list))
                for p in range(species_list[i].n_proj)]

t0 = time.perf_counter()
G_table = jnp.asarray(np.stack(G_rows_dummy), dtype=jnp.float64)
G_table.block_until_ready()
print(f"  [{time.perf_counter()-t0:6.3f}s]  jnp.asarray of G_table (np→GPU stack)")

# total_R for E_super sizing
total_R = 0
for isp, sp in enumerate(species_list):
    natoms = int(species_natoms[isp])
    if natoms == 0: continue
    for ip in range(sp.n_proj):
        l = int(sp.proj_l[ip])
        total_R += natoms * (2*l+1)
print(f"  total_R = {total_R}, building (nspinor={nspinor}, total_R, total_R) E_super")

t0 = time.perf_counter()
E_super = np.zeros((nspinor, nspinor, total_R, total_R), dtype=np.complex128)
E_super_j = jnp.asarray(E_super, dtype=jnp.complex128)
E_super_j.block_until_ready()
print(f"  [{time.perf_counter()-t0:6.3f}s]  E_super alloc + np→GPU")
print(f"  E_super size: {E_super.nbytes/1e6:.1f} MB")
