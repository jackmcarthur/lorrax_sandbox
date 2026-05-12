"""Sub-probe of build_vnl_setup — find where the 11s goes."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env; set_default_env()
import numpy as np
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.species import extract_species, build_atom_species_map
from psp.radial_tables import (
    build_all_tables, hankel_l, vloc_sr_table, core_charge_table,
    projector_table, projector_deriv_table,
)

wfn = WFNReader('WFN.h5')
pseudos = load_pseudopotentials('.')
nspinor = int(wfn.nspinor)
ecut = float(wfn.ecutwfc)
q_max = float(np.sqrt(ecut)) * 1.01
n_q = 4000
q_grid = np.linspace(0.0, q_max, n_q, dtype=np.float64)

t0 = time.perf_counter()
species = extract_species(pseudos, nspinor=nspinor)
print(f"  [{time.perf_counter()-t0:6.3f}s]  extract_species  (n_species={len(species)})")

# Per-species, per-table breakdown
total_n_proj = 0
for isp, sp in enumerate(species):
    print(f"\n  species {isp}: {sp.element}  n_proj={sp.n_proj}  n_r={len(sp.r)}")
    total_n_proj += sp.n_proj
    t0 = time.perf_counter()
    _ = vloc_sr_table(sp, q_grid)
    print(f"    [{time.perf_counter()-t0:6.3f}s]  vloc_sr_table")
    t0 = time.perf_counter()
    _ = core_charge_table(sp, q_grid)
    print(f"    [{time.perf_counter()-t0:6.3f}s]  core_charge_table")
    for ip in range(sp.n_proj):
        l = int(sp.proj_l[ip])
        t0 = time.perf_counter()
        _ = projector_table(sp, ip, q_grid)
        t1 = time.perf_counter()
        _ = projector_deriv_table(sp, ip, q_grid)
        t2 = time.perf_counter()
        print(f"    proj{ip} l={l}: "
              f"[{t1-t0:.3f}s] table   [{t2-t1:.3f}s] deriv_table")

print(f"\n  total projectors: {total_n_proj}")

# Time the umbrella build_all_tables for comparison
t0 = time.perf_counter()
tables = build_all_tables(species, q_max, n_q)
print(f"\n  [{time.perf_counter()-t0:6.3f}s]  build_all_tables (one-shot)")
