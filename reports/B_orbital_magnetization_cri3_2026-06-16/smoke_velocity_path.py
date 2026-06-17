"""Smoke test: can orbital_magnetization.velocity_at_k consume the band-path WFN?"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path
import numpy as np

SRC = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
sys.path.insert(0, str(SRC))
# import the extracted velocity module by path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "orbmag",
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16/orbital_magnetization_B.py")
orbmag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orbmag)

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
import psp.vnl_ops as vnl_ops
from psp.pseudos import load_pseudopotentials

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath/WFN.h5"
PSEUDO_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath"

nb = 120
nval, ncond = 70, 50
wfn = WFNReader(WFN)
print("nspinor:", wfn.nspinor, " nkpts:", wfn.nkpts)
sym = symmetry_maps.SymMaps(wfn)
print("nk_tot:", sym.nk_tot)
meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, False)

# build vnl
pseudos = load_pseudopotentials(PSEUDO_DIR)
print("pseudos:", list(pseudos))
vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=2)

for ik in (0, 20, 40, 80):
    v_kin, v_nl, eps, sz = orbmag.velocity_at_k(wfn, sym, meta, vnl_setup, ik, nb)
    print(f"ik={ik:3d}  k={np.round(sym.unfolded_kpts[ik],4)}  "
          f"v_kin|max|={np.abs(v_kin).max():.4e}  v_nl|max|={np.abs(v_nl).max():.4e}  "
          f"eps[0]={eps[0]*orbmag.RY2EV:.3f}eV  sz_sum(occ)={sz[:nval].sum():.3f}")
print("SMOKE OK")
