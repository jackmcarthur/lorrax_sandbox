"""Per-(n,k) orbital magnetization + spin along a Gamma-M-K-Gamma band-path WFN, for
any noncollinear+SOC system. Derives the k-path distance from the WFN reciprocal
lattice (no pre-made path_info needed). Sanity check: sum_occ <sz> ~ spin moment.
Usage: compute_orbmag_general.py <WFN.h5> <nval> <out.npz>"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path
import importlib.util
import numpy as np

WFNPATH, NVAL, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
REP = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16"
SRC = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("orbmag", f"{REP}/orbital_magnetization_B.py")
orbmag = importlib.util.module_from_spec(spec); spec.loader.exec_module(orbmag)
RY = orbmag.RY2EV
TOL = 1e-5

def per_nk_mz(vx, vy, E, mu):
    cross_z = vx * vy.T - vy * vx.T
    dE = E[:, None] - E[None, :]
    F = np.where(np.abs(dE) > TOL, (E[None, :] + E[:, None] - 2*mu) / np.where(np.abs(dE) > TOL, dE, 1.0)**2, 0.0)
    return -0.5 * np.imag(np.sum(cross_z * F, axis=1))

from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
import psp.vnl_ops as vnl_ops
from psp.pseudos import load_pseudopotentials

pdir = str(Path(WFNPATH).parent)
nb = 120; ncond = nb - NVAL
wfn = WFNReader(WFNPATH); sym = symmetry_maps.SymMaps(wfn)
meta = Meta.from_system(wfn, sym, NVAL, ncond, nb, 0, False)
vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, load_pseudopotentials(pdir), nspinor=2)
nk = int(sym.nk_tot)

E = np.zeros((nk, nb)); SZ = np.zeros((nk, nb)); V = np.zeros((nk, 3, nb, nb), complex)
for ik in range(nk):
    v_kin, v_nl, eps, sz = orbmag.velocity_at_k(wfn, sym, meta, vnl_setup, ik, nb)
    V[ik] = np.asarray(v_kin) + np.asarray(v_nl); E[ik] = np.asarray(eps); SZ[ik] = np.asarray(sz)
VBM = E[:, NVAL-1].max(); CBM = E[:, NVAL].min(); mu = 0.5*(VBM+CBM)
MZ = np.stack([per_nk_mz(V[ik, 0], V[ik, 1], E[ik], mu) for ik in range(nk)])
frame = 1.0 if SZ[:, :NVAL].sum() >= 0 else -1.0
MZ_spin = frame * MZ

# k-path distance from the WFN reciprocal lattice (only relative spacing matters)
B = np.asarray(wfn.bvec, float) * float(wfn.blat)
kc = np.asarray(sym.unfolded_kpts, float) @ B
dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
node = np.array([0, 40, 80, 120]); labels = np.array(['G', 'M', 'K', 'G'])

szsum = SZ[:, :NVAL].sum() / nk    # per-k-point average <sz> over occ -> ~ spin moment/cell
print(f"[{Path(WFNPATH).parts[-4]}] nk={nk} nb={nb} nocc={NVAL}  mu={mu*RY:.3f} eV  gap={(CBM-VBM)*RY:.3f} eV")
print(f"  SANITY sum_occ <sz>/k = {szsum:+.2f} (expect ~ spin moment/cell)  frame={frame:+.0f}")
print(f"  occ-summed m_z along spin (per-k mean) = {(MZ_spin[:, :NVAL].sum(axis=1)).mean():+.4f} mu_B")
np.savez(OUT, E=E*RY, MZ=MZ, MZ_spin=MZ_spin, SZ=SZ, mu=mu*RY, VBM=VBM*RY, CBM=CBM*RY,
         frame=frame, nocc=NVAL, dist=dist, node_idx=node, labels=labels)
print(f"saved {OUT}")
