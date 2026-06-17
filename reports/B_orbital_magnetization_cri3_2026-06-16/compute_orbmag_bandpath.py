"""Per-(n,k) orbital magnetization m_z(n,k) along Gamma-M-K-Gamma for FM CrI3.
SOS form (Ry a.u.):  m_z(n,k) = -1/2 Im sum_{m!=n} (vx_nm vy_mn - vy_nm vx_mn)(En+Em-2mu)/(En-Em)^2
v = v_kin + v_nl (physical p+vNL). Validates the formula on the 10x10 full-BZ npz first
(sum over occ bands & k must reproduce stored m_orb[2]), then runs the 121 path k-points."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path
import importlib.util
import numpy as np

REP = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath"
SRC = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("orbmag", f"{REP}/orbital_magnetization_B.py")
orbmag = importlib.util.module_from_spec(spec); spec.loader.exec_module(orbmag)
RY = orbmag.RY2EV
TOL = 1e-5  # Ry; mask near-degenerate/diagonal denominators

def per_nk_mz(vx, vy, E, mu):
    """m_z^orb per band n at one k (NOT summed over n, NOT occ-restricted). v[a,m,n]=v[a,bra,ket]."""
    cross_z = vx * vy.T - vy * vx.T                      # cross_z[n,m] = vx_nm vy_mn - vy_nm vx_mn
    dE = E[:, None] - E[None, :]                         # En - Em
    F = np.where(np.abs(dE) > TOL, (E[None, :] + E[:, None] - 2*mu) / np.where(np.abs(dE) > TOL, dE, 1.0)**2, 0.0)
    return -0.5 * np.imag(np.sum(cross_z * F, axis=1))   # (nb,)

# ---------------- 1. validate on 10x10 full-BZ npz ----------------
d = np.load(f"{REP}/orbmag_FM_10x10.npz", allow_pickle=True)
Vp, Vnl, Eg = d['Vp'], d['Vnl'], d['E']; mug = float(d['mu']); wk = float(d['w_k']); nocc = int(d['nocc'])
tot = 0.0
for k in range(Eg.shape[0]):
    v = Vp[k] + Vnl[k]                                   # (3,nb,nb), sign +1
    mz = per_nk_mz(v[0], v[1], Eg[k], mug)
    tot += wk * mz[:nocc].sum()
print(f"[validate 10x10] sum_occ m_z = {tot:+.5f}   stored m_orb[2] = {float(d['m_orb'][2]):+.5f}   "
      f"(diff {abs(tot-float(d['m_orb'][2])):.2e})")
assert abs(tot - float(d['m_orb'][2])) < 1e-4, "per-(n,k) formula does not reproduce stored total"
print("  OK: per-(n,k) decomposition reproduces the BZ total.")

# ---------------- 2. path WFN ----------------
from file_io import WfnLoader as WFNReader
from common import symmetry_maps, Meta
import psp.vnl_ops as vnl_ops
from psp.pseudos import load_pseudopotentials

nb, nval, ncond = 120, 70, 50
wfn = WFNReader(f"{RUN}/WFN.h5"); sym = symmetry_maps.SymMaps(wfn)
meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, False)
vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, load_pseudopotentials(RUN), nspinor=2)
nk = int(sym.nk_tot); assert nk == 121, nk

E = np.zeros((nk, nb)); SZ = np.zeros((nk, nb)); V = np.zeros((nk, 3, nb, nb), complex)
for ik in range(nk):
    v_kin, v_nl, eps, sz = orbmag.velocity_at_k(wfn, sym, meta, vnl_setup, ik, nb)
    V[ik] = np.asarray(v_kin) + np.asarray(v_nl)        # sign +1
    E[ik] = np.asarray(eps); SZ[ik] = np.asarray(sz)
VBM = E[:, nval-1].max(); CBM = E[:, nval].min(); mu = 0.5*(VBM+CBM)   # midgap
MZ = np.stack([per_nk_mz(V[ik, 0], V[ik, 1], E[ik], mu) for ik in range(nk)])

frame = 1.0 if SZ[:, :nval].sum() >= 0 else -1.0       # spin axis sign (path is -z -> frame=-1)
MZ_spin = frame * MZ                                    # >0 = parallel to spin (red), <0 = antiparallel (blue)
print(f"[path] nk={nk} nb={nb}  mu={mu*RY:.3f} eV  VBM={VBM*RY:.3f} CBM={CBM*RY:.3f}  gap={(CBM-VBM)*RY:.3f} eV")
print(f"  sum_occ <sz> = {SZ[:, :nval].sum():+.2f}  -> spin frame = {frame:+.0f}")
print(f"  occ-band-summed m_z along spin (per-k mean) = {(MZ_spin[:, :nval].sum(axis=1)).mean():+.4f} mu_B  "
      f"max|m_z(n,k)| = {np.abs(MZ_spin).max():.3f}")

pi = np.load(f"{RUN}/path_info.npz", allow_pickle=True)
np.savez(f"{REP}/orbmag_bandpath.npz",
         E=E*RY, MZ=MZ, MZ_spin=MZ_spin, SZ=SZ, mu=mu*RY, VBM=VBM*RY, CBM=CBM*RY, frame=frame,
         nocc=nval, dist=pi['dist'], node_idx=pi['node_idx'], labels=pi['labels'], kpts_crys=pi['kpts_crys'])
print(f"saved {REP}/orbmag_bandpath.npz")
