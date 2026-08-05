"""Stage-1 verification of the 80 Ry / 12x12x1 / 400-band MoS2 reference WFN.h5.

Reads the BGW mf_header with LORRAX's own reader (file_io.mf_header) — no
ad-hoc HDF5 parsing — and asserts the owner's acceptance criteria:
  mnband >= 400, ecutwfc == 80 Ry, kgrid == (12,12,1), K = (1/3,1/3,0) sampled.
Also emits n_rtot / ngkmax, the two numbers that drive the downstream GW
memory model.

Run under shifter on a compute node:
    srun ... python3 -u verify_wfn.py <path/to/WFN.h5>
"""
import sys
import os

sys.path.insert(0, os.environ.get(
    "LORRAX_SRC", "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src"))

import numpy as np
from file_io.mf_header import read_mf_header

path = sys.argv[1]
mf = read_mf_header(path)

fft = tuple(int(x) for x in mf.fft_grid)
n_rtot = int(np.prod(fft))
kgrid = tuple(int(x) for x in mf.kgrid)
nk = int(mf.nkpts)
mnband = int(mf.nbands)
ngkmax = int(mf.ngkmax)
ngk = np.asarray(mf.ngk)
kpts = np.asarray(mf.kpoints)          # (3, nrk) or (nrk, 3)
if kpts.shape[0] == 3 and kpts.shape[-1] != 3:
    kpts = kpts.T

print("=" * 72)
print("WFN.h5 :", path)
print("=" * 72)
print(f"  mnband (bands in file)  = {mnband}")
print(f"  ecutwfc                 = {float(mf.ecutwfc):.4f} Ry")
print(f"  ecutrho (gspace)        = {float(mf.ecutrho):.4f} Ry")
print(f"  nspin / nspinor         = {int(mf.nspin)} / {int(mf.nspinor)}")
print(f"  kgrid                   = {kgrid}")
print(f"  shift                   = {tuple(float(s) for s in mf.shift)}")
print(f"  nrk (k-points in file)  = {nk}")
print(f"  FFTgrid                 = {fft}   -> n_rtot = {n_rtot}")
print(f"  ngkmax                  = {ngkmax}   (ngk min/mean/max = "
      f"{ngk.min()}/{ngk.mean():.1f}/{ngk.max()})")
print(f"  ng (rho sphere)         = {int(mf.ng)}")
print(f"  cell_volume             = {float(mf.cell_volume):.4f} bohr^3")
print(f"  ntran (sym ops)         = {int(mf.ntran)}")
print(f"  nat                     = {int(mf.nat)}")

# --- K = (1/3, 1/3, 0) sampled? ---
target = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
d = kpts - target[None, :]
d -= np.round(d)                       # mod G
dist = np.linalg.norm(d, axis=1)
iK = int(np.argmin(dist))
print(f"  K=(1/3,1/3,0)           : nearest file k-point #{iK} = "
      f"{np.round(kpts[iK], 9).tolist()}  |dk| = {dist[iK]:.3e}")

# --- gap/occupation sanity at K ---
el = np.asarray(mf.energies)           # (nspin, nrk, mnband) Ry
if el.ndim == 3:
    elK = el[0, iK, :]
else:
    elK = el[iK, :]
ifmax = np.asarray(mf.ifmax).reshape(-1)[iK] if np.asarray(mf.ifmax).size else None
RY = 13.605693122994
if ifmax:
    vbm = float(elK[int(ifmax) - 1]) * RY
    cbm = float(elK[int(ifmax)]) * RY
    print(f"  at K: ifmax={int(ifmax)}  E_v={vbm:.4f} eV  E_c={cbm:.4f} eV  "
          f"direct DFT gap = {cbm - vbm:.4f} eV")
print(f"  highest band energy (k={iK}) = {float(elK[-1]) * RY:.3f} eV")

checks = [
    ("mnband >= 400", mnband >= 400),
    ("ecutwfc == 80 Ry", abs(float(mf.ecutwfc) - 80.0) < 1e-6),
    ("kgrid == (12,12,1)", kgrid == (12, 12, 1)),
    ("nrk == 144 (full BZ, unshifted)", nk == 144),
    ("shift == 0", all(abs(float(s)) < 1e-12 for s in mf.shift)),
    ("K=(1/3,1/3,0) sampled", dist[iK] < 1e-6),
]
print("-" * 72)
ok = True
for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    ok &= bool(passed)
print("-" * 72)
print("VERIFY:", "ALL PASS" if ok else "FAILURES PRESENT")
print(f"MACHINE-READABLE n_rtot={n_rtot} ngkmax={ngkmax} nk={nk} "
      f"mnband={mnband} nspinor={int(mf.nspinor)} "
      f"fft={fft[0]}x{fft[1]}x{fft[2]}")
sys.exit(0 if ok else 1)
