#!/usr/bin/env python3
"""Determine degeneracy-closed screening-window boundaries for the Si fixture.

Reads WFN energies (IBZ, Ry) and prints, for tol = 1e-6 Ry (BGW TOL_Degeneracy),
the closed boundaries near the production window top (nband=60, b3=nelec+ncond=60).
"""
import sys
import numpy as np
import h5py

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/tests/regression/si_cohsex_debug/WFN.h5"
RY = 13.6056980659
TOL = 1e-6  # Ry

with h5py.File(WFN, "r") as f:
    el = np.asarray(f["mf_header/kpoints/el"][:])  # (nspin, nk_ibz, nbands)
print("el shape (nspin, nk_ibz, nbands):", el.shape, flush=True)
e = el[0]  # (nk, nbands), spin 0, Ry
nk, nb = e.shape
print(f"nk_ibz={nk} nbands={nb}", flush=True)

# min-over-k gap below each boundary b (gap between band b-1 and band b)
print("\nboundary b : min_k(e[:,b]-e[:,b-1]) [Ry] [meV]  closed(>1e-6Ry)?", flush=True)
for b in range(50, nb):
    g = float(np.min(e[:, b] - e[:, b - 1]))
    print(f"  b={b:3d} : {g:.3e} Ry  {g*RY*1e3:9.3f} meV  {'CLOSED' if g > TOL else 'cut'}", flush=True)

def closed_down(b_hi, tol=TOL):
    """Largest b<=b_hi with min_k gap at boundary b > tol. Requires e[:,b] to exist for b<b_hi check."""
    b = int(b_hi)
    while b > 1:
        if b >= nb:  # boundary at/above the top edge: no band above to check
            b -= 1
            continue
        g = float(np.min(e[:, b] - e[:, b - 1]))
        if g > tol:
            return b
        b -= 1
    return b

for b_hi in (60, 62):
    print(f"\nclosed_down(b_hi={b_hi}) = {closed_down(b_hi)}", flush=True)

# Also: what is the multiplet structure at each k near the top?
print("\nPer-k gap at boundaries 58,59,60,61 (meV):", flush=True)
for b in (58, 59, 60, 61):
    if b < nb:
        gaps = (e[:, b] - e[:, b - 1]) * RY * 1e3
        print(f"  b={b}: " + " ".join(f"{g:7.2f}" for g in gaps), flush=True)

# b3 = nelec+ncond. nelec from ifmax
with h5py.File(WFN, "r") as f:
    ifmax = np.asarray(f["mf_header/kpoints/ifmax"][:])
nelec = int(np.max(ifmax))
print(f"\nnelec(ifmax)={nelec}  b3=nelec+ncond=8+52={nelec+52}  nband(b4_user)=60", flush=True)
