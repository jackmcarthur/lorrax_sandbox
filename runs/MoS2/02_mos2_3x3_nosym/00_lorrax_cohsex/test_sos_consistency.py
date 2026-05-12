"""Check that the SOS S-tensor (from chi_sos using v_cvkq[α,c,v,k,q=0])
matches the existing compute_S_omega (from chi_from_dipole using
dipole_cart) on the *kinetic* part of v.

We can't compare to the full p+vNL chi_from_dipole until VNL velocity is
wired into the finite-q path, but the test verifies the SOS prefactor
convention + sum structure are consistent.
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env; set_default_env()
import numpy as np
import h5py

from common.chi_sos import compute_S_tensor_sos
from file_io import WFNReader

with h5py.File('dipole.h5', 'r') as h5:
    deltaE_full = np.asarray(h5['deltaE'])              # (nk, nb, nb)
    fq = h5['finite_q']
    rho_cvkq = np.asarray(fq['rho_cvkq'])               # (nc, nv, nk, nq)
    v_cvkq   = np.asarray(fq['v_cvkq'])                 # (3, nc, nv, nk, nq)
    iq_list  = list(np.asarray(fq['iq_list']))
    n_occ = int(fq.attrs['n_occ'])
    v_lo  = int(fq.attrs['v_lo'])
    c_hi  = int(fq.attrs['c_hi'])

# Identify q=0 in iq_list
iq0 = iq_list.index(0)
v_alpha_q0 = v_cvkq[:, :, :, :, iq0]                    # (3, nc, nv, nk)
deltaE_cv_q0 = deltaE_full[:, n_occ:c_hi, v_lo:n_occ]   # (nk, nc, nv)
deltaE_cv_q0 = np.transpose(deltaE_cv_q0, (1, 2, 0))    # (nc, nv, nk)

wfn = WFNReader('WFN.h5')
nk_tot = 9
nv = n_occ - v_lo
nc = c_hi - n_occ
occ_v = np.ones(nv, dtype=np.float64)
occ_c = np.zeros(nc, dtype=np.float64)

S = compute_S_tensor_sos(
    v_alpha_q0, deltaE_cv_q0, occ_v, occ_c,
    omegas=np.array(0.0+0j),
    cell_volume=float(wfn.cell_volume),
    nk_tot=nk_tot, nspin=int(wfn.nspin), nspinor=int(wfn.nspinor),
)
S = np.asarray(S)[0]   # (3, 3) at ω=0

print(f"S-tensor at q=0 from chi_sos (KINETIC velocity only, nk_tot={nk_tot}):")
for r in S.real:
    print(f"  {r[0]:+.4e}  {r[1]:+.4e}  {r[2]:+.4e}")
print(f"  imag max: {np.max(np.abs(S.imag)):.2e}")
print(f"  trace: {np.trace(S).real:.4e}")

# Compare to compute_S_omega on dipole_cart (full velocity p + i[r, V_NL])
from common.chi_from_dipole import compute_S_omega

with h5py.File('dipole.h5', 'r') as h5:
    dipole_cart = np.asarray(h5['dipole_cart'])         # (3, nk, nb, nb)
    deltaE = np.asarray(h5['deltaE'])

# Build f_nk: 1 for valence (b < n_occ), 0 for cond
nb = dipole_cart.shape[-1]
f_nk = np.zeros((nk_tot, nb), dtype=np.float64)
f_nk[:, :n_occ] = 1.0

S_existing = compute_S_omega(
    dipole_cart, deltaE, f_nk,
    cell_volume=float(wfn.cell_volume),
    nk_tot=nk_tot, nspin=int(wfn.nspin), nspinor=int(wfn.nspinor),
    omegas=np.array(0.0+0j),
)
S_existing = np.asarray(S_existing)[0]

print(f"\nS-tensor from compute_S_omega (FULL velocity, ref):")
for r in S_existing.real:
    print(f"  {r[0]:+.4e}  {r[1]:+.4e}  {r[2]:+.4e}")
print(f"  imag max: {np.max(np.abs(S_existing.imag)):.2e}")
print(f"  trace: {np.trace(S_existing).real:.4e}")

# Sample one element
print(f"\nSample matrix elements at (k=0, c=0, v=last_v):")
v_kin = v_alpha_q0[0, 0, -1, 0]                          # x-comp
v_full = dipole_cart[0, 0, n_occ, n_occ - 1]
print(f"  v_kin (chi_sos)  = {v_kin:+.6e}")
print(f"  v_full (dipole)  = {v_full:+.6e}")
print(f"  ratio          = {v_kin/v_full if abs(v_full)>1e-30 else float('nan')}")
