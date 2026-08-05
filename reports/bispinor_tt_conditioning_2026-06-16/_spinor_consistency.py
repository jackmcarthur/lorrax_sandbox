"""Is sym.R_proper the EXACT SO(3) image of the U_spinor that unfold_psi uses?
The transverse current rho^i = psi^dag sig^i psi rotates by R s.t.
U^dag sig^i U = sum_j R[j,i] sig^j. The V_q Lorentz unfold uses sym.R_proper.
If they differ (esp. a transpose) the in-plane V_q unfold is wrong while z stays right.
"""
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=130)
W='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
with h5py.File(W,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
      sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn)
U=np.asarray(sym.U_spinor); Rp=np.asarray(sym.R_proper)
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
sig=[sx,sy,sz]
for sidx in [1,2]:                       # the two order-3 C3 ops
    Us=U[sidx]
    RfromU=np.zeros((3,3))
    for i in range(3):
        M=Us.conj().T@sig[i]@Us           # U^dag sig_i U
        for j in range(3):
            RfromU[j,i]=0.5*np.trace(sig[j]@M).real   # coeff of sig_j  => R[j,i]
    print(f'=== C3 op {sidx} (det U-image = {np.linalg.det(RfromU):+.2f}) ===')
    print('R_fromU (SO(3) image of U_spinor):\n',RfromU)
    print('sym.R_proper[op] (used by the V_q Lorentz unfold):\n',Rp[sidx])
    print(f'  |R_proper - R_fromU|   = {np.abs(Rp[sidx]-RfromU).max():.2e}')
    print(f'  |R_proper - R_fromU^T| = {np.abs(Rp[sidx]-RfromU.T).max():.2e}')
    print()
print("If R_proper == R_fromU: convention consistent (bug is elsewhere).")
print("If R_proper == R_fromU^T (and != R_fromU): the V_q Lorentz unfold uses the")
print("TRANSPOSED rotation vs what the spinor/wavefunction actually does -> in-plane wrong, z right.")
