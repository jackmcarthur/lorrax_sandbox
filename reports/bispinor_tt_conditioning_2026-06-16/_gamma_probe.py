import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=150)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
      sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn); U=np.asarray(sym.U_spinor); Rp=np.asarray(sym.R_proper)
from common.gamma_matrices import gamma1,gamma2,gamma3
gam=[None,np.asarray(gamma1),np.asarray(gamma2),np.asarray(gamma3)]
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
sig=[None,sx,sy,sz]
op=1; Us=U[op]; R=Rp[op]
print("U_spinor[op1]=\n",Us)
print("\nThe 4x4 gamma^i = gamma^0 gamma^i. Upper-right 2x2 block compared to sig_i.")
for i in (1,2,3):
    block = gam[i][:2,2:]
    print(f"\n  i={i}: gamma^i UR block =\n{block}\n        sig_{i}=\n{sig[i]}")
    UR_block = Us.conj().T @ block @ Us
    tgt = sum(R[j-1,i-1]*sig[j] for j in (1,2,3))
    print(f"     U^dag(block)U =\n{UR_block}")
    print(f"     sum_j R[j,i]sig_j =\n{tgt}")
    print(f"     |U^dag block U - sum_j R[j,i]sig_j| = {np.abs(UR_block-tgt).max():.2e}")
    cj=[(0.5*np.trace(sig[j]@UR_block)).real for j in (1,2,3)]
    print(f"     realized coeffs (sx,sy,sz)        = {np.array(cj)}")
    print(f"     R_proper[:,i] target              = {R[:,i-1]}")
    # also: does block == sig_i exactly, or +/- transpose/conj?
    print(f"     block==sig_i? {np.allclose(block,sig[i])}  block==sig_i^T? {np.allclose(block,sig[i].T)}"
          f"  block==-sig_i? {np.allclose(block,-sig[i])}  block==conj(sig_i)? {np.allclose(block,np.conj(sig[i]))}")
