"""Confirm the break is in the CROSS-channel mixing, not the diagonal.
The covariant law REQUIRES: g^{ij}_q = sum_ab R[a,i]R[b,j] g^{ab}_p.
For the C3 (op1, R = [[-1/2,-s],[s,-1/2]] in-plane, s=sqrt3/2):
  predicted g_xy(q) = R[a,1]R[b,2] g_ab(p)  mixes xx,yy,xy,yx of parent.
We already saw diag |zeta^i|^2 ~preserved. Test the off-diagonal closure directly:
print measured g_xy(q) vs predicted (R x R from parent), and the cross-channel overlap
<zeta^1_p|zeta^2_p> which the C3 law needs but the fit leaves arbitrary.
"""
import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=170)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'; RUN='runs/CrI3/C_cri3_full_bz_ref_2026-06-16'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],sym_matrices=s['mtrx'][()],translations=s['tnp'][()],
      ntran=int(s['ntran'][()]),kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),
      shift=k['shift'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn); Rp=np.asarray(sym.R_proper)
sidx=np.asarray(sym.sym_idx_q); iidx=np.asarray(sym.irr_idx_q); qfull=np.asarray(sym.q_irr_full_idx); kvec=np.asarray(sym.kvecs_asints)
nq=len(sidx); ntr=wfn.ntran; kg=np.asarray(wfn.kgrid,float); nkx,nky,nkz=[int(x) for x in kg]
zT=[]
for mu in (1,2,3):
    with h5py.File(f'{RUN}/tmp/zeta_q_mu{mu}.h5') as f:
        zT.append(f['zeta_q_G'][()]); ngkT=f['isdf_header']['ngk'][()]
zT=np.stack(zT,0)
nested=np.array([(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)],int)
key=lambda v:(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz); n2row={key(nested[r]):r for r in range(nq)}
def G(row):
    ng=int(ngkT[row]); z=zT[:,row,:,:ng]   # (3,102,ng)
    return np.einsum('img,jmg->ij',np.conj(z),z)  # (3,3) centroid-traced Gram
q=10; op=int(sidx[q]); p=int(qfull[iidx[q]])
Gp=G(n2row[key(kvec[p])]); Gq=G(n2row[key(kvec[q])]); R=Rp[op]
pred=np.einsum('ai,bj,ab->ij',R,R,Gp)
print(f"op{op} C3-about-z.  PARENT Gram (real part):\n{Gp.real}")
print(f"\nMEASURED child Gram g_q (real):\n{Gq.real}")
print(f"\nR x R PREDICTED child Gram (real):\n{pred.real}")
print(f"\n=> diagonal matches (xx,yy,zz preserved), but in-plane OFF-DIAG g_xy: measured {Gq[0,1].real:+.1f} vs predicted {pred[0,1].real:+.1f}")
print(f"   parent cross <z1|z2> = g_xy(p) = {Gp[0,1].real:+.3f}  (the fit leaves this ~arbitrary; C3 needs it to source g_xx-g_yy)")
print(f"   in-plane block measured eigenvalues: {np.linalg.eigvalsh(Gq[:2,:2].real)}")
print(f"   in-plane block parent  eigenvalues:  {np.linalg.eigvalsh(Gp[:2,:2].real)}  (a rotation CANNOT change these; if they differ, zeta basis is per-q gauge-arbitrary)")
