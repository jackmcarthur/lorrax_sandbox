"""Hand-replicate the bispinor TT unfold for the C3 q's and 3-way compare:
  MANUAL (my replay of unfold_v_q scalar stage + R^T V R)
  vs DIRECT (brute-force truth, full-BZ run)
  vs CODE  (the IBZ-run's actually-written unfolded tiles).
Replays unfold_v_q EXACTLY: V_full[q,mu,nu]=phase_mu*conj(phase_nu)*V_par[a(mu),a(nu)],
a=sym_perm[op], phase_mu=exp(2pi i q_irr . L_mu); then R^T V R with R=R_proper[op].
"""
import os
os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=140)
W='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
FFT=(45,45,120)
with h5py.File(W,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
      sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
from centroid.orbit_syms import compute_centroid_sym_perm
sym=SymMaps(wfn)
Rp=np.asarray(sym.R_proper); sidx=np.asarray(sym.sym_idx_q); iidx=np.asarray(sym.irr_idx_q)
qfull=np.asarray(sym.q_irr_full_idx); kvec=np.asarray(sym.kvecs_asints)
nq=len(sidx); ntr=wfn.ntran; kg=np.asarray(wfn.kgrid,float)
nkx,nky,nkz=[int(x) for x in kg]
# q_irr_frac per IBZ index (BGW wrap, same as _resolve_ibz_q_list)
qint=np.asarray(sym.q_irr_kgrid_int,float)
qwrap=np.where(qint>kg/2, qint-kg, qint); q_irr_frac=qwrap/kg          # (n_irr,3)

# transverse centroid FFT indices
cf=np.loadtxt('runs/CrI3/C_cri3_full_bz_ref_2026-06-16/centroids_frac_102_current.txt')
rmu=np.mod(np.rint(cf*np.array(FFT)).astype(int), np.array(FFT))       # (102,3)
sym_perm,L_table=compute_centroid_sym_perm(rmu, np.asarray(sym.sym_matrices[:ntr]),
    np.asarray(sym.translations[:ntr]), np.asarray(FFT,int), extend_trs=True)  # (2ntr,102),(2ntr,102,3)

UNIQUE=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
def load_tiles(path,nested):
    with h5py.File(path) as f:
        T={}
        for (i,j) in UNIQUE: T[(i,j)]=f[f'V_qmunu_TT_{i}{j}'][()]
    # build full 3x3 per q (synthesize lower)
    nqf=T[(1,1)].shape[0]; M=np.zeros((nqf,3,3,T[(1,1)].shape[1],T[(1,1)].shape[2]),complex)
    for (i,j) in UNIQUE:
        M[:,i-1,j-1]=T[(i,j)]
        if i!=j: M[:,j-1,i-1]=np.conj(np.swapaxes(T[(i,j)],-1,-2))
    return M
Mdir_raw=load_tiles('runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5',True)
Mcode  =load_tiles('runs/CrI3/C_cri3_ibz_active_2026-06-16/tmp/v_q_bispinor.h5',False)  # sym order
# align direct (nested) -> sym order
nested=[(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)]
lut={(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz):r for r,v in enumerate(nested)}
drow=[lut[(int(kvec[q][0])%nkx,int(kvec[q][1])%nky,int(kvec[q][2])%nkz)] for q in range(nq)]
Mdir=Mdir_raw[drow]    # sym order

def rel(a,b): return np.abs(a-b).max()/max(np.abs(b).max(),1e-30)
print(" q op par | z-z: man-vs-dir  code-vs-dir | inplane(1,1): man-vs-dir  code-vs-dir | (1,2): man-vs-dir code-vs-dir")
for q in range(nq):
    op=int(sidx[q])
    if op<ntr and op==0: continue          # identity
    par_irr=int(iidx[q]); par_row=int(qfull[par_irr])
    a=sym_perm[op]; L=L_table[op]; qfr=q_irr_frac[par_irr]
    ph=np.exp(2j*np.pi*(L@qfr))             # (102,) phase_mu
    Vpar=Mdir[par_row]                      # (3,3,102,102) parent (IBZ) tile
    # scalar unfold each (a,b): permute centroids + phase
    Vsu=Vpar[:,:,a][:,:,:,a]                # gather a(mu),a(nu) on last two axes
    Vsu=Vsu*ph[None,None,:,None]*np.conj(ph)[None,None,None,:]
    R=Rp[op]
    Vman=np.einsum('ai,bj,abmn->ijmn',R,R,Vsu)   # R^T V R
    zz_m=rel(Vman[2,2],Mdir[q][2,2]); zz_c=rel(Mcode[q][2,2],Mdir[q][2,2])
    xx_m=rel(Vman[0,0],Mdir[q][0,0]); xx_c=rel(Mcode[q][0,0],Mdir[q][0,0])
    xy_m=rel(Vman[0,1],Mdir[q][0,1]); xy_c=rel(Mcode[q][0,1],Mdir[q][0,1])
    if op>0:
        print(f"{q:2d} {op:2d} {par_row:3d} |  {zz_m:.2e}     {zz_c:.2e}  |   {xx_m:.2e}      {xx_c:.2e}    |  {xy_m:.2e}    {xy_c:.2e}")
print("\nman-vs-dir ~0 => my hand replay reproduces TRUTH => the CODE differs from the correct unfold (wiring bug).")
print("man==code (both != dir) => hand replay matches code => the unfold FORMULA/approach is wrong, not wiring.")
