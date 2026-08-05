"""Gauge-invariant V_q tile covariance test.

The centroid-trace was NOT gauge invariant (it sums diagonal in the per-q SVD
zeta basis).  A genuinely basis-rotation-invariant scalar built from the
transverse tiles is the Hilbert-Schmidt contraction over the 3x3 channel block:

   S(q) = sum_{i,j in xyz} sum_{mu,nu} V^{ij}(q)[mu,nu] * conj(V^{ij}(q)[mu,nu])
        = sum_ij || V^{ij}(q) ||_F^2

Under a per-q unitary zeta-basis change Q (V^{ij} -> Q V^{ij} Q^dag for all ij
simultaneously) the Frobenius norm of each block is invariant, so S(q) is gauge
invariant.  Under symmetry the CHANNEL indices rotate: V^{ij}(Sq)=R[a,i]R[b,j]V^{ab}(q),
and because R is orthogonal, sum_ij ||V^{ij}||_F^2 is ALSO invariant under the
channel rotation.  Therefore for a CORRECT (covariant) tile set:
   S(Sq) == S(q)   for every q in an orbit.

Test S(q) constancy across each C3 orbit for:
  (a) FULL-BZ-FIT tiles  (b) IBZ-unfold tiles, and cross-run S agreement.
If S is orbit-constant in BOTH and they agree -> tiles encode the same physics,
zeta-Gram non-covariance is pure gauge.  If S differs across an orbit -> a real
covariance bug survives the gauge quotient.
"""
import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=5,suppress=True,linewidth=160)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
with h5py.File(WFN,'r') as f:
    k=f['/mf_header/kpoints']; s=f['/mf_header/symmetry']
    wfn=SimpleNamespace(sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()],
      avec=f['/mf_header/crystal/avec'][()],atom_crys=f['/mf_header/crystal/apos'][()],
      atom_types=f['/mf_header/crystal/atyp'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn)
sidx=np.asarray(sym.sym_idx_q); iidx=np.asarray(sym.irr_idx_q)
qfull=np.asarray(sym.q_irr_full_idx); kvec=np.asarray(sym.kvecs_asints)
kg=np.asarray(wfn.kgrid,float); nkx,nky,nkz=[int(x) for x in kg]
nested=np.array([(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)],int)
keyf=lambda v:(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz)
n2row={keyf(nested[r]):r for r in range(nested.shape[0])}
qrow=lambda q: n2row[keyf(kvec[q])]

def load_tiles(run):
    V={}
    with h5py.File(f'{run}/tmp/v_q_bispinor.h5') as f:
        for (i,j) in [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]:
            V[(i,j)]=f[f'V_qmunu_TT_{i}{j}'][()]
    for (i,j) in [(1,2),(1,3),(2,3)]:
        V[(j,i)]=np.conj(np.transpose(V[(i,j)],(0,2,1)))
    return V

def Sq(V,row):
    return sum((np.abs(V[(i,j)][row])**2).sum() for i in (1,2,3) for j in (1,2,3))

# also the in-plane-only HS (xx,xy,yx,yy) and the z-only, to localize
def Sq_inplane(V,row):
    return sum((np.abs(V[(i,j)][row])**2).sum() for i in (1,2) for j in (1,2))
def Sq_z(V,row):
    return (np.abs(V[(3,3)][row])**2).sum()

for run,label in [('runs/CrI3/C_cri3_fix_validate_2026-06-16','FULL-BZ-FIT'),
                  ('runs/CrI3/C_cri3_ibz_active_2026-06-16','IBZ-UNFOLD')]:
    V=load_tiles(run)
    print("\n"+"#"*80); print(f"# {label}")
    # group q by IBZ parent (orbit); report S per orbit member
    orbits={}
    for q in range(len(iidx)):
        orbits.setdefault(int(iidx[q]),[]).append(q)
    nbad=0
    for irr,members in list(orbits.items())[:6]:
        Ss=[Sq(V,qrow(q)) for q in members]
        Sin=[Sq_inplane(V,qrow(q)) for q in members]
        Sz=[Sq_z(V,qrow(q)) for q in members]
        spread=(max(Ss)-min(Ss))/(np.mean(Ss)+1e-30)
        spin=(max(Sin)-min(Sin))/(np.mean(Sin)+1e-30)
        spz=(max(Sz)-min(Sz))/(np.mean(Sz)+1e-30)
        flag='' if spread<1e-6 else '  <-- ORBIT NON-CONSTANT'
        print(f"  IBZ {irr:2d} |orbit|={len(members):2d}: S spread={spread:.2e} (in-plane={spin:.2e}, z={spz:.2e}){flag}")
        if spread>=1e-6: nbad+=1
    print(f"  => {'all orbits S-constant (gauge-invariant physics is covariant)' if nbad==0 else f'{nbad} orbits NON-constant -> REAL covariance bug survives gauge quotient'}")

# cross-run gauge-invariant agreement
Vf=load_tiles('runs/CrI3/C_cri3_fix_validate_2026-06-16')
Vi=load_tiles('runs/CrI3/C_cri3_ibz_active_2026-06-16')
print("\n"+"#"*80); print("# CROSS-RUN gauge-invariant S(q): IBZ-unfold vs FULL-BZ-fit, per q")
rels=[]
for q in range(len(iidx)):
    r=qrow(q); a=Sq(Vf,r); b=Sq(Vi,r); rels.append(abs(a-b)/(abs(a)+1e-30))
rels=np.array(rels)
print(f"  rel |S_fullbzfit - S_ibzunfold|: max={rels.max():.3e} mean={rels.mean():.3e}")
print(f"  in-plane-only:")
rin=[]
for q in range(len(iidx)):
    r=qrow(q); a=Sq_inplane(Vf,r); b=Sq_inplane(Vi,r); rin.append(abs(a-b)/(abs(a)+1e-30))
print(f"     max={max(rin):.3e} mean={np.mean(rin):.3e}")
rz=[]
for q in range(len(iidx)):
    r=qrow(q); a=Sq_z(Vf,r); b=Sq_z(Vi,r); rz.append(abs(a-b)/(abs(a)+1e-30))
print(f"  z-only:  max={max(rz):.3e} mean={np.mean(rz):.3e}")
print("  -> if in-plane S agrees across runs: IBZ-unfold reproduces full-BZ physics (no bug).")
print("     if in-plane S DIFFERS but z agrees: REAL in-plane covariance bug (the reported symptom).")
