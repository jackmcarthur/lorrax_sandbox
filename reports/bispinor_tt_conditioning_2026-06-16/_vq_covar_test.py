"""Decide: is the LAB-FRAME zeta-Gram a GAUGE ARTIFACT of the per-q SVD solve,
or a real covariance bug?  Test the GAUGE-INVARIANT physical V_q tile trace.

For each transverse tile V^{ij}(q) (centroid x centroid, the (mu,nu) ISDF
metric), the centroid-trace  T^{ij}(q) = sum_mu V^{ij}(q)[mu, alpha_q(mu)] with
the proper centroid permute is gauge invariant (it's tr over the physical
ISDF basis).  Covariance law:  T^{ij}(Sq) = sum_{ab} R[a,i] R[b,j] T^{ab}(q).

We test the FULL-BZ-FIT tiles (C_cri3_full_bz_ref): each q fit independently.
If T is covariant there, the zeta basis non-covariance is a pure gauge artifact
(the SVD picks a per-q frame) and there is NO U_spinor loss.  We ALSO test the
IBZ-unfolded tiles (C_cri3_ibz_active) to confirm the R x R unfold reproduces
the full-BZ-fit T.
"""
import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=160)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
CENTS='runs/CrI3/C_cri3_ibz_active_2026-06-16/centroids_frac_102_current.txt'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
      sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
    fft_grid=f['/mf_header/gspace/FFTgrid'][()]
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn); Rp=np.asarray(sym.R_proper)
sidx=np.asarray(sym.sym_idx_q); iidx=np.asarray(sym.irr_idx_q)
qfull=np.asarray(sym.q_irr_full_idx); kvec=np.asarray(sym.kvecs_asints)
kg=np.asarray(wfn.kgrid,float); nkx,nky,nkz=[int(x) for x in kg]
from file_io.centroids import load_centroids
from centroid.orbit_syms import compute_centroid_sym_perm
_,cents_idx,n_rmu=load_centroids(CENTS,tuple(int(v) for v in fft_grid)); cents_idx=np.asarray(cents_idx,np.int64)
sym_perm,_=compute_centroid_sym_perm(cents_idx,wfn.sym_matrices,wfn.translations,fft_grid,validate=False,extend_trs=True)

# map kvec-as-int rows to the flat-q row index used in the tile file (nested kx,ky,kz)
nested=np.array([(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)],int)
keyf=lambda v:(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz)
n2row={keyf(nested[r]):r for r in range(nested.shape[0])}
qrow=lambda q: n2row[keyf(kvec[q])]

def load_tiles(run):
    p=f'{run}/tmp/v_q_bispinor.h5'
    V={}
    with h5py.File(p) as f:
        for (i,j) in [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]:
            V[(i,j)]=f[f'V_qmunu_TT_{i}{j}'][()]
    # symmetric fill
    for (i,j) in [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]:
        V[(j,i)]=np.conj(np.transpose(V[(i,j)],(0,2,1)))  # Hermitian tile
    return V   # V[(i,j)][qrow] = (n_rmu,n_rmu)

def Tij(V,row,i,j):
    # centroid trace tr_mu V^{ij}(row)[mu,mu]  (gauge invariant scalar)
    return np.trace(V[(i,j)][row])

def test_run(run,label):
    print("\n"+"#"*90); print(f"# {label}: {run}")
    V=load_tiles(run)
    # pick a C3 op (sidx==1 or 2) and a child q with nonzero parent
    for q in range(len(sidx)):
        op=int(sidx[q]);
        if op in (1,2):
            p=int(qfull[iidx[q]])
            if np.linalg.norm(kvec[p])>1e-6: break
    R=Rp[op]
    rc=qrow(q); rp=qrow(p)
    print(f"  C3 op={op}: child q-row={rc} (k={kvec[q]}) parent q-row={rp} (k={kvec[p]})  R=\n{R}")
    # build T(child) and the R x R prediction from T(parent)
    Tchild=np.array([[Tij(V,rc,i,j) for j in (1,2,3)] for i in (1,2,3)])
    Tpar  =np.array([[Tij(V,rp,i,j) for j in (1,2,3)] for i in (1,2,3)])
    pred=np.einsum('ai,bj,ab->ij',R,R,Tpar)
    e=np.abs(Tchild-pred).max(); elab=np.abs(Tchild-Tpar).max()
    den=np.abs(Tchild).max()+1e-30
    print(f"  centroid-trace T^ij(child) real=\n{Tchild.real}")
    print(f"  R x R T^ij(parent) pred real =\n{pred.real}")
    print(f"  T^ij(parent) [lab] real      =\n{Tpar.real}")
    print(f"  rel |T_child - RxR T_parent| = {e/den:.3e}   rel |T_child - T_parent(lab)| = {elab/den:.3e}")
    print(f"  -> {'COVARIANT (physical V tile)' if e/den<1e-3 else ('LAB-FRAME' if elab/den<1e-3 else 'NEITHER')}")
    return V

Vfull=test_run('runs/CrI3/C_cri3_full_bz_ref_2026-06-16','FULL-BZ-FIT (each q independent)')
Vibz =test_run('runs/CrI3/C_cri3_ibz_active_2026-06-16','IBZ-FIT + RxR UNFOLD')

# cross-run: do the IBZ-unfolded tiles match the full-BZ-fit tiles? (gauge-inv trace)
print("\n"+"#"*90); print("# CROSS-RUN: IBZ-unfold T vs FULL-BZ-fit T (per q, all channels, gauge-invariant trace)")
maxrel=0.0
for q in range(len(sidx)):
    r=qrow(q)
    Tf=np.array([[np.trace(Vfull[(i,j)][r]) for j in (1,2,3)] for i in (1,2,3)])
    Ti=np.array([[np.trace(Vibz [(i,j)][r]) for j in (1,2,3)] for i in (1,2,3)])
    rel=np.abs(Tf-Ti).max()/(np.abs(Tf).max()+1e-30)
    maxrel=max(maxrel,rel)
print(f"  max over all 36 q of rel |T_ibz_unfold - T_fullbz_fit| = {maxrel:.3e}")
print("  -> if ~0: RxR unfold reproduces full-BZ truth (gauge-inv); if large: unfold is wrong")
