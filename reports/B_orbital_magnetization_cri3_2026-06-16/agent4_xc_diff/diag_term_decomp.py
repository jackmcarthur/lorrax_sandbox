"""Term-by-term <v|O|v> decomposition at k=Gamma to LOCALIZE the band-dependent
residual.  For each occupied band compute:
  <T>, <V_loc>, <V_H>, <Vbar_xc>, <B.sigma>, <V_NL>, sum, eps, residual.
Then report which term's BAND-DEPENDENT variation tracks the residual.

Also runs a B-OFF variant (B_vec=0) to test whether the band-dependence lives in
B.sigma or in the charge channel (Vbar/V_H/V_loc/V_NL).
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, compute_ngkmax, compute_V_H_and_V_xc,
                               build_V_scf, build_G_cart, _bsigma_psi)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN=sys.argv[1]; RY2EV=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),
                                  q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)

rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(m_x**2+m_y**2+m_z**2)
M=jnp.array([m_x.sum(),m_y.sum(),m_z.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
seg=jnp.where(amag>1e-12,jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
rhoG_up=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rhoG_dn=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rhoG_up,rhoG_dn,G_cart)
Vbar=0.5*(Vu+Vd); Bmag=0.5*seg*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
B_vec=jnp.stack([Bmag*m_x*inv,Bmag*m_y*inv,Bmag*m_z*inv],axis=0)

ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                          float(wfn.ecutwfc),tuple(int(x) for x in fg)))
ik=0
kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
eps=np.asarray(wfn.energies[0,k_red,:nocc],float)*RY2EV
box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
psi_r=jnp.fft.ifftn(box,axes=(-3,-2,-1),norm='ortho')

V_scf=build_V_scf(V_loc,V_H,Vbar)
H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)

def expect_localG(field_r):
    Vpsi=jnp.fft.fftn(psi_r*field_r,axes=(-3,-2,-1),norm='ortho')[:,:,H_k.Gx,H_k.Gy,H_k.Gz]*H_k.mask[None,None,:]
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),Vpsi).real)/nrm*RY2EV
def expect_T():
    HT=H_k.T_diag[None,None,:]*U
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HT).real)/nrm*RY2EV
def expect_B():
    Bp=jnp.fft.fftn(_bsigma_psi(psi_r,B_vec),axes=(-3,-2,-1),norm='ortho')[:,:,H_k.Gx,H_k.Gy,H_k.Gz]*H_k.mask[None,None,:]
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),Bp).real)/nrm*RY2EV
def expect_VNL():
    P=jnp.einsum('RG,vsG->Rsv',jnp.conj(H_k.vnl_Z),U,optimize=True)
    D=jnp.einsum('stRQ,Qtv->Rsv',H_k.vnl_E,P,optimize=True)
    HV=jnp.einsum('RG,Rsv->vsG',H_k.vnl_Z,D,optimize=True)*H_k.mask[None,None,:]
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HV).real)/nrm*RY2EV

T=expect_T(); Vl=expect_localG(V_loc); VH=expect_localG(V_H); Vx=expect_localG(Vbar)
Bx=expect_B(); VN=expect_VNL()
tot=T+Vl+VH+Vx+Bx+VN
res=(tot-eps)*1000  # meV

print(f"{'b':>4} {'eps':>9} {'<T>':>8} {'<Vloc>':>9} {'<VH>':>8} {'<Vxc>':>8} {'<B>':>8} {'<VNL>':>8} {'res(meV)':>9}")
watch=[0,1,28,29,35,39,40,63,69,nocc-1]
for b in watch:
    print(f"{b:>4} {eps[b]:>9.3f} {T[b]:>8.3f} {Vl[b]:>9.3f} {VH[b]:>8.3f} {Vx[b]:>8.3f} {Bx[b]:>8.3f} {VN[b]:>8.3f} {res[b]:>9.2f}")
print(f"\nRESIDUAL over occ: RMS={np.sqrt((res**2).mean()):.2f} mean={res.mean():.2f} max|={np.abs(res).max():.2f} meV")
# Correlation of residual with each term's band-variation (de-meaned)
def corr(a):
    a=a-a.mean(); r=res-res.mean()
    return float((a*r).sum()/(np.sqrt((a**2).sum()*(r**2).sum())+1e-30))
print(f"corr(res, <B>)={corr(Bx):.3f}  corr(res,<Vxc>)={corr(Vx):.3f}  "
      f"corr(res,<VNL>)={corr(VN):.3f}  corr(res,<Vloc>)={corr(Vl):.3f}  "
      f"corr(res,<VH>)={corr(VH):.3f}  corr(res,<T>)={corr(T):.3f}")
print(f"std per term (meV): B={Bx.std()*1000:.1f} Vxc={Vx.std()*1000:.1f} "
      f"VNL={VN.std()*1000:.1f} Vloc={Vl.std()*1000:.1f} VH={VH.std()*1000:.1f} T={T.std()*1000:.1f}")
