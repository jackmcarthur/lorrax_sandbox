"""Refine: report SIGNED (diag-eps) residual per band so the constant alpha-Z
gauge offset (+~34 meV) can be subtracted, exposing the BAND-DEPENDENT physical
residual that is the real target. Compare L0 (no segni) vs QE (segni).
Also sanity-check the ux sign by trying both +z and -z."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart)
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
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(m_x**2+m_y**2+m_z**2)
def build(seg):
    n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
    rhoG_up=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2
    rhoG_dn=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
    Vu,Vd=compute_V_xc_spin(n_up,n_dn,rhoG_up,rhoG_dn,G_cart)
    Vbar=0.5*(Vu+Vd); Bmag=0.5*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
    Bv=jnp.stack([seg*Bmag*m_x*inv,seg*Bmag*m_y*inv,seg*Bmag*m_z*inv],axis=0)
    return Vbar,Bv
ones=jnp.ones_like(amag)
segz=jnp.where(amag>1e-12,jnp.sign(-m_z),1.0)  # ux=-z
ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def signed(ik,Vbar,Bv):
    kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
    eps=np.asarray(wfn.energies[0,k_red,:nocc],float)
    box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc); V_scf=build_V_scf(V_loc,V_H,Vbar)
    H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
    Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
    U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,Bv)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
    diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm
    return (diag-eps)*RY2EV*1000  # SIGNED meV
ik=0
for lab,seg in [("L0 (no segni, +|m|)",ones),("QE (segni, ux=-z)",segz)]:
    Vbar,Bv=build(seg); r=signed(ik,Vbar,Bv)
    off=np.median(r); rd=r-off
    print(f"{lab}: const offset(median)={off:.1f} meV | band-DEP residual after subtract: "
          f"max|={np.abs(rd).max():.2f} std={rd.std():.2f} band0={rd[0]:.2f} VBM={rd[-1]:.2f}")
    print(f"   raw first 5 bands: {np.round(r[:5],1)}  last 5: {np.round(r[-5:],1)}")
