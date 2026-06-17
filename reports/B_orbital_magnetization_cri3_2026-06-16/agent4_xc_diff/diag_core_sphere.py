import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
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
G2=jnp.sum(G_cart**2,axis=-1); ecutrho=4.0*float(wfn.ecutwfc); sphere=(G2<=ecutrho).astype(jnp.complex128)
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
# core box-corner power
out=float(jnp.sum(jnp.abs(rho_core_G*(1-sphere))**2)); tot=float(jnp.sum(jnp.abs(rho_core_G)**2))
print(f"rho_core_G |G|>ecutrho power fraction = {100*out/tot:.4f}%")
n=rho_val+rho_core; amag=jnp.sqrt(m_x**2+m_y**2+m_z**2)
M=jnp.array([m_x.sum(),m_y.sum(),m_z.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
seg=jnp.where(amag>1e-12,jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2; inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def run(core_G, label):
    core_grid=jnp.real(jnp.fft.ifftn(core_G))
    rgu=jnp.fft.fftn(n_up-core_grid/2)+core_G/2; rgd=jnp.fft.fftn(n_dn-core_grid/2)+core_G/2
    Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart)
    Vbar=0.5*(Vu+Vd); Bmag=0.5*seg*(Vu-Vd)
    B=jnp.stack([Bmag*m_x*inv,Bmag*m_y*inv,Bmag*m_z*inv],axis=0)
    allr=[]
    for ik in (0,4,8):
        kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
        eps=np.asarray(wfn.energies[0,k_red,:nocc],float); box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
        Vs=build_V_scf(V_loc,V_H,Vbar); H_k=setup_H_k_from_kvec(kv,Vs,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
        Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
        Uu=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
        HU=apply_H_k_from_G(Uu,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,B)
        nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),Uu).real)
        diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),HU).real)/nrm
        allr.append(np.abs(diag-eps)*RY2EV*1000)
    r=np.concatenate(allr); r0=allr[0]
    print(f"  [{label}] RMS={np.sqrt((r**2).mean()):.2f} max={r.max():.2f} | k0: b0={r0[0]:.2f} b28={r0[28]:.2f} b35={r0[35]:.2f} b63={r0[63]:.2f} VBM={r0[nocc-1]:.2f}")
print("=== base: analytic core full-box (current) ==="); run(rho_core_G,"base")
print("=== core_G masked to ecutrho sphere ==="); run(rho_core_G*sphere,"coreSphere")
