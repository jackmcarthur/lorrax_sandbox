"""Fully-consistent ecutrho-sphere reproduction of QE's V_xc.

QE's rho lives on the ngm sphere (rho%of_g has ZERO box-corner content).  Every
use of rho -- the LDA-level n in df_du, the gradient, V_H, V_xc -- sees the SAME
sphere-limited density.  LORRAX builds rho from sum|psi|^2 on the grid, which has
nonzero box-corner (aliased) content that QE never has.

Test: low-pass rho_val and m to the ecutrho sphere ONCE (the QE-faithful density),
then build Vbar,B,V_H entirely from that.  Re-measure <v|H|v>-eps.

Also quantify the box-corner content of the LORRAX density (how much |rho(G)| sits
outside the ecutrho sphere) to confirm it is non-negligible.
"""
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
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),
                                  q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
G2=jnp.sum(G_cart**2,axis=-1)
ecutrho=4.0*float(wfn.ecutwfc)
sphere=(G2<=ecutrho)
maskR=sphere.astype(jnp.float64)
def lowpass(field):
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(field)*sphere))

rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)

# Box-corner content of LORRAX valence density
rvG=jnp.fft.fftn(rho_val)
out=float(jnp.sum(jnp.abs(rvG*(~sphere))**2)); tot=float(jnp.sum(jnp.abs(rvG)**2))
print(f"rho_val |G|>ecutrho power fraction = {100*out/tot:.4f}%  (QE has EXACTLY 0)")
mzG=jnp.fft.fftn(m_z); outm=float(jnp.sum(jnp.abs(mzG*(~sphere))**2)); totm=float(jnp.sum(jnp.abs(mzG)**2))
print(f"m_z     |G|>ecutrho power fraction = {100*outm/totm:.4f}%")

ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                          float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def build_and_resid(rho_v,mx,my,mz,label):
    V_H,_=compute_V_H_and_V_xc(rho_v,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
    core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_v+rho_core
    amag=jnp.sqrt(mx**2+my**2+mz**2)
    M=jnp.array([mx.sum(),my.sum(),mz.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
    seg=jnp.where(amag>1e-12,jnp.sign(mx*u[0]+my*u[1]+mz*u[2]),1.0)
    n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
    rgu=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
    Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart)
    Vbar=0.5*(Vu+Vd); Bmag=0.5*seg*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
    B=jnp.stack([Bmag*mx*inv,Bmag*my*inv,Bmag*mz*inv],axis=0)
    allr=[]
    for ik in (0,4,8):
        kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
        eps=np.asarray(wfn.energies[0,k_red,:nocc],float)
        box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
        Vs=build_V_scf(V_loc,V_H,Vbar)
        H_k=setup_H_k_from_kvec(kv,Vs,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
        Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
        Uu=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
        HU=apply_H_k_from_G(Uu,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,B)
        nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),Uu).real)
        diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),HU).real)/nrm
        allr.append(np.abs(diag-eps)*RY2EV*1000)
    r=np.concatenate(allr); r0=allr[0]
    print(f"  [{label}] RMS={np.sqrt((r**2).mean()):.2f} max={r.max():.2f} | "
          f"k0: b0={r0[0]:.2f} b28={r0[28]:.2f} b35={r0[35]:.2f} b63={r0[63]:.2f} VBM={r0[nocc-1]:.2f}")
    return r

print("\n=== base: full-box density (current) ===")
build_and_resid(rho_val,m_x,m_y,m_z,"base")
print("=== sphere: rho_val + m low-passed to ecutrho sphere (QE-faithful density) ===")
build_and_resid(lowpass(rho_val),lowpass(m_x),lowpass(m_y),lowpass(m_z),"sphere")
