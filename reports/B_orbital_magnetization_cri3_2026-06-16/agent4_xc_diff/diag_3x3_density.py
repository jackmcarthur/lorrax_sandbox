"""DECISIVE root-cause test: the CrI3 residual is the SCF/NSCF k-grid MISMATCH.

QE's eps_nk in the NSCF WFN were computed with the V_scf of the 3x3 SCF density.
LORRAX rebuilds V_scf from the 6x6 NSCF WFN density (build_rho_val_from_wfn sums
over ALL 36 unfolded k).  6x6 density != 3x3 density -> band-dependent residual,
worst on localized Cr 3p/3d (which sample the regions where the two densities
differ most).  MoS2 is exact because its SCF and NSCF are BOTH 3x3.

Test: rebuild rho_val and m from ONLY the 9 k-points that form the 3x3 sub-grid
of the 6x6 mesh (matching the QE SCF density), and re-measure <v|H|v>-eps.
If the residual collapses toward MoS2 levels, the mismatch is the mechanism.
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
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2]); N_grid=nx*ny*nz; vol=float(wfn.cell_volume)
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)

def build_rho_m_subset(ksub):
    """rho_val and (mx,my,mz) summed over a SUBSET of unfolded k (uniform weight)."""
    rho=jnp.zeros((nx,ny,nz)); mx=jnp.zeros((nx,ny,nz)); my=jnp.zeros_like(mx); mz=jnp.zeros_like(mx)
    for ik in ksub:
        box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
        pr=jnp.fft.ifftn(box,axes=(-3,-2,-1),norm='ortho')
        rho=rho+jnp.sum(jnp.abs(pr)**2,axis=(0,1))
        up=pr[:,0]; dn=pr[:,1]; ud=jnp.sum(jnp.conj(up)*dn,axis=0)
        mx=mx+2*jnp.real(ud); my=my+2*jnp.imag(ud); mz=mz+jnp.sum(jnp.abs(up)**2-jnp.abs(dn)**2,axis=0)
    f=(1.0/len(ksub))*(N_grid/vol)
    return rho*f, mx*f, my*f, mz*f

ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def resid(rho_val,m_x,m_y,m_z,label):
    V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
    core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
    amag=jnp.sqrt(m_x**2+m_y**2+m_z**2)
    M=jnp.array([m_x.sum(),m_y.sum(),m_z.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
    seg=jnp.where(amag>1e-12,jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]),1.0)
    n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
    rgu=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
    Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart)
    Vbar=0.5*(Vu+Vd); Bmag=0.5*seg*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
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
    print(f"  [{label}] RMS={np.sqrt((r**2).mean()):.2f} mean={r.mean():.2f} max={r.max():.2f} | "
          f"k0: b0={r0[0]:.2f} b28={r0[28]:.2f} b35={r0[35]:.2f} b63={r0[63]:.2f} VBM={r0[nocc-1]:.2f}")

ksub_3x3=[0,2,4,12,14,16,24,26,28]
all36=list(range(int(sym.nk_tot)))
print(f"nk_tot={sym.nk_tot}  3x3-subset={ksub_3x3}")
rv6,mx6,my6,mz6=build_rho_m_subset(all36)
print(f"6x6 rho integral={float(rv6.sum())*vol/N_grid:.4f}  net mz={float(mz6.sum())*vol/N_grid:.4f}")
print("=== 6x6 density (current build_rho_val_from_wfn, all 36 k) ===")
resid(rv6,mx6,my6,mz6,"6x6")
rv3,mx3,my3,mz3=build_rho_m_subset(ksub_3x3)
print(f"3x3 rho integral={float(rv3.sum())*vol/N_grid:.4f}  net mz={float(mz3.sum())*vol/N_grid:.4f}")
print("=== 3x3 density (matches QE SCF grid) ===")
resid(rv3,mx3,my3,mz3,"3x3")
# how different are the two densities?
d=np.asarray(jnp.abs(rv6-rv3)); print(f"\n|rho_6x6 - rho_3x3|: max={d.max():.5f}  L1/N={d.mean():.6f}  "
      f"rel L1={float(jnp.sum(jnp.abs(rv6-rv3))/jnp.sum(jnp.abs(rv6))):.4%}")
