"""Charge-channel gate on NON-MAGNETIC CrI3 (scalar V_xc, no B). If a soft-Cr
(z=7) 80Ry non-mag CrI3 gives ~MoS2-level residual while the hard-Cr (z=14, 3s3p
semicore) FM-30Ry gives 24 meV, the residual is the hard semicore under-converged
at 30Ry -- a basis/cutoff effect, not a V_xc code bug."""
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
from psp.scf_potential import build_rho_val_from_wfn
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
WFN=sys.argv[1]; RY=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn)
nspinor=int(wfn.nspinor); nelec=int(wfn.nelec)
nocc = nelec if nspinor==2 else nelec//2
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=nspinor,q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float)); bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=True)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,V_xc=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
ngk=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))
print(f"ecutwfc={float(wfn.ecutwfc):.0f} Ry nspinor={nspinor} nocc={nocc}")
allr=[]
for ik in (0,4,8):
    kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
    eps=np.asarray(wfn.energies[0,k_red,:nocc],float); box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
    Vs=build_V_scf(V_loc,V_H,V_xc); H_k=setup_H_k_from_kvec(kv,Vs,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngk)
    Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
    Uu=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
    HU=apply_H_k_from_G(Uu,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,None)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),Uu).real)
    diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(Uu),HU).real)/nrm
    allr.append(np.abs(diag-eps)*RY*1000)
r=np.concatenate(allr)
print(f"[nonmag gate] RMS={np.sqrt((r**2).mean()):.2f} mean={r.mean():.2f} max={r.max():.2f} meV (over {len(r)} band-k)")
