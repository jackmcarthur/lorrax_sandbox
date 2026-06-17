"""ULTIMATE density control: build V_H, Vbar, B from QE's OWN charge-density.hdf5
(rhotot_g, m_x/y/z on the ngm sphere) -- QE's exact self-consistent density, the
SAME one its V_scf (hence eps) was built from.  This removes ALL LORRAX density-
reconstruction error.  If <v|H|v>-eps still shows the band-dependent residual,
the bug is purely in the V_xc functional/assembly (or V_loc/V_H), NOT the density.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax.numpy as jnp, h5py
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               poisson_potential_from_rhoG, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN=sys.argv[1]; RY2EV=13.605693122994; trunc=True
CHG=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(WFN))),
                 "scf","CrI3.save","charge-density.hdf5")
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2]); N_grid=nx*ny*nz; vol=float(wfn.cell_volume)
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)

# --- load QE density, scatter ngm-sphere G-values into the FFT box ---
f=h5py.File(CHG,'r')
mill=np.asarray(f['MillerIndices'])           # (ngm,3) integer G in crystal
def to_box(arr_ri):
    c=arr_ri.view(np.complex128) if arr_ri.dtype==np.float64 else arr_ri
    if c.shape[0]==2*mill.shape[0]: c=arr_ri[0::2]+1j*arr_ri[1::2]
    box=np.zeros((nx,ny,nz),dtype=np.complex128)
    gx=mill[:,0]%nx; gy=mill[:,1]%ny; gz=mill[:,2]%nz
    box[gx,gy,gz]=c
    return jnp.asarray(box)
# rho/m stored as flat (2*ngm) real = interleaved re/im OR (ngm,) complex; handle both
def load_cmplx(name):
    a=np.asarray(f[name])
    if a.ndim==1 and a.shape[0]==2*mill.shape[0]:
        return a[0::2]+1j*a[1::2]
    return a.astype(np.complex128)
rhotot_G_sphere=load_cmplx('rhotot_g')         # total charge (val+? ) in G, Hartree-norm
mx_G=load_cmplx('m_x'); my_G=load_cmplx('m_y'); mz_G=load_cmplx('m_z')
# QE rhotot_g convention: rho(G) such that rho(r)=sum_G rho(G) e^{iGr}; integral=N.
# Build box then ifft (numpy convention: ifftn = (1/N) sum). QE rho(G=0)=N/Omega*Omega? -> check integral.
def box_scatter(c):
    box=np.zeros((nx,ny,nz),dtype=np.complex128)
    box[mill[:,0]%nx,mill[:,1]%ny,mill[:,2]%nz]=c
    return jnp.asarray(box)
rhoG=box_scatter(rhotot_G_sphere)
# QE stores rho(G) as coefficients with rho(r)=Sum_G rho(G)exp(iGr); numpy ifftn includes 1/N,
# so rho(r)=N*ifftn(rhoG_box). Integral rho(r) dV = rho(G=0)*Omega.
rho_r=jnp.real(jnp.fft.ifftn(rhoG))*N_grid
integ=float(rho_r.sum())*vol/N_grid
print(f"QE rhotot integral = {integ:.4f} e  (expect ~{nocc} val; if ~{nocc} this is rho_VAL incl core?)")
mxr=jnp.real(jnp.fft.ifftn(box_scatter(mx_G)))*N_grid
myr=jnp.real(jnp.fft.ifftn(box_scatter(my_G)))*N_grid
mzr=jnp.real(jnp.fft.ifftn(box_scatter(mz_G)))*N_grid
print(f"QE net m_z integral = {float(mzr.sum())*vol/N_grid:.4f} muB")

# rhotot_g from QE = valence ONLY (core is added separately via rho_core for NLCC).
rho_val=rho_r
V_H=jnp.real(poisson_potential_from_rhoG(jnp.fft.fftn(rho_val,norm='ortho'),bdot,bvec,wfn.blat,truncation_2d=trunc))
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(mxr**2+myr**2+mzr**2)
M=jnp.array([mxr.sum(),myr.sum(),mzr.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
seg=jnp.where(amag>1e-12,jnp.sign(mxr*u[0]+myr*u[1]+mzr*u[2]),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
rgu=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart)
Vbar=0.5*(Vu+Vd); Bmag=0.5*seg*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
B=jnp.stack([Bmag*mxr*inv,Bmag*myr*inv,Bmag*mzr*inv],axis=0)

ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))
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
print(f"\n[QE-exact-density] RMS={np.sqrt((r**2).mean()):.2f} mean={r.mean():.2f} max={r.max():.2f} | "
      f"k0: b0={r0[0]:.2f} b28={r0[28]:.2f} b35={r0[35]:.2f} b63={r0[63]:.2f} VBM={r0[nocc-1]:.2f}")
print("=> If still ~22 meV: density is exonerated, bug is in Vxc functional/assembly or V_loc.")
