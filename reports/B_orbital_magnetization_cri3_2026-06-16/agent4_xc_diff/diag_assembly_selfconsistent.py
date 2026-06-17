"""Trustworthy assembly test on the REAL grid: build V_up/V_dn TWO ways, both from
LORRAX's OWN pbe_xc_spin energy (so no QE-transcription LDA-Slater trap):
  way A (LORRAX): V_s = df_ds - div(2*df_dss*gns + df_dud*gn_other)        [current code]
  way B (QE form): per-channel autodiff of the SAME energy but ASSEMBLE the flux as
        h_up = (v2x_up + v2c)*gnu + v2c*gnd  with v2x_up=2*dEx_duu, v2c=dEc(grho_tot).
If A==B pointwise, the LORRAX flux assembly is algebraically identical to QE's
gcx_spin/gcc_spin+gradcorr structure ON THE ACTUAL CrI3 density. (We already know
the per-point coeffs match QE analytic to 1e-8; this confirms it on-grid.)"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software"); sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from psp.dft_operators import build_G_cart
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin, _pbe_x_spin, _pbe_c_spin
from psp.pseudos import load_pseudopotentials
WFN=sys.argv[1]; RY=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
Gj=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float)); G=np.asarray(Gj)
rho_val=np.asarray(build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False))
mx,my,mz=[np.asarray(a) for a in build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)]
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
rho_core=np.asarray(rho_core); rho_core_G=np.asarray(rho_core_G); core_grid=np.real(np.fft.ifftn(rho_core_G))
n=rho_val+rho_core; amag=np.sqrt(mx**2+my**2+mz**2)
M=np.array([mx.sum(),my.sum(),mz.sum()]); u=M/(np.linalg.norm(M)+1e-30)
seg=np.where(amag>1e-12,np.sign(mx*u[0]+my*u[1]+mz*u[2]),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
rgu=np.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd=np.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
gnu=[np.real(np.fft.ifftn(1j*G[...,i]*rgu)) for i in range(3)]; gnd=[np.real(np.fft.ifftn(1j*G[...,i]*rgd)) for i in range(3)]
suu=sum(gnu[i]**2 for i in range(3)); sdd=sum(gnd[i]**2 for i in range(3)); sud=sum(gnu[i]*gnd[i] for i in range(3))
grho_tot=sum((gnu[i]+gnd[i])**2 for i in range(3))
nu=jnp.asarray(np.maximum(n_up,1e-10)); nd=jnp.asarray(np.maximum(n_dn,1e-10))
suuJ,sudJ,sddJ=jnp.asarray(suu),jnp.asarray(sud),jnp.asarray(sdd); gtJ=jnp.asarray(grho_tot)
# way B coeffs (QE form) from LORRAX energy: v2x_up=2*dEx/dsuu ; v2c=dEc/dgrho_tot (single arg)
def Ex(uu,dd,a,b,c): return jnp.sum((uu+dd)*2.0*_pbe_x_spin(uu,dd,a,c))   # Ry
def Ec_tot(uu,dd,gt): return jnp.sum((uu+dd)*2.0*_pbe_c_spin(uu,dd,gt))   # Ry, single grho_tot arg (QE)
v2x_up=np.array(2*np.asarray(jax.grad(Ex,2)(nu,nd,suuJ,sudJ,sddJ))); v2x_dn=np.array(2*np.asarray(jax.grad(Ex,4)(nu,nd,suuJ,sudJ,sddJ)))
v2c=np.array(np.asarray(jax.grad(Ec_tot,2)(nu,nd,gtJ)))
# v1: dEx/dn_s + dEc/dn_s (QE form)
v1u=np.asarray(jax.grad(Ex,0)(nu,nd,suuJ,sudJ,sddJ))+np.asarray(jax.grad(Ec_tot,0)(nu,nd,gtJ))
v1d=np.asarray(jax.grad(Ex,1)(nu,nd,suuJ,sudJ,sddJ))+np.asarray(jax.grad(Ec_tot,1)(nu,nd,gtJ))
mask=(n>1e-6)&((suu+2*sud+sdd)>1e-10)
for a in (v2x_up,v2x_dn,v2c): a[~mask]=0.0
def div(fx):
    o=np.zeros_like(n)
    for i in range(3): o=o+np.real(np.fft.ifftn(1j*G[...,i]*np.fft.fftn(fx[i])))
    return o
h_up=[(v2x_up+v2c)*gnu[i]+v2c*gnd[i] for i in range(3)]; h_dn=[(v2x_dn+v2c)*gnd[i]+v2c*gnu[i] for i in range(3)]
# NOTE: v1 here still has the LDA-mask issue; use LORRAX's own masked v1 by calling its code for fairness:
# compare ONLY the GGA flux divergence assembly (the term in question)
divB_u=-div(h_up); divB_d=-div(h_dn)
# way A (LORRAX) flux divergence: reproduce its internal assembly
def E(uu,dd,a,b,c):
    from psp.xc import pbe_xc_spin
    return jnp.sum((uu+dd)*pbe_xc_spin(uu,dd,a,b,c))
duu=np.array(np.asarray(jax.grad(E,2)(nu,nd,suuJ,sudJ,sddJ))); dud=np.array(np.asarray(jax.grad(E,3)(nu,nd,suuJ,sudJ,sddJ))); ddd=np.array(np.asarray(jax.grad(E,4)(nu,nd,suuJ,sudJ,sddJ)))
for a in (duu,dud,ddd): a[~mask]=0.0
fA_u=[2*duu*gnu[i]+dud*gnd[i] for i in range(3)]; fA_d=[2*ddd*gnd[i]+dud*gnu[i] for i in range(3)]
divA_u=-div(fA_u); divA_d=-div(fA_d)
print(f"GGA-flux-divergence (the antisym/phi terms) LORRAX-assembly vs QE-assembly, on REAL CrI3 grid:")
print(f"  up: max|A-B|={np.max(np.abs(divA_u-divB_u))*RY*1000:.4f} meV  rms={np.sqrt(((divA_u-divB_u)**2)[mask].mean())*RY*1000:.4f} meV")
print(f"  dn: max|A-B|={np.max(np.abs(divA_d-divB_d))*RY*1000:.4f} meV  rms={np.sqrt(((divA_d-divB_d)**2)[mask].mean())*RY*1000:.4f} meV")
Bdiv_A=0.5*(divA_u-divA_d); Bdiv_B=0.5*(divB_u-divB_d)
print(f"  B-channel flux-div diff: max|={np.max(np.abs(Bdiv_A-Bdiv_B))*RY*1000:.4f} meV (this is term-1+2 in B)")
