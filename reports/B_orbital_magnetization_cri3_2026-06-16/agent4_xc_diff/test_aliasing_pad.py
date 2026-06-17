"""ALIASING test (suspect 4). Re-evaluate <v|(V_scf + B.sigma)|v> with the real-
space products formed on a 2x ZERO-PADDED FFT grid (dealiased), vs the native
ecutrho grid. If the d-state residual drops on the padded grid, the surviving
~35 meV is aliasing of sharp V/B against localized d-states. If unchanged, it is
NOT aliasing (then it's a genuine V_xc/core/convention error).

We compute <v| V_scf |v> and <v| B.sigma |v> by brute force in real space:
  <v|O|v> = sum_r conj(psi_v(r)) O(r) psi_v(r)   (O multiplicative in r).
On native grid: psi on the ecutwfc sphere -> IFFT on ecutrho box -> multiply ->
  the diagonal <v|Vpsi> already truncates to the same box. To DEALIAS we instead
  upsample psi and V/B to a 2x grid, multiply there, and integrate -- which keeps
  the high-G tail of the product that the native grid folds back.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import compute_V_H_and_V_xc, build_V_scf, build_G_cart
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.pseudos import load_pseudopotentials
WFN=sys.argv[1]; RY2EV=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(m_x**2+m_y**2+m_z**2); seg=jnp.where(amag>1e-12,jnp.sign(-m_z),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
rhoG_up=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rhoG_dn=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rhoG_up,rhoG_dn,G_cart)
Vbar=0.5*(Vu+Vd); Bmag=0.5*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
Bz=seg*Bmag*m_z*inv; Bx=seg*Bmag*m_x*inv; By=seg*Bmag*m_y*inv

def upsample(field, f=2):
    """Zero-pad a real field's FFT to an f*N grid (dealias upsample)."""
    F=jnp.fft.fftn(field); Fs=jnp.fft.fftshift(F)
    pad=[( (f*s-s)//2, (f*s-s)-(f*s-s)//2 ) for s in field.shape]
    Fp=jnp.pad(Fs,pad); Fp=jnp.fft.ifftshift(Fp)
    return jnp.real(jnp.fft.ifftn(Fp))*(f**3)

ik=0
kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
eps=np.asarray(wfn.energies[0,k_red,:nocc],float)
box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)             # (nocc,2,nx,ny,nz) sphere-G box
psi_r=jnp.fft.ifftn(box,axes=(-3,-2,-1),norm='ortho')    # native grid real space
up=psi_r[:,0]; dn=psi_r[:,1]
nrm=np.asarray((jnp.sum(jnp.abs(up)**2+jnp.abs(dn)**2,axis=(-3,-2,-1))).real)

def Bsig_expect_native():
    # <v|B.sigma|v> native grid
    s_up=Bz*up+(Bx-1j*By)*dn; s_dn=(Bx+1j*By)*up-Bz*dn
    val=jnp.sum(jnp.conj(up)*s_up+jnp.conj(dn)*s_dn,axis=(-3,-2,-1)).real
    return np.asarray(val)/nrm

def Bsig_expect_padded(f=2):
    upU=jnp.stack([upsample(jnp.real(up[v]),f)+1j*upsample(jnp.imag(up[v]),f) for v in range(up.shape[0])])
    dnU=jnp.stack([upsample(jnp.real(dn[v]),f)+1j*upsample(jnp.imag(dn[v]),f) for v in range(dn.shape[0])])
    BzU=upsample(Bz,f); BxU=upsample(Bx,f); ByU=upsample(By,f)
    s_up=BzU*upU+(BxU-1j*ByU)*dnU; s_dn=(BxU+1j*ByU)*upU-BzU*dnU
    val=jnp.sum(jnp.conj(upU)*s_up+jnp.conj(dnU)*s_dn,axis=(-3,-2,-1)).real
    nrmU=jnp.sum(jnp.abs(upU)**2+jnp.abs(dnU)**2,axis=(-3,-2,-1)).real
    return np.asarray(val/nrmU)
Bn=Bsig_expect_native(); Bp=Bsig_expect_padded(2)
print("band: <v|B.sigma|v> native vs 2x-padded (dealiased), diff in meV")
for b in list(range(8))+[nocc-1]:
    print(f"  b{b:2d}: native={Bn[b]*RY2EV*1000:9.2f}  padded={Bp[b]*RY2EV*1000:9.2f}  "
          f"dealias_shift={(Bp[b]-Bn[b])*RY2EV*1000:7.2f} meV")
print(f"\nmax|dealias shift| over occ = {np.abs((Bp-Bn))*RY2EV*1000 .max():.2f} meV")
print("=> if dealias_shift ~ the 35 meV residual, ALIASING of B.sigma is the surviving bug.")
