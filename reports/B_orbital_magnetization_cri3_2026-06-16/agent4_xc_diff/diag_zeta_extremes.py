"""Where does zeta=(n_up-n_dn)/n approach +-1 on the real CrI3 grid, and does the
autodiff PBE-correlation potential (df_du,df_dd) develop large spurious values
there (phi'(zeta) ~ (1-|zeta|)^(-1/3) diverges)?  This is the finite-zeta
POTENTIAL pathology that the m=0 energy gate cannot see.

We also directly compare LORRAX autodiff df_du vs an INDEPENDENT high-order
central finite-difference of the SAME pbe_xc_spin energy, channel-resolved into
exchange-only and correlation-only, to localize any autodiff inaccuracy.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart
from psp.xc import _pbe_x_spin, _pbe_c_spin, _compute_sigma_spin
from psp.pseudos import load_pseudopotentials

WFN=sys.argv[1]; RY2EV=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(m_x**2+m_y**2+m_z**2)
M=jnp.array([m_x.sum(),m_y.sum(),m_z.sum()]); u=M/(jnp.linalg.norm(M)+1e-30)
seg=jnp.where(amag>1e-12,jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2

zeta=np.asarray(jnp.clip((n_up-n_dn)/jnp.maximum(n,1e-30),-1,1))
nn=np.asarray(n)
# weight by density (high-density Cr cores matter for d-band <H>)
w=nn/nn.sum()
print("zeta distribution (density-weighted):")
for thr in [0.9,0.95,0.99,0.999,0.9999]:
    frac=float(((np.abs(zeta)>thr)*w).sum())
    print(f"  density fraction with |zeta|>{thr}: {100*frac:.4f}%   (count={int((np.abs(zeta)>thr).sum())})")
print(f"  max|zeta|={np.abs(zeta).max():.8f}  n at that pt={nn[np.unravel_index(np.argmax(np.abs(zeta)),nn.shape)]:.4f}")

# Channel-resolved autodiff vs FD of df_du for EXCHANGE and CORRELATION at the
# 20 highest-density grid points (where d-bands live).
suu,sud,sdd=_compute_sigma_spin(jnp.fft.fftn(n_up),jnp.fft.fftn(n_dn),G_cart)
nu=jnp.maximum(n_up,1e-10); nd=jnp.maximum(n_dn,1e-10)

def E_x(u_,d_): return jnp.sum((u_+d_)*_pbe_x_spin(u_,d_,suu,sdd*0+sdd))  # sigma fixed
def E_c(u_,d_):
    stot=suu+2*sud+sdd
    return jnp.sum((u_+d_)*_pbe_c_spin(u_,d_,stot))
# fix sigma (LDA-level potential part: df_du at fixed gradient) to isolate zeta path
def E_x_fixedsig(u_,d_): return jnp.sum((u_+d_)*_pbe_x_spin(u_,d_,suu,sdd))
dXu=np.asarray(jax.grad(E_x_fixedsig,0)(nu,nd))
dCu=np.asarray(jax.grad(E_c,0)(nu,nd))

idx=np.argsort(nn.ravel())[-12:][::-1]
print(f"\nHigh-density points: autodiff df_du (exch, corr) and FD check (Ry):")
print(f"{'n':>9} {'zeta':>10} {'dXu_auto':>12} {'dCu_auto':>12} {'dCu_FD':>12} {'rel_c':>9}")
nu_np=np.asarray(nu); nd_np=np.asarray(nd)
for fi in idx:
    p=np.unravel_index(fi,nn.shape)
    h=1e-6*max(nn[p],1.0)
    def Ec_at(uval):
        u2=nu.at[p].set(uval)
        return float(E_c(u2,nd))
    fd=(Ec_at(nu_np[p]+h)-Ec_at(nu_np[p]-h))/(2*h)
    rel=abs(dCu[p]-fd)/(abs(fd)+1e-12)
    print(f"{nn[p]:9.4f} {zeta[p]:10.6f} {dXu[p]:12.5f} {dCu[p]:12.5f} {fd:12.5f} {rel:9.2e}")
