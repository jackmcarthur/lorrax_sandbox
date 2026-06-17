"""Critical: does 0.5*(V_up+V_dn) from compute_V_xc_spin EQUAL the scalar
compute_V_xc when zeta=0 on a REALISTIC grid?  If yes, the spin Vbar machinery is
self-consistent with the proven scalar path. Then test: at FINITE zeta, does
0.5*(V_up+V_dn) match QE's gradcorr v(k,1) GGA-charge?  We reuse LORRAX's OWN
scalar compute_V_xc on n_tot vs the spin path Vbar."""
import os; os.environ.setdefault("JAX_ENABLE_X64","1")
import sys; sys.path.insert(0,"/global/u2/j/jackm/software"); sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
import numpy as np, jax.numpy as jnp
from psp.xc import compute_V_xc, compute_V_xc_spin, XCLevel, pbe_functional
N=40; L=9.0; x=(np.arange(N)/N-0.5)*L
X,Y,Z=np.meshgrid(x,x,x,indexing='ij'); R=np.sqrt(X**2+Y**2+Z**2)
gk=2*np.pi/L*np.fft.fftfreq(N)*N; GX,GY,GZ=np.meshgrid(gk,gk,gk,indexing='ij')
G_cart=jnp.asarray(np.stack([GX,GY,GZ],-1))
RY=13.605693122994
n_tot=jnp.asarray(1.2*np.exp(-(R/1.5)**2)+0.05)
xc_fn,lvl=pbe_functional()
# scalar path
rgt=jnp.fft.fftn(n_tot)
V_scalar=compute_V_xc(n_tot, rgt, G_cart, xc_fn, XCLevel.GGA)
# spin path at zeta=0 (m=0)
n_up=n_tot/2; n_dn=n_tot/2; rgu=jnp.fft.fftn(n_up); rgd=jnp.fft.fftn(n_dn)
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart); Vbar0=0.5*(Vu+Vd)
print(f"zeta=0: max|Vbar_spin - V_scalar| = {float(jnp.max(jnp.abs(Vbar0-V_scalar)))*RY*1000:.4f} meV  (scale {float(jnp.max(jnp.abs(V_scalar)))*RY:.2f} eV)")
# finite zeta: spin Vbar with m, compare to scalar on SAME n_tot (should DIFFER, that's physical)
m=jnp.asarray(0.6*np.exp(-(R/1.0)**2)); n_up=(n_tot+m)/2; n_dn=(n_tot-m)/2
rgu=jnp.fft.fftn(n_up); rgd=jnp.fft.fftn(n_dn)
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart); Vbar=0.5*(Vu+Vd)
print(f"finite zeta: max|Vbar_spin - V_scalar(n_tot)| = {float(jnp.max(jnp.abs(Vbar-V_scalar)))*RY*1000:.1f} meV (PHYSICAL: spin Vbar != nonmag Vxc at zeta!=0)")
