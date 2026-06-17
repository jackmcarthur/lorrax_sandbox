"""GGA CORRELATION potential: LORRAX _pbe_c_spin (autodiff) vs QE pbec_spin
(analytic v1c_up, v1c_dw, v2c), transcribed from XClib/qe_funct_corr_gga.f90.

QE pbec_spin(rho, zeta, grho, iflag=1): rho=total, zeta=(nu-nd)/n, grho=|grad n_tot|^2.
Returns sc (energy*rho? -> sc=rho*h0), v1c_up, v1c_dw (dH/dn_s, GGA part only),
v2c (dH/d grho, i.e. d sc/d grho ... the gradient potential).
QE ALSO adds the LDA pw_spin part separately; here we compare ONLY the GGA-H part:
  LORRAX: v1c_s^gga = d(n*(2*_pbe_c_spin - 2*pw_lda_c))/dn_s  (Ry); and the flux coeffs.
We compare QE (v1c_up_gga, v2c) to LORRAX autodiff of the H-only correlation energy.
All in Ry (QE drivers are Hartree -> *2).
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys as _s; _s.path.insert(0, "/global/u2/j/jackm/software"); _s.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
import jax, jax.numpy as jnp, numpy as np
RY = 13.605693122994

# ---------- QE pw_spin (LDA correlation) analytic, Hartree ----------
def pw_spin(rs, zeta):
    rs12 = np.sqrt(rs); rs32 = rs*rs12; rs2 = rs**2
    def G(a, a1, b1, b2, b3, b4):
        om = 2*a*(b1*rs12+b2*rs+b3*rs32+b4*rs2)
        dom = 2*a*(0.5*b1*rs12+b2*rs+1.5*b3*rs32+2*b4*rs2)
        olog = np.log(1+1/om); e = -2*a*(1+a1*rs)*olog
        v = -2*a*(1+2/3*a1*rs)*olog - 2/3*a*(1+a1*rs)*dom/(om*(om+1))
        return e, v
    epwc, vpwc = G(0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    epwcp, vpwcp = G(0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    ae, va = G(0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671); alpha = -ae; vpwca = -va
    fz0 = 1.709921; p43 = 4/3; third = 1/3
    fz = ((1+zeta)**p43 + (1-zeta)**p43 - 2)/(2**p43-2)
    dfz = ((1+zeta)**third - (1-zeta)**third)*4/(3*(2**p43-2))
    ec = epwc + alpha*fz*(1-zeta**4)/fz0 + (epwcp-epwc)*fz*zeta**4
    vc = vpwc*(1-fz*zeta**4) + vpwcp*fz*zeta**4 + vpwca*fz*(1-zeta**4)/fz0
    dec = 4*zeta**3*fz*(epwcp-epwc-alpha/fz0) + dfz*(zeta**4*(epwcp-epwc)+(1-zeta**4)*alpha/fz0)
    vc_up = vc + dec*(1-zeta); vc_dn = vc - dec*(1+zeta)
    return ec, vc_up, vc_dn

# ---------- QE pbec_spin GGA-H part analytic, Hartree ----------
def pbec_spin(rho, zeta, grho):
    ga = 0.031091; be = 0.06672455060314922; third = 1/3
    pi34 = 0.6203504908994; xkf = 1.919158292677513; xks = 1.128379167095513
    rs = pi34/rho**third
    ec, vc_up, vc_dn = pw_spin(rs, zeta)
    kf = xkf/rs; ks = xks*np.sqrt(kf)
    fz = 0.5*((1+zeta)**(2/3)+(1-zeta)**(2/3)); fz3 = fz**3
    dfz = ((1+zeta)**(-1/3)-(1-zeta)**(-1/3))/3
    t = np.sqrt(grho)/(2*fz*ks*rho)
    expe = np.exp(-ec/(fz3*ga)); af = be/ga*(1/(expe-1))
    bfup = expe*(vc_up-ec)/fz3; bfdw = expe*(vc_dn-ec)/fz3
    y = af*t*t; xy = (1+y)/(1+y+y*y); qy = y*y*(2+y)/(1+y+y*y)**2
    s1 = 1+be/ga*t*t*xy; h0 = fz3*ga*np.log(s1)
    dh0up = be*t*t*fz3/s1*(-7/3*xy - qy*(af*bfup/be-7/3))
    dh0dw = be*t*t*fz3/s1*(-7/3*xy - qy*(af*bfdw/be-7/3))
    dh0zup = (3*h0/fz - be*t*t*fz**2/s1*(2*xy - qy*(3*af*expe*ec/fz3/be+2)))*dfz*(1-zeta)
    dh0zdw = -(3*h0/fz - be*t*t*fz**2/s1*(2*xy - qy*(3*af*expe*ec/fz3/be+2)))*dfz*(1+zeta)
    ddh0 = be*fz/(2*ks*ks*rho)*(xy-qy)/s1
    v1c_up = h0 + dh0up + dh0zup
    v1c_dw = h0 + dh0dw + dh0zdw
    v2c = ddh0   # d(rho*h0)/d grho  (per QE: h(ipol)= (..+v2c)*grad ; v2c=2*ddh0? check)
    return h0, v1c_up, v1c_dw, ddh0

# ---------- LORRAX _pbe_c_spin autodiff (Ry) of the FULL correlation (LDA+H) ----------
from psp.xc import _pbe_c_spin   # returns ec+H per electron, Hartree
def lrx_c(nu, nd, sigma_tot):
    def Ec(uu, dd, st):
        return jnp.sum((uu+dd)*2.0*_pbe_c_spin(uu, dd, st))  # Ry
    A = (jnp.array([nu]), jnp.array([nd]), jnp.array([sigma_tot]))
    v1u = float(jax.grad(Ec, 0)(*A)[0]); v1d = float(jax.grad(Ec, 1)(*A)[0])
    dst = float(jax.grad(Ec, 2)(*A)[0])  # d(n*2*eps_c)/d sigma_total
    return v1u, v1d, dst

print(f"{'n':>5}{'zeta':>6}{'sigma':>8} | {'v1c_up dLRX-QE(meV)':>20} {'v1c_dw d(meV)':>14} {'v2c LRX':>10} {'v2c QE':>10}")
for n0 in (0.5, 2.0):
  for z0 in (0.2, 0.5, 0.9):
    for sg in (0.1, 1.0):
      nu = n0*(1+z0)/2; nd = n0*(1-z0)/2
      # LORRAX full correlation potential (LDA+H), Ry
      v1u_L, v1d_L, dst_L = lrx_c(nu, nd, sg)
      # QE full = pw_spin(LDA) + pbec_spin(H); convert Hartree->Ry (*2)
      rs = (3/(4*np.pi*n0))**(1/3); ec_lda, vcu_lda, vcd_lda = pw_spin(rs, z0)
      h0, v1u_h, v1d_h, ddh0 = pbec_spin(n0, z0, sg)
      v1u_Q = 2*(vcu_lda + v1u_h); v1d_Q = 2*(vcd_lda + v1d_h)
      # QE v2c (d sc/d grho): the flux uses h = (v2x+v2c)*grad_tot; v2c=2*ddh0 (=d(rho*h0)/d|grad|^2 chain)
      v2c_Q = 2*ddh0   # Hartree-per-? -> compare to LORRAX dst (Ry). LORRAX flux= -div(2*dst*grad_tot? )
      print(f"{n0:5.2f}{z0:6.2f}{sg:8.2f} | {(v1u_L-v1u_Q)*RY*1000:20.3f} {(v1d_L-v1d_Q)*RY*1000:14.3f} {dst_L:10.5f} {v2c_Q*RY/RY:10.5f}")
print("\nv1c_up/v1c_dw diffs ~0 => correlation v1 potential matches QE.")
print("(v2c columns are in different normalizations; compare the flux assembly separately.)")
