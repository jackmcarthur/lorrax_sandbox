"""POINTWISE: LORRAX autodiff LDA spin V_up/V_dn  vs  QE exact analytic
slater_spin (exchange) + pw_spin (PW92 correlation) potentials.

Both fed the SAME (n, |m|) = (rho_val+rho_core, amag).  If they agree to ~ueV
pointwise, the LDA functional+autodiff is EXACT and the band-dependent residual
is NOT in the LDA V_xc.  We weight the pointwise diff by band 28's |psi|^2 (the
worst band) to see if any disagreement projects onto it.

QE LDA noncollinear: zeta = +amag/arho (POSITIVE), v(:,1)=0.5(vx_up+vx_dw+vc_up+vc_dw)*e2,
field magnitude vs = 0.5(vx_up+vc_up - vx_dw - vc_dw)*e2 along +m_hat.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import build_G_cart
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import pbe_xc_spin
from psp.pseudos import load_pseudopotentials

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
arho = jnp.maximum(jnp.abs(n), 1e-10)
zeta = jnp.clip(amag/arho, 0.0, 1.0-1e-12)   # QE: positive zeta = amag/arho
n_up = (n+amag)/2; n_dn = (n-amag)/2          # positive-amag spin densities (QE LDA)
nu = jnp.maximum(n_up, 1e-12); nd = jnp.maximum(n_dn, 1e-12)

# ---- LORRAX autodiff LDA (sigma=0) per-channel potentials ----
def E_lda(uu, dd):
    z = jnp.zeros_like(uu)
    return jnp.sum((uu+dd)*pbe_xc_spin(uu, dd, z, z, z))  # Ry
Vu_lrx = jax.grad(E_lda, 0)(nu, nd)   # Ry
Vd_lrx = jax.grad(E_lda, 1)(nu, nd)

# ---- QE exact analytic: slater_spin (x) + pw_spin (c), Hartree -> Ry (*2 = e2) ----
f_sl = -1.10783814957303361; alpha = 2.0/3.0; p43 = 4.0/3.0; third = 1.0/3.0
# slater_spin
rho13_up = ((1.0+zeta)*arho)**third
rho13_dn = ((1.0-zeta)*arho)**third
vx_up = p43*f_sl*alpha*rho13_up
vx_dw = p43*f_sl*alpha*rho13_dn
# pw_spin (PW92), zeta in [0,1]
rs = (3.0/(4.0*jnp.pi*arho))**third
rs12 = jnp.sqrt(rs); rs32 = rs*rs12; rs2 = rs**2
def Gpw(a, a1, b1, b2, b3, b4):
    om = 2.0*a*(b1*rs12+b2*rs+b3*rs32+b4*rs2)
    dom = 2.0*a*(0.5*b1*rs12+b2*rs+1.5*b3*rs32+2.0*b4*rs2)
    olog = jnp.log(1.0+1.0/om)
    e = -2.0*a*(1.0+a1*rs)*olog
    v = -2.0*a*(1.0+2.0/3.0*a1*rs)*olog - 2.0/3.0*a*(1.0+a1*rs)*dom/(om*(om+1.0))
    return e, v
epwc, vpwc = Gpw(0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
epwcp, vpwcp = Gpw(0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
alphae, valpha = Gpw(0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
alphac = -alphae; vpwca = -valpha
fz0 = 1.709921
fz = ((1.0+zeta)**p43 + (1.0-zeta)**p43 - 2.0)/(2.0**p43 - 2.0)
dfz = ((1.0+zeta)**third - (1.0-zeta)**third)*4.0/(3.0*(2.0**p43-2.0))
ec = epwc + alphac*fz*(1.0-zeta**4)/fz0 + (epwcp-epwc)*fz*zeta**4
dec_dz = (4.0*zeta**3*fz*(epwcp-epwc-alphac/fz0)
          + dfz*(zeta**4*(epwcp-epwc) + (1.0-zeta**4)*alphac/fz0))
vc_up = ec + dec_dz*(1.0-zeta)
vc_dw = ec - dec_dz*(1.0+zeta)
# total QE per-channel potential in Ry (e2=2 Hartree->Ry):  v_s = e2*(vx_s+vc_s)
Vu_qe = 2.0*(vx_up+vc_up)
Vd_qe = 2.0*(vx_dw+vc_dw)

# Mask to magnetized/dense region
mreg = (n > 1e-6)
du_pt = (Vu_lrx-Vu_qe); dd_pt = (Vd_lrx-Vd_qe)
print(f"LDA Vbar=0.5(Vu+Vd) LORRAX-QE: max|={float(jnp.max(jnp.abs(0.5*(du_pt+dd_pt)*mreg)))*RY2EV*1000:.4f} meV(pt) "
      f"rms={float(jnp.sqrt(jnp.mean((0.5*(du_pt+dd_pt))**2*mreg)))*RY2EV*1000:.4f}")
print(f"LDA B=0.5(Vu-Vd)   LORRAX-QE: max|={float(jnp.max(jnp.abs(0.5*(du_pt-dd_pt)*mreg)))*RY2EV*1000:.4f} meV(pt) "
      f"rms={float(jnp.sqrt(jnp.mean((0.5*(du_pt-dd_pt))**2*mreg)))*RY2EV*1000:.4f}")

# Project onto band 28 |psi|^2 (the worst band)
ik = 0
box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
for b in (0, 28, 35, 63, nocc-1):
    w = (jnp.abs(psi_r[b, 0])**2 + jnp.abs(psi_r[b, 1])**2)
    dVbar = 0.5*(du_pt+dd_pt)
    proj = float(jnp.sum(w*dVbar))*RY2EV*1000  # <b|dVbar_LDA|b> in meV
    print(f"  band{b}: <b|(Vbar_LDA_LORRAX - Vbar_LDA_QE)|b> = {proj:+.3f} meV")
print("=> If these are ~0, the LDA Vxc is exact pointwise; bug is GGA flux or V_loc.")
