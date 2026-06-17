"""Localize WHICH sub-term of the spin V_xc charge part Vbar carries the band-
dependent error, by toggling exchange-only vs correlation-only and LDA vs GGA,
and measuring the change in the worst bands' <psi|Vbar|psi>.

Vbar = 0.5*(V_up + V_dn) where V_s = df/dn_s - div(GGA flux).
Decompose V_up, V_dn into:
  x_lda  : exchange,    sigma=0
  c_lda  : correlation, sigma=0
  x_gga  : exchange    full-minus-lda (v1 part) and its flux
  c_gga  : correlation full-minus-lda (v1 part) and its flux
Build Vbar from each piece and measure <b|Vbar_piece|b> on the worst bands; the
piece whose value DIFFERS from a QE-faithful reference is the culprit.

Reference: rebuild Vbar a SECOND way using QE's exact spin-LDA potentials
(slater_spin exchange + pw_spin correlation, analytic v_up/v_dn) at the LDA level
to cross-check LORRAX's autodiff LDA Vbar pointwise.  If they agree, the LDA is
fine and the bug is in the GGA flux of one channel.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import _compute_sigma_spin, pbe_xc_spin
from psp.pseudos import load_pseudopotentials

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G)); n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
M = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u = M/(jnp.linalg.norm(M)+1e-30)
seg = jnp.where(amag > 1e-12, jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]), 1.0)
n_up = (n+seg*amag)/2; n_dn = (n-seg*amag)/2
nu = jnp.maximum(n_up, 1e-10); nd = jnp.maximum(n_dn, 1e-10)
rgu = jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd = jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
suu, sud, sdd = _compute_sigma_spin(rgu, rgd, G_cart)
gnu = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rgu)) for i in range(3)]
gnd = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rgd)) for i in range(3)]
def div(fields):
    out = jnp.zeros_like(n)
    for i in range(3):
        out = out + jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*jnp.fft.fftn(fields[i])))
    return out

# ---- LORRAX-faithful pieces via the SAME pbe_xc_spin used in production ----
# exchange-only and correlation-only energies
from jax_xc_local.pbe import pbe_x, pbe_c  # unpolarized building blocks (Hartree)
def _ex_spin(uu, dd, auu, add):
    nn = jnp.maximum(uu+dd, 1e-30)
    return (uu*pbe_x(2.0*uu, 4.0*auu) + dd*pbe_x(2.0*dd, 4.0*add))/nn
# correlation: reuse the production _pbe_c_spin via pbe_xc_spin minus exchange
from psp.xc import _pbe_x_spin, _pbe_c_spin
def E_x(uu, dd, auu, aud, add):
    return jnp.sum((uu+dd)*2.0*_pbe_x_spin(uu, dd, auu, add))
def E_c(uu, dd, auu, aud, add):
    return jnp.sum((uu+dd)*2.0*_pbe_c_spin(uu, dd, auu+2.0*aud+add))
zers = jnp.zeros_like(nu)

def chan_pot(Efn, with_gga):
    a1, a2, a3 = (suu, sud, sdd) if with_gga else (zers, zers, zers)
    dfu = jax.grad(Efn, 0)(nu, nd, a1, a2, a3)
    dfd = jax.grad(Efn, 1)(nu, nd, a1, a2, a3)
    if not with_gga:
        return dfu, dfd
    duu = jax.grad(Efn, 2)(nu, nd, a1, a2, a3)
    dud = jax.grad(Efn, 3)(nu, nd, a1, a2, a3)
    ddd = jax.grad(Efn, 4)(nu, nd, a1, a2, a3)
    mask = (n > 1e-6) & ((suu+2*sud+sdd) > 1e-10)
    duu = jnp.where(mask, duu, 0.0); dud = jnp.where(mask, dud, 0.0); ddd = jnp.where(mask, ddd, 0.0)
    dfu_g = jnp.where(mask, dfu, jax.grad(Efn, 0)(nu, nd, zers, zers, zers))
    dfd_g = jnp.where(mask, dfd, jax.grad(Efn, 1)(nu, nd, zers, zers, zers))
    Vu = dfu_g - div([2*duu*gnu[i] + dud*gnd[i] for i in range(3)])
    Vd = dfd_g - div([2*ddd*gnd[i] + dud*gnu[i] for i in range(3)])
    return Vu, Vd

Vu_xl, Vd_xl = chan_pot(E_x, False)
Vu_cl, Vd_cl = chan_pot(E_c, False)
Vu_xg, Vd_xg = chan_pot(E_x, True)
Vu_cg, Vd_cg = chan_pot(E_c, True)
Vbar_xl = 0.5*(Vu_xl+Vd_xl); Vbar_cl = 0.5*(Vu_cl+Vd_cl)
Vbar_xg = 0.5*(Vu_xg+Vd_xg); Vbar_cg = 0.5*(Vu_cg+Vd_cg)
Vbar_full = Vbar_xg + Vbar_cg

ik = 0
box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
def exp_field(field):
    Vp = psi_r*field[None, None]
    nrm = jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), psi_r).real
    return np.asarray(jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Vp).real/nrm)*RY2EV

print(f"{'b':>4} {'Vbar_full':>10} {'x_lda':>9} {'c_lda':>9} {'x_gga-x_lda':>12} {'c_gga-c_lda':>12}")
for b in (0, 28, 35, 39, 40, 63, 69, nocc-1):
    vf = exp_field(Vbar_full)[b]
    vxl = exp_field(Vbar_xl)[b]; vcl = exp_field(Vbar_cl)[b]
    vxg = exp_field(Vbar_xg)[b]; vcg = exp_field(Vbar_cg)[b]
    print(f"{b:>4} {vf:>10.4f} {vxl:>9.4f} {vcl:>9.4f} {vxg-vxl:>12.4f} {vcg-vcl:>12.4f}")
