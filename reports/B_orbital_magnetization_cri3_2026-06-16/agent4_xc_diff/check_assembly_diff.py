"""LORRAX-vs-QE noncollinear V_xc ASSEMBLY diff on the real CrI3 FM grid.

The V_s=dE/dn_s identity (check_finite_zeta_fd.py: 6e-8) PROVES (V_up,V_dn) is
exact. So the only remaining structural difference is the (V_up,V_dn) -> 4-comp
potential map. This script reconstructs BOTH maps on the SAME m(r),n(r) and
compares, isolating: (A) segni, (B) the |m|-cusp in the gradient source, and
(C) any sign of m_z flipping (which would activate segni!=+1).
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart
from psp.xc import compute_V_xc_spin
from psp.pseudos import load_pseudopotentials

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=True)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)

n = rho_val + rho_core
amag = jnp.sqrt(m_x**2 + m_y**2 + m_z**2)

# ---- (C) Does m_z (the dominant axis) change sign on the grid? ----
mz = np.asarray(m_z); mxn=np.asarray(m_x); myn=np.asarray(m_y); am=np.asarray(amag)
sig_phys = np.sign(mz)                          # physical sign of dominant component
# segni in QE (lsign=.TRUE., ux ~ z): sign(m.ux) = sign(m_z)
frac_neg = float(np.mean((am>1e-6) & (sig_phys<0)))
# where m has appreciable magnitude, is m_z ever negative?
big = am > 0.01*am.max()
print(f"\n[C] grid points with |m|>1e-6 and m_z<0 (segni=-1): {100*frac_neg:.3f}%")
print(f"    among |m|>1%max: m_z>0 {100*np.mean(mz[big]>0):.2f}%  m_z<0 {100*np.mean(mz[big]<0):.2f}%")
print(f"    max|m_x,m_y|={max(abs(mxn).max(),abs(myn).max()):.4f}  max|m_z|={abs(mz).max():.4f}")

# ---- (A) |m|-cusp: grad of |m| vs grad of (segni*m_z-ish). ----
# LORRAX builds n_up=(n+amag)/2; the GGA flux uses grad(n_up). Compare |grad amag|
# to |grad m_z| in regions where m_z>0 (where they should coincide up to in-plane).
def grad_mag(field):
    fG = jnp.fft.fftn(field); g2 = jnp.zeros((nx,ny,nz))
    for i in range(3):
        gi = jnp.real(jnp.fft.ifftn(1j*G_cart[...,i]*fG)); g2 = g2 + gi**2
    return np.asarray(jnp.sqrt(g2))
g_amag = grad_mag(amag); g_mz = grad_mag(jnp.asarray(mz))
# cusp metric: |grad amag| - |grad mz| is large only near m sign flips / in-plane twist
cusp = g_amag - g_mz
print(f"\n[A] |grad|m||: max={g_amag.max():.4f}  |grad m_z|: max={g_mz.max():.4f}")
print(f"    (|grad|m|| - |grad m_z|): max={cusp.max():.4e} mean|={np.abs(cusp).mean():.4e}")
print(f"    -> if ~0 everywhere, |m| has NO cusp (m_z single-signed): segni is a NON-issue.")

# ---- Build LORRAX B and QE B, compare directly ----
n_up=(n+amag)/2; n_dn=(n-amag)/2
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
rhoG_up = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
rhoG_dn = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
V_up, V_dn = compute_V_xc_spin(n_up, n_dn, rhoG_up, rhoG_dn, G_cart)
Bmag = 0.5*(V_up - V_dn)
inv = jnp.where(amag>1e-12, 1.0/(amag+1e-30), 0.0)
B_lorrax_z = np.asarray(Bmag * m_z * inv)        # z-component
# QE: v(ir,4)=segni*0.5*(vgg1-vgg2)*m_z/amag. vgg1=V_up,vgg2=V_dn here (same densities).
segni = np.where(am>1e-12, np.sign(mz), 1.0)      # ux~z
B_qe_z = segni * np.asarray(Bmag) * mz / np.where(am>1e-12, am, 1.0)
dz = np.abs(B_lorrax_z - B_qe_z)
print(f"\n[B] B_z(LORRAX) vs B_z(QE,segni): max|diff| = {dz.max()*RY2EV*1000:.4f} meV/bohr^3-ish")
print(f"    max|B_z| = {np.abs(B_lorrax_z).max()*RY2EV:.3f} eV  (sharp at Cr)")
print(f"    => nonzero ONLY where segni=-1.  If 0, segni is NOT the bug.")
