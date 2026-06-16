"""Definitive sign test for the analytic nonlocal velocity.

Compares compute_vnl_velocity_cart (analytic dZ-based dV_NL/dK) against a
finite-difference of <m|V_NL(k)|n> w.r.t. k (the unambiguous physical
derivative, psi held fixed).  Sign agreement => physical velocity is
p + compute_vnl_velocity_cart;  opposite => p - compute_vnl_velocity_cart.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax
_SRC = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src"
sys.path.insert(0, _SRC)
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import generate_gvectors_k
from psp.get_dipole_mtxels import compute_vnl_velocity_cart, compute_vnl_matrix_from_setup
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn)
nb = 80
meta = Meta.from_system(wfn, sym, int(wfn.nelec), nb - int(wfn.nelec), nb, 0, False)
pdir = os.path.dirname(WFN)
pseudos = vnl_ops.__dict__.get('load_pseudopotentials', None)
from psp.pseudos import load_pseudopotentials
pseudos = load_pseudopotentials(pdir)
vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=int(wfn.nspinor))

ik = 1
wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
Gk, _ = generate_gvectors_k(ik, sym, wfn, meta)
k_crys = np.asarray(sym.unfolded_kpts[ik], float)
B = np.asarray(wfn.bvec, float) * float(wfn.blat)
Binv = np.linalg.inv(B)

# analytic dV_NL/dK_cart : (3, nb, nb)
vNL_an = np.asarray(compute_vnl_velocity_cart(wfn_k, Gk, k_crys, vnl_setup))

# numeric d<V_NL>/dk_cart along x (alpha=0): psi fixed, shift projector k
alpha = 0
delta = 1e-3
dk_crys = delta * Binv[alpha]            # crystal shift for +delta along cart x
Vp = np.asarray(compute_vnl_matrix_from_setup(wfn_k, Gk, k_crys + dk_crys, vnl_setup))
Vm = np.asarray(compute_vnl_matrix_from_setup(wfn_k, Gk, k_crys - dk_crys, vnl_setup))
vNL_num = (Vp - Vm) / (2 * delta)        # physical d<V_NL>/dk_cart_x : (nb, nb)

an = vNL_an[alpha]
# compare the largest-magnitude off-diagonal elements (these drive orbital moment)
mag = np.abs(vNL_num)
np.fill_diagonal(mag, 0.0)
idx = np.dstack(np.unravel_index(np.argsort(mag.ravel())[::-1][:8], mag.shape))[0]
print("Largest |d<V_NL>/dk_x| off-diagonal elements: analytic vs numeric")
print(f"{'(m,n)':>10} {'analytic':>26} {'numeric(FD)':>26}  ratio an/num")
ratios = []
for m, n in idx:
    a = an[m, n]; nu = vNL_num[m, n]
    r = (a / nu) if abs(nu) > 1e-9 else float('nan')
    ratios.append(np.real(r))
    print(f"({m:3d},{n:3d}) {a.real:+.5e}{a.imag:+.5e}j  {nu.real:+.5e}{nu.imag:+.5e}j  {np.real(r):+.3f}")
rr = np.array([r for r in ratios if np.isfinite(r)])
print(f"\nmean Re(analytic/numeric) over top elements = {rr.mean():+.3f}")
print("=> analytic == +dV_NL/dk  (use p + vNL)" if rr.mean() > 0
      else "=> analytic == -dV_NL/dk  (use p - vNL)")
# also report diagonal magnitudes (why HF was inconclusive)
dnum = np.abs(np.diag(vNL_num)).mean(); doff = mag.max()
print(f"\nmean |diag numeric vNL| = {dnum:.3e}   max |offdiag numeric vNL| = {doff:.3e}"
      f"   (ratio {doff/max(dnum,1e-12):.1f}x -> diagonal HF test is insensitive)")
