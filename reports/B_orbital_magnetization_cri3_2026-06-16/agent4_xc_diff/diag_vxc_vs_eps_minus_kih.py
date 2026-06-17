"""DECISIVE & RELIABLE: compare LORRAX <Vbar+B.sigma> per band against QE's TRUE
V_xc matrix element = eps - kih_QE  (NOT the buggy vxc.dat).

We showed: LORRAX <T+Vloc+VH+VNL> == QE kih.dat to +33.3 meV (band-INDEPENDENT,
std 4.3 meV).  And QE's vxc.dat is internally inconsistent (eps - kih_QE differs
from vxc.dat by ~0.5 eV on band28).  Since eps = kih + vxc EXACTLY in QE's H,
the faithful QE V_xc matrix element is  vxc_QE_true(b) = eps(b) - kih_QE(b).

So:  diff(b) = <Vbar+B.sigma>_LORRAX(b) - (eps(b) - kih_QE(b)).
The +33 meV kih offset shows up here as a -33 meV CONSTANT (it cancels in the
full H: H = kih + vxc, and LORRAX_kih = QE_kih + 33, so to get H right LORRAX_vxc
must be QE_vxc - 33).  The BAND-DEPENDENT part of diff is the real bug, and its
de-meaned pattern must equal the -74/-41/+50 H-residual pattern.

Also split Vbar-only vs B-only contributions and report which one carries the
band dependence.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc, _bsigma_psi
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.pseudos import load_pseudopotentials

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
KIHFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "kih.dat")

def read_dat_diag(path, kx, ky, kz):
    out = {}
    lines = open(path).readlines(); i = 0
    while i < len(lines):
        h = lines[i].split()
        if len(h) >= 5 and abs(float(h[0])-kx) < 1e-6 and abs(float(h[1])-ky) < 1e-6 and abs(float(h[2])-kz) < 1e-6:
            ndiag = int(h[3]); noff = int(h[4])
            for j in range(ndiag):
                p = lines[i+1+j].split(); out[int(p[1])] = float(p[2])
            return out
        ndiag = int(h[3]); noff = int(h[4]); i += 1 + ndiag + noff
    raise RuntimeError("k not found")

wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)
nb = 180

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G)); n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
M = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u = M/(jnp.linalg.norm(M)+1e-30)
seg = jnp.where(amag > 1e-12, jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]), 1.0)
n_up = (n+seg*amag)/2; n_dn = (n-seg*amag)/2
rgu = jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd = jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu, Vd = compute_V_xc_spin(n_up, n_dn, rgu, rgd, G_cart)
Vbar = 0.5*(Vu+Vd); Bmag = 0.5*seg*(Vu-Vd); inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)

ik = 0
kv = np.asarray(sym.unfolded_kpts[ik], float)
qe_kih = read_dat_diag(KIHFILE, kv[0], kv[1], kv[2])
k_red = int(sym.irr_idx_k[ik])
eps = np.asarray(wfn.energies[0, k_red, :nb], float)*RY2EV

box = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
Vpsi = psi_r*Vbar[None, None]; Bpsi = _bsigma_psi(psi_r, B_vec)
vbar_b = np.asarray(jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Vpsi).real)*RY2EV
b_b = np.asarray(jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Bpsi).real)*RY2EV
vxc_lrx = vbar_b + b_b

vxc_qe_true = np.array([eps[b-1]-qe_kih[b] for b in range(1, nb+1) if b in qe_kih])  # eV
bands = [b for b in range(1, nb+1) if b in qe_kih]
diff = np.array([vxc_lrx[b-1] for b in bands])*1000 - vxc_qe_true*1000  # meV
print(f"vxc_LORRAX - (eps-kih_QE): mean={diff.mean():.2f} std={diff.std():.2f} "
      f"max|={np.abs(diff).max():.2f} meV")
de = diff - diff.mean()
print(f"DE-MEANED (band-dependent, the real bug): RMS={np.sqrt((de**2).mean()):.2f} "
      f"max|={np.abs(de).max():.2f} meV")
print(f"\n{'b':>4} {'eps':>9} {'vxc_LRX':>10} {'vxc_QEtrue':>11} {'diff':>8} {'demean':>8} {'Vbar':>9} {'B':>8}")
for i, b in enumerate(bands):
    if (b-1) in {0, 1, 28, 29, 35, 39, 40, 63, 69, nocc-1}:
        print(f"{b-1:>4} {eps[b-1]:>9.3f} {vxc_lrx[b-1]:>10.4f} {vxc_qe_true[i]:>11.4f} "
              f"{diff[i]:>8.2f} {de[i]:>8.2f} {vbar_b[b-1]:>9.4f} {b_b[b-1]:>8.4f}")
