"""DECISIVE: compare LORRAX <psi|Vbar + B.sigma|psi> per band against QE's OWN
vxc.dat (pw2bgw write_vxc_g, nspin=4) at k=Gamma.

The total QE eps = kih_diag + vxc_diag.  Note 3 established kih(LORRAX)-kih(QE)
= +34 meV band-INDEPENDENT.  So the ~24 meV BAND-DEPENDENT residual must be
entirely in  vxc(LORRAX) - vxc(QE).  This script measures exactly that, against
QE's analytic noncollinear V_xc -- no reimplementation of QE formulas.

Outputs per band: vxc_LORRAX, vxc_QE(.dat), diff(meV).  The diff should
reproduce the band-dependent pattern (worst on Cr 3p/3d idx 28,29,35,39,40,63).
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import build_G_cart, _bsigma_psi
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.pseudos import load_pseudopotentials

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
DATFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "vxc.dat")

def read_dat_diag(path, kx, ky, kz):
    """Read BGW-format .dat: return dict band-> Re(value, eV) for the k block."""
    out = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split()
        if len(h) >= 5 and abs(float(h[0])-kx) < 1e-6 and abs(float(h[1])-ky) < 1e-6 and abs(float(h[2])-kz) < 1e-6:
            ndiag = int(h[3]); noff = int(h[4])
            for j in range(ndiag):
                p = lines[i+1+j].split()
                out[int(p[1])] = float(p[2])   # col2 = band index; Re, eV
            return out
        # skip this block
        ndiag = int(h[3]); noff = int(h[4])
        i += 1 + ndiag + noff
    raise RuntimeError("k not found")

wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))

# number of bands in the .dat file (header) -- use that many bands
nb = 180

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=True)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)

n = rho_val + rho_core
amag = jnp.sqrt(m_x**2 + m_y**2 + m_z**2)
M_net = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()])
u_ax = M_net / (jnp.linalg.norm(M_net) + 1e-30)
segni = jnp.where(amag > 1e-12, jnp.sign(m_x*u_ax[0]+m_y*u_ax[1]+m_z*u_ax[2]), 1.0)

n_up = (n + segni*amag)/2; n_dn = (n - segni*amag)/2
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
rhoG_up = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
rhoG_dn = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
V_up, V_dn = compute_V_xc_spin(n_up, n_dn, rhoG_up, rhoG_dn, G_cart)
Vbar = 0.5*(V_up + V_dn)
Bmag = 0.5*segni*(V_up - V_dn)
inv = jnp.where(amag > 1e-12, 1.0/(amag + 1e-30), 0.0)
B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)

# k=Gamma is ik=0 (unfolded). Find the unfolded index whose kvec is (0,0,0).
ik = 0
kv = np.asarray(sym.unfolded_kpts[ik], float)
print(f"k[{ik}] = {kv}")
qe_diag = read_dat_diag(DATFILE, kv[0], kv[1], kv[2])
KIHFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "kih.dat")
qe_kih = read_dat_diag(KIHFILE, kv[0], kv[1], kv[2])

box = load_kpoint_fftbox(wfn, sym, meta, ik, nb)   # (nb, 2, nx,ny,nz)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
# <psi| Vbar |psi>  +  <psi| B.sigma |psi>, ortho convention (matches apply_H_k)
Vpsi = psi_r * Vbar[None, None]
Bpsi = _bsigma_psi(psi_r, B_vec)
vxc_lorrax = jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Vpsi + Bpsi).real
vxc_lorrax = np.asarray(vxc_lorrax) * RY2EV   # eV

# diagnostics: split Vbar vs B contribution
vbar_only = np.asarray(jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Vpsi).real) * RY2EV
b_only = np.asarray(jnp.einsum('bsxyz,bsxyz->b', jnp.conj(psi_r), Bpsi).real) * RY2EV

eps = np.asarray(wfn.energies[0, int(sym.irr_idx_k[ik]), :nb], float) * RY2EV

# sanity: QE kih+vxc should equal eps (validates band ordering + units)
chk = np.array([(qe_kih[b]+qe_diag[b]) - eps[b-1] for b in range(1, nb+1) if b in qe_diag and b in qe_kih])
print(f"\n[sanity] QE (kih+vxc) - eps_WFN: RMS={np.sqrt((chk**2).mean())*1000:.3f} meV "
      f"max|={np.abs(chk).max()*1000:.3f} meV  (should be ~0 if ordering/units OK)")

print(f"\n{'b':>4} {'eps':>10} {'vxc_LRX':>11} {'vxc_QE':>11} {'diff(meV)':>10} {'Vbar':>10} {'B':>9}")
diffs = []
watch = {28,29,35,39,40,63,0,nocc-1}
for b in range(1, nb+1):
    if b not in qe_diag:
        continue
    d = (vxc_lorrax[b-1] - qe_diag[b]) * 1000
    diffs.append(d)
    if (b-1) in watch or abs(d) > 30:
        print(f"{b-1:>4} {eps[b-1]:>10.4f} {vxc_lorrax[b-1]:>11.5f} {qe_diag[b]:>11.5f} {d:>10.2f} {vbar_only[b-1]:>10.4f} {b_only[b-1]:>9.4f}")
diffs = np.array(diffs)
print(f"\nvxc(LORRAX)-vxc(QE): RMS={np.sqrt((diffs**2).mean()):.2f} meV  "
      f"mean={diffs.mean():.2f}  max|={np.abs(diffs).max():.2f}  "
      f"(over {len(diffs)} bands)")
print(f"zero-mean band-dependent? std={diffs.std():.2f} meV")
