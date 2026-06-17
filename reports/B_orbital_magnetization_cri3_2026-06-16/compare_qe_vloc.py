"""Compare QE's bare V_loc (pp_2.cube) and total V_scf (pp_11.cube) to LORRAX's
reconstruction. Mean-removed (G=0 conventions differ). Localizes the 60 meV."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from psp.dft_operators import compute_V_H_and_V_xc, build_V_scf, build_G_cart
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.pseudos import load_pseudopotentials
RY2EV = 13.605693122994

def read_cube(path):
    with open(path) as f:
        lines = f.read().split("\n")
    nat = int(lines[2].split()[0])
    nx = int(lines[3].split()[0]); ny = int(lines[4].split()[0]); nz = int(lines[5].split()[0])
    vals = []
    for ln in lines[6 + nat:]:
        vals.extend(ln.split())
    a = np.array(vals[:nx*ny*nz], float).reshape(nx, ny, nz)  # C-order (ix,iy,iz)
    return a  # QE cube potential in Ry

WFN = sys.argv[1]; QESAVE = sys.argv[2]; OUT = os.path.dirname(os.path.realpath(__file__))
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B); bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=True)
V_loc = np.asarray(V_loc)

def cmp(qe, lx, name):
    qe = qe - qe.mean(); lx = lx - lx.mean()
    d = qe - lx
    print(f"  {name}: max|Δ|={np.abs(d).max()*RY2EV*1000:8.1f} meV  "
          f"rms={np.sqrt((d**2).mean())*RY2EV*1000:7.1f} meV  "
          f"corr={np.corrcoef(qe.ravel(),lx.ravel())[0,1]:.5f}")

vloc_qe = read_cube(f"{OUT}/pp_2.cube")
cmp(vloc_qe, V_loc, "V_loc (bare ionic)")

# V_scf charge channel = V_loc + V_H + Vbar
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
mx, my, mz = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=True)
n = rho_val + rho_core; amag = jnp.sqrt(mx**2+my**2+mz**2)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
rgu = jnp.fft.fftn((n+amag)/2 - core_grid/2) + rho_core_G/2
rgd = jnp.fft.fftn((n-amag)/2 - core_grid/2) + rho_core_G/2
V_up, V_dn = compute_V_xc_spin((n+amag)/2, (n-amag)/2, rgu, rgd, G_cart)
Vbar = 0.5*(V_up+V_dn)
V_scf = np.asarray(build_V_scf(V_loc, V_H, Vbar))
vscf_qe = read_cube(f"{OUT}/pp_11.cube")
cmp(vscf_qe, V_scf, "V_scf (V_loc+V_H+Vbar)")
cmp(vscf_qe - vloc_qe, np.asarray(V_H + Vbar), "V_H+Vxc (QE 11-2 vs LX)")
