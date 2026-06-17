"""Isolate V_NL: replace LORRAX's reconstructed local V_scf with QE's EXACT one
(pp_11 cube = V_loc+V_H+V_xc charge channel), keep LORRAX B_vec + V_NL.
Report residual SPREAD (residual - per-k mean) to remove any constant G=0
reference offset between QE's cube and the WFN eigenvalue convention.
If spread -> ~0: the LORRAX local reconstruction was the 60 meV.
If spread stays ~tens of meV: it's V_NL."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
RY2EV = 13.605693122994; trunc = True; OUT = os.path.dirname(os.path.realpath(__file__))

def read_cube(path, nx, ny, nz, nat=8):
    vals = []
    for ln in open(path).read().split("\n")[6+nat:]:
        vals.extend(ln.split())
    return jnp.asarray(np.array(vals[:nx*ny*nz], float).reshape(nx, ny, nz))  # Ry

WFN = sys.argv[1]
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B); bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
mx, my, mz = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=trunc)
n = rho_val + rho_core; amag = jnp.sqrt(mx**2+my**2+mz**2)
V_up, V_dn = compute_V_xc_spin((n+amag)/2, (n-amag)/2,
                               jnp.fft.fftn((n+amag)/2), jnp.fft.fftn((n-amag)/2), G_cart)
Vbar = 0.5*(V_up+V_dn); Bmag = 0.5*(V_up-V_dn)
inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
B_vec = jnp.stack([Bmag*mx*inv, Bmag*my*inv, Bmag*mz*inv], axis=0)

V_scf_lx = build_V_scf(V_loc, V_H, Vbar)
V_scf_qe = read_cube(f"{OUT}/pp_11.cube", nx, ny, nz)
# align constant: match means (eigenvalue reference is a separate constant handled by spread)
V_scf_qe = V_scf_qe - V_scf_qe.mean() + jnp.asarray(V_scf_lx).mean()
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))

def report(V_scf, label):
    for ik in (0, 4):
        kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
        eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
        box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
        H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
        Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
        HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                              H_k.vnl_Z, H_k.vnl_E, H_k.mask, B_vec)
        nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
        diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm
        d = (diag-eps)*RY2EV*1000           # signed meV
        spread = d - d.mean()               # remove constant reference offset
        print(f"  [{label}] k={ik}: |resid|max={np.abs(d).max():.1f}  "
              f"SPREAD(max-min)={d.max()-d.min():.1f}  rms_spread={np.sqrt((spread**2).mean()):.1f} meV")

print("=== LORRAX local V_scf (baseline) ===");  report(V_scf_lx, "LX")
print("=== QE EXACT local V_scf (cube) + LORRAX B + V_NL ==="); report(V_scf_qe, "QE")
