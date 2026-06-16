"""Localize the CrI3 V_scf bug: NLCC (rho_core) vs short-range V_loc.
Build V_scf with rho_core ON and OFF; measure <v|H|v>-eps_v each time."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
print(f"rho_core integral = {float(jnp.sum(rho_core))*float(wfn.cell_volume)/(nx*ny*nz):.4f} e")
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))
ik = 0; kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)

def resid(rc, rcG, label):
    V_H, V_xc = compute_V_H_and_V_xc(rho_val, rc, rcG, G_cart, jnp.asarray(wfn.bdot),
                                     jnp.asarray(wfn.bvec), wfn.blat, truncation_2d=trunc)
    V_scf = build_V_scf(V_loc, V_H, V_xc)
    H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                          H_k.vnl_Z, H_k.vnl_E, H_k.mask)
    nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real) / nrm
    r = np.abs(diag - eps) * RY2EV * 1000
    print(f"  [{label}] max={r.max():.1f} meV mean={r.mean():.1f}  "
          f"band0(deep)={r[0]:.1f}  VBM(b{nocc-1})={r[nocc-1]:.1f}")
    return r

print("=== NLCC ON (production) ===");  resid(rho_core, rho_core_G, "NLCC on")
print("=== NLCC OFF (rho_core=0) ==="); resid(jnp.zeros_like(rho_core), jnp.zeros_like(rho_core_G), "NLCC off")
