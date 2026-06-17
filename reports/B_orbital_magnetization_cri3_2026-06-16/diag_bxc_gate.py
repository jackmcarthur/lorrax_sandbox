"""GATE 2 — does B_xc reproduce CrI3's DFT H matrix elements?
Build V_scf + B_vec (spin V_xc), measure <v|H|v>-eps_v with B_xc OFF vs ON."""
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

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B)
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=True)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)

# V_H (spin-independent) reused from the scalar routine; discard its scalar V_xc.
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec,
                              wfn.blat, truncation_2d=trunc)

# spin V_xc: n_up/dn = (n ± |m|)/2 with n = rho_val + rho_core (core nonmagnetic).
n = rho_val + rho_core
amag = jnp.sqrt(m_x**2 + m_y**2 + m_z**2)
n_up = (n + amag) / 2; n_dn = (n - amag) / 2
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
rhoG_up = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2     # precise-core split
rhoG_dn = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
V_up, V_dn = compute_V_xc_spin(n_up, n_dn, rhoG_up, rhoG_dn, G_cart)
Vbar = 0.5 * (V_up + V_dn)
Bmag = 0.5 * (V_up - V_dn)                                    # <=0 in majority region
inv = jnp.where(amag > 1e-12, 1.0 / (amag + 1e-30), 0.0)
B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)
V_scf = build_V_scf(V_loc, V_H, Vbar)
print(f"B_xc field: max|B_vec| = {float(jnp.abs(B_vec).max())*RY2EV:.3f} eV")

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))

def resid(ik, bvec_arg, label):
    kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
    eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
    box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
    H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                          H_k.vnl_Z, H_k.vnl_E, H_k.mask, bvec_arg)
    nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real) / nrm
    r = np.abs(diag - eps) * RY2EV * 1000
    print(f"  [{label}] k={ik}: max={r.max():.1f} meV mean={r.mean():.1f}  "
          f"band0(deep)={r[0]:.1f}  VBM(b{nocc-1})={r[nocc-1]:.1f}")

print("=== B_xc OFF (old, broken) ==="); resid(0, None, "off"); resid(4, None, "off")
print("=== B_xc ON (fix) ===");          resid(0, B_vec, "on");  resid(4, B_vec, "on")
