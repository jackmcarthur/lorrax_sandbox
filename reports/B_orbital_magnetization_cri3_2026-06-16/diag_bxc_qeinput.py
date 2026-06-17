"""Discriminate the residual: m-reconstruction vs functional.
GATE 1: reconstructed m vs QE charge-density.hdf5 m.
QE-input gate: build V_scf+B_vec from QE's EXACT rho+m, measure residual.
If QE-input << recon-input(60meV) -> my m reconstruction is the culprit.
If both ~60meV -> the functional / noncollinear treatment is."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, h5py, jax.numpy as jnp
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

WFN, QESAVE = sys.argv[1], sys.argv[2]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2]); N = nx*ny*nz
vol = float(wfn.cell_volume)
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B); bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_L = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
mxL, myL, mzL = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)

# QE exact density + magnetization (G-space -> real)
f = h5py.File(os.path.join(QESAVE, "charge-density.hdf5"), "r")
mill = np.asarray(f["MillerIndices"])
def to_r(name):
    g = np.zeros((nx, ny, nz), np.complex128)
    g[mill[:,0]%nx, mill[:,1]%ny, mill[:,2]%nz] = np.asarray(f[name]).view(np.complex128)
    return jnp.asarray(np.fft.ifftn(g).real * N)
rho_Q = to_r("rhotot_g"); mxQ, myQ, mzQ = to_r("m_x"), to_r("m_y"), to_r("m_z")

# GATE 1: reconstructed m vs QE m.  Note LORRAX sign may differ globally; compare |m| too.
def integ(a): return float(jnp.sum(a)) * vol / N
print(f"GATE1  net m_z: LORRAX {integ(mzL):+.4f}  QE {integ(mzQ):+.4f}")
print(f"       |m| int: LORRAX {integ(jnp.sqrt(mxL**2+myL**2+mzL**2)):.4f}  "
      f"QE {integ(jnp.sqrt(mxQ**2+myQ**2+mzQ**2)):.4f}")
print(f"       max|mx,my| LORRAX {float(jnp.maximum(jnp.abs(mxL),jnp.abs(myL)).max()):.4f}  "
      f"QE {float(jnp.maximum(jnp.abs(mxQ),jnp.abs(myQ)).max()):.4f}  (collinearity)")
amagL = jnp.sqrt(mxL**2+myL**2+mzL**2); amagQ = jnp.sqrt(mxQ**2+myQ**2+mzQ**2)
print(f"       ||m|_L - |m|_Q|| / ||m_Q|| = {float(jnp.linalg.norm(amagL-amagQ)/jnp.linalg.norm(amagQ)):.2e}")

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))

def build_and_resid(rho_val, amag, mx, my, mz, label):
    V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec,
                                  wfn.blat, truncation_2d=trunc)
    n = rho_val + rho_core
    n_up = (n + amag) / 2; n_dn = (n - amag) / 2
    core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
    rgu = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
    rgd = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
    V_up, V_dn = compute_V_xc_spin(n_up, n_dn, rgu, rgd, G_cart)
    Vbar = 0.5*(V_up+V_dn); Bmag = 0.5*(V_up-V_dn)
    inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
    B_vec = jnp.stack([Bmag*mx*inv, Bmag*my*inv, Bmag*mz*inv], axis=0)
    V_scf = build_V_scf(V_loc, V_H, Vbar)
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
        diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real) / nrm
        r = np.abs(diag - eps) * RY2EV * 1000
        print(f"  [{label}] k={ik}: max={r.max():.1f} meV mean={r.mean():.1f}  band0={r[0]:.1f}  VBM={r[nocc-1]:.1f}")

print("=== recon rho+m ==="); build_and_resid(rho_L, amagL, mxL, myL, mzL, "recon")
# QE stores m in +z but the WFN spinors carry it in -z; negate QE m to the spinor frame.
print("=== QE exact rho+m (m sign-matched to spinor frame) ===")
build_and_resid(rho_Q, amagQ, -mxQ, -myQ, -mzQ, "QE")
