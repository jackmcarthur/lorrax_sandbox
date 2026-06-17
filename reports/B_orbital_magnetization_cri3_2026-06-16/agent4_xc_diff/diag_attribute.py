"""FINAL ATTRIBUTION: across occupied bands at k=Gamma, compute
  H_res(b)  = <b|H_LORRAX|b> - eps(b)             (the bug)
  dkih(b)   = <b|T+Vloc+VH+VNL>_LORRAX - kih_QE(b)  (charge/kin channel error)
  vxc_lrx(b)= <b|Vbar+B.sigma|b>_LORRAX
and verify  H_res = dkih + (vxc_lrx - (eps-kih_QE))  identically, then report
  corr(H_res, dkih) and corr(H_res, vxc_lrx)  over OCCUPIED bands,
and the std of dkih and of vxc_lrx-(eps-kih_QE) over OCCUPIED bands only.
This says definitively whether the band dependence is charge/kin (dkih) or Vxc.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart, _bsigma_psi)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
KIHFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "kih.dat")
def read_dat_diag(path, kx, ky, kz):
    out = {}; lines = open(path).readlines(); i = 0
    while i < len(lines):
        h = lines[i].split()
        if len(h) >= 5 and abs(float(h[0])-kx) < 1e-6 and abs(float(h[1])-ky) < 1e-6 and abs(float(h[2])-kz) < 1e-6:
            nd = int(h[3]); no = int(h[4])
            for j in range(nd):
                p = lines[i+1+j].split(); out[int(p[1])] = float(p[2])
            return out
        i += 1 + int(h[3]) + int(h[4])
    raise RuntimeError("k")

wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor), q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=trunc)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G)); n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
M = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u = M/(jnp.linalg.norm(M)+1e-30)
seg = jnp.where(amag > 1e-12, jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]), 1.0)
n_up = (n+seg*amag)/2; n_dn = (n-seg*amag)/2
rgu = jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd = jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu, Vd = compute_V_xc_spin(n_up, n_dn, rgu, rgd, G_cart)
Vbar = 0.5*(Vu+Vd); Bmag = 0.5*seg*(Vu-Vd); inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot), float(wfn.ecutwfc), tuple(int(x) for x in fg)))
ik = 0
kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
qe_kih = read_dat_diag(KIHFILE, kv[0], kv[1], kv[2])
eps = np.asarray(wfn.energies[0, k_red, :nocc], float)*RY2EV

box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
# Full H residual
V_scf = build_V_scf(V_loc, V_H, Vbar)
H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
U = _psi_box_to_G_sphere(box, Gk)[:nocc]*H_k.mask[None, None, :].astype(box.dtype)
HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz, H_k.vnl_Z, H_k.vnl_E, H_k.mask, B_vec)
nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
H_diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm*RY2EV
H_res = H_diag - eps  # eV

# kih channel
VxcOFF = build_V_scf(V_loc, V_H, None)
Hk0 = setup_H_k_from_kvec(kv, VxcOFF, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
HU0 = apply_H_k_from_G(U, Hk0.T_diag, Hk0.V_scf, Hk0.Gx, Hk0.Gy, Hk0.Gz, Hk0.vnl_Z, Hk0.vnl_E, Hk0.mask, None)
kih_lrx = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU0).real)/nrm*RY2EV
dkih = np.array([kih_lrx[b-1]-qe_kih[b] for b in range(1, nocc+1)])  # eV
# Vxc channel
vxc_lrx = H_diag - kih_lrx
vxc_qe_true = np.array([eps[b-1]-qe_kih[b] for b in range(1, nocc+1)])
dvxc = vxc_lrx - vxc_qe_true

print(f"OCC bands={nocc}")
print(f"H_res:  RMS={np.sqrt((H_res**2).mean())*1000:.2f} mean={H_res.mean()*1000:.2f} std={H_res.std()*1000:.2f} meV")
print(f"dkih :  mean={dkih.mean()*1000:.2f} std={dkih.std()*1000:.2f} meV  (charge/kin channel)")
print(f"dvxc :  mean={dvxc.mean()*1000:.2f} std={dvxc.std()*1000:.2f} meV  (Vxc channel)")
print(f"identity check  max|H_res - (dkih+dvxc)| = {np.abs(H_res-(dkih+dvxc)).max()*1000:.4f} meV")
def corr(a, b):
    a = a-a.mean(); b = b-b.mean()
    return float((a*b).sum()/(np.sqrt((a**2).sum()*(b**2).sum())+1e-30))
print(f"corr(H_res, dkih) = {corr(H_res, dkih):.3f}   corr(H_res, dvxc) = {corr(H_res, dvxc):.3f}")
print(f"=> the channel with high std AND high corr is the culprit.")
