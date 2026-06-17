"""Localize the GGA bug to EXCHANGE vs CORRELATION channel.

dvxc (LORRAX Vxc - QE true Vxc) std=23 meV corr 0.98 with H_res -> the bug is in
Vxc.  LDA exchange & B verified exact.  So it's the GGA potential.  Split:

  full     : full spin-PBE (x_gga + c_gga)  [current LORRAX]
  c_lda    : full GGA exchange, LDA-only correlation (drop GGA-c flux & v1c-gga)
  x_lda    : LDA-only exchange, full GGA correlation (drop GGA-x flux & v1x-gga)

We measure <v|H|v>-eps band-by-band for each.  The variant whose worst-band
pattern (28/35/63) MATCHES 'full' is NOT the culprit; the one that CHANGES the
pattern most localizes the bug.  (This is internal-consistency localization; the
absolute QE-correctness of each piece was checked via slater_spin/pw_spin LDA.)
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import _compute_sigma_spin, _pbe_x_spin, _pbe_c_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
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
nu = jnp.maximum(n_up, 1e-10); nd = jnp.maximum(n_dn, 1e-10)
rgu = jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd = jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
suu, sud, sdd = _compute_sigma_spin(rgu, rgd, G_cart)
gnu = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rgu)) for i in range(3)]
gnd = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rgd)) for i in range(3)]
zers = jnp.zeros_like(nu)
mask = (n > 1e-6) & ((suu+2*sud+sdd) > 1e-10)

def div(fields):
    out = jnp.zeros_like(n)
    for i in range(3):
        out = out + jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*jnp.fft.fftn(fields[i])))
    return out

def make_E(use_x_gga, use_c_gga):
    def E(uu, dd, a1, a2, a3):
        suu_ = a1 if use_x_gga else zers
        sdd_ = a3 if use_x_gga else zers
        stot_c = (a1+2*a2+a3) if use_c_gga else (zers)
        ex = _pbe_x_spin(uu, dd, suu_, sdd_)
        ec = _pbe_c_spin(uu, dd, stot_c)
        return jnp.sum((uu+dd)*2.0*(ex+ec))  # Ry
    return E

def assemble(use_x_gga, use_c_gga):
    E = make_E(use_x_gga, use_c_gga)
    dfu = jax.grad(E, 0)(nu, nd, suu, sud, sdd); dfd = jax.grad(E, 1)(nu, nd, suu, sud, sdd)
    duu = jax.grad(E, 2)(nu, nd, suu, sud, sdd); dud = jax.grad(E, 3)(nu, nd, suu, sud, sdd); ddd = jax.grad(E, 4)(nu, nd, suu, sud, sdd)
    # LDA fallback for v1
    El = make_E(False, False)
    dfu_l = jax.grad(El, 0)(nu, nd, zers, zers, zers); dfd_l = jax.grad(El, 1)(nu, nd, zers, zers, zers)
    dfu = dfu_l + jnp.where(mask, dfu-dfu_l, 0.0); dfd = dfd_l + jnp.where(mask, dfd-dfd_l, 0.0)
    duu = jnp.where(mask, duu, 0.0); dud = jnp.where(mask, dud, 0.0); ddd = jnp.where(mask, ddd, 0.0)
    Vu = dfu - div([2.0*duu*gnu[i] + dud*gnd[i] for i in range(3)])
    Vd = dfd - div([2.0*ddd*gnd[i] + dud*gnu[i] for i in range(3)])
    Vbar = 0.5*(Vu+Vd); Bmag = 0.5*seg*(Vu-Vd)
    B = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)
    return Vbar, B

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot), float(wfn.ecutwfc), tuple(int(x) for x in fg)))
def resid(Vbar_, B_, label):
    rs = []; pb0 = None
    for ik in (0, 4, 8):
        kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
        eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
        box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
        Vs = build_V_scf(V_loc, V_H, Vbar_)
        H_k = setup_H_k_from_kvec(kv, Vs, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
        Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        U = _psi_box_to_G_sphere(box, Gk)[:nocc]*H_k.mask[None, None, :].astype(box.dtype)
        HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz, H_k.vnl_Z, H_k.vnl_E, H_k.mask, B_)
        nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
        diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm
        r = (diag-eps)*RY2EV*1000; rs.append(r)
        if ik == 0: pb0 = r
    ra = np.concatenate([np.abs(x) for x in rs])
    print(f"  [{label}] RMS={np.sqrt((ra**2).mean()):.2f} max={ra.max():.2f} | "
          f"b0={pb0[0]:+.1f} b28={pb0[28]:+.1f} b35={pb0[35]:+.1f} b63={pb0[63]:+.1f}")
    return pb0

print("=== full: spin-PBE x_gga + c_gga (current) ===")
resid(*assemble(True, True), "full")
print("=== c_lda: full GGA exchange, LDA-only correlation ===")
resid(*assemble(True, False), "c_lda")
print("=== x_lda: LDA-only exchange, full GGA correlation ===")
resid(*assemble(False, True), "x_lda")
print("=== both_lda: pure LDA (no GGA at all) ===")
resid(*assemble(False, False), "both_lda")
