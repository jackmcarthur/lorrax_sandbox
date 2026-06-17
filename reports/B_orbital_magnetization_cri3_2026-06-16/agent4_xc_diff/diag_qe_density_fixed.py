"""CORRECTED ultimate density control: build V_H, Vbar, B from QE's OWN
charge-density.hdf5 (rhotot_g valence + m_x/y/z), with verified normalization
(rho integral=70, |m|=7, m matches LORRAX to 0.1% sign-independent).

If <v|H|v>-eps is STILL ~22 meV band-dependent with QE's exact density, the
density is exonerated and the residual is purely in the V_xc functional/assembly.
If it DROPS to <1 meV, the LORRAX density reconstruction (the 0.1% m error or the
sign flip) was the cause.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp, h5py
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               poisson_potential_from_rhoG, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor), q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2]); N = nx*ny*nz; vol = float(wfn.cell_volume)
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)

CHG = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(WFN))), "scf", "CrI3.save", "charge-density.hdf5")
f = h5py.File(CHG, 'r'); mill = np.asarray(f['MillerIndices'])
def cmplx(name):
    a = np.asarray(f[name]); return a[0::2]+1j*a[1::2]
def scat(c):
    b = np.zeros((nx, ny, nz), np.complex128); b[mill[:, 0] % nx, mill[:, 1] % ny, mill[:, 2] % nz] = c; return jnp.asarray(b)
rho_val = jnp.real(jnp.fft.ifftn(scat(cmplx('rhotot_g'))))*N      # valence, integral=70
m_x = jnp.real(jnp.fft.ifftn(scat(cmplx('m_x'))))*N
m_y = jnp.real(jnp.fft.ifftn(scat(cmplx('m_y'))))*N
m_z = jnp.real(jnp.fft.ifftn(scat(cmplx('m_z'))))*N
print(f"QE density: rho={float(rho_val.sum())*vol/N:.3f}e net_mz={float(m_z.sum())*vol/N:+.3f}")

V_H = jnp.real(poisson_potential_from_rhoG(jnp.fft.fftn(rho_val, norm='ortho'), bdot, bvec, wfn.blat, truncation_2d=trunc))
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G)); n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
M = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u = M/(jnp.linalg.norm(M)+1e-30)
seg = jnp.where(amag > 1e-12, jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]), 1.0)
n_up = (n+seg*amag)/2; n_dn = (n-seg*amag)/2
rgu = jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rgd = jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu, Vd = compute_V_xc_spin(n_up, n_dn, rgu, rgd, G_cart)
Vbar = 0.5*(Vu+Vd); Bmag = 0.5*seg*(Vu-Vd); inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
B = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot), float(wfn.ecutwfc), tuple(int(x) for x in fg)))
allr = []
for ik in (0, 4, 8):
    kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
    eps = np.asarray(wfn.energies[0, k_red, :nocc], float); box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
    Vs = build_V_scf(V_loc, V_H, Vbar); H_k = setup_H_k_from_kvec(kv, Vs, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc]*H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz, H_k.vnl_Z, H_k.vnl_E, H_k.mask, B)
    nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm
    allr.append((diag-eps)*RY2EV*1000)
r = np.concatenate([np.abs(x) for x in allr]); r0 = allr[0]
print(f"[QE-exact-density] RMS={np.sqrt((r**2).mean()):.2f} max={r.max():.2f} | "
      f"k0: b0={r0[0]:+.2f} b28={r0[28]:+.2f} b35={r0[35]:+.2f} b63={r0[63]:+.2f} VBM={r0[nocc-1]:+.2f}")
print("=> ~22 meV: density exonerated, bug in Vxc functional. <1 meV: density was the cause.")
