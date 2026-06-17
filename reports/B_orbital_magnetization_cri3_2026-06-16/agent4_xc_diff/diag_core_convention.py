"""DECISIVE: does the GGA gradient's CORE-DENSITY convention explain the
band-dependent residual on the sharp Cr 3p/3d bands?

QE gradcorr (nspin==4, domag): grho(:,is) = grad(rho_r2g(n_is + (1/2)rho_core_GRIDDED)).
  i.e. it FFTs the REAL-SPACE GRIDDED spin density (which already contains the
  gridded NLCC core) and gradients THAT.  rho_r2g is a plain forward FFT.

LORRAX compute_V_xc_spin is fed rho_G_up/dn built with the "precise-core trick":
  rho_G_up = fft(n_up - core_grid/2) + rho_core_G/2
  -> the gridded core is SUBTRACTED and the ANALYTIC G-space core (sharper,
     un-aliased) is substituted.  This changes the gradient near the sharp Cr
     core vs QE's gridded-core gradient.

Variants (all use segni + B, full spin-GGA, measured by full <v|H|v>-eps):
  precise : current LORRAX (analytic precise-core in rho_G_up/dn) AND in V_H/Vbar charge
  gridded : QE-gradcorr convention -> rho_G_up = fft(n_up), rho_G_dn = fft(n_dn)
            (pure FFT of the gridded spin density; gridded NLCC core in the gradient)
            -- this is EXACTLY what QE's gradcorr does for the field.

If 'gridded' << 'precise', the precise-core trick is the bug for the spin field.
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
                                    q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x, m_y, m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=trunc)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
n = rho_val + rho_core
amag = jnp.sqrt(m_x**2 + m_y**2 + m_z**2)
M = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u = M/(jnp.linalg.norm(M)+1e-30)
seg = jnp.where(amag > 1e-12, jnp.sign(m_x*u[0]+m_y*u[1]+m_z*u[2]), 1.0)
n_up = (n + seg*amag)/2; n_dn = (n - seg*amag)/2
inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)

def assemble(rhoG_up, rhoG_dn):
    Vu, Vd = compute_V_xc_spin(n_up, n_dn, rhoG_up, rhoG_dn, G_cart)
    Vbar = 0.5*(Vu+Vd); Bmag = 0.5*seg*(Vu-Vd)
    B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)
    return Vbar, B_vec

# precise : analytic precise-core (current LORRAX)
rhoG_up_p = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
rhoG_dn_p = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
Vbar_p, B_p = assemble(rhoG_up_p, rhoG_dn_p)

# gridded : QE gradcorr convention -> pure FFT of the gridded spin density
rhoG_up_g = jnp.fft.fftn(n_up)
rhoG_dn_g = jnp.fft.fftn(n_dn)
Vbar_g, B_g = assemble(rhoG_up_g, rhoG_dn_g)

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))

def resid(Vbar_, B_, label, verbose_bands=False):
    rs = []
    perband0 = None
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
        r = (diag-eps)*RY2EV*1000
        rs.append(r)
        if ik == 0:
            perband0 = r
    r = np.concatenate([np.abs(x) for x in rs])
    print(f"  [{label}] RMS={np.sqrt((r**2).mean()):.2f} mean(signed,k0)={perband0.mean():.2f} max={r.max():.2f} meV")
    if verbose_bands:
        for b in (0, 28, 29, 35, 39, 40, 63, 69):
            print(f"        band{b}: {perband0[b]:+.2f} meV")
    return perband0

print("=== precise: analytic precise-core in spin-density gradient (current LORRAX) ===")
rp = resid(Vbar_p, B_p, "precise", verbose_bands=True)
print("=== gridded: QE gradcorr convention (pure FFT of gridded spin density) ===")
rg = resid(Vbar_g, B_g, "gridded", verbose_bands=True)
print(f"\nband-wise change precise->gridded (k0, signed meV):")
for b in (0, 28, 29, 35, 39, 40, 63, 69):
    print(f"   band{b}: {rp[b]:+.2f} -> {rg[b]:+.2f}  (delta {rg[b]-rp[b]:+.2f})")
