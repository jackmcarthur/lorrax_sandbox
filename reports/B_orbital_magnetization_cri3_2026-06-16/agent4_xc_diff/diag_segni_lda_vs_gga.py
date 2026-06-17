"""DECISIVE: QE applies segni ONLY to the GGA gradient correction, NOT to the
LDA part of the noncollinear V_xc.

Source evidence:
  - xc_wrapper_lda_lsda.f90 CASE(4): LDA spin densities = (rho +/- |amag|)/2
    (POSITIVE amag, NO segni).  v_of_rho.f90:537-539: v(:,2:4)=vs*m_a/amag along
    +m_hat (vs from positive zeta).  => LDA field has NO segni.
  - gradcorr.f90:86 compute_rho builds n_up/dn with segni; :233-235 the GGA field
    v(k,2:4) += segni*0.5*(vgg_up-vgg_dn)*m_a/amag.  => GGA field HAS segni.

LORRAX currently applies segni to the WHOLE compute_V_xc_spin (LDA-fallback AND
GGA together).  Where segni=-1 (core polarization near Cr: 3p/3s semicore have
spin opposite the net 3d moment), LORRAX swaps up<->dn for the LDA part too,
which QE does NOT.  Test the four combinations of (LDA segni) x (GGA segni).

We split compute_V_xc_spin into its LDA piece (df_du/df_dd, no -div) and its GGA
flux piece (the -div term), then assemble the field with segni applied to each
piece independently.  Measured by the full <v|H|v>-eps.
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
from psp.xc import _compute_sigma_spin, pbe_xc_spin
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
inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)

# fraction of magnetized volume with segni=-1 (sanity: is core polarization real?)
neg_frac = float(jnp.sum((seg < 0) & (amag > 1e-6))) / float(jnp.sum(amag > 1e-6))
print(f"segni=-1 fraction of magnetized cells: {neg_frac*100:.2f}%")
print(f"max |amag| where segni=-1: {float(jnp.max(jnp.where(seg<0, amag, 0.0))):.4e}")

def vxc_pieces(s_lda, s_gga):
    """Return (Vbar, B_vec) with segni-sign s_lda applied to the LDA piece and
    s_gga to the GGA flux piece, each in {+1 array of segni, +1 ones}."""
    n_up = (n + s_lda*amag)/2; n_dn = (n - s_lda*amag)/2  # LDA spin densities
    nu = jnp.maximum(n_up, 1e-10); nd = jnp.maximum(n_dn, 1e-10)
    # --- LDA piece (no -div): df_du, df_dd at sigma=0 baseline + full df_du masked ---
    # Build gradient G-space for the GGA piece from the s_gga-projected densities.
    ngu = (n + s_gga*amag)/2; ngd = (n - s_gga*amag)/2
    rhoG_up = jnp.fft.fftn(ngu - core_grid/2) + rho_core_G/2
    rhoG_dn = jnp.fft.fftn(ngd - core_grid/2) + rho_core_G/2
    suu, sud, sdd = _compute_sigma_spin(rhoG_up, rhoG_dn, G_cart)

    # LDA potential uses the s_lda densities WITHOUT sigma (sigma=0):
    def E_lda(uu, dd):
        z = jnp.zeros_like(uu)
        return jnp.sum((uu+dd)*pbe_xc_spin(uu, dd, z, z, z))
    nu_l = jnp.maximum((n + s_lda*amag)/2, 1e-10)
    nd_l = jnp.maximum((n - s_lda*amag)/2, 1e-10)
    dfu_l = jax.grad(E_lda, 0)(nu_l, nd_l)
    dfd_l = jax.grad(E_lda, 1)(nu_l, nd_l)
    Vbar_lda = 0.5*(dfu_l + dfd_l)
    # LDA field along +m_hat, magnitude vs = 0.5*(Vup-Vdn) for s_lda densities,
    # then projected with s_lda (so the field direction = s_lda * m_hat * vs):
    Bmag_lda = 0.5*s_lda*(dfu_l - dfd_l)

    # --- GGA piece: full spin-GGA minus its LDA, on the s_gga densities ---
    nug = jnp.maximum(ngu, 1e-10); ndg = jnp.maximum(ngd, 1e-10)
    def E_full(uu, dd, a, b, c):
        return jnp.sum((uu+dd)*pbe_xc_spin(uu, dd, a, b, c))
    def E_gga_lda(uu, dd):
        z = jnp.zeros_like(uu)
        return jnp.sum((uu+dd)*pbe_xc_spin(uu, dd, z, z, z))
    dfu_f = jax.grad(E_full, 0)(nug, ndg, suu, sud, sdd)
    dfd_f = jax.grad(E_full, 1)(nug, ndg, suu, sud, sdd)
    dfu_gl = jax.grad(E_gga_lda, 0)(nug, ndg)
    dfd_gl = jax.grad(E_gga_lda, 1)(nug, ndg)
    duu = jax.grad(E_full, 2)(nug, ndg, suu, sud, sdd)
    dud = jax.grad(E_full, 3)(nug, ndg, suu, sud, sdd)
    ddd = jax.grad(E_full, 4)(nug, ndg, suu, sud, sdd)
    rho_raw = n; sigma_tot = suu + 2.0*sud + sdd
    mask = (rho_raw > 1e-6) & (sigma_tot > 1e-10)
    duu = jnp.where(mask, duu, 0.0); dud = jnp.where(mask, dud, 0.0); ddd = jnp.where(mask, ddd, 0.0)
    df_du_gga = jnp.where(mask, dfu_f - dfu_gl, 0.0)  # GGA-only v1 (full - lda)
    df_dd_gga = jnp.where(mask, dfd_f - dfd_gl, 0.0)
    gnu = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rhoG_up)) for i in range(3)]
    gnd = [jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*rhoG_dn)) for i in range(3)]
    def div(fields):
        out = jnp.zeros_like(n)
        for i in range(3):
            hG = jnp.fft.fftn(fields[i])
            out = out + jnp.real(jnp.fft.ifftn(1j*G_cart[..., i]*hG))
        return out
    Vu_gga = df_du_gga - div([2.0*duu*gnu[i] + dud*gnd[i] for i in range(3)])
    Vd_gga = df_dd_gga - div([2.0*ddd*gnd[i] + dud*gnu[i] for i in range(3)])
    Vbar_gga = 0.5*(Vu_gga + Vd_gga)
    Bmag_gga = 0.5*s_gga*(Vu_gga - Vd_gga)

    Vbar = Vbar_lda + Vbar_gga
    Bmag = Bmag_lda + Bmag_gga
    B_vec = jnp.stack([Bmag*m_x*inv, Bmag*m_y*inv, Bmag*m_z*inv], axis=0)
    return Vbar, B_vec

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))
ones = jnp.ones_like(seg)

def resid(Vbar_, B_, label, verbose_bands=False):
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
    print(f"  [{label}] RMS={np.sqrt((ra**2).mean()):.2f} max={ra.max():.2f} meV")
    if verbose_bands:
        for b in (0, 28, 29, 35, 39, 40, 63, 69):
            print(f"        band{b}: {pb0[b]:+.2f} meV")
    return pb0

print("\n=== seg_lda=seg, seg_gga=seg : current LORRAX (segni everywhere) ===")
resid(*vxc_pieces(seg, seg), "lda=S,gga=S", verbose_bands=True)
print("=== seg_lda=+1, seg_gga=seg : QE convention (LDA no segni, GGA segni) ===")
resid(*vxc_pieces(ones, seg), "lda=+,gga=S", verbose_bands=True)
print("=== seg_lda=+1, seg_gga=+1 : no segni anywhere ===")
resid(*vxc_pieces(ones, ones), "lda=+,gga=+", verbose_bands=True)
print("=== seg_lda=seg, seg_gga=+1 : segni only on LDA (sanity) ===")
resid(*vxc_pieces(seg, ones), "lda=S,gga=+")
