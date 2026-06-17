"""Split the spin V_xc into LDA-level vs GGA-gradient parts and measure each
one's per-band contribution to <psi|V_xc|psi>, vs the full residual. Tells us
whether the ~22 meV (worst on Cr semicore band 28) lives in the GGA-gradient
spin terms (antisymmetric exchange flux + phi(zeta) corr gradient) or the
LDA-level spin V_xc / B assembly. (charge V_bar kept at PBE throughout.)"""
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
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin, pbe_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
RY2EV = 13.605693122994; trunc = True

WFN = sys.argv[1]
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart = build_G_cart(nx,ny,nz, float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_x,m_y,m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc,rho_core,rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H,_ = compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)

amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)
M_net = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()]); u_ax = M_net/(jnp.linalg.norm(M_net)+1e-30)
segni = jnp.where(amag>1e-12, jnp.sign(m_x*u_ax[0]+m_y*u_ax[1]+m_z*u_ax[2]), 1.0)
n_tot = rho_val + rho_core
n_up,n_dn = (n_tot+segni*amag)/2, (n_tot-segni*amag)/2
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
rgu = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
rgd = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2

# PBE (full GGA) spin V_xc
V_up,V_dn = compute_V_xc_spin(n_up,n_dn,rgu,rgd,G_cart)
# LDA-level spin V_xc (sigma=0): pure local autodiff of E_lda
nu=jnp.maximum(n_up,1e-10); nd=jnp.maximum(n_dn,1e-10)
def E_lda(u,d):
    z=jnp.zeros_like(u); return jnp.sum((u+d)*pbe_xc_spin(u,d,z,z,z))
V_up_lda=jax.grad(E_lda,0)(nu,nd); V_dn_lda=jax.grad(E_lda,1)(nu,nd)

def fields(Vu,Vd):
    Vbar=0.5*(Vu+Vd); Bmag=0.5*segni*(Vu-Vd)
    inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
    Bvec=jnp.stack([Bmag*m_x*inv,Bmag*m_y*inv,Bmag*m_z*inv],axis=0)
    return Vbar,Bvec
Vbar_p,Bv_p = fields(V_up,V_dn)       # PBE
Vbar_l,Bv_l = fields(V_up_lda,V_dn_lda)  # LDA

ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                          float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def resid(Vbar,Bvec,ik):
    kv=np.asarray(sym.unfolded_kpts[ik],float); kr=int(sym.irr_idx_k[ik])
    eps=np.asarray(wfn.energies[0,kr,:nocc],float)
    box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
    V_scf=build_V_scf(V_loc,V_H,Vbar)
    H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
    Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
    U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,Bvec)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
    return (np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm - eps)*RY2EV*1000

ik=0
r_full = resid(Vbar_p, Bv_p, ik)     # full PBE spin (= the 22 meV)
r_Blda = resid(Vbar_p, Bv_l, ik)     # GGA killed in the FIELD only
r_Vlda = resid(Vbar_l, Bv_p, ik)     # GGA killed in V_bar only
worst = [28,29,35,39,40,63,nocc-1]
print(f"k={ik}: full PBE-spin: max={np.abs(r_full).max():.1f} mean={np.abs(r_full).mean():.1f}")
print(f"   GGA-in-B  killed: max={np.abs(r_Blda).max():.1f} mean={np.abs(r_Blda).mean():.1f}  (shift vs full: rms {np.sqrt(((r_full-r_Blda)**2).mean()):.1f})")
print(f"   GGA-in-Vbar killed: max={np.abs(r_Vlda).max():.1f} mean={np.abs(r_Vlda).mean():.1f}  (shift vs full: rms {np.sqrt(((r_full-r_Vlda)**2).mean()):.1f})")
print("   per-band (meV):  full | GGA-B-contrib(full-Blda) | GGA-Vbar-contrib(full-Vlda)")
for b in worst:
    print(f"     band {b:2d}: {r_full[b]:+7.1f} | {r_full[b]-r_Blda[b]:+7.1f} | {r_full[b]-r_Vlda[b]:+7.1f}")
