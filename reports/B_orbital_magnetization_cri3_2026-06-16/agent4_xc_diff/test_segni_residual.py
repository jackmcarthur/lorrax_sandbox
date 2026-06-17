"""DECISIVE: does QE's segni in BOTH the spin-density build AND the B-field fix
the ~15 meV band-dependent residual <v|H|v>-eps_v on CrI3?

Three configs, same k-points, measuring max/mean/band0/VBM residual:
  L0  : LORRAX-current. n_up=(n+|m|)/2, B=+Bmag*m/|m|.          (no segni)
  QE  : QE-correct.     n_up=(n+segni|m|)/2, B=segni*Bmag*m/|m|, segni=sign(m_z proj).
Both use the SAME compute_V_xc_spin (proven exact). The ONLY difference is segni.
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
fg = wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart = build_G_cart(nx,ny,nz, float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
m_x,m_y,m_z = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=True)
V_loc,rho_core,rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H,_ = compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
n = rho_val + rho_core
amag = jnp.sqrt(m_x**2+m_y**2+m_z**2)

# ux ~ majority axis. Net m_z<0 here, so ux=-z => segni=sign(-m_z)=sign(m.ux).
# QE picks ux from starting_magnetization of first magnetic atom; for CrI3 FM that
# is the Cr moment direction = the net direction = -z. segni = sign(m . ux).
ux = jnp.array([0.0,0.0,-1.0])   # majority direction (net m_z<0)
m_dot_ux = m_x*ux[0]+m_y*ux[1]+m_z*ux[2]
segni = jnp.where(amag>1e-12, jnp.sign(m_dot_ux), 1.0)

def build_B_and_Vbar(use_segni):
    s = segni if use_segni else jnp.ones_like(segni)
    n_up=(n+s*amag)/2; n_dn=(n-s*amag)/2
    rhoG_up = jnp.fft.fftn(n_up - core_grid/2) + rho_core_G/2
    rhoG_dn = jnp.fft.fftn(n_dn - core_grid/2) + rho_core_G/2
    V_up,V_dn = compute_V_xc_spin(n_up,n_dn,rhoG_up,rhoG_dn,G_cart)
    Vbar = 0.5*(V_up+V_dn); Bmag=0.5*(V_up-V_dn)
    inv = jnp.where(amag>1e-12, 1.0/(amag+1e-30), 0.0)
    # QE: v(2..4)=segni*Bmag*m_alpha/amag ; with n_up built using s, Bmag already
    # corresponds to the s-channel. The field along m: B_vec = s*Bmag*m/|m|.
    B_vec = jnp.stack([s*Bmag*m_x*inv, s*Bmag*m_y*inv, s*Bmag*m_z*inv],axis=0)
    return Vbar, B_vec

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                            float(wfn.ecutwfc),tuple(int(x) for x in fg)))
def resid(ik, Vbar, B_vec, label):
    kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
    eps=np.asarray(wfn.energies[0,k_red,:nocc],float)
    box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
    V_scf=build_V_scf(V_loc,V_H,Vbar)
    H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
    Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
    U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,B_vec)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
    diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm
    r=np.abs(diag-eps)*RY2EV*1000
    print(f"  [{label}] k={ik}: max={r.max():.2f} mean={r.mean():.2f} band0(deep)={r[0]:.2f} VBM={r[nocc-1]:.2f} meV")
    return r

Vbar0,B0 = build_B_and_Vbar(False)   # LORRAX current (no segni)
Vbar1,B1 = build_B_and_Vbar(True)    # QE-correct (segni)
print("\n=== L0: LORRAX current (+|m|, no segni) ===")
for ik in (0,4,8): resid(ik,Vbar0,B0,"L0")
print("=== QE: segni-correct (segni|m|, segni*B) ===")
for ik in (0,4,8): resid(ik,Vbar1,B1,"QE")
