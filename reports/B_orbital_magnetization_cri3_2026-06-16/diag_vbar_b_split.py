"""Split the band-dependent residual into charge V̄ vs field B·σ, vs QE.
QE references (pw2bgw): kih.dat = <T+Vloc+VNL+VH>; vxc.dat = <V̄> (charge only,
B dropped for noncollinear). So:
  QE <V̄>     = vxc_q
  QE <B.sigma>= eps - kih_q - vxc_q   (what's left in the eigenvalue)
LORRAX (segni-fixed build_dft_potentials):
  <V̄>     = <H[V_scf,B=None]> - <H[Vloc+VH,B=None]>
  <V̄+B>   = <H[V_scf,B=B_vec]> - <H[Vloc+VH,B=None]>
  <B.sigma>= <V̄+B> - <V̄>
Whichever (V̄ or B) is band-dependent vs QE is the bug."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import compute_V_H_and_V_xc
from psp.scf_potential import (build_rho_val_from_wfn, build_magnetization_from_wfn,
                               build_dft_potentials)
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
RY2EV = 13.605693122994; trunc = True

def parse_bgw(path):
    out = {}; toks = open(path).read().split("\n"); i = 0
    while i < len(toks):
        t = toks[i].split()
        if len(t) == 5:
            k = tuple(round(float(x), 5) for x in t[:3]); nd = int(t[3])
            vals = np.zeros(nd, complex)
            for j in range(nd):
                d = toks[i+1+j].split(); vals[int(d[1])-1] = float(d[2])+1j*float(d[3])
            out[k] = vals; i += 1+nd
        else: i += 1
    return out

WFN = sys.argv[1]; ND = os.path.dirname(os.path.realpath(WFN))
vxc_qe = parse_bgw(f"{ND}/vxc.dat"); kih_qe = parse_bgw(f"{ND}/kih.dat")
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(ND)
fg = wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart = build_G_cart(nx,ny,nz, float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_vec = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
# segni-fixed V_scf (incl V̄) + B_vec via the production path
V_scf, V_loc, vnl_setup, B_vec = build_dft_potentials(wfn, pseudos, rho_val,
                                  truncation_2d=trunc, m_vec=m_vec, verbose=False)
# V_loc+V_H only (no V_xc): rebuild V_H
_, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H,_ = compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
V_noxc = build_V_scf(V_loc, V_H)
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                            float(wfn.ecutwfc),tuple(int(x) for x in fg)))

def diag_of(V_scf_arg, Bv, ik):
    kv=np.asarray(sym.unfolded_kpts[ik],float)
    box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
    H_k=setup_H_k_from_kvec(kv,V_scf_arg,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
    Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
    U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,Bv)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm*RY2EV

for ik in (0,4):
    k = tuple(round(float(x),5) for x in sym.unfolded_kpts[ik])
    if k not in vxc_qe: k = min(vxc_qe, key=lambda q: sum((a-b)**2 for a,b in zip(q,k)))
    kr=int(sym.irr_idx_k[ik]); eps=np.asarray(wfn.energies[0,kr,:nocc],float)*RY2EV
    vxc_q=vxc_qe[k][:nocc].real; kih_q=kih_qe[k][:nocc].real
    full=diag_of(V_scf,B_vec,ik); vbar=diag_of(V_scf,None,ik); noxc=diag_of(V_noxc,None,ik)
    Vbar_lx = vbar - noxc; B_lx = full - vbar; kih_lx = noxc
    dVbar = (Vbar_lx - vxc_q)*1000
    dB    = (B_lx - (eps - kih_q - vxc_q))*1000
    dkih  = (kih_lx - kih_q)*1000
    print(f"k={ik} {k}:")
    print(f"   d<Vbar>  (charge V_xc): mean {dVbar.mean():+.1f} std {dVbar.std():.1f} max|{np.abs(dVbar).max():.1f}|")
    print(f"   d<B.sig> (field)      : mean {dB.mean():+.1f} std {dB.std():.1f} max|{np.abs(dB).max():.1f}|")
    print(f"   d<kih>   (T+Vl+VH+VNL): mean {dkih.mean():+.1f} std {dkih.std():.1f} max|{np.abs(dkih).max():.1f}|")
    tot = dVbar+dB+dkih
    print(f"   -> total: mean {tot.mean():+.1f} std {tot.std():.1f}  (worst-band dVbar {dVbar[np.argmax(np.abs(tot))]:+.0f} dB {dB[np.argmax(np.abs(tot))]:+.0f} dkih {dkih[np.argmax(np.abs(tot))]:+.0f})")
