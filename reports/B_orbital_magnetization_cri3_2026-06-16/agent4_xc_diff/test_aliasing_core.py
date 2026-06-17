"""Probe the residual that survives the segni fix. Two hypotheses:
 (4) ALIASING of (B.sigma)psi and V_scf*psi: products of sharp potentials with
     localized states have G-content beyond the wavefunction cutoff. QE evaluates
     V*psi on the DENSE (ecutrho) grid then keeps only |G|<ecutwfc components --
     same as LORRAX (FFT on full grid, gather sphere). BUT the spinor d-states at
     Cr are sharp. Test: how much does <v|H|v> change if we DOUBLE-pad the FFT grid
     before forming V*psi and B*psi (kills aliasing)?  If band residual drops ->
     aliasing. If unchanged -> not aliasing.
 (core) The spin split puts HALF the (nonmagnetic) NLCC core into each channel.
     The GGA flux of n_up=(n_val+|m|+n_core)/2 mixes core+|m| gradients. QE adds
     fac*rho_core to rhoaux (fac=1/2 for nspin0=2) -- SAME. Verify by checking the
     residual with vs without the precise-core G split already in the harness.

This isolates whether the surviving ~35-73 meV is an APPLICATION (grid) artifact.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64","1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0,"/global/u2/j/jackm/software")
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
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
WFN=sys.argv[1]; RY2EV=13.605693122994; trunc=True
wfn=WfnLoader(WFN); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup=vnl_ops.build_vnl_setup(wfn,pseudos=pseudos,nspinor=int(wfn.nspinor),q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
G_cart=build_G_cart(nx,ny,nz,float(wfn.blat)*np.asarray(wfn.bvec,float))
bdot=jnp.asarray(wfn.bdot); bvec=jnp.asarray(wfn.bvec)
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_x,m_y,m_z=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_loc,rho_core,rho_core_G=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=trunc)
V_H,_=compute_V_H_and_V_xc(rho_val,rho_core,rho_core_G,G_cart,bdot,bvec,wfn.blat,truncation_2d=trunc)
core_grid=jnp.real(jnp.fft.ifftn(rho_core_G)); n=rho_val+rho_core
amag=jnp.sqrt(m_x**2+m_y**2+m_z**2); seg=jnp.where(amag>1e-12,jnp.sign(-m_z),1.0)
n_up=(n+seg*amag)/2; n_dn=(n-seg*amag)/2
rhoG_up=jnp.fft.fftn(n_up-core_grid/2)+rho_core_G/2; rhoG_dn=jnp.fft.fftn(n_dn-core_grid/2)+rho_core_G/2
Vu,Vd=compute_V_xc_spin(n_up,n_dn,rhoG_up,rhoG_dn,G_cart)
Vbar=0.5*(Vu+Vd); Bmag=0.5*(Vu-Vd); inv=jnp.where(amag>1e-12,1.0/(amag+1e-30),0.0)
B_vec=jnp.stack([seg*Bmag*m_x*inv,seg*Bmag*m_y*inv,seg*Bmag*m_z*inv],axis=0)
V_scf=build_V_scf(V_loc,V_H,Vbar)
ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))

# Decompose <v|H|v>-eps into the B.sigma part vs the rest, per band, to see WHERE
# the surviving residual lives (is it in B.sigma application, or V_scf, or vnl?)
ik=0
kv=np.asarray(sym.unfolded_kpts[ik],float); k_red=int(sym.irr_idx_k[ik])
eps=np.asarray(wfn.energies[0,k_red,:nocc],float)
box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)
U=_psi_box_to_G_sphere(box,Gk)[:nocc]*H_k.mask[None,None,:].astype(box.dtype)
nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
def diagval(Bv):
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,Bv)
    return np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm
d_on=diagval(B_vec); d_off=diagval(None)
r_on=(d_on-eps)*RY2EV*1000; r_off=(d_off-eps)*RY2EV*1000
B_contrib=(d_on-d_off)*RY2EV*1000   # <v|B.sigma|v> per band
print("per band: residual(B on) | residual(B off) | <v|B.sigma|v>")
for b in range(min(nocc,12)):
    print(f"  b{b:2d}: on={r_on[b]:7.2f} off={r_off[b]:7.2f} Bsig={B_contrib[b]:8.2f} meV")
print(f"...\n  VBM b{nocc-1}: on={r_on[-1]:.2f} off={r_off[-1]:.2f} Bsig={B_contrib[-1]:.2f}")
print("\n(B.sigma per-band decomposition above is the key diagnostic)")
