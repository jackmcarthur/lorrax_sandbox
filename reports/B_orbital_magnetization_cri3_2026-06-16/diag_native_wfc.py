"""Apply the verified H to QE's NATIVE spinors (CrI3.save/wfc1.hdf5) at Gamma,
vs the BGW-exported WFN.h5. Same H (segni-fixed V_scf+B_vec+V_NL), same eps
(QE native, from XML). If native << BGW(22 meV) -> the BGW export re-gauges the
spinors and the residual is an export artifact (H vindicated)."""
import os, sys, numpy as np, h5py, xml.etree.ElementTree as ET
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax
from psp.scf_potential import (build_rho_val_from_wfn, build_magnetization_from_wfn,
                               build_dft_potentials)
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
RY=13.605693122994; HA2RY=2.0
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf"
wfn=WfnLoader(f"{RUN}/WFN.h5"); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(RUN)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
rho_val=build_rho_val_from_wfn(wfn,sym,meta,nocc,verbose=False)
m_vec=build_magnetization_from_wfn(wfn,sym,meta,nocc,verbose=False)
V_scf,V_loc,vnl_setup,B_vec=build_dft_potentials(wfn,pseudos,rho_val,truncation_2d=True,m_vec=m_vec,verbose=False)
ngkmax=int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),float(wfn.ecutwfc),tuple(int(x) for x in fg)))

# Gamma index in the unfolded BGW list
ik=[i for i in range(int(sym.nk_tot)) if np.allclose(np.asarray(sym.unfolded_kpts[i]),[0,0,0],atol=1e-6)][0]
kv=np.asarray(sym.unfolded_kpts[ik],float)
H_k=setup_H_k_from_kvec(kv,V_scf,vnl_setup,wfn,meta,V_loc_r=V_loc,ngkmax=ngkmax)
Gk=jnp.stack([H_k.Gx,H_k.Gy,H_k.Gz],axis=-1).astype(jnp.int32)

# eps (QE native, Gamma) in Ry
root=ET.parse(f"{RUN}/CrI3.save/data-file-schema.xml").getroot()
for kse in root.find('.//band_structure').findall('ks_energies'):
    if np.allclose([float(x) for x in kse.find('k_point').text.split()],[0,0,0],atol=1e-6):
        eps=np.array([float(x) for x in kse.find('eigenvalues').text.split()])[:nocc]*HA2RY; break

def resid(U,label):
    U=U[:nocc]*H_k.mask[None,None,:].astype(U.dtype)
    HU=apply_H_k_from_G(U,H_k.T_diag,H_k.V_scf,H_k.Gx,H_k.Gy,H_k.Gz,H_k.vnl_Z,H_k.vnl_E,H_k.mask,B_vec)
    nrm=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U).real)
    diag=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),HU).real)/nrm
    r=np.abs(diag-eps)*RY*1000
    mz=np.asarray(jnp.einsum('vsG,vsG->v',jnp.conj(U),U*jnp.array([1.,-1.])[None,:,None]).real)/nrm
    print(f"  [{label}] max={r.max():.1f} mean={r.mean():.1f} band0={r[0]:.1f} VBM={r[nocc-1]:.1f} meV | sum<sz>={mz.sum():+.2f}")

# BGW WFN spinors at Gamma
boxB=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
UB=_psi_box_to_G_sphere(boxB,Gk)
print("BGW WFN.h5 spinors:"); resid(UB,"BGW")

# QE native spinors at Gamma: wfc1.hdf5
f=h5py.File(f"{RUN}/CrI3.save/wfc1.hdf5","r")
mill=np.asarray(f["MillerIndices"]); igwx=mill.shape[0]
evc=np.asarray(f["evc"]).view(np.complex128)          # (nbnd, 2*igwx)
nb=evc.shape[0]
box=np.zeros((nocc,2,nx,ny,nz),complex)
ix,iy,iz=mill[:,0]%nx,mill[:,1]%ny,mill[:,2]%nz
for b in range(nocc):
    box[b,0,ix,iy,iz]=evc[b,:igwx]      # up
    box[b,1,ix,iy,iz]=evc[b,igwx:]      # dn
UN=_psi_box_to_G_sphere(jnp.asarray(box),Gk)
print("QE native wfc1.hdf5 spinors:"); resid(UN,"native")
print("(if native << BGW -> BGW export re-gauges spinors; H is fine)")
