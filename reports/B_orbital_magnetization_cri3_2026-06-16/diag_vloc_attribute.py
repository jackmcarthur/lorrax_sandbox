"""Attribute the residual to V_loc directly: compute <psi_v|V_loc_LX - V_loc_QE|psi_v>
per band at Gamma, using QE's actual 2D-treated V_loc (pp.x plot_num=2 = pp_2.cube),
and compare to the known residual pattern (band 28 -74, 29 -60, VBM ~13). If
<psi|dV_loc|psi> reproduces that pattern, the 2D V_loc treatment is the bug."""
import os, sys, numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.ionic_gspace import build_ionic_and_core
from psp.pseudos import load_pseudopotentials
RY=13.605693122994
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf"
OUT=os.path.dirname(os.path.realpath(__file__))

def read_cube(path,nx,ny,nz,nat=8):
    vals=[];
    for ln in open(path).read().split("\n")[6+nat:]: vals.extend(ln.split())
    return np.array(vals[:nx*ny*nz],float).reshape(nx,ny,nz)

wfn=WfnLoader(f"{RUN}/WFN.h5"); sym=symmetry_maps.SymMaps(wfn); nocc=int(wfn.nelec)
meta=Meta.from_system(wfn,sym,nocc,0,nocc,0,False)
pseudos=load_pseudopotentials(RUN)
fg=wfn.fft_grid; nx,ny,nz=int(fg[0]),int(fg[1]),int(fg[2])
V_loc_lx,_,_=build_ionic_and_core(wfn,pseudos,fg,truncation_2d=True)
V_loc_lx=np.asarray(V_loc_lx)
V_loc_qe=read_cube(f"{OUT}/pp_2.cube",nx,ny,nz)
dV=V_loc_lx - V_loc_qe
dV=dV - dV.mean()                       # remove constant (gauge); keep the shape
print(f"dV_loc (LX-QE, mean-removed): rms={dV.std()*RY*1000:.1f} meV  max|={np.abs(dV).max()*RY*1000:.0f} meV")
# decompose the cube diff: is it nucleus-localized (cusp) or in-plane/smooth (2D)?
# nucleus voxels = where |V_loc_qe| is most negative (deepest)
deep = V_loc_qe < np.percentile(V_loc_qe, 0.5)
print(f"   rms in deep-nucleus voxels: {dV[deep].std()*RY*1000:.0f} meV ; rms elsewhere: {dV[~deep].std()*RY*1000:.1f} meV")

ik=[i for i in range(int(sym.nk_tot)) if np.allclose(np.asarray(sym.unfolded_kpts[i]),[0,0,0],atol=1e-6)][0]
box=load_kpoint_fftbox(wfn,sym,meta,ik,nocc)
psi_r=np.asarray(jnp.fft.ifftn(box,axes=(-3,-2,-1),norm='ortho'))   # (nb,2,nx,ny,nz), <psi|psi>=1
dVj=jnp.asarray(dV)
rho_v = np.sum(np.abs(psi_r)**2, axis=1)        # (nb,nx,ny,nz) per-band density, sums to 1
contrib = np.array([float(np.sum(rho_v[b]*dV)) for b in range(nocc)])*RY*1000  # <psi|dV_loc|psi> meV
print("\n<psi_v|dV_loc|psi_v> (meV) vs residual pattern:")
resid_pat={28:-73.8,29:-60.0,35:-41.2,39:-40.6,40:-41.4,63:+50.2,nocc-1:+12.9,0:+1.9}
for b in sorted(resid_pat):
    print(f"   band {b:2d}: <dV_loc> = {contrib[b]:+7.1f}    (full residual {resid_pat[b]:+.1f})")
print(f"   corr(<dV_loc>, residual) over those bands: ",
      np.corrcoef([contrib[b] for b in sorted(resid_pat)],[resid_pat[b] for b in sorted(resid_pat)])[0,1].round(3))
