"""Per-band sig_c at Γ for 01_lorrax_bse_vcoul."""
import h5py, numpy as np, sys, re
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul')
ryd2ev = 13.6056980659
with h5py.File(base/'sigma_mnk_ppm.h5','r') as f:
    omega_ev=f['omega_ev'][...]; sigma_c_omega=f['sigma_c_kij_ev'][...]
ol,oh=float(omega_ev.min()),float(omega_ev.max())
with h5py.File(base/'WFN.h5','r') as f:
    el_ry=f['mf_header/kpoints/el'][0]; nelec=int(f['mf_header/kpoints/ifmax'][0,0])
el_ev=el_ry*ryd2ev
vbm=max(np.max(el_ev[k,:nelec]) for k in range(el_ev.shape[0]))
cbm=min(np.min(el_ev[k,nelec:]) for k in range(el_ev.shape[0]))
ef=0.5*(vbm+cbm)

sys.path.insert(0,'/global/u2/j/jackm/software/lorrax_A/src')
from file_io import WFNReader
from common import symmetry_maps
w=WFNReader(str(base/'WFN.h5')); sym=symmetry_maps.SymMaps(w)
unfolded=np.asarray(sym.unfolded_kpts); wfn_kpts=np.asarray(w.kpoints)

def find_kfull(kbgw,tol=1e-4):
    target=np.array(kbgw)
    for i,k in enumerate(unfolded):
        d=(k-target+0.5)%1.0-0.5
        if np.max(np.abs(d))<tol: return i
    return None
def irr_idx(kf):
    k=unfolded[kf]
    for i,kk in enumerate(wfn_kpts):
        d=(kk-k+0.5)%1.0-0.5
        if np.max(np.abs(d))<1e-4: return i
    return 0

bgw_ppm={}; bgw_csx={}
for which,path in [('ppm','/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log'),('csx','/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/00_bgw_cohsex/sigma_hp.log')]:
    cur=None
    for raw in open(path):
        s=raw.strip()
        m=re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)',s)
        if m: cur=(float(m.group(1)),float(m.group(2)),float(m.group(3))); continue
        if cur is None: continue
        p=s.split()
        if len(p)>=15 and p[0].isdigit():
            n=int(p[0])
            (bgw_ppm if which=='ppm' else bgw_csx)[(cur,n-1)]=float(p[4])+float(p[10])

kf=find_kfull((0,0,0)); irrk=irr_idx(kf)
print(f'Γ (kfull={kf}, irrk={irrk})')
print(' n  E_DFT     ω_rel    LORRAX_sigc   BGW_PPM    BGW_CSX   Δ_PPM    Δ_CSX')
for n0 in range(min(16,sigma_c_omega.shape[2])):
    edft=el_ev[irrk,n0]; omr=edft-ef
    if not (ol<=omr<=oh):
        print(f' {n0+1:2d}  {edft:+8.3f}  {omr:+7.3f}   (out of grid; NaN)')
        continue
    sigc=float(np.interp(omr,omega_ev,sigma_c_omega[:,kf,n0,n0].real))
    sim=float(np.interp(omr,omega_ev,sigma_c_omega[:,kf,n0,n0].imag))
    bp=bgw_ppm.get(((0,0,0),n0),float('nan'))
    bc=bgw_csx.get(((0,0,0),n0),float('nan'))
    print(f' {n0+1:2d}  {edft:+8.3f}  {omr:+7.3f}   {sigc:+8.3f}   {bp:+8.3f}  {bc:+8.3f}  {sigc-bp:+7.3f}  {sigc-bc:+7.3f}  Im={sim:+.1f}')
