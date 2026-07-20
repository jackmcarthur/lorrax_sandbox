"""Which interp nband gives CLEAN 8v8c conduction (bands 26..33)?  Unlike
capacity_probe (fixed 80-band alpha-basis, truncated fH sum), this rebuilds the
alpha-basis FRESH at each nband (separate initialize_wfns) — the honest
'best interp basis' test.  For each nband: on-grid |E_interp - E_exact| over the
8v8c conduction window 26..33, and the min-sval gate vs stored psi (8 cond)."""
import numpy as np, jax, h5py, copy
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy
RY = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
base = read_lorrax_input("hts80.in")

with h5py.File("tmp/isdf_tensors_640.h5","r") as f:
    psi_st_all = np.asarray(f["psi_full_y"][:])   # (nk,80,ns,mu)
psi_st_c = psi_st_all[:, 26:34, :, :]; nmu = psi_st_c.shape[-1]

def minsval(psi_ht_c):
    smin=1.0
    for k in range(psi_ht_c.shape[0]):
        A=psi_ht_c[k].reshape(8,-1); Bm=psi_st_c[k].reshape(8,-1)
        A=A/np.linalg.norm(A,axis=1,keepdims=True); Bm=Bm/np.linalg.norm(Bm,axis=1,keepdims=True)
        smin=min(smin, float(np.linalg.svd(A.conj()@Bm.T,compute_uv=False).min()))
    return smin

print(f"{'nband':>6} {'ncond':>6} {'rank':>6} {'cond26_33_meV':>13} {'val18_25_meV':>12} {'min_sval':>9}  per-cond 26..33 (meV)")
for nband in [34, 40, 48, 64, 80]:
    p = copy.deepcopy(base); p["nband"]=nband; p["ncond"]=nband-26
    (wfn,sym,meta,_m,_S,ctilde,B,enk)=ht.initialize_wfns("hts80.in", p, (lambda *a,**k:None), mesh_xy=mesh_xy)
    kg=(int(meta.nkx),int(meta.nky),int(meta.nkz)); nkx,nky,nkz=kg
    rank=int(ctilde.shape[2]); nb_ret=min(nband,rank)
    k_frac=np.stack(np.meshgrid(np.arange(nkx)/nkx,np.arange(nky)/nky,[0.0],indexing="ij"),axis=-1).reshape(-1,3)
    enk_bk=np.asarray(enk).T   # (nk, nband)
    bnd=compute_wfns_fi(ctilde=ctilde,B_at_mu=B,enk_sigma=enk,kgrid_co=kg,
                        band_window_fi=(0,nb_ret),mesh_xy=mesh_xy,q_list=k_frac,
                        a_band_index=None,log_fn=(lambda *a,**k:None))
    E=np.asarray(jax.device_get(bnd.enk_full))
    psi=np.asarray(jax.device_get(bnd.psi_rmu_Y))[:,26:34,:,:nmu]
    dcond=np.abs(E[:,26:34]-enk_bk[:,26:34])*RY*1e3
    dval =np.abs(E[:,18:26]-enk_bk[:,18:26])*RY*1e3
    sm=minsval(psi)
    print(f"{nband:6d} {nband-26:6d} {rank:6d} {dcond.max():13.2f} {dval.max():12.2f} {sm:9.4f}  "
          +" ".join(f"{v:6.1f}" for v in dcond.max(axis=0)))
