"""SP htransform reconciliation figure (the owner deliverable):
  Panel A: SP conduction bands along Gamma-M-K-Gamma, nband=40 interp (clean)
           vs nband=80 interp (broken) — shows the 80-band basis rings/shifts
           the conduction bands by ~eV even where on-grid points are exact.
  Panel B: on-grid |E_interp - E_exact| for the 8v8c conduction window 26..33
           vs interp nband (fresh alpha-basis each) — the cliff at 80.
Self-contained: fresh initialize_wfns per nband; saves sp_reconcile.npz."""
import numpy as np, jax, copy
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy
RY = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
base = read_lorrax_input("hts80.in")
wfn0,_s0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac,x_path,node_idx,node_labels,_ = ht.initialize_kpath(wfn0, base)
kpath=np.asarray(kpath_frac); x_path=np.asarray(x_path); node_idx=[int(n) for n in node_idx]
nQ=kpath.shape[0]

NBANDS=[34,40,48,64,80]
ongrid_err={}; path_bands={}
for nband in NBANDS:
    p=copy.deepcopy(base); p["nband"]=nband; p["ncond"]=nband-26
    (wfn,sym,meta,_m,_S,ctilde,B,enk)=ht.initialize_wfns("hts80.in",p,(lambda *a,**k:None),mesh_xy=mesh_xy)
    kg=(int(meta.nkx),int(meta.nky),int(meta.nkz)); nkx,nky,nkz=kg
    rank=int(ctilde.shape[2]); nb_ret=min(nband,rank)
    k_frac=np.stack(np.meshgrid(np.arange(nkx)/nkx,np.arange(nky)/nky,[0.0],indexing="ij"),axis=-1).reshape(-1,3)
    enk_bk=np.asarray(enk).T
    # on-grid error
    bg=compute_wfns_fi(ctilde=ctilde,B_at_mu=B,enk_sigma=enk,kgrid_co=kg,band_window_fi=(0,nb_ret),
                       mesh_xy=mesh_xy,q_list=k_frac,a_band_index=None,log_fn=(lambda *a,**k:None))
    Eg=np.asarray(jax.device_get(bg.enk_full))
    ongrid_err[nband]=float(np.abs(Eg[:,26:34]-enk_bk[:,26:34]).max())*RY*1e3
    # path bands (only for 40 and 80, the contrast)
    if nband in (40,80):
        bp=compute_wfns_fi(ctilde=ctilde,B_at_mu=B,enk_sigma=enk,kgrid_co=kg,band_window_fi=(0,nb_ret),
                           mesh_xy=mesh_xy,q_list=kpath,a_band_index=None,log_fn=(lambda *a,**k:None))
        path_bands[nband]=np.asarray(jax.device_get(bp.enk_full))*RY
    print(f"nband={nband}: on-grid cond26_33 = {ongrid_err[nband]:.2f} meV", flush=True)

np.savez("sp_reconcile.npz", x_path=x_path, node_idx=node_idx,
         nbands=np.array(NBANDS), ongrid=np.array([ongrid_err[n] for n in NBANDS]),
         path40=path_bands[40], path80=path_bands[80])

# reference VBM from the clean (nband=40) run
vbm40=path_bands[40][:, :26].max()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.5,6.0),dpi=150,gridspec_kw={"width_ratios":[1.5,1]})
# Panel A: conduction bands 26..33 along path, clean(40) vs broken(80)
for b in range(26,34):
    a1.plot(x_path, path_bands[40][:,b]-vbm40, color="#2c6e8f", lw=1.4,
            label="nband=40 interp (clean, 1.0 meV on-grid)" if b==26 else None)
    a1.plot(x_path, path_bands[80][:,b]-vbm40, color="#b5432c", lw=1.2, ls="--",
            label="nband=80 interp (broken, 955 meV on-grid)" if b==26 else None)
for n in node_idx: a1.axvline(x_path[n],color="0.8",lw=.6,zorder=0)
a1.set_xticks([x_path[n] for n in node_idx]); a1.set_xticklabels(["Γ","M","K","Γ"])
a1.set_xlim(x_path[0],x_path[-1]); a1.set_ylabel("E − E$_{VBM}$ (eV)")
a1.set_title("MoS₂ SP conduction bands (8v8c window 26–33): nband=40 vs nband=80 interp")
a1.legend(loc="upper right",fontsize=8)
# Panel B: on-grid conduction error vs interp nband (the cliff)
a2.semilogy(NBANDS,[ongrid_err[n] for n in NBANDS],"o-",color="#333",lw=1.6,ms=7)
a2.axhline(50,ls=":",color="#999",lw=1); a2.text(35,60,"~50 meV usable ceiling",fontsize=8,color="#666")
for n in NBANDS:
    a2.annotate(f"{ongrid_err[n]:.2f}" if ongrid_err[n]<100 else f"{ongrid_err[n]:.0f}",
                (n,ongrid_err[n]),textcoords="offset points",xytext=(4,6),fontsize=8)
a2.set_xlabel("interp nband (fresh α-basis; 640 centroids)")
a2.set_ylabel("on-grid |E$_{interp}$−E$_{exact}$| cond 26–33 (meV)")
a2.set_title("640-centroid capacity: conduction energy vs interp band count")
a2.grid(True,which="both",alpha=0.3)
fig.tight_layout(); fig.savefig("sp_reconcile_80_vs_40.png")
print("SAVED sp_reconcile_80_vs_40.png",flush=True)
