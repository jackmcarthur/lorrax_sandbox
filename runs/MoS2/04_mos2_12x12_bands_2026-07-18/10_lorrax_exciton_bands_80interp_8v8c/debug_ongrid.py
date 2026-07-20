"""Disambiguate the ~1000 meV on-grid conduction error: compare compute_wfns_fi
energies at grid k against (a) enk = initialize_wfns build input (the true
round-trip reference) and (b) restart enk_full.  Print actual values."""
import numpy as np, jax, h5py
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy
RY = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input("hts80.in")
(wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns("hts80.in", params, (lambda *a,**k:None), mesh_xy=mesh_xy)
kg=(int(meta.nkx),int(meta.nky),int(meta.nkz)); nkx,nky,nkz=kg
rank=int(ctilde.shape[2]); nb_ret=min(int(ctilde.shape[1]),rank)
print(f"kgrid={kg} rank={rank} nb_ret={nb_ret}  enk(build) shape={tuple(enk.shape)}")
# enk is (nb, nk) per initialize_wfns (transpose(1,0)); confirm
enk_np = np.asarray(enk)   # (nb, nk)
with h5py.File("tmp/isdf_tensors_640.h5","r") as f:
    enk_st = np.asarray(f["enk_full"][:])   # (nk, nb)
print(f"enk(build)={enk_np.shape}  enk_st(restart)={enk_st.shape}")

k_frac = np.stack(np.meshgrid(np.arange(nkx)/nkx, np.arange(nky)/nky, [0.0], indexing="ij"),axis=-1).reshape(-1,3)
nk=k_frac.shape[0]
bnd = compute_wfns_fi(ctilde=ctilde,B_at_mu=B,enk_sigma=enk,kgrid_co=kg,
                      band_window_fi=(0,nb_ret),mesh_xy=mesh_xy,q_list=k_frac,
                      a_band_index=34,log_fn=(lambda *a,**k:None))
E=np.asarray(jax.device_get(bnd.enk_full))   # (nk, nb_ret) Ry, ascending
# build reference in (nk, nb): enk_np.T
enk_bk = enk_np.T   # (nk, nb)

# which grid index is k=(0,0,0)?
j0=int(np.argmin(np.linalg.norm(k_frac,axis=1)))
print(f"\nk=Gamma grid idx j0={j0}  k_frac[j0]={k_frac[j0]}")
print("band |  E_interp   enk_build   enk_restart   (Ry)")
for b in list(range(24,36)):
    print(f"{b:4d} | {E[j0,b]:10.5f} {enk_bk[j0,b]:10.5f} {enk_st[j0,b]:12.5f}")

# errors vs each reference, conduction 26..34
dE_build = np.abs(E - enk_bk[:, :nb_ret])
dE_rest  = np.abs(E - enk_st[:, :nb_ret])
print(f"\nvs BUILD enk: cond(26:34) max={dE_build[:,26:34].max()*RY*1e3:.3f} meV  "
      f"all max={dE_build.max()*RY*1e3:.2f} meV  argmax band={np.unravel_index(dE_build.argmax(),dE_build.shape)[1]}")
print(f"vs RESTART enk: cond(26:34) max={dE_rest[:,26:34].max()*RY*1e3:.3f} meV  "
      f"all max={dE_rest.max()*RY*1e3:.2f} meV")
# is enk_build vs enk_restart itself different?
print(f"enk_build vs enk_restart: cond(26:34) max={np.abs(enk_bk-enk_st)[:,26:34].max()*RY*1e3:.3f} meV  "
      f"all max={np.abs(enk_bk-enk_st).max()*RY*1e3:.2f} meV")
# per-band on-grid error vs BUILD (the true round-trip), to see WHICH bands break
print("\nper-band on-grid |E_interp-enk_build| max over k (meV):")
pb = dE_build.max(axis=0)*RY*1e3
for b in range(0,nb_ret,4):
    print("  b%2d-%2d: "%(b,min(b+3,nb_ret-1))+" ".join(f"{pb[bb]:8.2f}" for bb in range(b,min(b+4,nb_ret))))
