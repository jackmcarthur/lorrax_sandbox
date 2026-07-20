"""DECISIVE reconciliation: is nband=80 an f-transform tuning artifact or a real
640-centroid capacity/conditioning wall for the top conduction bands?

(1) ctilde orthonormality: the fH eigenvalue->energy map assumes ctilde rows are
    orthonormal in the alpha-metric (fH=sum_n f(eps_n) c_n c_n^H -> eigvals=f).
    Measure ||C_k C_k^H - I|| per band -> where does it break?
(2) Truncation sweep: build fH over the FIRST nb_fH bands of the SAME 80-band
    alpha-basis, evaluate on the 12x12 grid, and measure the on-grid round-trip
    error of the 8v8c conduction bands 26..33 vs the exact WFN energies.  This
    isolates the effect of packing more (higher, more oscillatory) bands into the
    fH sum, at fixed centroid count.  Answers: how many bands can 640 centroids
    carry before the conduction energies break?
"""
import numpy as np, jax
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
ct = np.asarray(jax.device_get(ctilde))   # (nk, nb, rank)
enk_np = np.asarray(enk)                   # (nb, nk)
nk, nb, rank = ct.shape
print(f"ctilde {ct.shape}  enk {enk_np.shape}  rank(alpha)={rank}  ns*n_mu=1280")

# (1) orthonormality of ctilde rows per k: ||C_k C_k^H - I||, and per-band diag
diag_err = np.zeros(nb); offd_max = 0.0
for k in range(nk):
    G = ct[k] @ ct[k].conj().T            # (nb, nb)
    diag_err += np.abs(np.diag(G).real - 1.0)
    offd_max = max(offd_max, float(np.max(np.abs(G - np.diag(np.diag(G))))))
diag_err /= nk
print(f"\n[orthonormality] mean_k |G_kk - 1| per band group, max offdiag={offd_max:.3e}")
for b0 in range(0, nb, 8):
    print("  b%2d-%2d: "%(b0,min(b0+7,nb-1))+" ".join(f"{diag_err[b]:.3e}" for b in range(b0,min(b0+8,nb))))

# (2) truncation sweep at fixed 80-band alpha-basis
k_frac = np.stack(np.meshgrid(np.arange(nkx)/nkx, np.arange(nky)/nky, [0.0], indexing="ij"),axis=-1).reshape(-1,3)
enk_bk = enk_np.T                          # (nk, nb)
print("\n[truncation] fH over first nb_fH bands (80-band alpha-basis); "
      "on-grid |E_interp - E_exact| max over k, conduction bands 26..33 (meV):")
print(f"{'nb_fH':>6} {'a_band':>7} {'shift(Ry)':>10} {'cond26_33':>10} {'val_maxerr':>10} {'per-cond-band 26..33 (meV)'}")
for nb_fH in [34, 40, 48, 56, 64, 72, 80]:
    ct_t = jax.numpy.asarray(ct[:, :nb_fH, :])
    en_t = jax.numpy.asarray(enk_np[:nb_fH, :])
    for a_band in (min(33, nb_fH-1), None):
        bnd = compute_wfns_fi(ctilde=ct_t, B_at_mu=B, enk_sigma=en_t, kgrid_co=kg,
                              band_window_fi=(0, nb_fH), mesh_xy=mesh_xy, q_list=k_frac,
                              a_band_index=a_band, log_fn=(lambda *a,**k:None))
        E = np.asarray(jax.device_get(bnd.enk_full))     # (nk, nb_fH) Ry
        from bandstructure.htransform import f_transform_eigs
        _f,_a,_n,shift = f_transform_eigs(en_t, a_band_index=a_band)
        dcond = np.abs(E[:, 26:34] - enk_bk[:, 26:34])*RY*1e3   # (nk, 8) meV
        dval  = np.abs(E[:, 18:26] - enk_bk[:, 18:26])*RY*1e3
        perb = dcond.max(axis=0)
        ab = "top" if a_band is None else str(a_band)
        print(f"{nb_fH:6d} {ab:>7} {shift:10.4f} {dcond.max():10.2f} {dval.max():10.2f}   "
              + " ".join(f"{v:6.1f}" for v in perb))
