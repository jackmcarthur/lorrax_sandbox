"""MoS2 12x12 nband=80 htransform reconciliation: sweep the f-transform
a_band_index and, for each setting, measure the THREE things that decide
whether the prior 'min-sval -> 0.2175 at nband=80' is real HARM or an
f-transform tuning artifact:

  (A) min-sval  : gauge-free conduction-subspace overlap min singular value of
                  <psi_ht | psi_stored> over the 8v8c BSE conduction window
                  (bands 26..33), replicating exciton_bands.gate_htransform_vs_stored
                  EXACTLY, but against the restart psi_full_y read directly.
  (B) on-grid   : |E_interp(k) - E_stored(k)| at the 144 coarse-grid k, per band
                  group (BSE conduction 26..33 ; high guards 34..79 ; all).  This
                  is the direct on-grid exactness check (the WFN/GW energies at
                  grid k).  ~0 => the basis DOES represent the band; large =>
                  genuine capacity failure for that band.
  (C) ringing   : max |2nd finite-difference| of the conduction bands 26..33
                  along the OFF-grid Gamma-M-K-Gamma path (node kinks excluded).

fH is built over ALL 80 bands (single-source with the SP path); band_window_fi
only selects what is returned.  Nothing here touches the exciton solver — it is
purely the htransform basis quality vs a_band_index.

Saves sp80_aband_sweep.npz (all metrics + E_path per a_band) BEFORE plotting.
"""
import os, sys, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import h5py
from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.htransform import f_transform_eigs
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY = 13.6056980659
INP = "hts80.in"
RESTART = "tmp/isdf_tensors_640.h5"
NVAL = 26                       # CBM = band index 26
BSE_C0, BSE_C1 = 26, 34         # 8v8c conduction window [26,34)
A_BANDS = [None, 28, 30, 33, 34, 36, 40, 54]   # None = default (top band 79)

mesh_xy = _create_mesh_xy(1, 1)
params = read_lorrax_input(INP)

t0 = time.time()
(wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(INP, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz))
nkx, nky, nkz = kg
rank = int(ctilde.shape[2])
nb_win = int(ctilde.shape[1])
nb_ret = min(nb_win, rank)      # bands we can return
print(f"[cfg] kgrid={kg} nband(window)={nb_win} rank(alpha)={rank} nb_ret={nb_ret}", flush=True)
print(f"[cfg] init {time.time()-t0:.1f}s  ns*n_mu = {int(B.shape[1])*int(B.shape[2])} (per-k capacity)", flush=True)
assert nb_ret >= BSE_C1, f"rank {rank} < BSE conduction top {BSE_C1}; cannot return 8v8c window"

# ── coarse k grid (driver 'ij' convention) + the 40-pt path ──────────────────
k_frac = np.stack(np.meshgrid(np.arange(nkx)/nkx, np.arange(nky)/nky,
                              np.arange(nkz)/nkz, indexing="ij"),
                  axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]
wfn0, _s0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac); x_path = np.asarray(x_path)
nQ = kpath.shape[0]
node_idx = [int(n) for n in node_idx]
print(f"[cfg] nk={nk} path nQ={nQ} nodes={node_idx}", flush=True)

# ── stored reference (paired psi + energies from the SAME restart) ───────────
with h5py.File(RESTART, "r") as f:
    psi_st_all = np.asarray(f["psi_full_y"][:])   # (nk, 80, ns, mu)
    enk_st_all = np.asarray(f["enk_full"][:])     # (nk, 80)  Ry
psi_st_c = psi_st_all[:, BSE_C0:BSE_C1, :, :]     # (nk, 8, ns, mu)  stored conduction
nmu = int(psi_st_c.shape[-1])
print(f"[ref] restart psi_full_y {psi_st_all.shape}  enk_full {enk_st_all.shape}", flush=True)

def minsval_gate(psi_ht_c, psi_st_c):
    """Exact replica of gate_htransform_vs_stored's subspace SVD (bands x bands,
    spin+mu contracted, rows normalized) -> min over k of min singular value."""
    smin = 1.0
    ncb = psi_ht_c.shape[1]
    for k in range(psi_ht_c.shape[0]):
        A = psi_ht_c[k].reshape(ncb, -1)
        Bm = psi_st_c[k].reshape(ncb, -1)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bm = Bm / np.linalg.norm(Bm, axis=1, keepdims=True)
        s = np.linalg.svd(A.conj() @ Bm.T, compute_uv=False)
        smin = min(smin, float(s.min()))
    return smin

def path_ringing(E_path_c, node_idx):
    """max |E[i+1]-2E[i]+E[i-1]| over interior, node points excluded (meV)."""
    d2 = E_path_c[2:] - 2*E_path_c[1:-1] + E_path_c[:-2]     # (nQ-2, ncb)
    mask = np.ones(d2.shape[0], bool)
    for n in node_idx:
        for j in (n-1, n, n+1):
            if 0 <= j-1 < d2.shape[0]:
                mask[j-1] = False
    return float(np.max(np.abs(d2[mask]))) * 1e3 if mask.any() else float("nan")

# q for the sweep: path (0:nQ) then coarse grid (nQ:nQ+nk)
q_sweep = np.concatenate([kpath, k_frac], axis=0)

rows = []
E_path_by_a = {}
for a_band in A_BANDS:
    t1 = time.time()
    # f-transform params for the record (a-compression is visible directly)
    _fe, a_f, n_f, shift = f_transform_eigs(enk, a_band_index=a_band)
    bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk, kgrid_co=kg,
                          band_window_fi=(0, nb_ret), mesh_xy=mesh_xy,
                          q_list=q_sweep, a_band_index=a_band, log_fn=(lambda *a, **k: None))
    E = np.asarray(jax.device_get(bnd.enk_full))          # (nQ+nk, nb_ret) Ry
    psi = np.asarray(jax.device_get(bnd.psi_rmu_Y))       # (nQ+nk, nb_ret, ns, mu)
    E_path = E[:nQ] * RY                                   # (nQ, nb_ret) eV
    E_grid = E[nQ:nQ+nk]                                   # (nk, nb_ret) Ry
    psi_grid_c = psi[nQ:nQ+nk, BSE_C0:BSE_C1, :, :nmu]     # (nk, 8, ns, mu)

    # (A) min-sval gate vs stored
    smin = minsval_gate(psi_grid_c, psi_st_c)
    # (B) on-grid exactness (Ry -> meV) vs the paired restart energies
    dE = np.abs(E_grid - enk_st_all[:, :nb_ret])
    err_c    = float(np.max(dE[:, BSE_C0:BSE_C1])) * RY * 1e3     # BSE conduction 26..33
    err_high = float(np.max(dE[:, BSE_C1:]))       * RY * 1e3     # guards 34..79
    err_all  = float(np.max(dE))                   * RY * 1e3
    # (C) ringing on the path conduction bands
    ring = path_ringing(E_path[:, BSE_C0:BSE_C1], node_idx)

    E_path_by_a[str(a_band)] = E_path
    rows.append(dict(a_band=(-1 if a_band is None else a_band),
                     a_ry=a_f, shift_ry=shift, min_sval=smin,
                     ongrid_cond_meV=err_c, ongrid_high_meV=err_high,
                     ongrid_all_meV=err_all, ring_cond_meV=ring))
    print(f"[a_band={str(a_band):>4}] a={a_f:7.3f}Ry shift={shift:7.3f}Ry | "
          f"min-sval={smin:.4f} | on-grid cond={err_c:8.3f} high={err_high:9.2f} "
          f"all={err_all:9.2f} meV | ring(cond)={ring:8.3f} meV | {time.time()-t1:.1f}s",
          flush=True)

# ── save BEFORE plotting (expensive compute preserved) ───────────────────────
np.savez("sp80_aband_sweep.npz",
         a_bands=np.array([-1 if a is None else a for a in A_BANDS]),
         a_ry=np.array([r["a_ry"] for r in rows]),
         shift_ry=np.array([r["shift_ry"] for r in rows]),
         min_sval=np.array([r["min_sval"] for r in rows]),
         ongrid_cond_meV=np.array([r["ongrid_cond_meV"] for r in rows]),
         ongrid_high_meV=np.array([r["ongrid_high_meV"] for r in rows]),
         ongrid_all_meV=np.array([r["ongrid_all_meV"] for r in rows]),
         ring_cond_meV=np.array([r["ring_cond_meV"] for r in rows]),
         x_path=x_path, node_idx=np.array(node_idx), nb_ret=nb_ret,
         **{f"Epath_{k}": v for k, v in E_path_by_a.items()})
print("SAVED sp80_aband_sweep.npz", flush=True)

# ── summary table + verdict ──────────────────────────────────────────────────
print("\n==== nband=80 a_band SWEEP SUMMARY ====", flush=True)
print(f"{'a_band':>7} {'a(Ry)':>8} {'min-sval':>9} {'ongrid_c':>9} {'ongrid_hi':>10} {'ring_c':>9}", flush=True)
for r in rows:
    ab = "None(79)" if r["a_band"] == -1 else str(r["a_band"])
    print(f"{ab:>7} {r['a_ry']:8.3f} {r['min_sval']:9.4f} {r['ongrid_cond_meV']:9.3f} "
          f"{r['ongrid_high_meV']:10.2f} {r['ring_cond_meV']:9.3f}", flush=True)
best = max(rows, key=lambda r: (r["min_sval"], -r["ongrid_cond_meV"]))
bb = "None(79)" if best["a_band"] == -1 else str(best["a_band"])
print(f"\nBEST by min-sval: a_band={bb}  min-sval={best['min_sval']:.4f}  "
      f"on-grid cond={best['ongrid_cond_meV']:.3f} meV  ring={best['ring_cond_meV']:.3f} meV", flush=True)

# ── plots: SP-80 bands (best a_band) + metrics vs a_band ──────────────────────
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Ebest = E_path_by_a[("None" if best["a_band"] == -1 else str(best["a_band"]))]
vbm = Ebest[:, :NVAL].max()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.4, 1]})
for b in range(nb_ret):
    col = "#2c6e8f" if b < NVAL else ("#b5432c" if b < BSE_C1 else "#c9a24a")
    lw = 1.3 if BSE_C0 <= b < BSE_C1 else 0.7
    a1.plot(x_path, Ebest[:, b] - vbm, lw=lw, color=col)
a1.set_ylim(-8, 14); a1.set_ylabel("E - E$_{VBM}$ (eV)")
a1.set_title(f"MoS2 12x12 SP-80 htransform bands (a_band={bb})\n"
             f"blue=val red=8v8c-cond(26-33) gold=guard(34-79)")
for n in node_idx:
    a1.axvline(x_path[n], color="0.8", lw=.6, zorder=0)
a1.set_xticks([x_path[n] for n in node_idx]); a1.set_xticklabels(["Γ","M","K","Γ"])
a1.set_xlim(x_path[0], x_path[-1])

abx = [79 if r["a_band"] == -1 else r["a_band"] for r in rows]
a2.plot(abx, [r["min_sval"] for r in rows], "o-", color="#2c6e8f", label="min-sval (cond)")
a2.axhline(0.885, ls=":", color="#2c6e8f", lw=1, label="nband=40 baseline 0.885")
a2.axhline(0.5, ls="--", color="#999", lw=1, label="gate floor 0.5")
a2.set_xlabel("a_band_index (79 = default top band)"); a2.set_ylabel("min-sval", color="#2c6e8f")
a2.set_ylim(0, 1.02)
a2b = a2.twinx()
a2b.semilogy(abx, [max(r["ongrid_cond_meV"], 1e-4) for r in rows], "s--", color="#b5432c",
             label="on-grid cond err (meV)")
a2b.set_ylabel("on-grid conduction error (meV)", color="#b5432c")
a2.set_title("min-sval & on-grid exactness vs a_band")
l1, la1 = a2.get_legend_handles_labels(); l2, la2 = a2b.get_legend_handles_labels()
a2.legend(l1+l2, la1+la2, loc="center right", fontsize=8)
fig.tight_layout(); fig.savefig("sp80_aband_sweep.png")
print("SAVED sp80_aband_sweep.png", flush=True)
