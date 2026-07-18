"""v4: SP-bands upgrade with the 1000-centroid basis.

The 1000-mu alpha-basis lifts the htransform capacity to nb <= 13, opening
the (24,36) window: BOTH boundaries at healthy between-pair gaps (23|24 =
241 meV, 35|36 = 243 meV) and gap-edge bands 24-31 all deep (a_band=31 ->
near-linear f').  This removes the v3 valence blemishes (val(20,28) had
isolated 100-250 meV off-grid ringing on bands 24/25 at iQ 20/30 from its
64.7/75.8 meV boundaries).

Composition of the deliverable:
  bands 22,23   val(20,28) @640c   (no clean-boundary window exists below
                                    band 24 at ANY reachable capacity;
                                    isolated O(100 meV) off-grid blemishes)
  bands 24-31   (24,36) @1000c    (clean)
  D_min ref     (24,32) @640c     REUSED from v3 dmin dat (driver-identical)
  D_min clean   (24,36) @1000c    (also cross-checked vs v3's (26,34)@640c
                                    clean column -> basis-convergence gate)
"""
import time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY2EV = 13.6056980659
timings = {}


def stage(name, t0):
    timings[name] = time.time() - t0
    print(f"[t] {name}: {timings[name]:.2f} s", flush=True)


def run_window(input_file, mesh_xy, q_list, band_window, a_band_index=None):
    params = read_lorrax_input(input_file)
    (wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(
        input_file, params, print, mesh_xy=mesh_xy)
    kg = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                          kgrid_co=kg, band_window_fi=band_window,
                          mesh_xy=mesh_xy, q_list=q_list,
                          a_band_index=a_band_index, log_fn=print)
    E = np.asarray(jax.device_get(bnd.enk_full))
    return E, (wfn, meta, enk, ctilde, B, params, kg)


t_all = time.time()
mesh_xy = _create_mesh_xy(1, 1)

# path from any input (same K_POINTS block everywhere)
params0 = read_lorrax_input("hts_1000c.in")
wfn0, _sym0 = ht.setup_wfn_and_sym("WFN.h5")
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(
    wfn0, params0)
kpath = np.asarray(kpath_frac)
nq = kpath.shape[0]
x_path = np.asarray(x_path)

# ── 1000c window (24,36): path + D_min conduction + on-grid valence ─────
t0 = time.time()
E_w_path, hw = run_window("hts_1000c.in", mesh_xy, kpath, (0, 12),
                          a_band_index=7)                     # 24..35
stage("galerkin+path_1000c_2436", t0)
(_w, _m2, enk_w, ctilde_w, B_w, _p, kgrid_co) = hw

nkx, nky, _ = kgrid_co
k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                              [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]
q_big = (kpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde_w, B_at_mu=B_w, enk_sigma=enk_w,
                      kgrid_co=kgrid_co, band_window_fi=(2, 8),
                      mesh_xy=mesh_xy, q_list=q_big, a_band_index=7,
                      log_fn=print)
E_cQ = np.asarray(jax.device_get(bnd.enk_full)).reshape(nq, nk, 6)  # 26..31
stage("dmin_cond_kQ_1000c", t0)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde_w, B_at_mu=B_w, enk_sigma=enk_w,
                      kgrid_co=kgrid_co, band_window_fi=(0, 2),
                      mesh_xy=mesh_xy, q_list=k_frac, a_band_index=7,
                      log_fn=print)
E_vk = np.asarray(jax.device_get(bnd.enk_full))               # (nk,2) 24,25
stage("val_ongrid_1000c", t0)

enk_w_np = np.asarray(enk_w)
d_og = np.abs(np.sort(E_vk[:, 1]) - np.sort(enk_w_np[1, :])) * RY2EV * 1e3
print(f"[gate] on-grid htransform vs stored DFT (band 25, sorted): "
      f"max|d| = {d_og.max():.4f} meV")

D_clean = ((E_cQ[:, :, :, None] - E_vk[None, :, None, :])
           .min(axis=(2, 3)).min(axis=1)) * RY2EV
k_arg = ((E_cQ[:, :, :, None] - E_vk[None, :, None, :])
         .min(axis=(2, 3)).argmin(axis=1))

# ── 640c valence window for bands 22,23 ─────────────────────────────────
t0 = time.time()
E_val_path, _hv = run_window("hts_val.in", mesh_xy, kpath, (0, 8))  # 20..27
stage("galerkin+path_val640_2028", t0)

# ── gates vs the committed v3 .dat ──────────────────────────────────────
v3 = np.loadtxt("sp_bands_12x12_GMKG.dat")
dm3 = np.loadtxt("dmin_12x12_GMKG.dat")
vbm = max(float(E_vk[:, 1].max()), float(E_w_path[:, 1].max()))
bands_ev = np.concatenate([E_val_path[:, 2:4], E_w_path[:, 0:8]], axis=1)
bands_ev = (bands_ev - vbm) * RY2EV
print("[gate] v4(1000c) vs v3(640c) plotted bands, |d| meV over path:")
for j, b_abs in enumerate(range(22, 32)):
    d = np.abs(bands_ev[:, j] - v3[:, 5 + j]) * 1e3
    print(f"    band {b_abs}: max {d.max():9.3f} @iQ {int(d.argmax()):2d}, "
          f"median {np.median(d):8.3f}")
d_ab = np.abs(D_clean - dm3[:, 6]) * 1e3
print(f"[gate] clean D_min: (24,36)@1000c vs (26,34)@640c: "
      f"max|d| = {d_ab.max():.3f} meV @iQ {int(d_ab.argmax())}, "
      f"median = {np.median(d_ab):.3f} meV")
print(f"D_min(Gamma) = {D_clean[0]:.4f} eV (direct gap 1.7001)")

# ── write v4 .dat (same deliverable names; v3 preserved in git) ─────────
nodes_str = " ".join(f"{int(i)}:{l}" for i, l in zip(node_idx, node_labels))
with open("sp_bands_12x12_GMKG.dat", "w", encoding="utf8") as fh:
    fh.write("# htransform single-particle bands, MoS2 12x12 (v4)\n"
             "# bands 24-31 from window (24,36) @1000 centroids (boundaries "
             "23|24=241/35|36=243 meV, a_band=31 -> near-linear f')\n"
             "# bands 22,23 from window (20,28) @640 centroids (no "
             "clean-boundary window exists below band 24; isolated O(100 "
             "meV) off-grid blemishes possible)\n"
             f"# energies in eV relative to VBM (band 25, {vbm:.6f} Ry)\n"
             f"# nodes: {nodes_str}\n"
             "# iQ  s_path  kx  ky  kz  E_b22..E_b31 (eV)\n")
    for i in range(nq):
        fh.write(f"{i:4d} {x_path[i]:9.6f} " +
                 " ".join(f"{c: .6f}" for c in kpath[i]) + " " +
                 " ".join(f"{e: .6f}" for e in bands_ev[i]) + "\n")
with open("dmin_12x12_GMKG.dat", "w", encoding="utf8") as fh:
    fh.write("# free-pair floor D_min(Q) = min_{k,c,v}[eps_c(k+Q)-eps_v(k)] "
             "(v4), 12x12 coarse k set, c in 26-31, v in 24-25\n"
             "# Dmin_eV: htransform window (24,32) @640c — IDENTICAL to the "
             "exciton driver's conduction caches (reused from v3)\n"
             "# Dmin_clean_eV: (24,36) @1000c clean window (cross-checked "
             "vs (26,34)@640c)\n"
             f"# nodes: {nodes_str}\n"
             "# iQ  s_path  Qx  Qy  Qz  Dmin_eV  Dmin_clean_eV  "
             "argmin_kx  argmin_ky\n")
    for i in range(nq):
        fh.write(f"{i:4d} {x_path[i]:9.6f} " +
                 " ".join(f"{c: .6f}" for c in kpath[i]) +
                 f" {dm3[i, 5]: .6f} {D_clean[i]: .6f} "
                 f"{k_frac[k_arg[i], 0]: .6f} {k_frac[k_arg[i], 1]: .6f}\n")
print("Wrote sp_bands_12x12_GMKG.dat, dmin_12x12_GMKG.dat (v4)")

timings["TOTAL"] = time.time() - t_all
print("\n--- timings (s) ---")
for k, v in timings.items():
    print(f"  {k:28s} {v:10.2f}")
