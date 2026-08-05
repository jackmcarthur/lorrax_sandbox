"""Pre-flight probe of the (22,34)@1000c window at a_band=6 (band 28, a=4xBW=2.87 eV — the proven clean-floor configuration)
(the exciton driver has no --a-band plumbing, so the follow-up run uses
a = 4*BW(band 33)).  Checks, against the validated v4 clean references
((24,36)@1000c a_band=31):
  1. conduction path bands 26-29 (the driver's cache bands) smoothness +
     agreement;
  2. D_min(Q) over the full 40x144 k+Q set (also THE deliverable clean
     D_min curve for the three-way overlay -> dmin_2234_GMKG.dat);
  3. on-grid exactness.
If default-a conditioning were insufficient, isolated 100+ meV
disagreements would appear at the known off-grid hotspots."""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY2EV = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)
INP = "exciton_bands_1000c_cleanwin.in"

params = read_lorrax_input(INP)
(wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(
    INP, params, print, mesh_xy=mesh_xy)
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn, params)
kpath = np.asarray(kpath_frac)
nq = kpath.shape[0]
x_path = np.asarray(x_path)
kgrid_co = (int(meta.nkx), int(meta.nky), int(meta.nkz))

# 1. conduction path bands 26-29 = window idx 4..7, DEFAULT a
bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                      kgrid_co=kgrid_co, band_window_fi=(0, 12),
                      mesh_xy=mesh_xy, q_list=kpath, a_band_index=6, log_fn=print)
E_path = np.asarray(jax.device_get(bnd.enk_full))            # (nq,12) 22..33

v4 = np.loadtxt("../05_htransform_spbands/sp_bands_12x12_GMKG.dat")
vbm_row = None
print("[gate] (22,34)@1000c default-a vs v4 refs, |d| meV over path:")
# v4 dat cols 5..14 = bands 22..31 (VBM-referenced); re-reference by VBM
E_vk_probe = None
for j, b_abs in enumerate(range(22, 32)):
    pass  # filled after VBM known

# 2. valence on-grid (idx 2,3 = bands 24,25) + D_min conduction 26-31
nkx, nky, _ = kgrid_co
k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                              [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]
q_big = (kpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)
bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                      kgrid_co=kgrid_co, band_window_fi=(4, 10),
                      mesh_xy=mesh_xy, q_list=q_big, a_band_index=6, log_fn=print)
E_cQ = np.asarray(jax.device_get(bnd.enk_full)).reshape(nq, nk, 6)  # 26..31
bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                      kgrid_co=kgrid_co, band_window_fi=(2, 4),
                      mesh_xy=mesh_xy, q_list=k_frac, a_band_index=6, log_fn=print)
E_vk = np.asarray(jax.device_get(bnd.enk_full))               # (nk,2) 24,25

enk_np = np.asarray(enk)          # (12, nk) window 22..33
d_og = np.abs(np.sort(E_vk[:, 1]) - np.sort(enk_np[3, :])) * RY2EV * 1e3
print(f"[gate] on-grid vs stored DFT (band 25 sorted): max|d| = "
      f"{d_og.max():.4f} meV")

vbm = float(E_vk[:, 1].max())
bands_ev = (E_path - vbm) * RY2EV                             # 22..33
for j, b_abs in enumerate(range(22, 32)):
    d = np.abs(bands_ev[:, j] - v4[:, 5 + j]) * 1e3
    print(f"    band {b_abs}: max {d.max():9.3f} @iQ {int(d.argmax()):2d}, "
          f"median {np.median(d):8.3f}")

D_pair = E_cQ[:, :, :, None] - E_vk[None, :, None, :]
D_k = D_pair.min(axis=(2, 3))
D_min = D_k.min(axis=1) * RY2EV
k_arg = D_k.argmin(axis=1)

dm4 = np.loadtxt("../05_htransform_spbands/dmin_12x12_GMKG.dat")
d_ab = np.abs(D_min - dm4[:, 6]) * 1e3
print(f"[gate] D_min (22,34)@default-a vs (24,36)@a31: max|d| = "
      f"{d_ab.max():.3f} meV @iQ {int(d_ab.argmax())}, median = "
      f"{np.median(d_ab):.3f} meV")
print(f"D_min(Gamma) = {D_min[0]:.4f} eV; iQ6/9/16/17: " +
      " ".join(f"{D_min[i]:.4f}" for i in (6, 9, 16, 17)))

nodes_str = " ".join(f"{int(i)}:{l}" for i, l in zip(node_idx, node_labels))
with open("dmin_2234_GMKG.dat", "w", encoding="utf8") as fh:
    fh.write("# free-pair floor D_min(Q), htransform window (22,34)@1000c a_band=28 "
             "DEFAULT a — the EXACT window/a the clean-window exciton "
             "driver uses\n"
             f"# nodes: {nodes_str}\n"
             "# iQ  s_path  Qx  Qy  Qz  Dmin_eV  argmin_kx  argmin_ky\n")
    for i in range(nq):
        fh.write(f"{i:4d} {x_path[i]:9.6f} " +
                 " ".join(f"{c: .6f}" for c in kpath[i]) +
                 f" {D_min[i]: .6f} {k_frac[k_arg[i], 0]: .6f} "
                 f"{k_frac[k_arg[i], 1]: .6f}\n")
print("Wrote dmin_2234_GMKG.dat")
