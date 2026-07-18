"""htransform SINGLE-PARTICLE bandstructure + free-pair floor D_min(Q).

Owner deliverable: DFT bands eps_n(k) along the SAME Gamma-M-K-Gamma path
(40 points, 15/8/16) used by the exciton run, plus the free-pair
continuum floor

    D_min(Q) = min_{k,c,v} [ eps_c(k+Q) - eps_v(k) ]

on the 12x12 coarse k set — THE diagnostic for the exciton-band smoothness
question: exciton wiggles that track D_min(Q) are single-particle
kinematics (e.g. the Lambda-valley dip near Q=(0,0.3), diag_dip.py);
wiggles absent from BOTH panels indicate ISDF/interp artifacts.

Machinery: standard htransform (streaming_galerkin_solve via
ht.initialize_wfns) + bse_setup.compute_wfns_fi with explicit q-lists —
identical to what the exciton driver uses for its conduction caches.

Window strategy (12x12/640mu capacity: nk*nb <= ns*n_mu = 1280 -> nb <= 8;
plus TWO lessons quantified by gap_scan.py on the stored energies):
  (a) a window boundary must NOT cut a Kramers pair anywhere in the BZ
      (pairs are exactly degenerate; cutting one breaks fH k-smoothness ->
      eV-scale off-grid failures — the w2331/w2533 mechanism, same as the
      Si degeneracy root-cause), and boundary min-gaps should be healthy;
  (b) plotted bands must be conditioned: error amplification on inversion
      is 1/f', and f'->0 toward the window top (compression zone).
Windows:
  hts_val.in    (20,28): plot valence 22-25 (idx 2-5; f' >= 0.8),
                boundaries 64.7/75.8 meV
  hts_cond2.in  (26,34): plot conduction 26-31 (idx 0-5), boundaries
                1700/2194 meV (the two largest gaps in the spectrum);
                a_band_index=2 (band 28) keeps the plotted-top pair 30,31
                at f' ~ 0.31
  hts_cond.in   (24,32): D_min(Q) caches — IDENTICAL to the exciton
                driver's window — and the cross-window gate reference
Gates: per-band cross-window on 24-25 and 26-29 (all deep on both sides);
on-grid htransform vs stored DFT; D_min A/B across the two structurally
clean windows (24,32) vs (26,34).
"""
import time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY2EV = 13.6056980659
# dataviz palette (validated categorical slots, light surface)
C_VAL, C_COND, C_DMIN, C_EXC = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"

timings = {}


def stage(name, t0):
    timings[name] = time.time() - t0
    print(f"[t] {name}: {timings[name]:.2f} s", flush=True)


def run_window(input_file, mesh_xy, q_list, band_window, a_band_index=None):
    """Galerkin-solve a window and evaluate enk at q_list. Returns (nq, nb) Ry
    plus the htransform handles for reuse."""
    params = read_lorrax_input(input_file)
    (wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(
        input_file, params, print, mesh_xy=mesh_xy)
    kgrid_co = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                          kgrid_co=kgrid_co, band_window_fi=band_window,
                          mesh_xy=mesh_xy, q_list=q_list,
                          a_band_index=a_band_index, log_fn=print)
    E = np.asarray(jax.device_get(bnd.enk_full))
    return E, (wfn, meta, enk, ctilde, B, params, kgrid_co)


t_all = time.time()
mesh_xy = _create_mesh_xy(1, 1)

# ── path from the reference (24,32) window input ─────────────────────────
params_c = read_lorrax_input("hts_cond.in")
t0 = time.time()
(wfn, sym, meta, _m, _S, ctilde_c, B_c, enk_c) = ht.initialize_wfns(
    "hts_cond.in", params_c, print, mesh_xy=mesh_xy)
stage("galerkin_cond_2432", t0)

kpath_frac, x_path, node_idx, node_labels, _gp = ht.initialize_kpath(wfn, params_c)
kpath = np.asarray(kpath_frac)
nq = kpath.shape[0]
x_path = np.asarray(x_path)
print(f"Q path: {nq} pts, nodes {list(map(int, node_idx))} labels {node_labels}")
nelec = int(wfn.nelec)
kgrid_co = (int(meta.nkx), int(meta.nky), int(meta.nkz))
assert nelec == 26 and kgrid_co == (12, 12, 1), (nelec, kgrid_co)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde_c, B_at_mu=B_c, enk_sigma=enk_c,
                      kgrid_co=kgrid_co, band_window_fi=(0, 8),
                      mesh_xy=mesh_xy, q_list=kpath, log_fn=print)
E_ref_path = np.asarray(jax.device_get(bnd.enk_full))        # (nq,8) 24..31
stage("path_cond_2432", t0)

# ── plot windows: valence (20,28), conduction (26,34) ───────────────────
t0 = time.time()
E_val_path, _ = run_window("hts_val.in", mesh_xy, kpath, (0, 8))   # 20..27
stage("galerkin+path_val_2028", t0)
t0 = time.time()
E_c2_path, hc2 = run_window("hts_cond2.in", mesh_xy, kpath, (0, 8),
                            a_band_index=2)                        # 26..33
stage("galerkin+path_cond2_2634", t0)
(_wc2, _mc2, enk_c2, ctilde_c2, B_c2, _pc2, _kg2) = hc2

# ── cross-window gates (per band, meV, max over the 40-pt path) ─────────
print("[gate] cross-window per-band |d| over path (meV):")
for b_abs, (Ea, ia, Eb, ib, tag) in {
        24: (E_val_path, 4, E_ref_path, 0, "val(20,28) vs ref(24,32), both deep"),
        25: (E_val_path, 5, E_ref_path, 1, "val(20,28) vs ref(24,32), both deep"),
        26: (E_c2_path, 0, E_ref_path, 2, "cond2(26,34) vs ref(24,32), both deep"),
        27: (E_c2_path, 1, E_ref_path, 3, "cond2(26,34) vs ref(24,32), both deep"),
        28: (E_c2_path, 2, E_ref_path, 4, "cond2(26,34) vs ref(24,32), both deep"),
        29: (E_c2_path, 3, E_ref_path, 5, "cond2(26,34) vs ref(24,32), both deep"),
        30: (E_c2_path, 4, E_ref_path, 6, "cond2 vs ref [ref side: top pair + 5.9 meV 31|32 boundary]"),
        31: (E_c2_path, 5, E_ref_path, 7, "cond2 vs ref [ref side: top pair + 5.9 meV 31|32 boundary]"),
}.items():
    d = np.abs(Ea[:, ia] - Eb[:, ib]) * RY2EV * 1e3
    print(f"    band {b_abs}: max {d.max():9.3f} @iQ {int(d.argmax()):2d}, "
          f"median {np.median(d):8.3f}   ({tag})")

# ── D_min(Q) on the 12x12 coarse k set — driver-identical (24,32) ───────
nkx, nky, _ = kgrid_co
k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                              [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
nk = k_frac.shape[0]

t0 = time.time()
q_big = (kpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)
bnd = compute_wfns_fi(ctilde=ctilde_c, B_at_mu=B_c, enk_sigma=enk_c,
                      kgrid_co=kgrid_co, band_window_fi=(2, 8),
                      mesh_xy=mesh_xy, q_list=q_big, log_fn=print)
E_cQ = np.asarray(jax.device_get(bnd.enk_full)).reshape(nq, nk, 6)  # c1..c6
stage("dmin_cond_kQ", t0)

t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde_c, B_at_mu=B_c, enk_sigma=enk_c,
                      kgrid_co=kgrid_co, band_window_fi=(0, 2),
                      mesh_xy=mesh_xy, q_list=k_frac, log_fn=print)
E_vk = np.asarray(jax.device_get(bnd.enk_full))              # (nk,2), 24..25
stage("val_ongrid", t0)

# on-grid gate: htransform at grid k must reproduce the stored DFT energies
# (order-insensitive: compare the sorted top-valence sets)
enk_c_np = np.asarray(enk_c)                                  # (8, nk) window
d_ongrid = np.abs(np.sort(E_vk[:, 1]) - np.sort(enk_c_np[1, :])) * RY2EV * 1e3
print(f"[gate] on-grid htransform vs stored DFT (band 25, sorted): "
      f"max|d| = {d_ongrid.max():.4f} meV")

D_pair = E_cQ[:, :, :, None] - E_vk[None, :, None, :]         # (nq,nk,6,2) Ry
D_k = D_pair.min(axis=(2, 3))                                 # (nq,nk)
D_min = D_k.min(axis=1) * RY2EV                               # (nq,) eV
k_arg = D_k.argmin(axis=1)

# ── D_min A/B: same floor from the structurally clean (26,34) window ────
t0 = time.time()
bnd = compute_wfns_fi(ctilde=ctilde_c2, B_at_mu=B_c2, enk_sigma=enk_c2,
                      kgrid_co=kgrid_co, band_window_fi=(0, 6),
                      mesh_xy=mesh_xy, q_list=q_big, a_band_index=2,
                      log_fn=print)
E_cQ2 = np.asarray(jax.device_get(bnd.enk_full)).reshape(nq, nk, 6)
stage("dmin_cond_kQ_w2634", t0)
D_min2 = ((E_cQ2[:, :, :, None] - E_vk[None, :, None, :])
          .min(axis=(2, 3)).min(axis=1)) * RY2EV
d_ab = np.abs(D_min - D_min2) * 1e3
print(f"[gate] D_min A/B (24,32) vs (26,34): max|d| = {d_ab.max():.3f} meV "
      f"@iQ {int(d_ab.argmax())}, median = {np.median(d_ab):.3f} meV")

# VBM reference for the bands panel (band 25 over grid + path)
vbm = max(float(E_vk[:, 1].max()), float(E_val_path[:, 5].max()))
print(f"VBM (band 25) = {vbm:.6f} Ry; direct-gap floor D_min(Gamma) = "
      f"{D_min[0]:.4f} eV; min over path = {D_min.min():.4f} eV at iQ "
      f"{int(D_min.argmin())}")

# ── .dat outputs ─────────────────────────────────────────────────────────
bands_ev = np.concatenate([E_val_path[:, 2:6], E_c2_path[:, 0:6]], axis=1)
bands_ev = (bands_ev - vbm) * RY2EV                           # (nq,10) eV
nodes_str = " ".join(f"{int(i)}:{l}" for i, l in zip(node_idx, node_labels))
with open("sp_bands_12x12_GMKG.dat", "w", encoding="utf8") as fh:
    fh.write("# htransform single-particle bands, MoS2 12x12 (640 centroids)\n"
             "# valence 22-25 from window (20,28) [hts_val.in]; conduction "
             "26-31 from window (26,34) [hts_cond2.in, a_band=28]\n"
             "# (windows chosen so no boundary cuts a Kramers pair and every "
             "plotted band is f-transform-conditioned; see gap_scan.py)\n"
             f"# energies in eV relative to VBM (band 25, {vbm:.6f} Ry)\n"
             f"# nodes: {nodes_str}\n"
             "# iQ  s_path  kx  ky  kz  E_b22..E_b31 (eV)\n")
    for i in range(nq):
        fh.write(f"{i:4d} {x_path[i]:9.6f} " +
                 " ".join(f"{c: .6f}" for c in kpath[i]) + " " +
                 " ".join(f"{e: .6f}" for e in bands_ev[i]) + "\n")
with open("dmin_12x12_GMKG.dat", "w", encoding="utf8") as fh:
    fh.write("# free-pair floor D_min(Q) = min_{k,c,v}[eps_c(k+Q)-eps_v(k)], "
             "12x12 coarse k set, c in 26-31, v in 24-25\n"
             "# Dmin_eV: htransform window (24,32) — IDENTICAL to the "
             "exciton driver's conduction caches\n"
             "# Dmin_w2634_eV: A/B from the structurally clean (26,34) "
             "window (both boundaries at the two largest gaps)\n"
             f"# nodes: {nodes_str}\n"
             "# iQ  s_path  Qx  Qy  Qz  Dmin_eV  Dmin_w2634_eV  "
             "argmin_kx  argmin_ky\n")
    for i in range(nq):
        fh.write(f"{i:4d} {x_path[i]:9.6f} " +
                 " ".join(f"{c: .6f}" for c in kpath[i]) +
                 f" {D_min[i]: .6f} {D_min2[i]: .6f} "
                 f"{k_frac[k_arg[i], 0]: .6f} {k_frac[k_arg[i], 1]: .6f}\n")
print("Wrote sp_bands_12x12_GMKG.dat, dmin_12x12_GMKG.dat")

# ── exciton E_1 overlay data (sibling's delivered .dat) ─────────────────
exc_s, exc_E1 = None, None
try:
    rows = []
    with open("../01_lorrax_exciton_bands/exciton_bands_12x12_GMKG.dat",
              encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if t[5] == "interp":
                rows.append((int(t[0]), float(t[1]), float(t[6])))
    rows.sort()
    exc_s = np.array([r[1] for r in rows])
    exc_E1 = np.array([r[2] for r in rows])
    assert np.allclose(exc_s, x_path, atol=1e-5), "path mismatch vs exciton run"
except FileNotFoundError:
    print("exciton .dat not found; panel 2 will omit the E_1 overlay")

# ── plot: two stacked panels, shared x ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(7.2, 7.0),
    gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.08})
for b in range(4):
    ax1.plot(x_path, bands_ev[:, b], lw=1.4, color=C_VAL,
             label="valence 22-25" if b == 0 else None)
for b in range(4, 10):
    ax1.plot(x_path, bands_ev[:, b], lw=1.4, color=C_COND,
             label="conduction 26-31" if b == 4 else None)
ax1.axhline(0.0, color="0.6", lw=0.7, ls=":")
ax1.set_ylabel(r"$\varepsilon_n(k) - E_\mathrm{VBM}$ (eV)")
ax1.set_title("MoS$_2$ 12$\\times$12 — htransform single-particle bands "
              "(640 centroids)", fontsize=11)
ax1.legend(loc="center left", fontsize="small", framealpha=0.9)
ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

ax2.plot(x_path, D_min, lw=1.8, color=C_DMIN, label=r"$D_\min(Q)$ free-pair floor")
if exc_E1 is not None:
    ax2.plot(exc_s, exc_E1, lw=1.2, ls="--", color=C_EXC,
             label=r"$E_1(Q)$ exciton (640c interp)")
iq_dip = 9
ax2.annotate("$\\Lambda$-valley dip\n(single-particle)",
             xy=(x_path[iq_dip], D_min[iq_dip]),
             xytext=(x_path[iq_dip] + 0.9, D_min[iq_dip] - 0.25),
             fontsize=8, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax2.set_ylabel("energy (eV)")
ax2.legend(loc="upper right", fontsize="small", framealpha=0.9)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

node_x = x_path[np.asarray(node_idx, dtype=int)]
for ax in (ax1, ax2):
    for xv in node_x:
        ax.axvline(xv, color="k", lw=0.6, alpha=0.25)
ax2.set_xticks(node_x, [l or "" for l in node_labels])
ax2.set_xlim(x_path[0], x_path[-1])

fig.text(0.5, -0.02,
         "Exciton-band wiggles that track $D_\\min(Q)$ (e.g. the dips near "
         "$Q=(0,0.2)$ and $Q=(0,0.3)$) are single-particle kinematics;\n"
         "wiggles absent from both panels would indicate ISDF/interpolation "
         "artifacts.",
         ha="center", fontsize=8)
fig.savefig("sp_bands_12x12_GMKG.png", dpi=180, bbox_inches="tight")
print("Wrote sp_bands_12x12_GMKG.png")

timings["TOTAL"] = time.time() - t_all
print("\n--- timings (s) ---")
for k, v in timings.items():
    print(f"  {k:26s} {v:10.2f}")
