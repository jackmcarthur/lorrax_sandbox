"""offgrid_path_htr — OWNER-REDESIGNED off-grid test, part (a)+(partial b):
PHYSICAL observables along Gamma -> (1/6,0,0)x-hat via htransform'd
wavefunctions on the SAME 6x6 dataset (arbitrary_q_bse.md sec 1 contract:
bandstructure.htransform.initialize_wfns + bse_setup.compute_wfns_fi with
kgrid_fi = 24x24x1, which contains every k - q(t) for t in {0,1/4,1/2,3/4,1}).

Per t: TRUE pair rows M(t) (spin-traced, 3v x 3c x all 36 k) from htransform
psi(r_mu) at k - q(t); D(t) from htransform energies; H_dir(t) from
htransform psi + the stored W0 (q_kk' is t-independent); B(t) = M(t)^H
V_interp(t) M(t) with V from the rank-cut ingredient-interp scheme
(nR=36 exact stencil, G0-excluded union superset, analytic slab v).
Deliverables: lowest-4 eigenvalues of the swap-H(t) as functions of t
(smoothness + endpoint anchoring vs the stored-fit truth contracted with
the SAME M(t)), and the B-relF endpoint anchors in htransform content.

Content notes (measured by the gates, not assumed):
- htransform is trained on WFN.h5 content; gates compare on-grid psi(r_mu)
  vs WFN-at-centroids (pure htransform fidelity) AND vs psi_full_y (the
  known band-span trap, KNOWN_SANDBOX_ERRORS 2026-07-17 item 2).
- psi_htr is rescaled once by the global median per-band norm ratio to the
  psi_full_y convention (V/W0 units); the ratio distribution is a gate.
- The full zeta-refit ground truth at midpoints (fit RHS on the full r-grid
  from htransform'd full-grid psi) is NOT built here — centroids-only
  contract; marked as the follow-up in the docs.

Run: JID=<jid> ./proto1_run.sh python3 -u offgrid_path_htr.py
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import offgrid_prep as op
from offgrid_prep import (Fixture, relF, truncR_weights, fix_sphere_wrap,
                          svd_herm, sorted_stencil, RY2MEV)

t00 = time.time()
NPZ = {}
BASE = "/pscratch/sd/j/jackm/lorrax_sandbox"
LORRAX_DIR = f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax"
QDIR = np.array([1.0 / 6.0, 0.0, 0.0])
TJ = [0, 1, 2, 3, 4]                 # t = j/4; q(t) = (j/24, 0, 0)
RUNGS = [("rankcut", 1e-3), ("rankcut", 1e-4), ("rankcut", 1e-5)]
NVW = NCW = 3
NB_HTR = 36                          # htransform band window (0, 36)
WLO, WHI = 20, 33                    # gate band window (covers 23..28 + slack)

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx, verbose=False)
C_q = fx.build_Cq()
Zr = np.empty((fx.nq, fx.n_mu * fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    Zr[q] = (C_q[q] @ fx.recon(q)).ravel()
print(f"[htr] fixture + Z_r ready ({time.time()-t00:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# htransform setup + 24x24 fine bundle (production entry points, read-only)
# ---------------------------------------------------------------------------
from bandstructure.htransform import initialize_wfns
from bandstructure.bse_setup import compute_wfns_fi

params = {"wfn_file": "WFN.h5", "centroids_file": "centroids_frac_640.txt",
          "nval": 26, "ncond": 54, "nband": 80, "bispinor": False}
wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma = initialize_wfns(
    f"{LORRAX_DIR}/cohsex.in", params, print)
print(f"[htr] galerkin rank = {ctilde.shape[2]}", flush=True)
bundle = compute_wfns_fi(
    ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
    kgrid_co=(6, 6, 1), kgrid_fi=(24, 24, 1),
    band_window_fi=(0, NB_HTR), mesh_xy=mesh_xy,
    a_band_index=NB_HTR - 1, batch_size=32, log_fn=print)
psi_htr = np.asarray(bundle.psi_rmu_Y)               # (576, NB_HTR, 2, n_mu)
enk_htr = np.asarray(bundle.enk_full)                # (576, NB_HTR) Ry
print(f"[htr] fine bundle {psi_htr.shape} ({time.time()-t00:.0f}s)", flush=True)


def idx24(i, j):
    return (i % 24) * 24 + (j % 24)


# ---------------------------------------------------------------------------
# gates: on-grid fidelity + content + normalization
# ---------------------------------------------------------------------------
print("\n[htr gates]")
kmap = [(idx24(4 * a, 4 * b), 6 * a + b) for a in range(6) for b in range(6)]
de = np.array([np.abs(enk_htr[i24, WLO:WHI] - fx.enk[k6, WLO:WHI])
               for i24, k6 in kmap])
print(f"  [gate] on-grid energies, bands {WLO}..{WHI-1}: "
      f"med {np.median(de)*RY2MEV:.3f} meV  max {de.max()*RY2MEV:.3f} meV")
de_all = np.array([np.abs(enk_htr[i24, :NB_HTR] - fx.enk[k6, :NB_HTR])
                   for i24, k6 in kmap])
print(f"  [gate] on-grid energies, bands 0..{NB_HTR-1}: "
      f"med {np.median(de_all)*RY2MEV:.3f} meV  max {de_all.max()*RY2MEV:.3f} meV")

# subspace fidelity vs WFN-at-centroids and vs psi_full_y (window bands)
sv_wfn, sv_psi, ratios = [], [], []
for a in range(6):
    for b in range(6):
        i24, k6 = idx24(4 * a, 4 * b), 6 * a + b
        Ph = psi_htr[i24, WLO:WHI].reshape(WHI - WLO, -1)
        ug = fx.u_grid(k6, nbmax=WHI)[WLO:, :, fx.rmu_flat].reshape(WHI - WLO, -1)
        Pf = fx.psi[k6, WLO:WHI].reshape(WHI - WLO, -1)
        for A, B_, sink in ((Ph, ug, sv_wfn), (Ph, Pf, sv_psi)):
            Qa = np.linalg.qr(A.T.conj())[0]
            Qb = np.linalg.qr(B_.T.conj())[0]
            sink.append(np.linalg.svd(Qa.conj().T @ Qb, compute_uv=False))
        ratios.append(np.linalg.norm(Pf, axis=1) / np.linalg.norm(Ph, axis=1))
sv_wfn, sv_psi = np.array(sv_wfn), np.array(sv_psi)
ratios = np.array(ratios)
print(f"  [gate] window-subspace cos vs WFN-centroids: "
      f"min {sv_wfn.min():.6f} med {np.median(sv_wfn):.6f}")
print(f"  [gate] window-subspace cos vs psi_full_y:    "
      f"min {sv_psi.min():.6f} med {np.median(sv_psi):.6f}  (band-span trap)")
scale = float(np.median(ratios))
print(f"  [gate] norm ratio psi_full_y/psi_htr: med {scale:.6f} "
      f"spread [{ratios.min():.4f}, {ratios.max():.4f}]")
psi_htr = psi_htr * scale            # -> psi_full_y normalization convention

# ---------------------------------------------------------------------------
# interp machinery on the union superset (same as offgrid_path.py)
# ---------------------------------------------------------------------------
gr = [np.arange(-(n // 2), n - n // 2) for n in (fx.nx, fx.ny, fx.nz)]
GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0).astype(np.int64)
m = np.full(Gall.shape[1], np.inf)
for t in np.linspace(0, 1, 33):
    K = fx.bvec.T @ (t * QDIR[:, None] + Gall.astype(np.float64))
    m = np.minimum(m, np.sum(K * K, axis=0))
keep = (m <= fx.zeta_cutoff + 1e-9) & ~np.all(Gall == 0, axis=0)
GS = Gall[:, keep]


def to_superset(zr_rows, qfrac):
    ph = np.exp(-2j * np.pi * (fx.rfrac @ qfrac))
    box = np.fft.fftn((zr_rows * ph[None, :]).reshape(-1, fx.nx, fx.ny, fx.nz),
                      axes=(1, 2, 3), norm="backward").reshape(zr_rows.shape[0],
                                                               fx.n_rtot)
    return box[:, fx.flat_idx(GS)]


def v_slab(qfrac):
    K = fx.bvec.T @ (qfrac[:, None] + GS.astype(np.float64))
    K2 = np.sum(K * K, axis=0)
    zc = np.pi / fx.bvec[2, 2]
    f2d = 1.0 - np.exp(-zc * np.sqrt(K[0]**2 + K[1]**2)) * np.cos(K[2] * zc)
    return 8.0 * np.pi / K2 * f2d / fx.celvol


R36 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4) for j in range(-2, 4)])
train = list(range(fx.nq))


def zeta_interp(qt):
    """rank-cut ladder solutions on the superset at momentum qt."""
    w = truncR_weights(fx.qfr[train], qt, R36)
    C0 = np.tensordot(w, C_q, axes=(0, 0))
    Z0 = np.zeros(fx.n_mu * fx.n_rtot, dtype=np.complex128)
    for wi, tr in zip(w, train):
        Z0 += wi * Zr[tr]
    Zt0 = to_superset(Z0.reshape(fx.n_mu, fx.n_rtot), qt)
    U, s, Vh = svd_herm(C0)
    UZ = np.conj(U.T) @ Zt0
    return {op.rung_label(mo, la): (np.conj(Vh.T) * op.rung_sinv(s, mo, la)) @ UZ
            for mo, la in RUNGS}


# ---------------------------------------------------------------------------
# swap-H(t) trajectory
# ---------------------------------------------------------------------------
cs = list(range(fx.nv, fx.nv + NCW))
vs = list(range(fx.nv - NVW, fx.nv))
bs = NCW * NVW
npair = fx.nk * bs
kg = fx.kgrid
qkk = np.array([[fx.k_lookup[tuple((fx.k_int[k] - fx.k_int[kp]) % kg)]
                 for kp in range(fx.nk)] for k in range(fx.nk)])
k24 = np.array([idx24(4 * a, 4 * b) for a in range(6) for b in range(6)])


def swap_H_pieces(j):
    """M(t), D(t), H_dir(t) at q = (j/24, 0, 0), all from psi_htr."""
    kq24 = np.array([idx24(4 * a - j, 4 * b) for a in range(6)
                     for b in range(6)])
    psic = np.ascontiguousarray(psi_htr[:, cs])       # (576, 3, 2, nmu)
    psiv = np.ascontiguousarray(psi_htr[:, vs])
    pc_kq = psic[kq24]                                # (36, 3, 2, nmu) by k
    pv_k = psiv[k24]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc_kq), pv_k).reshape(-1, fx.n_mu)
    D = np.array([enk_htr[kq24[k], c] - enk_htr[k24[k], v]
                  for k in range(fx.nk) for c in cs for v in vs])
    H = np.zeros((npair, npair), dtype=np.complex128)
    for k in range(fx.nk):
        Tc = np.einsum("csm,KCsm->KcCm", np.conj(pc_kq[k]), pc_kq)
        Tv = np.einsum("vsm,KVsm->KvVm", pv_k[k], np.conj(pv_k))
        Wg = fx.W0[qkk[k]]
        blk = np.einsum("KcCm,Kmn,KvVn->KcvCV", Tc, Wg, Tv, optimize=True)
        H[k * bs:(k + 1) * bs] = blk.transpose(1, 2, 0, 3, 4).reshape(bs, npair)
    return M, D, H / fx.nk


def B_of(M, zt, qt):
    A = (M @ zt) * np.sqrt(v_slab(qt))[None, :]
    return np.conj(A) @ A.T


def swap_eigs(D, Hdir, B):
    H = np.diag(D).astype(np.complex128) - Hdir + B / fx.nk
    H = 0.5 * (H + np.conj(H.T))
    return np.linalg.eigvalsh(H)[:4]


traj = {op.rung_label(mo, la): [] for mo, la in RUNGS}
anchors = {}
Brel = {}
for j in TJ:
    tq = time.time()
    t = j / 4.0
    qt = t * QDIR
    M, D, Hdir = swap_H_pieces(j)
    zts = zeta_interp(qt)
    for lbl, zt in zts.items():
        traj[lbl].append(swap_eigs(D, Hdir, B_of(M, zt, qt)))
    if j in (0, 4):                   # on-grid endpoints: stored-fit truth
        qi = 0 if j == 0 else 6
        zt_true = to_superset(fx.recon(qi), qt)
        B_true = B_of(M, zt_true, qt)
        ev_true = swap_eigs(D, Hdir, B_true)
        anchors[t] = {lbl: float(np.max(np.abs(
            swap_eigs(D, Hdir, B_of(M, zts[lbl], qt)) - ev_true)) * RY2MEV)
            for lbl in zts}
        Brel[t] = {lbl: relF(B_of(M, zts[lbl], qt), B_true) for lbl in zts}
        NPZ[f"htr__truth_eigs_t{t:.2f}"] = ev_true
    print(f"  t={t:.2f} done ({time.time()-tq:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
print("\n========== swap-H(t) lowest-4 eigenvalue trajectories (meV, rel to "
      "t=0 lowest) ==========")
tvals = np.array(TJ) / 4.0
for lbl in sorted(traj):
    E = np.array(traj[lbl]) * RY2MEV                 # (5, 4)
    E0 = E[0, 0]
    print(f"\n  ---- {lbl} ----")
    for s in range(4):
        y = E[:, s] - E0
        d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
        print(f"    ev{s}(t) [meV]: " + " ".join(f"{x:10.4f}" for x in y)
              + f"   max|d2| {d2.max():.4f} meV")
    print(f"    endpoint swap anchors [meV]: t=0 {anchors[0.0][lbl]:.4f}  "
          f"t=1 {anchors[1.0][lbl]:.4f}")
    print(f"    endpoint B relF (htransform-M): t=0 {Brel[0.0][lbl]:.3e}  "
          f"t=1 {Brel[1.0][lbl]:.3e}")
    NPZ[f"htr__{lbl}__eigs_meV"] = E
    NPZ[f"htr__{lbl}__anchor_meV"] = np.array([anchors[0.0][lbl],
                                               anchors[1.0][lbl]])
    NPZ[f"htr__{lbl}__Brel"] = np.array([Brel[0.0][lbl], Brel[1.0][lbl]])
NPZ["htr__tvals"] = tvals

np.savez(f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/"
         f"primer_response_study/offgrid_path_htr_results.npz", **NPZ)
print(f"\n[offgrid_path_htr] ALL DONE in {time.time()-t00:.0f}s")
