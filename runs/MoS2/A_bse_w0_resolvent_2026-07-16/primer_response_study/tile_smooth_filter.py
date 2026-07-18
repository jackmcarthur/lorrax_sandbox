"""tile_smooth_filter — OWNER AMENDMENT (2026-07-17): smooth spectral
filters vs hard rank cuts in the tile-cleaning, + two operator-theory
checks on the "frame chaos" claim.

Theory being tested (Davis-Kahan; Benzi-Boito-Razouk-class analytic-filter
bounds): C_q's spectrum is GAPLESS, so hard-cut spectral projectors P_r(q)
are lawless — the subspace rotation between adjacent q is unbounded as the
local gap -> 0, and the C2 campaign's "random floor" subspace overlaps may
be exactly this artifact, not physics.  Analytic filters f_eps obey
||f_eps(C_q) - f_eps(C_q')|| <= L_eps ||C_q - C_q'|| with L_eps set by the
filter's resolvent width (for the cleaning weight g_eps(lam) =
lam^2/(lam^2+eps^2), L ~ 0.65/eps), i.e. PROVABLE q-smoothness given the
measured-small ingredient variation ||DeltaC||.

CHECK A  matrix-function continuity: ||Delta f_eps(C)||_F / ||Delta C||_F
         and ||Delta g_eps(C)||_F / ||Delta C||_F across all adjacent-q
         (+x-hat) 6x6 pairs, eps swept across the inert window; contrast
         with the hard-cut projector distance ||Delta P_r||_F / sqrt(2r).
         Plus the tile-level version: ||Delta V_clean|| / ||Delta V_ref||
         for hard vs Tikhonov cleaning.
CHECK B  re-audit of the C2 subspace-overlap probe: principal cosines of
         the top-m PLAIN eigen-subspaces (no whitening, no transport, no
         zeta — C_q is psi-level and label-free) between adjacent q, for
         m = 10..480, with the Davis-Kahan gap lam_m - lam_{m+1} vs
         ||Delta C|| printed.  Gap-separated top blocks CANNOT sit at the
         random floor if C_q is smooth.
T1-TIK   the cleaned-tile LOO ladder rerun with Tikhonov cleaning
         S_eps = R g_eps(lam) R^H (Z-free: cleaned zeta = S zeta_stored
         because f_eps(C) Z = S zeta_stored), head-to-head with the hard
         cut, including the winner F-composition (clean SR + F-channel LR).

Run: JID=<jid> ./proto1_run.sh python3 -u tile_smooth_filter.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel, build_Hdir, exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile

t00 = time.time()
NPZ = {}
RCSTAR = 1e-4
ASTAR = 0.30
EPSREL = [1e-3, 1e-4, 1e-5, 1e-6]
EPSSTAR = 1e-4

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
kg = fx.kgrid
pairs = [(q, fx.k_lookup[tuple((fx.k_int[q] + np.array([1, 0, 0])) % kg)])
         for q in range(fx.nq)]


def gmat(q, eps_rel):
    """Tikhonov cleaning weight S = R diag(lam^2/(lam^2+eps^2)) R^H."""
    lam, R = ts.eig[q]
    e = eps_rel * lam[0]
    g = lam * lam / (lam * lam + e * e)
    return (R * g[None, :]) @ R.conj().T


def fmat(q, eps_rel):
    """Regularized inverse f_eps(C) = R diag(lam/(lam^2+eps^2)) R^H."""
    lam, R = ts.eig[q]
    e = eps_rel * lam[0]
    f = lam / (lam * lam + e * e)
    return (R * f[None, :]) @ R.conj().T


# ===========================================================================
# CHECK A — matrix-function continuity vs hard-cut projector distance
# ===========================================================================
print("\n[checkA] adjacent-q (+x) matrix-function continuity, 36 pairs")
dC = np.array([np.linalg.norm(C_q[a] - C_q[b]) for a, b in pairs])
nC = np.array([np.linalg.norm(C_q[a]) for a, b in pairs])
print(f"  ||DeltaC||_F/||C||_F: med {np.median(dC/nC):.3e} "
      f"max {np.max(dC/nC):.3e}")
NPZ["A_dC_rel"] = dC / nC
for er in EPSREL:
    rf, rg = [], []
    for a, b in pairs:
        rf.append(np.linalg.norm(fmat(a, er) - fmat(b, er))
                  / np.linalg.norm(C_q[a] - C_q[b]))
        rg.append(np.linalg.norm(gmat(a, er) - gmat(b, er))
                  / np.linalg.norm(C_q[a] - C_q[b]))
    lam0 = np.median([ts.eig[q][0][0] for q in range(fx.nq)])
    print(f"  eps_rel={er:.0e}: ||Df_eps||/||DC|| med {np.median(rf):.3e} "
          f"(x eps_abs = {np.median(rf)*er*lam0:.3f}); "
          f"||Dg_eps||/||DC|| med {np.median(rg):.3e} "
          f"(x eps_abs = {np.median(rg)*er*lam0:.3f}; Lipschitz bound 0.65)")
    NPZ[f"A_f_ratio_{er:.0e}"] = np.array(rf)
    NPZ[f"A_g_ratio_{er:.0e}"] = np.array(rg)
for rc in (1e-3, 1e-4, 1e-5):
    dP = []
    for a, b in pairs:
        Pa, Pb = ts.P(a, rc), ts.P(b, rc)
        r = ts.rank(a, rc)
        dP.append(np.linalg.norm(Pa - Pb) / np.sqrt(2 * r))
    print(f"  hard-cut rc={rc:.0e}: ||DeltaP||_F/sqrt(2r) med "
          f"{np.median(dP):.3f} max {np.max(dP):.3f}  (1 = orthogonal, "
          f"random floor ~ sqrt(1-r/n) = "
          f"{np.sqrt(1-ts.rank(0, rc)/fx.n_mu):.3f})")
    NPZ[f"A_dP_{rc:.0e}"] = np.array(dP)
# tile-level smoothness: Delta V_clean / Delta V_ref, hard vs tik
dVr = np.array([np.linalg.norm(ts.V_ref[a] - ts.V_ref[b]) for a, b in pairs])
Vc_hard = ts.Vc(RCSTAR)
Stik = [gmat(q, EPSSTAR) for q in range(fx.nq)]
Vc_tik = np.stack([np.conj(Stik[q]) @ ts.V_ref[q] @ np.conj(Stik[q])
                   for q in range(fx.nq)])
for tag, Vc in (("hard rc1e-4", Vc_hard), ("tik eps1e-4", Vc_tik)):
    dV = np.array([np.linalg.norm(Vc[a] - Vc[b]) for a, b in pairs])
    print(f"  tile smoothness {tag:<12s}: ||DVc||/||DVref|| med "
          f"{np.median(dV/dVr):.3f} max {np.max(dV/dVr):.3f}")
    NPZ[f"A_dV_{tag.split()[0]}"] = dV / dVr

# ===========================================================================
# CHECK B — plain top-m eigen-subspace overlaps (no whitening/transport)
# ===========================================================================
print("\n[checkB] adjacent-q top-m plain subspace principal cosines")
lam_med = np.median([ts.eig[q][0] for q in range(fx.nq)], axis=0)
dC_med = float(np.median(dC))
print(f"  med ||DeltaC||_F = {dC_med:.4e}; lam_0 = {lam_med[0]:.4e}")
for m in (10, 25, 50, 100, 200, 266, 480):
    cos_med, cos_min, aff = [], [], []
    for a, b in pairs:
        Ra = ts.eig[a][1][:, :m]
        Rb = ts.eig[b][1][:, :m]
        s = np.linalg.svd(Ra.conj().T @ Rb, compute_uv=False)
        cos_med.append(np.median(s))
        cos_min.append(s.min())
        aff.append(np.sqrt(np.mean(s ** 2)))
    gap = float(lam_med[m - 1] - lam_med[m]) if m < fx.n_mu else 0.0
    print(f"  m={m:>3d}: cos med {np.median(cos_med):.4f} "
          f"min {np.median(cos_min):.4f} affinity {np.median(aff):.4f} | "
          f"DK gap lam_m-lam_m+1 = {gap:.3e} "
          f"(||DC||/gap = {dC_med/max(gap,1e-300):.1f}) "
          f"random floor ~ {np.sqrt(m/fx.n_mu):.3f}")
    NPZ[f"B_cosmed_m{m}"] = np.array(cos_med)
    NPZ[f"B_aff_m{m}"] = np.array(aff)

# ===========================================================================
# T1-TIK — LOO ladder, smooth vs hard cleaning
# ===========================================================================
print("\n[T1tik] LOO ladder (36 targets, nR7): Tikhonov vs hard cleaning")
R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]
# caches: tik-cleaned tiles per eps; tik SR/F at eps* for the F-composition
Vc_tik_e = {EPSSTAR: Vc_tik}
for er in EPSREL:
    if er not in Vc_tik_e:
        Vc_tik_e[er] = np.stack([
            (lambda S: np.conj(S) @ ts.V_ref[q] @ np.conj(S))(gmat(q, er))
            for q in range(fx.nq)])
GS = ts.gset(ASTAR)
VLR_tik = np.stack([np.conj(Stik[q])
                    @ fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=ASTAR)
                    @ np.conj(Stik[q]) for q in range(fx.nq)])
F_tik = np.empty((fx.nq, fx.n_mu, GS.shape[1]), dtype=np.complex128)
for q in range(fx.nq):
    zt = Stik[q] @ fx.ZG[q]
    idx = ts.sphere_slot(q, GS)
    zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1), np.complex128)], 1)
    qG = fx.qfr[q][None, :] + GS.T.astype(np.float64)
    F_tik[q] = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T)) * zt_ext[:, idx]
VLR_hard = ts.VLR_exact_c(RCSTAR, ASTAR)
F_hard = ts.F_channels(RCSTAR, ASTAR)
print(f"  [T1tik] caches built ({time.time()-t00:.0f}s)")

EXC = {f"Btik_eps{EPSSTAR:.0e}", f"Ftik_eps{EPSSTAR:.0e}_a{ASTAR}",
       f"Fhard_rc{RCSTAR:.0e}_a{ASTAR}"}
res = {}
for q0 in range(fx.nq):
    train = [q for q in range(fx.nq) if q != q0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_tile(x, ts.V_ref[q0])
    D_diag, Hdir = build_Hdir(fx, q0)
    ev_true = exciton_evs(fx, D_diag, Hdir, B_true)
    preds = {}
    for er in EPSREL:
        preds[f"Btik_eps{er:.0e}"] = np.tensordot(w, Vc_tik_e[er][train],
                                                  axes=(0, 0))
    preds[f"Bhard_rc{RCSTAR:.0e}"] = np.tensordot(w, Vc_hard[train],
                                                  axes=(0, 0))
    Fi_t = np.tensordot(w, F_tik[train], axes=(0, 0))
    preds[f"Ftik_eps{EPSSTAR:.0e}_a{ASTAR}"] = \
        np.tensordot(w, Vc_tik[train] - VLR_tik[train], axes=(0, 0)) \
        + ts.V_from_F(Fi_t, fx.qfr[q0], GS, ASTAR)
    Fi_h = np.tensordot(w, F_hard[train], axes=(0, 0))
    preds[f"Fhard_rc{RCSTAR:.0e}_a{ASTAR}"] = \
        np.tensordot(w, Vc_hard[train] - VLR_hard[train], axes=(0, 0)) \
        + ts.V_from_F(Fi_h, fx.qfr[q0], GS, ASTAR)
    for lbl, Vp in preds.items():
        Bp = B_tile(x, Vp)
        met = {"B": relF(Bp, B_true), "Bdec": top_decile_rel(Bp, B_true)}
        if lbl in EXC:
            met["exc_meV"] = float(np.max(np.abs(
                exciton_evs(fx, D_diag, Hdir, Bp) - ev_true)) * RY2MEV)
        res.setdefault(lbl, {})[q0] = met
    print(f"  q0={q0} done", flush=True)

print(f"\n  ========== T1tik LOO: median / max over {fx.nq} targets "
      f"==========")
print(f"    {'label':<36s} {'B med':>10s} {'B max':>10s} {'Bdec md':>9s} "
      f"{'exc med':>8s} {'exc max':>8s}")
for lbl in sorted(res):
    rows = res[lbl]
    Bm = [rows[q]["B"] for q in rows]
    em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
    em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
            else "      --       --")
    print(f"    {lbl:<36s} {np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
          f"{np.median([rows[q]['Bdec'] for q in rows]):>9.2e} {em_s}")
for lbl in res:
    qs = sorted(res[lbl])
    NPZ[f"tik__{lbl}__q0"] = np.array(qs)
    for key in ("B", "Bdec", "exc_meV"):
        NPZ[f"tik__{lbl}__{key}"] = np.array(
            [res[lbl][q].get(key, np.nan) for q in qs], dtype=float)

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "tile_smooth_filter_results.npz", **NPZ)
print(f"\n[tile_smooth_filter] ALL DONE in {time.time()-t00:.0f}s")
