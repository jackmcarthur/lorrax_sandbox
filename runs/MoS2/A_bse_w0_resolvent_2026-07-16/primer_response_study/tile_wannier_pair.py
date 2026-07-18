"""tile_wannier_pair — OWNER AMENDMENT 2 (2026-07-17): the Wannier-theory
testable consequence on existing 6x6 data.  NO htransform anywhere.

Theory (writeup in arbitrary_q_bse.md sec 12): for gapped band groups an
analytic periodic Bloch gauge exists (Panati; Marzari-Vanderbilt) giving
exponentially localized Wannier functions; the momentum-q pair space is
spanned by Bloch sums of WANNIER PAIR PRODUCTS w_R w_R' — a canonical
q-ANALYTIC frame.  BBR 2013 decay bounds apply to the gapped objects
(density matrices / analytic filters), NOT to hard spectral cuts inside
C_q's gapless spectrum.  Consequence tested here: pair-level LR channel
matrix elements expressed in a SMOOTH (projection-gauge) band frame,
        Mtil(q)[(k,tc,tv), j] = sum_mu xtil_q[(k,tc,tv), mu]
                                        zeta_c,mu(q+G_j),
must vary at INGREDIENT level across q — while the same content in the
centroid dual frame (F-channels, moments, g0) is q-rough — because the only
q-dependence left is the analytic Bloch-gauge dependence of the orbitals,
not the lawless eigenframe or the LSQ dual basis.

Gauge construction (cheapest standard route, item 5 of the amendment):
Gamma-anchored PROJECTION gauge on the two gap-window trios separately
(top-3 valence, bottom-3 conduction): trials = the Gamma states of each
trio; A_k[n,t] = <psi_nk|T_t> by centroid quadrature on the stored
psi_full_y; Loewdin U_k = A (A^H A)^{-1/2} (3x3 unitary).  min singular
values of A_k printed — the honest nonsingularity diagnostic (the trios are
NOT isolated band groups; Panati's theorem licenses the full valence group,
the trio construction is the practical projection gauge, judged
empirically).  Row transform per k (left leg k-q):
    xtil[(k,tc,tv)] = sum_cv conj(Uc[k-q][c,tc]) Uv[k][v,tv] x[(k,c,v)].

D1  smoothness: adjacent-q rel diff + R-falloff of Mtil (smooth gauge) vs
    the SAME object in the raw stored gauge (U = I) vs the centroid-frame
    F-channels (tile_t1t2 benchmark ~0.47) — plus the G=0 pair-head column.
D2  scheme consequence (LOO, all 36 q): variant W = cleaned-SR tile A
    interp + PAIR-LEVEL LR kernel from interpolated Mtil rotated back to
    the stored gauge at the target (U's are per-k, q-independent, all
    on-grid: no off-grid info needed).  Controls: same with raw gauge.
    Gate: pair-level LR(own) == B_tile(x, V_LR^Gset) at every q.

Run: JID=<jid> ./proto1_run.sh python3 -u tile_wannier_pair.py
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel, build_Hdir, exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile

t00 = time.time()
NPZ = {}
RCSTAR = 1e-4
ALPHAS = [0.30, 0.45]
ASTAR = 0.30
# window size per leg.  NW=3 (the gap-window trio) SPLITS KRAMERS DOUBLETS
# at both edges (C3 metric-hygiene flag) — the window subspace is then
# k-DISCONTINUOUS and no smooth gauge exists for it even in principle.
# NW=2 is the Kramers-clean window (one doublet per leg).
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 3

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
kg = fx.kgrid
nv = fx.nv
vs = list(range(nv - NW, nv))
cs = list(range(nv, nv + NW))

# ---------------------------------------------------------------------------
# Gamma-anchored projection gauge on the two trios
# ---------------------------------------------------------------------------
def loewdin_gauges(bands):
    T = fx.psi[0][bands]                       # (NW, ns, n_mu) Gamma trials
    U, smin = [], []
    for k in range(fx.nk):
        A = np.einsum("nsm,tsm->nt", np.conj(fx.psi[k][bands]), T)
        u, s, vh = np.linalg.svd(A)
        U.append(u @ vh)                       # Loewdin unitary factor
        smin.append(s.min() / s.max())
    return np.array(U), np.array(smin)


Uv, sv = loewdin_gauges(vs)
Uc, sc = loewdin_gauges(cs)
print(f"[gauge] projection-gauge nonsingularity (smin/smax of A_k): "
      f"valence trio med {np.median(sv):.3f} min {sv.min():.3f} | "
      f"conduction trio med {np.median(sc):.3f} min {sc.min():.3f}")
NPZ["gauge_sv"] = sv
NPZ["gauge_sc"] = sc

kqs_all = {q: np.array([fx.kq_index(k, q)[0] for k in range(fx.nk)])
           for q in range(fx.nq)}


def rows_gauged(q, smooth=True):
    """Gap-window pair rows in the smooth (or raw) gauge, (nk*NW*NW, n_mu).
    Row block per k: m[c, v] -> Uc[k-q]^H m Uv[k]."""
    kqs = kqs_all[q]
    out = np.empty((fx.nk, NW, NW, fx.n_mu), dtype=np.complex128)
    for k in range(fx.nk):
        row = np.einsum("csm,vsm->cvm", np.conj(fx.psi[kqs[k]][cs]),
                        fx.psi[k][vs])
        if smooth:
            row = np.einsum("cC,cvm,vV->CVm", np.conj(Uc[kqs[k]]), row,
                            Uv[k], optimize=True)
        out[k] = row
    return out.reshape(-1, fx.n_mu)


def rotate_back(Mrows, q):
    """(nk*NW*NW, nch) smooth-gauge channel rows -> stored gauge at q:
    m = Uc[k-q] mtil Uv[k]^H, per k and channel."""
    kqs = kqs_all[q]
    M4 = Mrows.reshape(fx.nk, NW, NW, -1)
    out = np.empty_like(M4)
    for k in range(fx.nk):
        out[k] = np.einsum("cC,CVj,vV->cvj", Uc[kqs[k]], M4[k],
                           np.conj(Uv[k]), optimize=True)
    return out.reshape(Mrows.shape)


# ---------------------------------------------------------------------------
# pair-level LR channels Mtil(q) on gset(alpha), cleaned zeta
# ---------------------------------------------------------------------------
def zt_on_gset(q, GS):
    zt = ts.P(q, RCSTAR) @ fx.ZG[q]
    idx = ts.sphere_slot(q, GS)
    zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1), np.complex128)], 1)
    return zt_ext[:, idx]


Mtil, Mraw = {}, {}
for a in ALPHAS:
    GS = ts.gset(a)
    Mtil[a] = np.empty((fx.nq, fx.nk * NW * NW, GS.shape[1]),
                       dtype=np.complex128)
    Mraw[a] = np.empty_like(Mtil[a])
    for q in range(fx.nq):
        zg = zt_on_gset(q, GS)
        Mtil[a][q] = rows_gauged(q, True) @ zg
        Mraw[a][q] = rows_gauged(q, False) @ zg
print(f"[Mtil] pair-level LR channels built ({time.time()-t00:.0f}s)")

# gate: pair-level LR(own, rotated back) == B_tile(x, V_LR^Gset) at every q
gerr = []
for q in range(fx.nq):
    GS = ts.gset(ASTAR)
    v = ts.v_on_set(fx.qfr[q], GS, kind="slab_lr", alpha=ASTAR)
    m = rotate_back(Mtil[ASTAR][q], q)
    A = m * np.sqrt(v)[None, :]
    B_pair = np.conj(A) @ A.T
    x = fx.gap_window_pairs(q, NW, NW)
    zg = zt_on_gset(q, GS)
    Aref = (x @ zg) * np.sqrt(v)[None, :]
    gerr.append(relF(B_pair, np.conj(Aref) @ Aref.T))
print(f"  [gate] pairLR_own_rotateback_vs_direct max {max(gerr):.3e}"
      + ("  OK" if max(gerr) < 1e-10 else "  ** FAIL **"))
assert max(gerr) < 1e-10

# ---------------------------------------------------------------------------
# D1 — smoothness of the pair-level channels, smooth vs raw gauge
# ---------------------------------------------------------------------------
print("\n[D1] adjacent-q (+x) rel diff of pair-level LR channels")
pairs = [(q, fx.k_lookup[tuple((fx.k_int[q] + np.array([1, 0, 0])) % kg)])
         for q in range(fx.nq)]


def adjdiff(F):
    out = [np.linalg.norm(F[a] - F[b]) / np.linalg.norm(F[a])
           for a, b in pairs]
    return float(np.median(out)), float(np.max(out))


GS0 = ts.gset(ASTAR)
g0col = np.where(np.all(GS0 == 0, axis=0))[0]
for tag, M in (("smooth", Mtil), ("raw-gauge", Mraw)):
    med, mx = adjdiff(M[ASTAR].reshape(fx.nq, -1))
    medh, mxh = adjdiff(M[ASTAR][:, :, g0col].reshape(fx.nq, -1))
    print(f"  [D1] {tag:>9s} Mtil(a*): med {med:.3f} max {mx:.3f} | "
          f"G=0 pair-head column: med {medh:.3f} max {mxh:.3f}")
    NPZ[f"D1_adj_{tag}"] = np.array([med, mx, medh, mxh])
Rw = fx.Rw
dR = np.sqrt(np.einsum("ri,ij,rj->r", Rw, fx.adot, Rw))
shells = np.unique(np.round(dR, 6))
EqR = np.exp(2j * np.pi * (fx.qfr @ Rw.T)) / fx.nq
CRn = np.array([np.linalg.norm(fx.C_R_full[np.isclose(dR, s)])
                for s in shells])
CRn = CRn / CRn[0]
print("  [D1] R shells (bohr): " + " ".join(f"{s:.2f}" for s in shells[:8]))
print("  [D1] C_R shells     : " + " ".join(f"{v:.2e}" for v in CRn[:8]))
for tag, M in (("smooth", Mtil), ("raw-gauge", Mraw)):
    FR = EqR.T @ M[ASTAR].reshape(fx.nq, -1)
    sn = np.array([np.linalg.norm(FR[np.isclose(dR, s)]) for s in shells])
    sn = sn / sn[0]
    print(f"  [D1] {tag:>9s} Mtil R-falloff: "
          + " ".join(f"{v:.2e}" for v in sn[:8]))
    NPZ[f"D1_Rfall_{tag}"] = sn
NPZ["D1_Rfall_CR"] = CRn

# ---------------------------------------------------------------------------
# D2 — LOO variant W: cleaned-SR tile interp + pair-level LR from Mtil
# ---------------------------------------------------------------------------
print(f"\n[D2] LOO variant W ({fx.nq} targets, nR7)")
R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]
VLRc = {a: ts.VLR_exact_c(RCSTAR, a) for a in ALPHAS}
EXC = {f"W_pairLR_a{ASTAR}"}
res = {}
for q0 in range(fx.nq):
    train = [q for q in range(fx.nq) if q != q0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
    x = fx.gap_window_pairs(q0, NW, NW)
    B_true = B_tile(x, ts.V_ref[q0])
    D_diag, Hdir = build_Hdir(fx, q0, NW, NW)
    ev_true = exciton_evs(fx, D_diag, Hdir, B_true)
    preds = {}
    for a in ALPHAS:
        SRi = np.tensordot(w, ts.Vc(RCSTAR)[train] - VLRc[a][train],
                           axes=(0, 0))
        B_SR = B_tile(x, SRi)
        v = ts.v_on_set(fx.qfr[q0], ts.gset(a), kind="slab_lr", alpha=a)
        for tag, M in (("W_pairLR", Mtil), ("Wraw_pairLR", Mraw)):
            Mi = np.tensordot(w, M[a][train], axes=(0, 0))
            m = rotate_back(Mi, q0) if tag == "W_pairLR" else Mi
            A = m * np.sqrt(v)[None, :]
            preds[f"{tag}_a{a}"] = ("B", B_SR + np.conj(A) @ A.T)
    for lbl, (_, Bp) in preds.items():
        met = {"B": relF(Bp, B_true), "Bdec": top_decile_rel(Bp, B_true)}
        if lbl in EXC:
            met["exc_meV"] = float(np.max(np.abs(
                exciton_evs(fx, D_diag, Hdir, Bp) - ev_true)) * RY2MEV)
        res.setdefault(lbl, {})[q0] = met
    print(f"  q0={q0} done", flush=True)

print(f"\n  ========== Wannier-pair LOO: median / max over {fx.nq} "
      f"targets ==========")
print(f"    {'label':<28s} {'B med':>10s} {'B max':>10s} {'Bdec md':>9s} "
      f"{'exc med':>8s} {'exc max':>8s}")
for lbl in sorted(res):
    rows = res[lbl]
    Bm = [rows[q]["B"] for q in rows]
    em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
    em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
            else "      --       --")
    print(f"    {lbl:<28s} {np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
          f"{np.median([rows[q]['Bdec'] for q in rows]):>9.2e} {em_s}")
for lbl in res:
    qs = sorted(res[lbl])
    NPZ[f"W__{lbl}__q0"] = np.array(qs)
    for key in ("B", "Bdec", "exc_meV"):
        NPZ[f"W__{lbl}__{key}"] = np.array(
            [res[lbl][q].get(key, np.nan) for q in qs], dtype=float)

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "tile_wannier_pair_results.npz", **NPZ)
print(f"\n[tile_wannier_pair] ALL DONE in {time.time()-t00:.0f}s")
