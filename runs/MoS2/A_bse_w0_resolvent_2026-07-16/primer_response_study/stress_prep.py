"""stress_prep — shared machinery for the sec-14 STRESS/EDGE campaign on the
winning arbitrary-Q pipeline (Tikhonov-gauge clean + Gaussian split + SR tile
stencil + b26p global K-ball LR fit; arbitrary_q_bse.md secs 12-13).

Factors out of lr_basis_ladder.py the Tikhonov-gauge builder (verbatim
arithmetic; the ladder's --tik block), plus q-class labeling for edge
statistics and the path smoothness stat. Nothing new is derived here — every
routine is a lift of logged, gate-anchored code so stress results stay
bit-comparable with the sec-13 anchors.

READ-ONLY on all fixtures and on sources/lorrax_A. No Z_r object anywhere.
"""
import numpy as np

from proto1_prep import relF
from lr_prep import LRSamples


def build_tik(fx, ts, eps_rel):
    """Tikhonov cleaning operators S_eps = R g_eps(lam) R^H per coarse q
    (g = lam^2/(lam^2 + (eps_rel*lam0)^2)) — lr_basis_ladder TIK block."""
    Stik = []
    for q in range(fx.nq):
        lam, R = ts.eig[q]
        g = lam ** 2 / (lam ** 2 + (eps_rel * lam[0]) ** 2)
        Stik.append((R * g[None, :]) @ R.conj().T)
    return Stik


def tik_objects(fx, ts, Stik, alpha):
    """(Vc, VLRc, lr) in the Tikhonov gauge at window alpha; lr.Fch is the
    phase-factored cleaned channel array. Verbatim lr_basis_ladder TIK
    arithmetic; gate the F own-rebuild in the caller."""
    Vc = np.stack([np.conj(Stik[q]) @ ts.V_ref[q] @ np.conj(Stik[q])
                   for q in range(fx.nq)])
    VLRc = np.stack([np.conj(Stik[q])
                     @ fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=alpha)
                     @ np.conj(Stik[q]) for q in range(fx.nq)])
    lr = LRSamples(ts, None, alpha)
    for q in range(fx.nq):
        zt = Stik[q] @ fx.ZG[q]
        idx = ts.sphere_slot(q, lr.GS)
        zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1),
                                              np.complex128)], 1)
        qG = fx.qfr[q][None, :] + lr.GS.T.astype(np.float64)
        ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
        lr.Fch[q] = ph * zt_ext[:, idx]
    return Vc, VLRc, lr


def gate_F_rebuild(ts, fx, lr, VLRc, alpha, tol=1e-6, tag=""):
    r = [relF(ts.V_from_F(lr.Fch[q], fx.qfr[q], lr.GS, alpha), VLRc[q])
         for q in range(fx.nq)]
    ok = np.max(r) < tol
    print(f"  [gate] F_own_rebuild_vs_cleaned_VLR{tag} max {np.max(r):.3e}"
          + ("  OK" if ok else "  ** FAIL **"))
    assert ok
    return float(np.max(r))


def q_class(fx, q):
    """Edge class of a coarse 6x6 q (wrapped frac): Gamma / M (zone-boundary
    midpoint, a component at 1/2) / K (hexagonal corner, (1/3,1/3)-type) /
    edge (on the 1/2 boundary line) / interior."""
    f = fx.qfr[q]
    n = np.abs(f[:2])
    if np.max(n) < 1e-9:
        return "Gamma"
    half = np.isclose(n, 0.5, atol=1e-9)
    third = np.isclose(n, 1.0 / 3.0, atol=1e-9)
    if half.all() or (half.any() and np.isclose(min(n), 0.0, atol=1e-9)):
        return "M"
    if third.all():
        return "K"
    if half.any():
        return "edge"
    return "interior"


def smooth_stat(y):
    """max |second difference| / range — tile_path.py convention."""
    y = np.asarray(y, dtype=float)
    d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
    return float(d2.max() / max(y.max() - y.min(), 1e-300))
