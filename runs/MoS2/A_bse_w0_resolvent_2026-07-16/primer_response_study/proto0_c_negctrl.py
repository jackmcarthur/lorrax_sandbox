"""proto0 script C — negative control (Si 4x4x4) + aux LOO (MoS2 4x4).

No zeta on disk for either fixture, so the interpolated object is the
whitened DISK tile  Vc_q = S R^H V_disk R S  (the alpha->infinity SR body:
disk V has the head channel zeroed / handled rank-1).  Truth = disk V.
This is the same transported-target-frame machinery as C1 with the SR/LR
split degenerate (LR == 0 relative to the stored truth), which is exactly
the right form for a negative control on the transport+frame claim:
Si ingredients fail (C_q LOO 33% in Sec 3.5) -> C1 SHOULD fail here; a pass
is an overfitting alarm.  Band overlaps: centroid quadrature (no WFN wired;
the 3x3 run measured WFN-vs-centroid overlap deltas ~ its report).
"""
import numpy as np
import jax
import jax.numpy as jnp
import time

import prep
from prep import (load_fixture, relF, eigh_frame, polar_unitary,
                  gap_window_rows, B_from_tile, B_metrics, w_nn_avg,
                  w_fourier)

flags = prep.load_flags()
CONJ_LEFT, T_CONJ = flags["CONJ_LEFT"], flags["T_CONJ"]
assert flags.get("FLAVOR") == "response"
print(f"[negctrl] conventions: {flags}")
OUT = {}


def run_fixture(name, target_subset=None, weight_schemes=("nn",),
                ranks=(1.0, 0.5, 0.25)):
    fx = load_fixture(name)
    print(f"\n===== {name}: nk={fx.nk} nb={fx.nb} nmu={fx.nmu} nocc={fx.nocc} "
          f"kgrid={fx.kgrid}")
    nmu = fx.nmu
    psi_j = jnp.asarray(fx.psi)

    # C via row-Grams on GPU (the fit metric Gram; == falloff formula)
    t0 = time.time()
    C_all = np.stack([prep.gram_C_from_rows(fx, q, CONJ_LEFT)
                      for q in range(fx.nq)])
    print(f"[{name}] C built ({time.time()-t0:.0f}s)")
    R_by_q, S_by_q = {}, {}
    conds = []
    for q in range(fx.nq):
        R, lam = eigh_frame(C_all[q])
        R_by_q[q] = R
        S_by_q[q] = np.sqrt(np.maximum(lam, 1e-300))
        conds.append(lam[0] / max(lam[-1], 1e-300))
    print(f"[{name}] cond(C): med={np.median(conds):.2e} max={np.max(conds):.2e}")

    # whitened disk tiles — RESPONSE flavor: Vc_R = S R^H conj(V_disk) R S
    # (V_disk is in the make-convention V[mu,nu]=sum conj(zt) v zt; its
    # "natural" PSD counterpart is the conjugate), dressed as
    # B = conj( l Vc_R l^H ), l = M R S^-1 (rows of L, norm <= 1).
    Vc_by_q = {}
    for q in range(fx.nq):
        R, S = R_by_q[q], S_by_q[q]
        Vc = (S[:, None] * (R.conj().T @ np.conj(fx.Vdisk[q]) @ R)) * S[None, :]
        Vc_by_q[q] = 0.5 * (Vc + Vc.conj().T)

    Mrows = {q: gap_window_rows(fx, q) for q in range(fx.nq)}
    B_true = {q: B_from_tile(Mrows[q], fx.Vdisk[q]) for q in range(fx.nq)}

    def l_rows(Mr, R, S, r):
        return (Mr @ R[:, :r]) / S[:r][None, :]

    # null test: no-interp reconstruction through the whitened frame
    q = 1
    lM = l_rows(Mrows[q], R_by_q[q], S_by_q[q], nmu)
    nul = relF(np.conj(lM @ Vc_by_q[q] @ lM.conj().T), B_true[q])
    print(f"[{name}] null (no-interp conj(l Vc l^H) vs B_true): {nul:.2e}")

    # centroid-quadrature overlaps + transport
    O_cache = {}
    P = fx.psi.reshape(fx.nk, fx.nb, fx.ns * fx.nmu)
    def overlap(ka, kb):
        key = (ka, kb)
        if key not in O_cache:
            O_cache[key] = np.conj(P[ka]) @ P[kb].T
        return O_cache[key]

    @jax.jit
    def _edge_k(psiA, psiB, psiP, t):
        if CONJ_LEFT:
            Xq = jnp.einsum("asm,bsm->abm", jnp.conj(psiA), psiB)
            Xp = jnp.einsum("asm,bsm->abm", jnp.conj(psiP), psiB)
        else:
            Xq = jnp.einsum("asm,bsm->abm", psiA, jnp.conj(psiB))
            Xp = jnp.einsum("asm,bsm->abm", psiP, jnp.conj(psiB))
        tt = jnp.conj(t) if T_CONJ else t
        Xr = jnp.einsum("mM,Mnv->mnv", tt, Xp)
        return jnp.einsum("abm,abn->mn", jnp.conj(Xq), Xr)

    H_cache = {}
    def edge_H(q, qp):
        key = (q, qp)
        if key not in H_cache:
            H = jnp.zeros((nmu, nmu), dtype=jnp.complex128)
            for k in range(fx.nk):
                ka, kb = int(fx.kmq_idx[q, k]), int(fx.kmq_idx[qp, k])
                t, _ = polar_unitary(overlap(ka, kb))
                H = H + _edge_k(psi_j[ka], psi_j[k], psi_j[kb], jnp.asarray(t))
            H_cache[key] = np.asarray(H)
        return H_cache[key]

    def edge_T(q, qp, r):
        H = edge_H(q, qp)
        M = (R_by_q[q][:, :r].conj().T @ H @ R_by_q[qp][:, :r]) \
            / S_by_q[q][:r][:, None] / S_by_q[qp][:r][None, :]
        U, cos, Vh = np.linalg.svd(M)
        return U @ Vh, cos

    targets = (list(range(fx.nq)) if target_subset is None
               else sorted(target_subset))
    rr = [int(round(f * nmu)) for f in ranks]
    res = {}
    cos_rec = []
    for wname in weight_schemes:
        for r in rr:
            tots, srs = [], []
            for q0 in targets:
                if wname == "nn":
                    wts = w_nn_avg(fx, q0)
                else:
                    tr = [q for q in range(fx.nq) if q != q0]
                    wv = w_fourier(fx, fx.qfr[q0], fx.qfr[tr], 7)
                    wts = {qi: wv[j] for j, qi in enumerate(tr)}
                Vc = np.zeros((r, r), dtype=np.complex128)
                for qi, w in wts.items():
                    T, cos = edge_T(q0, qi, r)
                    if r == rr[0]:
                        cos_rec.append(cos)
                    Vc = Vc + w * (T @ Vc_by_q[qi][:r, :r] @ T.conj().T)
                Vc = 0.5 * (Vc + Vc.conj().T)
                lM = l_rows(Mrows[q0], R_by_q[q0], S_by_q[q0], r)
                Bp = np.conj(lM @ Vc @ lM.conj().T)
                tots.append(B_metrics(Bp, B_true[q0])[0])
            res[(wname, r)] = tots
            print(f"[{name}] w={wname} r={r}: B relF med={np.median(tots):.3e} "
                  f"max={np.max(tots):.3e}")
    cos_rec = np.concatenate([c for c in cos_rec]) if cos_rec else np.array([])
    if cos_rec.size:
        print(f"[{name}] principal cosines: min={cos_rec.min():.4f} "
              f"med={np.median(cos_rec):.4f} frac<0.9={np.mean(cos_rec < 0.9):.3f} "
              f"frac<0.5={np.mean(cos_rec < 0.5):.3f}")
    OUT[name] = {"res": {str(k): v for k, v in res.items()},
                 "cond": conds, "null": nul,
                 "cos_min": float(cos_rec.min()) if cos_rec.size else None,
                 "cos_med": float(np.median(cos_rec)) if cos_rec.size else None,
                 "cos_frac_09": float(np.mean(cos_rec < 0.9)) if cos_rec.size else None}
    return fx


# Si negative control: 16-target subset (every 4th), nn6 stencil
run_fixture("si_4x4x4", target_subset=list(range(0, 64, 4)),
            weight_schemes=("nn",), ranks=(1.0, 0.5, 0.25))
# MoS2 4x4 aux: all 16 targets, nn4 + f7
run_fixture("mos2_4x4", target_subset=None, weight_schemes=("nn", "f7"),
            ranks=(1.0, 0.5, 0.25))

import json
with open(f"{prep.STUDY}/proto0_c_results.json", "w") as f:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    json.dump(clean(OUT), f)
print("\n[proto0_c] DONE")
