#!/usr/bin/env python3
"""Where does symmetry break in Si BSE?  v2 — uses the CODE's own SymMaps so the
k-order, sym convention (mtrx^T applied as sym_mats_k@k), the 96-row TRS-augmented
op set, and the irr_idx_k star grouping EXACTLY match what unfold_psi used to build
psi_full_y.  (v1 guessed the convention and mis-grouped stars → 3 eV spurious spread.)

Rungs, each answered with NUMBERS:
  1. enk_full covariance: spread of e_nk within each code-defined k-star (group by
     sym.irr_idx_k), per band.  Also transition-energy (e_c-e_v) diagonal spread.
  2. term decomposition: lambda_i = <i|D|i> + <i|Kx|i> - <i|Kd|i> exactly; the
     spread of each piece across a near-degenerate manifold attributes the split.
  3. eigenvector k-weights: sum_k |A_cvk|^2 per k-star in the split manifold.
  4. gauge-invariant operator covariance: block Frobenius ||B_{Pk,Pk'}||_F vs
     ||B_{k,k'}||_F for each of the 96 ops (Frobenius is invariant under the
     left/right unitary band-rotation AND under TRS conjugation), SEPARATELY for
     D / Kx / Kd.  Split spatial(symmorphic) / spatial(nonsymmorphic) / TRS.

Usage (inside srun+shifter): python3 rung_probe2.py <arm> <nv> <nc>
"""
import sys, os, json
import numpy as np
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
sys.path.insert(0, os.path.join(LROOT, "tests"))
import jax
jax.config.update("jax_enable_x64", True)
from bse import bse_io
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")


def build_terms(data):
    psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
    eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
    V_q0 = np.asarray(data["V_q0"]); W_q = np.asarray(data["W_q"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    grid = (nkx, nky, nkz); nk = nkx*nky*nkz
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    Dcvk = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0, optimize=True)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M, optimize=True) / nk
    Wflat = W_q.reshape(nmu, nmu, nk)
    ck = np.array(np.unravel_index(np.arange(nk), grid)).T
    qidx = np.empty((nk, nk), dtype=int)
    for k in range(nk):
        q = (ck[k][None, :] - ck) % np.array(grid)
        qidx[k] = np.ravel_multi_index(q.T, grid)
    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    for k in range(nk):
        Wq_k = np.transpose(Wflat[:, :, qidx[k]], (2, 0, 1))
        Pc = np.einsum("csm,KCsm->KcCm", np.conj(psi_c[k]), psi_c, optimize=True)
        Pv = np.einsum("vsn,KVsn->KvVn", psi_v[k], np.conj(psi_v), optimize=True)
        tmp = np.einsum("KcCm,Kmn->KcCn", Pc, Wq_k, optimize=True)
        Kd[:, :, k, :, :, :] = np.einsum("KcCn,KvVn->cvCVK", tmp, Pv, optimize=True) / nk
    return dict(Dcvk=Dcvk, Kx=Kx.reshape(nc*nv*nk, -1), Kd=Kd.reshape(nc*nv*nk, -1),
                nc=nc, nv=nv, nk=nk, grid=grid, eps_c=eps_c, eps_v=eps_v)


def wrap01(x):
    y = np.mod(x, 1.0)
    y[np.abs(y - 1.0) < 1e-6] = 0.0
    return y


def build_kperms(sym):
    """P[s, k] = full-BZ index of wrap(sym_mats_k[s] @ unfolded_kpts[k]).  96 ops."""
    kpts = np.asarray(sym.unfolded_kpts, dtype=np.float64)   # (nk,3)
    nk = kpts.shape[0]
    S = np.asarray(sym.sym_mats_k)                            # (96,3,3)
    key = {tuple(np.round(wrap01(kpts[k]), 6)): k for k in range(nk)}
    P = np.full((S.shape[0], nk), -1, dtype=np.int64)
    for s in range(S.shape[0]):
        for k in range(nk):
            kp = wrap01(S[s] @ kpts[k])
            P[s, k] = key[tuple(np.round(kp, 6))]
    assert (P >= 0).all(), "kperm: unmatched"
    for s in range(S.shape[0]):
        assert len(set(P[s].tolist())) == nk, f"kperm op {s} not a bijection"
    return P


def cluster(ev_eV, gap_meV=1.0):
    gap = gap_meV*1e-3
    groups, cur = [], [0]
    for i in range(1, len(ev_eV)):
        if ev_eV[i]-ev_eV[i-1] > gap:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "old"
    nv = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    print(f"[{arm} {nv}v{nc}c] restart={os.path.basename(restart)}", flush=True)

    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    tnp = np.asarray(wfn.translations[:ntran], dtype=np.float64)
    irr = np.asarray(sym.irr_idx_k)          # (nk,)
    symidx = np.asarray(sym.sym_idx_k)       # (nk,) op used to build each full k
    P = build_kperms(sym)                     # (96, nk)
    nk = P.shape[1]; nops = P.shape[0]
    stars = [sorted(np.where(irr == u)[0].tolist()) for u in sorted(set(irr.tolist()))]
    print(f"  ntran={ntran} nops(inc TRS)={nops} nk={nk} #stars={len(stars)} "
          f"star sizes={[len(s) for s in stars]}", flush=True)
    print(f"  sym_idx_k used to build full-BZ ψ (unique)={sorted(set(symidx.tolist()))}",
          flush=True)

    data = bse_io._load_ring_subset(restart, n_val=nv, n_cond=nc, px=1, py=1,
                                    input_file=inp)
    t = build_terms(data)
    out = {"arm": arm, "window": f"{nv}v{nc}c", "ntran": ntran, "nops": nops,
           "star_sizes": [len(s) for s in stars]}

    # ---- RUNG 1: enk_full covariance within code-defined k-stars ----
    eps_c = t["eps_c"]; eps_v = t["eps_v"]     # (nk,nc),(nk,nv)  Ry
    worst = (0.0, None)
    per_band = {}
    for lbl, arr in (("cond", eps_c), ("val", eps_v)):
        for b in range(arr.shape[1]):
            sp = max((arr[st, b].max()-arr[st, b].min()) for st in stars)
            per_band[f"{lbl}{b}"] = float(sp*RY*1e6)
            if sp*RY*1e6 > worst[0]:
                worst = (sp*RY*1e6, f"{lbl}{b}")
    out["rung1_energy_star_spread_ueV"] = {"worst": {"band": worst[1], "ueV": worst[0]},
                                           "per_band": per_band}
    print(f"  RUNG1 max e_nk star-spread: {worst[0]:.4f} ueV (band {worst[1]})",
          flush=True)

    # ---- RUNG 4: gauge-invariant block-Frobenius covariance ----
    ncc = t["nc"]; nvv = t["nv"]
    Kx6 = t["Kx"].reshape(ncc, nvv, nk, ncc, nvv, nk)
    Kd6 = t["Kd"].reshape(ncc, nvv, nk, ncc, nvv, nk)
    Dcvk = t["Dcvk"]
    def block_fro(T6):
        return np.sqrt(np.einsum("cvkCVK->kK", np.abs(T6)**2, optimize=True))
    Fx = block_fro(Kx6); Fd = block_fro(Kd6)
    FD = np.zeros((nk, nk))
    for k in range(nk):
        FD[k, k] = np.sqrt(np.sum(np.abs(Dcvk[:, :, k])**2))

    is_trs = np.arange(nops) >= ntran
    nonsym = np.array([np.linalg.norm(tnp[s % ntran]) > 1e-6 for s in range(nops)])
    def covar(F):
        scale = F.max() if F.max() > 0 else 1.0
        per = []
        for s in range(nops):
            p = P[s]; d = np.abs(F[np.ix_(p, p)] - F)
            per.append((s, float(d.max()), float(d.max()/scale)))
        return scale, per
    for nm, F in (("D", FD), ("Kx", Fx), ("Kd", Fd)):
        scale, per = covar(F)
        def cls(mask):
            sub = [pp for pp in per if mask[pp[0]]]
            if not sub:
                return (0.0, 0.0, None)
            mx = max(sub, key=lambda z: z[1])
            return (mx[1], mx[2], mx[0])
        sym_symm = cls((~is_trs) & (~nonsym))
        sym_nsym = cls((~is_trs) & (nonsym))
        trs = cls(is_trs)
        allmx = max(per, key=lambda z: z[1])
        out[f"rung4_{nm}"] = {
            "scale": scale, "max_abs": allmx[1], "max_rel": allmx[2], "worst_op": allmx[0],
            "spatial_symmorphic": {"max_abs": sym_symm[0], "max_rel": sym_symm[1], "op": sym_symm[2]},
            "spatial_nonsymmorphic": {"max_abs": sym_nsym[0], "max_rel": sym_nsym[1], "op": sym_nsym[2]},
            "trs": {"max_abs": trs[0], "max_rel": trs[1], "op": trs[2]},
        }
        print(f"  RUNG4 {nm:>3}: scale={scale:.4e} | symm rel={sym_symm[1]:.2e} "
              f"nonsym rel={sym_nsym[1]:.2e} trs rel={trs[1]:.2e}", flush=True)

    # ---- RUNG 2+3: manifold term decomposition + eigenvector k-star weights ----
    N = ncc*nvv*nk
    Ddiag = Dcvk.reshape(-1).astype(np.complex128)
    H = np.diag(Ddiag) + t["Kx"] - t["Kd"]
    Hh = 0.5*(H + H.conj().T)
    ev, evec = np.linalg.eigh(Hh)
    ev_eV = ev*RY
    groups = cluster(ev_eV, 1.0)
    Kx = t["Kx"]; Kd = t["Kd"]
    star_of_k = np.zeros(nk, dtype=int)
    for si, st in enumerate(stars):
        for k in st:
            star_of_k[k] = si
    man = []
    for gi, g in enumerate(groups[:6]):
        idx = np.array(g)
        lam = ev[idx]
        dd = np.array([np.real(np.vdot(evec[:, i], Ddiag*evec[:, i])) for i in g])
        xx = np.array([np.real(np.vdot(evec[:, i], Kx@evec[:, i])) for i in g])
        ww = np.array([np.real(np.vdot(evec[:, i], Kd@evec[:, i])) for i in g])
        # k-star weight (sum |A|^2 per star) for the WHOLE manifold and per-state
        wstar = np.zeros((len(g), len(stars)))
        for j, i in enumerate(g):
            A = evec[:, i].reshape(ncc, nvv, nk)
            wk = np.sum(np.abs(A)**2, axis=(0, 1))
            for si, st in enumerate(stars):
                wstar[j, si] = wk[st].sum()
        entry = {
            "manifold": gi, "size": len(g), "mean_eV": float(ev_eV[idx].mean()),
            "split_ueV": float((ev_eV[idx].max()-ev_eV[idx].min())*1e6),
            "spread_D_ueV": float((dd.max()-dd.min())*RY*1e6),
            "spread_Kx_ueV": float((xx.max()-xx.min())*RY*1e6),
            "spread_Kd_ueV": float((ww.max()-ww.min())*RY*1e6),
            "spread_minusKd_ueV": float((ww.max()-ww.min())*RY*1e6),
            "per_state_kstar_weight": wstar.tolist(),
        }
        man.append(entry)
        print(f"  RUNG2 mfd{gi} sz={len(g)} mean={entry['mean_eV']:.6f}eV "
              f"split={entry['split_ueV']:.2f}ueV | dD={entry['spread_D_ueV']:.2f} "
              f"dKx={entry['spread_Kx_ueV']:.2f} d(Kd)={entry['spread_Kd_ueV']:.2f} ueV",
              flush=True)
    out["rung23_manifolds"] = man
    op = f"{RUN}/diag/probe2_{arm}_{nv}v{nc}c.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"  wrote {op}", flush=True)


if __name__ == "__main__":
    main()
