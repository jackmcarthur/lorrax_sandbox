#!/usr/bin/env python3
"""RUNG 8 — localize the split with the CORRECT BSE window (nspinor=2 ⇒ n_occ=8:
valence [4,8), conduction [8,12)).  Two questions:

(I) Is ψ at centroids covariant on the BSE-relevant bands?  Band-Gram eigenvalue
    R-invariance within k-stars for B = top-valence[4,8), conduction[8,12),
    full-occ[0,8).  Also the k-DIAGONAL transition-density Gram (identity metric)
    and the k-diagonal Kx/Kd blocks (V0/W0 metric).

(II) Does the split live on-site (Γ block) or in the inter-k coupling?
    Diagonalize H with (a) full Kx,Kd; (b) only k-DIAGONAL blocks; (c) only
    k-OFF-diagonal blocks + D.
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax; jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
GRID = (4, 4, 4); NK = 64


def star_eig_rel(vals_per_k, stars):
    """vals_per_k: (nk, m) sorted eigenvalues per k. Return worst within-star rel-spread."""
    worst = 0.0
    for st in stars:
        v = vals_per_k[st]
        sp = (v.max(axis=0) - v.min(axis=0))
        sc = np.abs(v).mean() + 1e-30
        worst = max(worst, float(sp.max()/sc))
    return worst


def lowest_manifolds(H, ntop=6):
    ev = np.sort(np.linalg.eigvalsh(H))*RY
    groups, cur = [], [0]
    for i in range(1, len(ev)):
        if ev[i]-ev[i-1] > 1e-3:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return [{"size": len(g), "mean_eV": float(np.array(ev)[g].mean()),
             "split_ueV": float((np.array(ev)[g].max()-np.array(ev)[g].min())*1e6)}
            for g in groups[:ntop]]


def main():
    arm = "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager"); sym = symmetry_maps.SymMaps(wfn)
    irr = np.asarray(sym.irr_idx_k)
    stars = [sorted(np.where(irr == u)[0].tolist()) for u in sorted(set(irr.tolist()))]

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])          # (nk,nb,s,μ)
        V0 = np.asarray(f["V_qmunu"][0]); W0 = np.asarray(f["W0_qmunu"][0])
    n_occ = 8
    out = {"arm": arm, "n_occ": n_occ}

    # ---- (I) band-Gram covariance for the correct windows ----
    def band_gram_rel(bands):
        pb = psi[:, bands, :, :]
        G = np.einsum("knsm,klsm->knl", np.conj(pb), pb, optimize=True)
        G = 0.5*(G + np.conj(np.transpose(G, (0, 2, 1))))
        ev = np.linalg.eigvalsh(G)
        return star_eig_rel(ev, stars)
    for nm, bs in (("val[4,8)", [4, 5, 6, 7]), ("cond[8,12)", [8, 9, 10, 11]),
                   ("occ[0,8)", list(range(8))), ("deepval[0,4)", [0, 1, 2, 3])):
        r = band_gram_rel(bs)
        out[f"bandGram_{nm}"] = r
        print(f"  (I) bandGram {nm:>12}: worst star eig rel-spread = {r:.3e}", flush=True)

    # transition-density objects on the BSE window
    val = [4, 5, 6, 7]; cond = [8, 9, 10, 11]
    pv = psi[:, val, :, :]; pc = psi[:, cond, :, :]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), pv, optimize=True)  # (k,c,v,μ)
    nc = len(cond); nv = len(val); nsub = nc*nv
    Mf = M.reshape(NK, nsub, -1)                                     # (k, cv, μ)
    # k-diagonal Gram, identity metric: T_k[cv,c'v'] = Σ_μ conj(M) M
    Tk_id = np.einsum("kam,kbm->kab", np.conj(Mf), Mf, optimize=True)
    # k-diagonal Kx block: Σ_μν conj(M) V0 M
    MV = np.einsum("kam,mn->kan", np.conj(Mf), V0, optimize=True)
    Kx_diag = np.einsum("kan,kbn->kab", MV, Mf, optimize=True)
    # k-diagonal Kd block uses pair densities at q=0:
    #   Kd_k[cv,c'v'] = Σ_μν conj(ψ_c)ψ_c'(μ) W0(μν) ψ_v conj(ψ_v')(ν)  (same k)
    Pc = np.einsum("kcsm,kCsm->kcCm", np.conj(pc), pc, optimize=True)  # (k,c,C,μ)
    Pv = np.einsum("kvsn,kVsn->kvVn", pv, np.conj(pv), optimize=True)  # (k,v,V,ν)
    tmp = np.einsum("kcCm,mn->kcCn", Pc, W0, optimize=True)
    Kd_diag = np.einsum("kcCn,kvVn->kcvCV", tmp, Pv, optimize=True).reshape(NK, nsub, nsub)
    for nm, Tk in (("transGram_id", Tk_id), ("Kx_diag(V0)", Kx_diag),
                   ("Kd_diag(W0)", Kd_diag)):
        Th = 0.5*(Tk + np.conj(np.transpose(Tk, (0, 2, 1))))
        ev = np.linalg.eigvalsh(Th)
        r = star_eig_rel(ev, stars)
        out[f"kdiag_{nm}"] = r
        print(f"  (I) k-diag {nm:>14}: worst star eig rel-spread = {r:.3e}", flush=True)

    # ---- (II) on-site vs inter-k split ----
    data = bse_io._load_ring_subset(restart, n_val=4, n_cond=4, px=1, py=1, input_file=inp)
    psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
    eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
    V0h = np.asarray(data["V_q0"]); Wf = np.asarray(data["W_q"]).reshape(792, 792, NK)
    Mh = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    Dcvk = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(Mh), V0h, optimize=True)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, Mh, optimize=True) / NK
    ck = np.array(np.unravel_index(np.arange(NK), GRID)).T
    qidx = np.empty((NK, NK), int)
    for k in range(NK):
        qidx[k] = np.ravel_multi_index(((ck[k][None]-ck) % np.array(GRID)).T, GRID)
    Kd = np.zeros((nc, nv, NK, nc, nv, NK), complex)
    for k in range(NK):
        Wq_k = np.transpose(Wf[:, :, qidx[k]], (2, 0, 1))
        Pc2 = np.einsum("csm,KCsm->KcCm", np.conj(psi_c[k]), psi_c, optimize=True)
        Pv2 = np.einsum("vsn,KVsn->KvVn", psi_v[k], np.conj(psi_v), optimize=True)
        t2 = np.einsum("KcCm,Kmn->KcCn", Pc2, Wq_k, optimize=True)
        Kd[:, :, k, :, :, :] = np.einsum("KcCn,KvVn->cvCVK", t2, Pv2, optimize=True) / NK
    N = nc*nv*NK
    Ddiag = np.diag(Dcvk.reshape(-1).astype(complex))
    Kx = Kx.reshape(N, N); Kd = Kd.reshape(N, N)
    # k-block masks
    kk = np.repeat(np.arange(NK)[None, :], nc*nv, axis=0)  # placeholder
    kidx = np.tile(np.arange(NK), nc*nv)  # index order is (c,v,k) C-order ⇒ k is fastest
    # rebuild proper k-label per flat index: flat = ((c*nv)+v)*NK + k ⇒ k = flat % NK
    klab = np.arange(N) % NK
    same_k = (klab[:, None] == klab[None, :])
    Hfull = Ddiag + Kx - Kd
    Hdiag = Ddiag + np.where(same_k, Kx, 0) - np.where(same_k, Kd, 0)
    Hoff = Ddiag + np.where(~same_k, Kx, 0) - np.where(~same_k, Kd, 0)
    for nm, H in (("full", Hfull), ("kdiag_only", Hdiag), ("koff_only", Hoff)):
        Hh = 0.5*(H + H.conj().T)
        man = lowest_manifolds(Hh)
        out[f"II_{nm}"] = man
        print(f"  (II) {nm:>11}: mfd0 sz={man[0]['size']} split={man[0]['split_ueV']:.2f} "
              f"| mfd2 split={man[2]['split_ueV']:.2f} ueV", flush=True)

    json.dump(out, open(f"{RUN}/diag/rung8_localize.json", "w"), indent=2)
    print("  wrote rung8_localize.json", flush=True)


if __name__ == "__main__":
    main()
