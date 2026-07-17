#!/usr/bin/env python3
"""RUNG 6 — isolate the Kx covariance violation to the metric, on the SYM arm
(ψ at centroids validated covariant to 2e-15 in rung5, so ANY violation here is
the μν metric, not ψ).

(A) Build the Q=0 exchange three ways and measure block-Frobenius covariance
    (uses the code's SymMaps k-perm; ψ-side is covariant so this is clean):
      Kx_id    : identity metric  Σ_μ conj(M(μ)) M(μ)
      Kx_rawV  : raw ISDF tile     Σ_μν conj(M(μ)) V0_raw(μ,ν) M(ν)   (NO head)
      Kx_headV : head-injected V0  (the actual BSE exchange)
(B) Direct tile test — q=0 identity from compute_centroid_sym_perm:
      covariant tile  ⟺  V0(μ,ν) == V0(α_s(μ), α_s(ν))  for every spatial op s
      (q=0 ⇒ umklapp phase exp(2πi·0·(L_μ−L_ν)) = 1).  Report raw and head.
      Same for W0(q=0).

Usage: python3 rung6_tile.py <arm>   (meaningful on sym; old fails α-closure)
"""
import sys, os, json
import numpy as np
import h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from centroid.orbit_syms import compute_centroid_sym_perm
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24])


def wrap01(x):
    y = np.mod(x, 1.0); y[np.abs(y - 1.0) < 1e-6] = 0.0
    return y


def build_kperms(sym):
    kpts = np.asarray(sym.unfolded_kpts, float); nk = kpts.shape[0]
    S = np.asarray(sym.sym_mats_k)
    key = {tuple(np.round(wrap01(kpts[k]), 6)): k for k in range(nk)}
    P = np.full((S.shape[0], nk), -1, np.int64)
    for s in range(S.shape[0]):
        for k in range(nk):
            P[s, k] = key[tuple(np.round(wrap01(S[s] @ kpts[k]), 6))]
    return P


def block_fro_covar(K6, P, ntran, nops):
    nk = K6.shape[2]
    F = np.sqrt(np.einsum("cvkCVK->kK", np.abs(K6)**2, optimize=True))
    scale = F.max() if F.max() > 0 else 1.0
    rels = []
    for s in range(nops):
        p = P[s]; d = np.abs(F[np.ix_(p, p)] - F)
        rels.append(d.max()/scale)
    rels = np.array(rels)
    return scale, float(rels.max()), float(rels[:ntran].max()), float(rels[ntran:].max())


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    P = build_kperms(sym); nops = P.shape[0]; nk = P.shape[1]

    # centroids → fft idx → α perm (spatial ops only; q=0 ⇒ no phase)
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_{'792' if arm=='sym' else '960'}.txt")
    ridx = np.rint(cfrac * FFT[None, :]).astype(np.int64) % FFT[None, :]
    try:
        alpha, Lwrap = compute_centroid_sym_perm(
            ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT,
            validate=(arm == "sym"))
        alpha_ok = True
    except Exception as e:
        print(f"  α-closure FAILED ({arm}): {str(e)[:100]}", flush=True)
        alpha = None; alpha_ok = False

    # raw tiles (NO head) from restart
    with h5py.File(restart, "r") as f:
        V0_raw = np.asarray(f["V_qmunu"][0])        # (μ,ν) q=0
        W0_raw = np.asarray(f["W0_qmunu"][0])
        psi = np.asarray(f["psi_full_y"][:])         # (nk,nb,s,μ)
    nmu = V0_raw.shape[0]
    print(f"[{arm}] restart={os.path.basename(restart)} nmu={nmu} α_ok={alpha_ok}",
          flush=True)

    # head-injected V0 via the production loader (px=py=1 ⇒ no μ pad at 792/960)
    data = bse_io._load_ring_subset(restart, n_val=4, n_cond=4, px=1, py=1,
                                    input_file=inp)
    V0_head = np.asarray(data["V_q0"])
    assert V0_head.shape[0] == nmu, f"pad mismatch {V0_head.shape} vs {nmu}"

    out = {"arm": arm, "nmu": nmu, "alpha_ok": alpha_ok}

    # ---------- (B) direct tile perm-invariance (sym only) ----------
    if alpha_ok:
        def tile_perm_viol(T):
            sc = np.abs(T).max()
            worst = 0.0; worst_s = None
            for s in range(ntran):
                a = alpha[s]
                d = np.abs(T[np.ix_(a, a)] - T).max()
                if d/sc > worst:
                    worst = d/sc; worst_s = s
            return float(worst), worst_s, float(sc)
        for nm, T in (("V0_raw", V0_raw), ("V0_head", V0_head), ("W0_raw", W0_raw)):
            w, ws, sc = tile_perm_viol(T)
            tnp = np.linalg.norm(wfn.translations[ws]/(2*np.pi)) if ws is not None else 0.0
            out[f"tileperm_{nm}"] = {"max_rel": w, "worst_op": ws, "scale": sc,
                                     "worst_op_nonsym": bool(tnp > 1e-6)}
            print(f"  (B) tile[{nm}]  V(αμ,αν)=V(μ,ν)?  max_rel={w:.3e} "
                  f"worst_op={ws} nonsym={tnp>1e-6} scale={sc:.3e}", flush=True)

    # ---------- (A) Kx three ways, block-Frobenius covariance ----------
    val = [0, 1, 2, 3]; cond = [4, 5, 6, 7]
    psi_v = psi[:, val, :, :]; psi_c = psi[:, cond, :, :]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)  # (k,c,v,μ)
    nc = len(cond); nv = len(val)
    def kx_from_metric(V):
        if V is None:  # identity
            lhs = np.conj(M)
        else:
            lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V, optimize=True)
        Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M, optimize=True) / nk
        return Kx
    for nm, V in (("Kx_id", None), ("Kx_rawV", V0_raw), ("Kx_headV", V0_head)):
        K6 = kx_from_metric(V)
        scale, rmax, rspat, rtrs = block_fro_covar(K6, P, ntran, nops)
        out[f"covar_{nm}"] = {"scale": scale, "max_rel": rmax,
                              "spatial_rel": rspat, "trs_rel": rtrs}
        print(f"  (A) {nm:>9}: scale={scale:.4e} covar max_rel={rmax:.3e} "
              f"(spatial={rspat:.3e} trs={rtrs:.3e})", flush=True)

    json.dump(out, open(f"{RUN}/diag/rung6_{arm}.json", "w"), indent=2)
    print(f"  wrote rung6_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
