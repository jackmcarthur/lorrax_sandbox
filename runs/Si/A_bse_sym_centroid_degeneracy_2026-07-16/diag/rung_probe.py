#!/usr/bin/env python3
"""Where does symmetry break in Si BSE? Term/ingredient decomposition.

Rungs (all answered with NUMBERS, per-element formulas, no einsum magic beyond
the VALIDATED fast_H builder that is bit-equal to the gate _build_dense_H):

  1. enk_full covariance: energy spread within k-stars, per band.
  2. term decomposition: split of a chosen near-degenerate manifold into
     diagonal-D / exchange-Kx / direct-Kd contributions (exact: lambda_i =
     <i|D|i> + <i|Kx|i> - <i|Kd|i>).
  3. eigenvector k-weights: |A_cvk|^2 summed per k in the split manifold.
  4. gauge-invariant operator covariance: for each WFN sym op R with k-perm P_R,
     compare block Frobenius norms ||B_{P k, P k'}||_F vs ||B_{k,k'}||_F,
     SEPARATELY for D, Kx, Kd.  Covariance H = U(R)^dag H U(R) with
     U(R)=P_R (x) (unitary band-rotation) implies ||block||_F is R-invariant
     (Frobenius norm is invariant under left/right unitary mult) -- so this
     needs NO band-rotation matrices and is fully gauge-free.

Usage (inside srun+shifter): python3 rung_probe.py <arm> <nv> <nc>
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
import h5py

RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")


# ----------------------------------------------------------------------
# fast_H terms -- factored from analyze_fast_all.fast_H (VALIDATED bit-equal
# to the gate _build_dense_H at 2v2c, rel-err 4.8e-17).  Same per-element
# formulas; returns D (diag, as (nc,nv,nk)), Kx, Kd as (N,N).
# ----------------------------------------------------------------------
def build_terms(data):
    psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
    eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
    V_q0 = np.asarray(data["V_q0"]); W_q = np.asarray(data["W_q"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    grid = (nkx, nky, nkz); nk = nkx*nky*nkz
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
    N = nc*nv*nk

    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    Dcvk = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))  # (nc,nv,nk)
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

    return dict(Dcvk=Dcvk, Kx=Kx.reshape(N, N), Kd=Kd.reshape(N, N),
                nc=nc, nv=nv, nk=nk, grid=grid,
                eps_c=eps_c, eps_v=eps_v)


def assemble_H(t):
    N = t["nc"]*t["nv"]*t["nk"]
    D = np.diag(t["Dcvk"].reshape(-1).astype(np.complex128))
    return D + t["Kx"] - t["Kd"]


# ----------------------------------------------------------------------
# k-permutation P_R for the 4x4x4 C-order MP grid, from WFN mtrx.
# Determine mtrx convention (k' = M k  vs  M^T k, mod grid) by demanding a
# valid bijection for EVERY op.  ck index = ix*nky*nkz + iy*nkz + iz.
# ----------------------------------------------------------------------
def kperms(grid):
    nkx, nky, nkz = grid
    nk = nkx*nky*nkz
    with h5py.File(WFN, "r") as f:
        mtrx = np.asarray(f["/mf_header/symmetry/mtrx"][:]).astype(np.int64)  # (48,3,3)
        ntran = int(f["/mf_header/symmetry/ntran"][()])
        tnp = np.asarray(f["/mf_header/symmetry/tnp"][:])
    mtrx = mtrx[:ntran]; tnp = tnp[:ntran]
    ck = np.array(np.unravel_index(np.arange(nk), grid)).T  # (nk,3) in {0..n-1}
    gvec = np.array(grid)

    def try_conv(use_T):
        perms = []
        for s in range(ntran):
            Ms = mtrx[s].T if use_T else mtrx[s]
            kp = (ck @ Ms.T) % gvec       # row k': (M @ k) per row  == ck @ Ms.T
            idx = np.ravel_multi_index(kp.T, grid)
            if len(set(idx.tolist())) != nk:
                return None
            perms.append(idx)
        return np.array(perms)  # (ntran, nk)

    # k' = M @ k  -> row-vector form ck @ M.T ; try M and M^T
    for use_T in (False, True):
        P = try_conv(use_T)
        if P is not None:
            return P, mtrx, tnp, ntran, ("M^T" if use_T else "M")
    raise RuntimeError("no valid k-permutation convention found")


# ----------------------------------------------------------------------
def kstars(P):
    """Orbits of full-BZ k under the perm group P (ntran, nk)."""
    ntran, nk = P.shape
    seen = np.zeros(nk, bool); stars = []
    for k in range(nk):
        if seen[k]:
            continue
        orb = sorted(set(P[:, k].tolist()))
        for j in orb:
            seen[j] = True
        stars.append(orb)
    return stars


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

    data = bse_io._load_ring_subset(restart, n_val=nv, n_cond=nc, px=1, py=1,
                                    input_file=inp)
    psi = np.asarray(data["psi_c"])
    print(f"  psi_c shape {psi.shape}; spin-comp norms "
          f"|s0|={np.linalg.norm(psi[:,:,0,:]):.4e} "
          f"|s1|={np.linalg.norm(psi[:,:,1,:]):.4e} "
          f"|s0-s1|={np.linalg.norm(psi[:,:,0,:]-psi[:,:,1,:]):.4e}", flush=True)

    t = build_terms(data)
    grid = t["grid"]; nk = t["nk"]; ncc = t["nc"]; nvv = t["nv"]
    P, mtrx, tnp, ntran, conv = kperms(grid)
    print(f"  k-perm convention: {conv}; ntran={ntran}; closure OK", flush=True)
    stars = kstars(P)
    print(f"  #k-stars={len(stars)}  sizes={sorted(set(len(s) for s in stars))}",
          flush=True)

    out = {"arm": arm, "window": f"{nv}v{nc}c", "conv": conv, "ntran": ntran}

    # ---- RUNG 1: enk_full covariance (energy spread within k-stars) ----
    eps_c = t["eps_c"]; eps_v = t["eps_v"]   # (nk, nc), (nk, nv)  [Ry]
    r1 = {"band_max_star_spread_ueV": {}, "worst": None}
    worst = (0.0, None)
    for lbl, arr in (("cond", eps_c), ("val", eps_v)):
        for b in range(arr.shape[1]):
            spreads = []
            for st in stars:
                vals = arr[st, b]
                spreads.append((vals.max()-vals.min()))
            mx = float(max(spreads)*RY*1e6)
            r1["band_max_star_spread_ueV"][f"{lbl}{b}"] = mx
            if mx > worst[0]:
                worst = (mx, f"{lbl}{b}")
    r1["worst"] = {"band": worst[1], "ueV": worst[0]}
    out["rung1_energy_covariance"] = r1
    print(f"  RUNG1 max energy star-spread: {worst[0]:.3f} ueV "
          f"(band {worst[1]})", flush=True)

    # also the transition-energy (e_c - e_v) diagonal spread within manifolds
    Dcvk = t["Dcvk"]  # (nc,nv,nk) Ry

    # ---- RUNG 4: gauge-invariant block-Frobenius covariance ----
    # block B_{k,k'} = term[:, :, k, :, :, k'] reshaped (ncnv, ncnv).
    Kx6 = t["Kx"].reshape(ncc, nvv, nk, ncc, nvv, nk)
    Kd6 = t["Kd"].reshape(ncc, nvv, nk, ncc, nvv, nk)
    # D is diagonal in (cvk): only B_{k,k} nonzero, itself diagonal in (cv).
    def block_fro(T6):
        # returns F[k,k'] = ||T6[:,:,k,:,:,k']||_F
        return np.sqrt(np.einsum("cvkCVK->kK", np.abs(T6)**2, optimize=True))
    Fx = block_fro(Kx6); Fd = block_fro(Kd6)
    # D block-fro: diagonal only
    FD = np.zeros((nk, nk))
    for k in range(nk):
        FD[k, k] = np.sqrt(np.sum(np.abs(Dcvk[:, :, k])**2))

    def covar_viol(F):
        scale = F.max() if F.max() > 0 else 1.0
        per_op = []
        for s in range(ntran):
            p = P[s]
            Fp = F[np.ix_(p, p)]
            d = np.abs(Fp - F)
            per_op.append((s, float(d.max()), float(d.max()/scale)))
        mx = max(per_op, key=lambda z: z[1])
        return scale, per_op, mx
    for nm, F in (("D", FD), ("Kx", Fx), ("Kd", Fd)):
        scale, per_op, mx = covar_viol(F)
        # tag nonsymmorphic ops (|tnp| > 1e-6)
        nonsym = [s for s in range(ntran) if np.linalg.norm(tnp[s]) > 1e-6]
        viol_ops = sorted([(s, ad, rel) for (s, ad, rel) in per_op if rel > 1e-9],
                          key=lambda z: -z[1])
        out[f"rung4_covar_{nm}"] = {
            "block_fro_scale": scale,
            "max_abs_viol": mx[1], "max_rel_viol": mx[2], "worst_op": mx[0],
            "worst_op_nonsymmorphic": bool(mx[0] in nonsym),
            "n_ops_violating_rel>1e-9": len(viol_ops),
            "top5_violating_ops": [
                {"op": s, "abs": ad, "rel": rel, "nonsym": bool(s in nonsym),
                 "tnp": [float(x) for x in tnp[s]]}
                for (s, ad, rel) in viol_ops[:5]],
        }
        print(f"  RUNG4 {nm:>3}: scale={scale:.4e} max|Δ‖B‖|={mx[1]:.4e} "
              f"rel={mx[2]:.3e} worst_op={mx[0]} "
              f"nonsym={mx[0] in nonsym} nviol={len(viol_ops)}", flush=True)

    # ---- RUNG 2+3: manifold term decomposition + eigenvector k-weights ----
    H = assemble_H(t)
    Hh = 0.5*(H + H.conj().T)
    ev, evec = np.linalg.eigh(Hh)   # ascending
    order = np.argsort(ev); ev = ev[order]; evec = evec[:, order]
    ev_eV = ev*RY
    groups = cluster(ev_eV, 1.0)
    Kx = t["Kx"]; Kd = t["Kd"]; Ddiag = t["Dcvk"].reshape(-1).astype(np.complex128)

    def decomp(idx_list):
        res = []
        for i in idx_list:
            a = evec[:, i]
            d = float(np.real(np.vdot(a, Ddiag*a)))
            x = float(np.real(np.vdot(a, Kx@a)))
            w = float(np.real(np.vdot(a, Kd@a)))
            res.append((ev[i], d, x, w))
        return res

    man = []
    for gi, g in enumerate(groups[:6]):
        idx = np.array(g)
        # eigenvector k-weights: |A_cvk|^2 per k
        wk = np.zeros(nk)
        for i in g:
            A = evec[:, i].reshape(ncc, nvv, nk)
            wk += np.sum(np.abs(A)**2, axis=(0, 1))
        dc = decomp(g)
        lam = np.array([z[0] for z in dc])*RY*1e6   # ueV
        dd = np.array([z[1] for z in dc])*RY*1e6
        xx = np.array([z[2] for z in dc])*RY*1e6
        ww = np.array([z[3] for z in dc])*RY*1e6
        entry = {
            "manifold": gi, "size": len(g),
            "mean_eV": float(ev_eV[idx].mean()),
            "split_ueV": float((ev_eV[idx].max()-ev_eV[idx].min())*1e6),
            "spread_D_ueV": float(dd.max()-dd.min()),
            "spread_Kx_ueV": float(xx.max()-xx.min()),
            "spread_Kd_ueV": float(ww.max()-ww.min()),
            "kweight_min": float(wk.min()), "kweight_max": float(wk.max()),
            "kweight_star_summary": None,
        }
        # k-weight per star (should be equal across a star if covariant)
        star_w = []
        for st in stars:
            star_w.append(float(wk[st].sum()))
        entry["kweight_star_summary"] = {
            "n_stars_occupied": int(np.sum(np.array(star_w) > 1e-6)),
        }
        man.append(entry)
        print(f"  RUNG2 mfd{gi} sz={len(g)} mean={entry['mean_eV']:.6f}eV "
              f"split={entry['split_ueV']:.2f}ueV | dD={entry['spread_D_ueV']:.2f} "
              f"dKx={entry['spread_Kx_ueV']:.2f} dKd={entry['spread_Kd_ueV']:.2f} ueV",
              flush=True)
    out["rung23_manifolds"] = man

    op = f"{RUN}/diag/probe_{arm}_{nv}v{nc}c.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"  wrote {op}", flush=True)


if __name__ == "__main__":
    main()
