#!/usr/bin/env python3
"""DIAG2 Task-1: closed-window BSE residual attribution.

The CUT 4v4c window gives ~518 μeV splits (window truncation of degenerate
multiplets, FINDINGS rung10/11).  The Γ-on-site CLOSED block [2,8)x[8,14)
restores multiplets to ≤36 μeV (rung10) — but 36 μeV is still 18x the BGW ~2
μeV scale.  QUESTION: is that 36 μeV floor driven by the ~3% non-covariant
ISDF V0/W0 tiles?  Test by re-measuring with little-group-SYMMETRIZED tiles.

We measure a 2x2 grid {raw tiles, sym tiles} x {head off, head on} on:
  (A) the Γ-on-site exciton block (q=0 only; cheap, decisive)
  (B) the FULL-H over all 64 k with a degeneracy-CLOSED window (boundaries
      gap-separated at EVERY k of the 4x4x4 grid).

If SYM-tiles collapse the residual → tiles own the remaining symmetry breaker.
If SYM-tiles do NOT move it → the residual is intrinsic (kernel / remaining
window closure), tiles are subdominant even here.

Usage: python3 closed_window.py sym
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from centroid.orbit_syms import compute_centroid_sym_perm
from gw.head_correction import apply_q0_head_rank1_sharded
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24]); GRID = (4, 4, 4); NK = 64
CELL = 270.11  # Si cell volume (matches rung7)


def clusters(ev, gap=1e-3):
    g = [[0]]
    for i in range(1, len(ev)):
        (g.append([i]) if ev[i] - ev[i - 1] > gap else g[-1].append(i))
    return g


def manifolds(H, ntop=6, gap_meV=1.0):
    ev = np.sort(np.linalg.eigvalsh(H)) * RY
    cl = clusters(ev, gap_meV * 1e-3)
    return [{"size": len(c), "mean_eV": float(ev[np.array(c)].mean()),
             "split_ueV": float((ev[c[-1]] - ev[c[0]]) * 1e6)} for c in cl[:ntop]]


def symmetrize_q0(T0, alpha, ntran):
    """q=0 tile: Stab = all spatial ops, phase = 1."""
    nmu = T0.shape[0]
    acc = np.zeros((nmu, nmu), complex)
    for s in range(ntran):
        a = alpha[s]
        acc += T0[np.ix_(a, a)]
    return acc / ntran


def symmetrize_allq(Tq, alpha, Lwrap, ntran, S):
    """Tq: (nq, nmu, nmu).  Little-group-symmetrize every q with umklapp phase."""
    nmu = Tq.shape[1]
    ck = np.array(np.unravel_index(np.arange(NK), GRID)).T          # (nk,3) int q
    Sk = S.transpose(0, 2, 1)                                        # mtrx^T acts on q
    qfrac = ck.astype(float) / np.array(GRID)
    Tsym = np.zeros_like(Tq)
    for iq in range(NK):
        q = ck[iq]
        stab = []
        for s in range(ntran):
            Rq = np.rint((Sk[s] @ qfrac[iq]) * np.array(GRID)).astype(int) % np.array(GRID)
            if np.array_equal(Rq, q):
                stab.append(s)
        acc = np.zeros((nmu, nmu), complex)
        Tqi = Tq[iq]
        for s in stab:
            a = alpha[s]
            ph_mu = np.exp(2j * np.pi * (Lwrap[s].astype(float) @ qfrac[iq]))  # (nmu,)
            phase = np.outer(ph_mu, np.conj(ph_mu))
            acc += phase * Tqi[np.ix_(a, a)]
        Tsym[iq] = acc / len(stab)
    return Tsym


def head_inject(V0, Wflat, G0, vhead, whead):
    """Inject the q=0 head onto V0 (μ,ν) and Wflat[...,0] via the production helper."""
    W3d = Wflat.reshape(V0.shape[0], V0.shape[1], *GRID)
    V0c, Wc = apply_q0_head_rank1_sharded(
        jnp.asarray(V0), jnp.asarray(W3d), jnp.asarray(G0), jnp.asarray(G0),
        complex(vhead), np.asarray(whead, complex), CELL, omega_index=0)
    return np.asarray(V0c), np.asarray(Wc).reshape(V0.shape[0], V0.shape[1], NK)


def build_full_H(psi_c, psi_v, eps_c, eps_v, V0, Wflat):
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    Dcvk = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V0, optimize=True)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M, optimize=True) / NK
    ck = np.array(np.unravel_index(np.arange(NK), GRID)).T
    qidx = np.empty((NK, NK), int)
    for k in range(NK):
        qidx[k] = np.ravel_multi_index(((ck[k][None] - ck) % np.array(GRID)).T, GRID)
    Kd = np.zeros((nc, nv, NK, nc, nv, NK), complex)
    for k in range(NK):
        Wq_k = np.transpose(Wflat[:, :, qidx[k]], (2, 0, 1))
        Pc = np.einsum("csm,KCsm->KcCm", np.conj(psi_c[k]), psi_c, optimize=True)
        Pv = np.einsum("vsn,KVsn->KvVn", psi_v[k], np.conj(psi_v), optimize=True)
        tmp = np.einsum("KcCm,Kmn->KcCn", Pc, Wq_k, optimize=True)
        Kd[:, :, k, :, :, :] = np.einsum("KcCn,KvVn->cvCVK", tmp, Pv, optimize=True) / NK
    N = nc * nv * NK
    H = np.diag(Dcvk.reshape(-1).astype(complex)) + Kx.reshape(N, N) - Kd.reshape(N, N)
    return 0.5 * (H + H.conj().T)


def gamma_onsite_H(psi, enk, V0, W0, vb, cb):
    """Γ-only exciton block (k=0)."""
    pv = psi[0][vb]; pc = psi[0][cb]; ev = enk[0][vb]; ec = enk[0][cb]
    nv = len(vb); nc = len(cb)
    M = np.einsum("csm,vsm->cvm", np.conj(pc), pv, optimize=True)
    Kx = np.einsum("cvm,mn,CVn->cvCV", np.conj(M), V0, M, optimize=True)
    Pc = np.einsum("csm,Csm->cCm", np.conj(pc), pc, optimize=True)
    Pv = np.einsum("vsn,Vsn->vVn", pv, np.conj(pv), optimize=True)
    Kd = np.einsum("cCn,vVn->cvCV", np.einsum("cCm,mn->cCn", Pc, W0, optimize=True), Pv, optimize=True)
    D = np.zeros((nc, nv, nc, nv), complex)
    for c in range(nc):
        for v in range(nv):
            D[c, v, c, v] = ec[c] - ev[v]
    H = (D + Kx - Kd).reshape(nc * nv, nc * nv)
    return 0.5 * (H + H.conj().T)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    S = np.asarray(wfn.sym_matrices[:ntran])
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
    alpha, Lwrap = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])         # (nk, nb, ns, nmu)
        enk = np.asarray(f["enk_full"][:])           # (nk, nb) Ry
        Vq = np.asarray(f["V_qmunu"][:])             # (nq, μ, ν) raw
        Wq = np.asarray(f["W0_qmunu"][:])            # (nq, μ, ν) raw
        G0 = np.asarray(f["G0_mu_nu"][:])
        vhead = complex(f["vhead"][()])
        whead = np.asarray(f["whead"][:], complex)
    nk, nb, ns, nmu = psi.shape
    print(f"[{arm}] psi{psi.shape} nmu={nmu} ntran={ntran}", flush=True)

    # ---- find degeneracy-closed conduction upper bounds (gap at EVERY k) ----
    thr = 20.0  # meV
    closed_upper = []
    for b in range(9, min(nb, 30)):
        gmin = float(np.min(enk[:, b] - enk[:, b - 1])) * RY * 1e3  # meV, gap below b
        if gmin > thr:
            closed_upper.append((b, gmin))
    print(f"  degeneracy-closed conduction boundaries [8,b) with min-k gap>{thr}meV:", flush=True)
    for b, g in closed_upper:
        print(f"     b={b}: min gap(b-1|b)={g:.1f} meV", flush=True)
    # valence: full occupied [0,8) is closed by the 2.54 eV fundamental gap.
    # pick smallest closed conduction window that contains the Γ15 6-fold (b>=14)
    nc_hi = next((b for b, _ in closed_upper if b >= 14), (closed_upper[-1][0] if closed_upper else 14))
    print(f"  chosen full-H window: valence [0,8), conduction [8,{nc_hi})  "
          f"(N={8 * (nc_hi - 8) * NK})", flush=True)

    out = {"arm": arm, "nmu": nmu, "closed_conduction_upper": closed_upper,
           "full_window": {"v": [0, 8], "c": [8, nc_hi]}}

    # ---- symmetrized tiles ----
    V0_raw = Vq[0].copy(); W0_raw = Wq[0].copy()
    V0_sym_q0 = symmetrize_q0(V0_raw, alpha, ntran)
    W0_sym_q0 = symmetrize_q0(W0_raw, alpha, ntran)
    dV0 = np.abs(V0_sym_q0 - V0_raw).max() / np.abs(V0_raw).max()
    dW0 = np.abs(W0_sym_q0 - W0_raw).max() / np.abs(W0_raw).max()
    print(f"  q=0 symmetrization change: V0 rel={dV0:.3e} W0 rel={dW0:.3e}", flush=True)
    out["symm_change_q0"] = {"V0": float(dV0), "W0": float(dW0)}

    # =====================================================================
    # (A) Γ-on-site block — 2x2 grid {raw,sym} x {head off, head on}
    # =====================================================================
    vb = [2, 3, 4, 5, 6, 7]; cb = [8, 9, 10, 11, 12, 13]  # closed at Γ (rung10)
    gamma_res = {}
    for tile_tag, (V0t, W0t) in (("raw", (V0_raw, W0_raw)),
                                 ("sym", (V0_sym_q0, W0_sym_q0))):
        for head_tag in ("nohead", "head"):
            if head_tag == "head":
                # inject head onto this q=0 tile (Wflat only needs q=0 slice)
                Wf = np.zeros((nmu, nmu, NK), complex); Wf[:, :, 0] = W0t
                V0h, Wfh = head_inject(V0t, Wf, G0, vhead, whead)
                V0u, W0u = V0h, Wfh[:, :, 0]
            else:
                V0u, W0u = V0t, W0t
            H = gamma_onsite_H(psi, enk, V0u, W0u, vb, cb)
            ev = np.sort(np.linalg.eigvalsh(H)) * RY
            cl = clusters(ev, 1e-3)
            splits = [float((ev[c[-1]] - ev[c[0]]) * 1e6) for c in cl]
            key = f"{tile_tag}_{head_tag}"
            gamma_res[key] = {"mult_sizes": [len(c) for c in cl],
                              "splits_ueV": [round(s, 3) for s in splits],
                              "max_split_ueV": float(max(splits))}
            print(f"  (A) Γ-block [{key:>11}]: sizes={[len(c) for c in cl]} "
                  f"max_split={max(splits):.3f} μeV  splits={[round(s,2) for s in splits]}",
                  flush=True)
    out["A_gamma_onsite"] = gamma_res

    # =====================================================================
    # (B) FULL-H closed window — 2x2 grid
    # =====================================================================
    vlo, vhi, clo, chi = 0, 8, 8, nc_hi
    psi_v = psi[:, vlo:vhi]; psi_c = psi[:, clo:chi]
    eps_v = enk[:, vlo:vhi]; eps_c = enk[:, clo:chi]
    V0_sym_all = symmetrize_allq(Vq, alpha, Lwrap, ntran, S)  # (nq,μ,ν)
    W0_sym_all = symmetrize_allq(Wq, alpha, Lwrap, ntran, S)
    full_res = {}
    for tile_tag, (Vall, Wall) in (("raw", (Vq, Wq)), ("sym", (V0_sym_all, W0_sym_all))):
        Wflat = np.transpose(Wall, (1, 2, 0)).copy()   # (μ,ν,nq)
        for head_tag in ("nohead", "head"):
            V0u = Vall[0].copy()
            Wf = Wflat.copy()
            if head_tag == "head":
                V0u, Wf = head_inject(V0u, Wf, G0, vhead, whead)
            H = build_full_H(psi_c, psi_v, eps_c, eps_v, V0u, Wf)
            mfd = manifolds(H, ntop=6)
            key = f"{tile_tag}_{head_tag}"
            full_res[key] = mfd
            print(f"  (B) full-H [{key:>11}]: "
                  + " | ".join(f"mfd{i}(sz{m['size']}) {m['split_ueV']:.2f}μeV"
                               for i, m in enumerate(mfd[:4])), flush=True)
    out["B_full_H"] = full_res

    json.dump(out, open(f"{RUN}/diag2/closed_window_{arm}.json", "w"), indent=2)
    print(f"  wrote closed_window_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
