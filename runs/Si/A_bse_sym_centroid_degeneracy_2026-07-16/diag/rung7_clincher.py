#!/usr/bin/env python3
"""RUNG 7 — CLINCHER (sym arm).  Does enforcing TILE covariance collapse the BSE
splitting toward BGW's ~2 μeV?  If yes, the non-covariant ISDF tiles are THE cause.

Little-group symmetrization (projection onto the covariant subspace) of each tile:
    T_sym[q,μ,ν] = (1/|Stab(q)|) Σ_{s: R_s q ≡ q}
                     exp(2πi q_frac·(L^s_μ − L^s_ν)) · T[q, α_s(μ), α_s(ν)]
with (α_s, L^s) from compute_centroid_sym_perm (BGW r-action, validated closure).
For q=0 the phase is 1 and Stab = all 48 spatial ops.  If T is already covariant
T_sym == T; the change measures non-covariance; using T_sym in H tests causality.

Variants diagonalised (4v4c, lowest manifolds):
    A baseline           : head-injected V0, raw W_q             (= production BSE)
    B no head            : raw V0, raw W_q
    C sym tiles + head   : symmetrize(raw V0,W_q) then re-inject head
    D sym tiles, no head : symmetrize(raw V0,W_q)
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
GRID = (4, 4, 4); NK = 64


def build_H(psi_c, psi_v, eps_c, eps_v, V0, Wflat):
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
    N = nc*nv*NK
    H = np.diag(Dcvk.reshape(-1).astype(complex)) + Kx.reshape(N, N) - Kd.reshape(N, N)
    return 0.5*(H + H.conj().T)


def lowest_manifolds(H, ntop=6, gap_meV=1.0):
    ev = np.sort(np.linalg.eigvalsh(H))*RY
    groups, cur = [], [0]
    for i in range(1, len(ev)):
        if ev[i]-ev[i-1] > gap_meV*1e-3:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    res = []
    for g in groups[:ntop]:
        gi = np.array(g)
        res.append({"size": len(g), "mean_eV": float(ev[gi].mean()),
                    "split_ueV": float((ev[gi].max()-ev[gi].min())*1e6)})
    return res


def symmetrize_q0(V0, alpha, ntran):
    """q=0 tile: Stab=all spatial ops, phase=1.  V0_sym = mean_s V0[α_s(μ),α_s(ν)]."""
    nmu = V0.shape[0]
    acc = np.zeros((nmu, nmu), complex)
    for s in range(ntran):
        a = alpha[s]
        acc += V0[np.ix_(a, a)]
    return acc/ntran


def symmetrize(Tflat, alpha, Lwrap, ntran):
    """Tflat: (nmu,nmu,nk).  Return little-group-symmetrized copy."""
    nmu = Tflat.shape[0]
    ck = np.array(np.unravel_index(np.arange(NK), GRID)).T          # (nk,3) int q
    S = np.asarray(WfnLoader(WFN, backend="eager").sym_matrices[:ntran])
    Sk = S.transpose(0, 2, 1)                                        # mtrx^T acts on q
    qfrac = ck.astype(float)/np.array(GRID)                          # (nk,3)
    Tsym = np.zeros_like(Tflat)
    for iq in range(NK):
        q = ck[iq]
        stab = []
        for s in range(ntran):
            Rq = np.rint((Sk[s] @ qfrac[iq]) * np.array(GRID)).astype(int) % np.array(GRID)
            if np.array_equal(Rq, q):
                stab.append(s)
        acc = np.zeros((nmu, nmu), complex)
        Tq = Tflat[:, :, iq]
        for s in stab:
            a = alpha[s]
            # phase[μ,ν] = exp(2πi qfrac·(L_μ - L_ν))
            ph_mu = np.exp(2j*np.pi*(Lwrap[s].astype(float) @ qfrac[iq]))   # (nmu,)
            phase = np.outer(ph_mu, np.conj(ph_mu))
            acc += phase * Tq[np.ix_(a, a)]
        Tsym[:, :, iq] = acc/len(stab)
    return Tsym


def main():
    arm = "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager"); ntran = int(wfn.ntran)
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac*FFT[None]).astype(np.int64) % FFT[None]
    alpha, Lwrap = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)

    with h5py.File(restart, "r") as f:
        V0_raw = np.asarray(f["V_qmunu"][0])                 # (μ,ν)
        Wq_raw = np.asarray(f["W0_qmunu"][:])                # (nq,μ,ν)
    nmu = V0_raw.shape[0]
    Wflat_raw = np.transpose(Wq_raw, (1, 2, 0)).copy()       # (μ,ν,nq)

    # CORRECT BSE window (nspinor=2 ⇒ n_occ=8): pull the sliced ψ/eps AND the
    # head-injected tiles from the production loader — identical to rung_probe2.
    data = bse_io._load_ring_subset(restart, n_val=4, n_cond=4, px=1, py=1, input_file=inp)
    psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
    eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
    V0_head = np.asarray(data["V_q0"])
    Whead = np.asarray(data["W_q"])                           # (μ,ν,nkx,nky,nkz)
    Wflat_head = Whead.reshape(nmu, nmu, NK)
    print(f"  window: psi_c{psi_c.shape} psi_v{psi_v.shape} "
          f"eps_v[0]={np.asarray(eps_v)[0]*RY} eV", flush=True)

    print(f"[clincher sym] nmu={nmu}", flush=True)
    out = {"arm": arm}

    # A baseline (production): head V0, head W
    HA = build_H(psi_c, psi_v, eps_c, eps_v, V0_head, Wflat_head)
    evA = np.sort(np.linalg.eigvalsh(HA))*RY
    print(f"  [sanity] HA lowest-6 eV: {[f'{x:.6f}' for x in evA[:6]]}", flush=True)
    out["A_baseline"] = lowest_manifolds(HA)
    print(f"  A baseline (head)     mfd0 split={out['A_baseline'][0]['split_ueV']:.2f} "
          f"mfd2 split={out['A_baseline'][2]['split_ueV']:.2f} ueV", flush=True)

    # B no head (raw)
    HB = build_H(psi_c, psi_v, eps_c, eps_v, V0_raw, Wflat_raw)
    out["B_nohead"] = lowest_manifolds(HB)
    print(f"  B no-head (raw)       mfd0 split={out['B_nohead'][0]['split_ueV']:.2f} "
          f"mfd2 split={out['B_nohead'][2]['split_ueV']:.2f} ueV", flush=True)

    # symmetrize raw tiles
    V0_sym = symmetrize_q0(V0_raw, alpha, ntran)
    Wflat_sym = symmetrize(Wflat_raw, alpha, Lwrap, ntran)
    dV = np.abs(V0_sym - V0_raw).max()/np.abs(V0_raw).max()
    dW = np.abs(Wflat_sym - Wflat_raw).max()/np.abs(Wflat_raw).max()
    print(f"  symmetrization change: V0 rel={dV:.3e}  W rel={dW:.3e}", flush=True)
    out["symm_change"] = {"V0_rel": float(dV), "W_rel": float(dW)}

    # D sym tiles, no head
    HD = build_H(psi_c, psi_v, eps_c, eps_v, V0_sym, Wflat_sym)
    out["D_sym_nohead"] = lowest_manifolds(HD)
    print(f"  D sym-tiles no-head   mfd0 split={out['D_sym_nohead'][0]['split_ueV']:.2f} "
          f"mfd2 split={out['D_sym_nohead'][2]['split_ueV']:.2f} ueV", flush=True)

    # C sym tiles + head: re-inject head onto symmetrized V0/W via the same helper
    import jax.numpy as jnp
    from gw.head_correction import apply_q0_head_rank1_sharded
    with h5py.File(restart, "r") as f:
        G0 = np.asarray(f["G0_mu_nu"][:]); vhd = complex(f["vhead"][()])
        whd = np.asarray(f["whead"][:], complex)
    cell = 270.11
    Wsym_3d = Wflat_sym.reshape(nmu, nmu, *GRID)
    V0c, Wc = apply_q0_head_rank1_sharded(
        jnp.asarray(V0_sym), jnp.asarray(Wsym_3d),
        jnp.asarray(G0), jnp.asarray(G0),
        vhd, whd, cell, omega_index=0)
    HC = build_H(psi_c, psi_v, eps_c, eps_v, np.asarray(V0c),
                 np.asarray(Wc).reshape(nmu, nmu, NK))
    out["C_sym_head"] = lowest_manifolds(HC)
    print(f"  C sym-tiles + head    mfd0 split={out['C_sym_head'][0]['split_ueV']:.2f} "
          f"mfd2 split={out['C_sym_head'][2]['split_ueV']:.2f} ueV", flush=True)

    json.dump(out, open(f"{RUN}/diag/rung7_clincher.json", "w"), indent=2)
    print("  wrote rung7_clincher.json", flush=True)


if __name__ == "__main__":
    main()
