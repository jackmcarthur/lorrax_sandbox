#!/usr/bin/env python3
"""DIAG2 Task-2 bisection: WHICH ingredient first violates covariance.

Ladder (all from the sym-arm restart; centroids orbit-closed ⇒ α exists):

  L0  ψ at centroids              — rung5 said 1e-15 (re-confirm on screening window)
  L1  C0 = CCT pair-density Gram over the SCREENING window [0,8)x[8,nband)
        at centroids: C0[α(μ),α(ν)] == C0[μ,ν] iff the (occ,cond) windows are
        each degeneracy-CLOSED.  TAU-INDEPENDENT (Gram over closed sets is a
        trace ⇒ D-rotations cancel).  If C0 breaks ⇒ screening band-window cut.
  L2  G0 = ζ̃_{q=0}(G=0)          — a pure ζ-fit OUTPUT component.  TAU-INDEPENDENT
        (phase e^{-iG·τ}=1 at G=0).  If G0 breaks ⇒ ζ non-covariant already in a
        tau-blind mode (band-cut / centroid), NOT a phase bug.  If G0 is clean
        but V0 breaks ⇒ the violation is in a G≠0 / tau-phase mode of ζ.
  L3  V0 = Σ_G conj(ζ̃_μ) v ζ̃_ν  — 3% (rung6).  q=0 is the IBZ parent ⇒ DIRECT
        contraction (no unfold), so V0 non-covariance ⟺ ζ̃ non-covariance
        (the τ phases cancel LEG-TO-LEG exactly; proven in FINDINGS2).
  L4  W0 = screened tile           — 3% (rung6).  If ~= V0 ⇒ W-solve adds nothing.
  L5  ΔW = W0 - V0 (screening corr)— covariance of the delta isolates the W solve.

Also: at G=0 the τ phase is unity, so degeneracy between L1/L2 (both τ-blind) vs
L3/L4 (τ-sensitive) is the discriminant for band-cut-vs-phase.

Usage: python3 bisect.py sym
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from centroid.orbit_syms import compute_centroid_sym_perm
from bse import bse_io
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24])
RY = 13.6056980659


def perm_viol(T, alpha, ntran, tnp, two_leg):
    """Max relative covariance violation over spatial ops.
    two_leg=True: T[ix(a,a)] vs T (matrix).  False: T[a] vs T (vector)."""
    sc = np.abs(T).max()
    worst = 0.0; worst_s = -1
    per = []
    for s in range(ntran):
        a = alpha[s]
        if two_leg:
            d = np.abs(T[np.ix_(a, a)] - T).max()
        else:
            d = np.abs(T[a] - T).max()
        r = float(d / sc)
        per.append(r)
        if r > worst:
            worst = r; worst_s = s
    nonsym = bool(np.linalg.norm(tnp[worst_s]) > 1e-6) if worst_s >= 0 else False
    return worst, worst_s, nonsym, sc, per


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    tnp = np.asarray(wfn.translations[:ntran]) / (2 * np.pi)
    n_nonsym = int((np.linalg.norm(tnp, axis=1) > 1e-6).sum())

    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
    alpha, Lwrap = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT,
        validate=True)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])      # (nk, nb, ns, nmu)
        enk = np.asarray(f["enk_full"][:])         # (nk, nb)
        V0 = np.asarray(f["V_qmunu"][0])           # (nmu, nmu) raw q=0
        W0 = np.asarray(f["W0_qmunu"][0])
        G0 = np.asarray(f["G0_mu_nu"][:])          # (nmu,) = ζ̃_{0,μ}(G=0)
    nk, nb, ns, nmu = psi.shape
    print(f"[{arm}] restart={os.path.basename(restart)} psi{psi.shape} "
          f"nmu={nmu} ntran={ntran} ({n_nonsym} nonsymmorphic)", flush=True)

    out = {"arm": arm, "nmu": nmu, "ntran": ntran, "n_nonsym": n_nonsym}

    # screening window from input
    NVAL, NCOND, NBAND = 8, 52, 60
    occ = list(range(0, NVAL))          # [0,8)
    cond = list(range(NVAL, NBAND))     # [8,60)

    # ---- band-boundary degeneracy at nband=60 across all k ----
    gap_top = float(np.min(enk[:, NBAND] - enk[:, NBAND - 1])) * RY if nb > NBAND else float('nan')
    gap_top_ueV = gap_top * 1e6
    # smallest gap at the (occ|cond) split band 7|8
    gap_fund = float(np.min(enk[:, NVAL] - enk[:, NVAL - 1])) * RY
    out["gap_band59_60_min_meV"] = gap_top * 1e3
    out["gap_fund_min_eV"] = gap_fund
    print(f"  min gap at cond-top boundary (band {NBAND-1}|{NBAND}): "
          f"{gap_top*1e3:.3f} meV  (fundamental 7|8: {gap_fund:.3f} eV)", flush=True)

    # ---- L1: C0 CCT pair-density Gram over the screening window ----
    # ρ_cvk(μ) = Σ_s conj(ψ_{ck,s}(μ)) ψ_{vk,s}(μ)  (spin-summed pair density)
    # C0[μ,ν] = Σ_{cvk} conj(ρ_cvk(μ)) ρ_cvk(ν)
    psi_c = psi[:, cond, :, :]          # (nk, nc, ns, nmu)
    psi_v = psi[:, occ, :, :]           # (nk, nv, ns, nmu)
    rho = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)  # (k,c,v,μ)
    C0 = np.einsum("kcvm,kcvn->mn", np.conj(rho), rho, optimize=True)          # (μ,ν)
    w, ws, nonsym, sc, per = perm_viol(C0, alpha, ntran, tnp, two_leg=True)
    out["L1_C0_CCT"] = {"max_rel": w, "worst_op": ws, "worst_nonsym": nonsym, "scale": sc}
    print(f"  L1 C0 CCT [tau-blind]   max_rel={w:.3e} worst_op={ws} nonsym={nonsym}", flush=True)

    # also: pure occ-manifold Gram (closed for sure) as a control
    Gv = np.einsum("kvsm,kvsn->mn", np.conj(psi_v), psi_v, optimize=True)
    w2, ws2, ns2, sc2, _ = perm_viol(Gv, alpha, ntran, tnp, two_leg=True)
    out["L0_occ_Gram_control"] = {"max_rel": w2, "worst_op": ws2, "worst_nonsym": ns2}
    print(f"  L0 occ-Gram control     max_rel={w2:.3e} worst_op={ws2} nonsym={ns2}", flush=True)

    # ---- L2: G0 = ζ̃(G=0) [tau-blind because e^{-iG·τ}=1 at G=0] ----
    w, ws, nonsym, sc, per = perm_viol(G0, alpha, ntran, tnp, two_leg=False)
    out["L2_G0_zeta_head"] = {"max_rel": w, "worst_op": ws, "worst_nonsym": nonsym, "scale": sc}
    print(f"  L2 G0 ζ(G=0) [tau-blind] max_rel={w:.3e} worst_op={ws} nonsym={nonsym}", flush=True)

    # ---- L3: V0 raw ----
    w, ws, nonsym, sc, per = perm_viol(V0, alpha, ntran, tnp, two_leg=True)
    out["L3_V0_raw"] = {"max_rel": w, "worst_op": ws, "worst_nonsym": nonsym, "scale": sc}
    # split: violation on symmorphic vs nonsymmorphic ops
    per = np.array(per)
    sym_mask = np.linalg.norm(tnp, axis=1) <= 1e-6
    out["L3_V0_raw"]["max_symmorphic"] = float(per[sym_mask].max())
    out["L3_V0_raw"]["max_nonsymmorphic"] = float(per[~sym_mask].max())
    print(f"  L3 V0 raw               max_rel={w:.3e} worst_op={ws} nonsym={nonsym}"
          f"  [symm={per[sym_mask].max():.3e} nonsymm={per[~sym_mask].max():.3e}]", flush=True)

    # ---- L4: W0 raw ----
    w, ws, nonsym, sc, per4 = perm_viol(W0, alpha, ntran, tnp, two_leg=True)
    per4 = np.array(per4)
    out["L4_W0_raw"] = {"max_rel": w, "worst_op": ws, "worst_nonsym": nonsym, "scale": sc,
                        "max_symmorphic": float(per4[sym_mask].max()),
                        "max_nonsymmorphic": float(per4[~sym_mask].max())}
    print(f"  L4 W0 raw               max_rel={w:.3e} worst_op={ws} nonsym={nonsym}"
          f"  [symm={per4[sym_mask].max():.3e} nonsymm={per4[~sym_mask].max():.3e}]", flush=True)

    # ---- L5: ΔW = W0 - V0 (screening correction) ----
    dW = W0 - V0
    w, ws, nonsym, sc, perd = perm_viol(dW, alpha, ntran, tnp, two_leg=True)
    perd = np.array(perd)
    out["L5_dW_screening"] = {"max_rel": w, "worst_op": ws, "worst_nonsym": nonsym, "scale": sc,
                              "max_symmorphic": float(perd[sym_mask].max()),
                              "max_nonsymmorphic": float(perd[~sym_mask].max())}
    print(f"  L5 ΔW=(W0-V0)           max_rel={w:.3e} worst_op={ws} nonsym={nonsym}"
          f"  [symm={perd[sym_mask].max():.3e} nonsymm={perd[~sym_mask].max():.3e}]", flush=True)

    # correlation: is W0's per-op violation ~ V0's? (W inherits V, adds nothing)
    corr = float(np.corrcoef(np.array(per), per4)[0, 1])
    out["V0_W0_perop_corr"] = corr
    print(f"  corr(V0 per-op viol, W0 per-op viol) = {corr:.4f}"
          f"  (≈1 ⇒ W just inherits V's non-covariance)", flush=True)

    json.dump(out, open(f"{RUN}/diag2/bisect_{arm}.json", "w"), indent=2)
    print(f"  wrote bisect_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
