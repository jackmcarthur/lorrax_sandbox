#!/usr/bin/env python3
"""DIAG2 Task-2 depth: localize the tile non-covariance INSIDE the ζ-fit.

Established by cov_ladder.py:
  * ψ@centroids covariant 1e-15; C0 CCT (fit INPUT) 0.4%; G0=ζ(G=0) (fit OUTPUT)
    8.6%; V0 3.2%; W0 3.0% (W inherits V, corr 0.997).
  * The 8.6% at G=0 is TAU-BLIND (phase=1) yet worst on a nonsymmorphic op —
    so "nonsymmorphic-worst" is base rate (36/48 ops), NOT a phase bug.

This probe tests the TWO-PART root cause:
  (a) SEED: the screening band window [0,8)x[8,60) cuts degenerate multiplets at
      its TOP (band 59), giving the 0.4% C0 non-covariance.  Test: rebuild C0
      with the conduction top truncated to each degeneracy-CLOSED boundary; the
      violation should fall toward machine precision for a fully closed window.
  (b) AMPLIFIER: the ζ solve ζ = C^{-1} Z is ill-conditioned; a 0.4% input
      non-covariance is amplified ~8-20x.  Test: cond(C0) and the spectral
      location of the covariance-violation (does it live in C0's small-λ modes?).

Usage: python3 zeta_probe.py sym
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
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24])


def build_C0(psi, occ, cond):
    """q=0 CCT pair-density Gram over occ x cond, spin-summed pair density."""
    psi_c = psi[:, cond, :, :]; psi_v = psi[:, occ, :, :]
    rho = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    return np.einsum("kcvm,kcvn->mn", np.conj(rho), rho, optimize=True)


def cov_viol(T, alpha, ntran, tnp):
    sc = np.abs(T).max(); worst = 0.0; ws = -1
    for s in range(ntran):
        a = alpha[s]
        d = np.abs(T[np.ix_(a, a)] - T).max()
        if d / sc > worst:
            worst = d / sc; ws = s
    return worst, ws, bool(np.linalg.norm(tnp[ws]) > 1e-6)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    ntran = int(wfn.ntran)
    tnp = np.asarray(wfn.translations[:ntran]) / (2 * np.pi)
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
    alpha, _ = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])
        enk = np.asarray(f["enk_full"][:])
        G0 = np.asarray(f["G0_mu_nu"][:])
    nk, nb, ns, nmu = psi.shape
    out = {"arm": arm}

    # ---- (a) SEED: C0 covariance vs screening conduction-top closure ----
    # closed conduction boundaries (gap>20meV at every k)
    thr = 20.0
    closed_b = [b for b in range(9, nb)
                if float(np.min(enk[:, b] - enk[:, b - 1])) * RY * 1e3 > thr]
    print(f"[{arm}] closed conduction tops (gap>{thr}meV all k): {closed_b}", flush=True)
    occ = list(range(0, 8))
    seed_scan = []
    # full production screening window uses ALL bands [8,nb)
    for ctop in closed_b + [nb]:
        cond = list(range(8, ctop))
        if not cond:
            continue
        C0 = build_C0(psi, occ, cond)
        w, ws, nonsym = cov_viol(C0, alpha, ntran, tnp)
        seed_scan.append({"cond_top": ctop, "closed": ctop in closed_b,
                          "n_cond": len(cond), "cov_viol": float(w),
                          "worst_op": int(ws), "nonsym": nonsym})
        print(f"  C0 screening [8,{ctop}) closed={ctop in closed_b:d}: "
              f"cov_viol={w:.3e} worst_op={ws} nonsym={nonsym}", flush=True)
    out["seed_C0_vs_screening_closure"] = seed_scan

    # ---- (b) AMPLIFIER: conditioning of the PRODUCTION C0 ----
    cond_full = list(range(8, nb))
    C0f = build_C0(psi, occ, cond_full)
    C0f = 0.5 * (C0f + C0f.conj().T)
    evals = np.linalg.eigvalsh(C0f)  # ascending
    evals = np.clip(evals, 0, None)
    lam_max = float(evals[-1]); lam_min = float(evals[0])
    # effective rank / conditioning at a few thresholds
    condition = lam_max / max(lam_min, 1e-300)
    rank_1e8 = int((evals > lam_max * 1e-8).sum())
    rank_1e10 = int((evals > lam_max * 1e-10).sum())
    rank_1e12 = int((evals > lam_max * 1e-12).sum())
    print(f"  cond(C0)={condition:.3e}  λmax={lam_max:.3e} λmin={lam_min:.3e}  "
          f"eff-rank(>1e-8/1e-10/1e-12·λmax)={rank_1e8}/{rank_1e10}/{rank_1e12} of {nmu}",
          flush=True)
    out["amplifier_C0_cond"] = {"cond": condition, "lam_max": lam_max, "lam_min": lam_min,
                                "n_mu": nmu, "eff_rank_1e8": rank_1e8,
                                "eff_rank_1e10": rank_1e10, "eff_rank_1e12": rank_1e12}

    # spectral location of the covariance violation: project the WORST-op
    # violation ΔC0[μ,ν] = C0[α(μ),α(ν)] - C0[μ,ν] onto C0's eigenbasis; measure
    # how much of ‖ΔC0‖ lives in the small-λ (≤1e-6·λmax) subspace.
    w, ws, _ = cov_viol(C0f, alpha, ntran, tnp)
    a = alpha[ws]
    dC = C0f[np.ix_(a, a)] - C0f
    U = np.linalg.eigh(C0f)[1]  # columns = eigenvectors, ascending λ
    dC_eig = U.conj().T @ dC @ U          # violation in eigenbasis (i,j)
    lam = evals
    small = lam <= lam_max * 1e-6
    tot = float(np.linalg.norm(dC_eig))
    # fraction of Frobenius norm with at least one small-λ leg
    small_mask = small[:, None] | small[None, :]
    frac_small = float(np.linalg.norm(dC_eig[small_mask]) / (tot + 1e-300))
    print(f"  ΔC0 (worst op {ws}) fraction of ‖·‖ touching small-λ (<1e-6·λmax) "
          f"modes: {frac_small:.3f}  (n_small={int(small.sum())})", flush=True)
    out["violation_in_small_lambda"] = {"worst_op": int(ws), "frac_small": frac_small,
                                        "n_small_modes": int(small.sum())}

    # ---- amplification ratio: ζ-output (G0) viol / C0-input viol ----
    # (G0 is a vector; measure its covariance directly)
    scG = np.abs(G0).max(); wG = 0.0
    for s in range(ntran):
        d = np.abs(G0[alpha[s]] - G0).max()
        wG = max(wG, d / scG)
    amp = wG / max(w, 1e-30)
    out["amplification_G0_over_C0"] = {"G0_viol": float(wG), "C0_viol": float(w),
                                       "ratio": float(amp)}
    print(f"  amplification: G0 viol {wG:.3e} / C0 viol {w:.3e} = {amp:.1f}x", flush=True)

    json.dump(out, open(f"{RUN}/diag2/zeta_probe_{arm}.json", "w"), indent=2)
    print(f"  wrote zeta_probe_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
