#!/usr/bin/env python3
"""Measure q=0 tile covariance (G0, V0, W0) under the centroid sym permutation
for both Si arms: work_sym (fix OFF, 60-band screening) vs work_demo (fix ON,
40-band closed screening).  Covariance viol = max_op ||T[α,α] - T|| / ||T||.
"""
import os, sys, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm
from bse import bse_io
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24])


def viol_two_leg(T, alpha, ntran):
    sc = np.abs(T).max(); worst = 0.0; ws = -1
    for s in range(ntran):
        a = alpha[s]
        d = np.abs(T[np.ix_(a, a)] - T).max() / sc
        if d > worst:
            worst = d; ws = s
    return float(worst), ws


def viol_one_leg(v, alpha, ntran):
    sc = np.abs(v).max(); worst = 0.0
    for s in range(ntran):
        worst = max(worst, np.abs(v[alpha[s]] - v).max() / sc)
    return float(worst)


def main():
    wfn = WfnLoader(WFN, backend="eager")
    ntran = int(wfn.ntran)
    out = {}
    for arm in ("sym", "demo"):
        inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
        restart = bse_io._find_restart_file(inp)
        cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
        ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
        alpha, _ = compute_centroid_sym_perm(
            ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)
        with h5py.File(restart, "r") as f:
            nb = f["psi_full_y"].shape[1]
            V0 = np.asarray(f["V_qmunu"][0])
            W0 = np.asarray(f["W0_qmunu"][0])
            G0 = np.asarray(f["G0_mu_nu"][:])
        g = viol_one_leg(G0, alpha, ntran)
        v, vs = viol_two_leg(V0, alpha, ntran)
        w, wsn = viol_two_leg(W0, alpha, ntran)
        out[arm] = {"nb_screen": int(nb), "G0_viol": g, "V0_viol": v, "W0_viol": w,
                    "V0_worst_op": vs, "W0_worst_op": wsn}
        print(f"[{arm}] screening bands=[0,{nb})  G0 viol={g:.3e}  "
              f"V0 viol={v:.3e} (op {vs})  W0 viol={w:.3e} (op {wsn})", flush=True)
    json.dump(out, open(f"{RUN}/fix_validate/tile_cov.json", "w"), indent=2)
    print("wrote tile_cov.json", flush=True)


if __name__ == "__main__":
    main()
