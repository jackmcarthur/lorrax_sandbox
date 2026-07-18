#!/usr/bin/env python3
"""Ridge-regularized ζ-fit A/B — analysis over the MoS2 gnppm + Si COHSEX arms.

Inputs (disk provenance):
  MoS2 arms: runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/
      {02_lorrax_gnppm_stock_ridgewt, 03_..ridge1em4, 04_..ridge1em5,
       05_..ridge1em6}/{eqp0.dat, eqp1.dat, sigma_diag.dat,
       tmp/isdf_tensors_642.h5, gw.out}
  Si arms:   runs/Si/B_zeta_ridge_covariance_2026-07-17/
      {work_stock, work_r1em4, work_r1em5, work_r1em6}/
      {eqp_si_test.dat, tmp/isdf_tensors_792.h5, gw.out}

Measurements:
  1. eqp0/eqp1 max|Δ| + MAE (meV) vs the stock arm (same worktree, same
     device count — isolates the ridge).
  2. sigma_diag: max|Δ| sigX / Re sigC (eV) vs stock.
  3. Covariance ladder under the centroid permutation α (the diag2
     machinery of runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16):
     L2 G0=ζ̃_0(G=0), L3 V0, L4 W0, L5 ΔW=W0−V0; per-arm max_rel.
  4. cond(C_q) / cond after ridge: exact eigh of the reconstructed fit
     CCT at q=0 (pair windows read from gw.out are full-band L=R), plus
     the power-iteration λ̂ from the gw.out ridge banner as cross-check.
"""
import json
import os
import re
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")  # orbit_syms needs int64

import numpy as np
import h5py

SB = "/pscratch/sd/j/jackm/lorrax_sandbox"
SRC = f"{SB}/sources/worktrees/lorrax_A_ridge_wt/src"
sys.path.insert(0, SRC)

RY = 13.6056980659

MOS2_RUN = f"{SB}/runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02"
MOS2_ARMS = {
    "stock": "02_lorrax_gnppm_stock_ridgewt",
    "1e-4":  "03_lorrax_gnppm_ridge1em4",
    "1e-5":  "04_lorrax_gnppm_ridge1em5",
    "1e-6":  "05_lorrax_gnppm_ridge1em6",
}
SI_RUN = f"{SB}/runs/Si/B_zeta_ridge_covariance_2026-07-17"
SI_ARMS = {
    "stock": "work_stock",
    "1e-4":  "work_r1em4",
    "1e-5":  "work_r1em5",
    "1e-6":  "work_r1em6",
}


def parse_eqp(path):
    """LORRAX eqp0/eqp1.dat (BGW eqp layout): k-header `kx ky kz nb`
    then `spin band Edft Eqp` rows.  Returns (nk*nb, 2) [Edft, Eqp]."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 4 and "." in parts[0]:
                continue  # k header
            if len(parts) == 4:
                rows.append((float(parts[2]), float(parts[3])))
    return np.asarray(rows)


_SIG_RE = re.compile(
    r"n=(\d+)\s+sigX=\s*(-?\d+\.\d+)\s+sigC=\s*(-?\d+\.\d+)\+\s*(-?\d+\.\d+)i")


def parse_sigma_diag(path):
    """Rows (sigX, Re sigC, Im sigC) in file order."""
    out = []
    with open(path) as f:
        for line in f:
            m = _SIG_RE.match(line.strip())
            if m:
                out.append((float(m.group(2)), float(m.group(3)),
                            float(m.group(4))))
    return np.asarray(out)


def perm_viol(T, alpha, two_leg):
    """Max relative covariance violation over spatial ops (diag2
    machinery: T[ix(a,a)] vs T for tiles, T[a] vs T for vectors)."""
    sc = np.abs(T).max()
    worst = 0.0
    per = []
    for a in alpha:
        if two_leg:
            d = np.abs(T[np.ix_(a, a)] - T).max()
        else:
            d = np.abs(T[a] - T).max()
        r = float(d / sc)
        per.append(r)
        worst = max(worst, r)
    return worst, per, float(sc)


def centroid_alpha(wfn_path, cent_path, fft):
    from file_io.wfn_loader import WfnLoader
    from centroid.orbit_syms import compute_centroid_sym_perm
    wfn = WfnLoader(wfn_path, backend="eager")
    ntran = int(wfn.ntran)
    cfrac = np.loadtxt(cent_path)
    ridx = np.rint(cfrac * fft[None]).astype(np.int64) % fft[None]
    alpha, _ = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran],
        np.asarray(fft, dtype=np.int64), validate=True)
    return alpha, ntran


def restart_path(arm_dir):
    tmp = os.path.join(arm_dir, "tmp")
    cands = [f for f in os.listdir(tmp)
             if f.startswith("isdf_tensors_") and f.endswith(".h5")]
    assert len(cands) == 1, (arm_dir, cands)
    return os.path.join(tmp, cands[0])


def cov_ladder(arm_dir, alpha):
    with h5py.File(restart_path(arm_dir), "r") as f:
        V0 = np.asarray(f["V_qmunu"][0])
        W0 = np.asarray(f["W0_qmunu"][0])
        G0 = np.asarray(f["G0_mu_nu"][:])
    out = {}
    for tag, T, two in (("G0", G0, False), ("V0", V0, True),
                        ("W0", W0, True), ("dW", W0 - V0, True)):
        worst, per, sc = perm_viol(T, alpha, two)
        out[tag] = {"max_rel": worst, "scale": sc}
    return out


def ridge_banner(arm_dir, fname="gw.out"):
    """Parse the factor_c_q ridge banner: eps_rel + λ̂ range."""
    txt = open(os.path.join(arm_dir, fname)).read()
    m = re.search(
        r"ζ-ridge: eps_rel=([\d.e+-]+), λ̂_max\(C_q\) ∈ "
        r"\[([\d.e+-]+), ([\d.e+-]+)\]", txt)
    if not m:
        return None
    return {"eps_rel": float(m.group(1)),
            "lam_min_q": float(m.group(2)),
            "lam_max_q": float(m.group(3))}


def fit_c0_eig(arm_dir):
    """Exact fit-window CCT at q=0 from the restart ψ(centroids):
    ρ^k_ab(μ) = Σ_s ψ*_{a,k,s}(μ) ψ_{b,k,s}(μ) over the FULL band
    window both legs (gw.out: '80 left + 80 right' / Si '60 + 60'),
    C0[μ,ν] = Σ_{kab} ρ^k_ab(μ)* ρ^k_ab(ν).  Returns eigvalsh."""
    with h5py.File(restart_path(arm_dir), "r") as f:
        psi = np.asarray(f["psi_full_y"][:])  # (nk, nb, ns, nmu)
    nk, nb, ns, nmu = psi.shape
    rho = np.einsum("kasm,kbsm->kabm", np.conj(psi), psi, optimize=True)
    C0 = np.einsum("kabm,kabn->mn", np.conj(rho), rho, optimize=True)
    C0 = 0.5 * (C0 + C0.conj().T)
    return np.linalg.eigvalsh(C0)


def si_sigma_table(run, arms, fname, nb, gap_lo, gap_hi):
    """Si COHSEX Σ file (sigSX/sigCOH/sigTOT rows) vs stock, via the
    harness's own parse_eqp_rows.  Gap window on the 0-based band col
    (VBM n=7, CBM n=8 for nval=8): 1-based [gap_lo, gap_hi]."""
    sys.path.insert(0, os.path.join(os.path.dirname(SRC), "tests"))
    from harness import parse_eqp_rows
    from pathlib import Path
    labels = ("sigSX", "sigCOH", "sigTOT")
    stock = parse_eqp_rows(Path(run) / arms["stock"] / fname, labels)
    band_1b = stock[:, 1].astype(int) + 1
    sel = (band_1b >= gap_lo) & (band_1b <= gap_hi)
    out = {}
    for eps, arm in arms.items():
        if eps == "stock":
            continue
        cur = parse_eqp_rows(Path(run) / arm / fname, labels)
        assert cur.shape == stock.shape
        d = np.abs(cur[:, 2:5] - stock[:, 2:5]) * 1e3  # meV, 3 Σ cols
        out[eps] = {
            "max_dSigma_meV": float(d.max()),
            "max_dSigTOT_meV": float(d[:, 2].max()),
            "gap_max_dSigTOT_meV": float(d[sel, 2].max()),
            "mae_dSigTOT_meV": float(d[:, 2].mean()),
        }
    return out


def eqp_table(run, arms, eqp_names, nb, gap_lo, gap_hi, clean_mask=None):
    """Δeqp vs the stock arm.  Per file: all-band max/MAE, gap-window
    (1-based bands [gap_lo, gap_hi]) max, and (when clean_mask given)
    max over PPM-clean bands (|Im Σc| < 1 eV at Edft — pole-adjacent
    bands have meaningless eqp sensitivity)."""
    stock_dir = os.path.join(run, arms["stock"])
    out = {}
    for name in eqp_names:
        ref = parse_eqp(os.path.join(stock_dir, name))
        nrows = ref.shape[0]
        band_1b = (np.arange(nrows) % nb) + 1
        gap_sel = (band_1b >= gap_lo) & (band_1b <= gap_hi)
        for eps, arm in arms.items():
            if eps == "stock":
                continue
            cur = parse_eqp(os.path.join(run, arm, name))
            assert cur.shape == ref.shape
            # Edft must be identical (same WFN/V_xc path)
            dE0 = np.abs(cur[:, 0] - ref[:, 0]).max()
            d = (cur[:, 1] - ref[:, 1]) * 1e3  # meV
            ent = {
                "max_abs_meV": float(np.abs(d).max()),
                "mae_meV": float(np.abs(d).mean()),
                "gap_window_max_meV": float(np.abs(d[gap_sel]).max()),
                "edft_check_meV": float(dE0 * 1e3),
            }
            if clean_mask is not None:
                ent["ppm_clean_max_meV"] = float(np.abs(d[clean_mask]).max())
                ent["n_ppm_clean"] = int(clean_mask.sum())
            out.setdefault(eps, {})[name] = ent
    return out


def eqp_pairwise(run, arms, name, nb, gap_lo, gap_hi):
    """Arm-to-arm Δeqp among the ridge arms — discriminates filter
    magnitude (scales with ε) from chaotic junk-mode noise (flat)."""
    keys = [k for k in arms if k != "stock"]
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra = parse_eqp(os.path.join(run, arms[a], name))
            rb = parse_eqp(os.path.join(run, arms[b], name))
            d = np.abs(ra[:, 1] - rb[:, 1]) * 1e3
            band_1b = (np.arange(len(d)) % nb) + 1
            sel = (band_1b >= gap_lo) & (band_1b <= gap_hi)
            out[f"{a}_vs_{b}"] = {"max_abs_meV": float(d.max()),
                                  "gap_window_max_meV": float(d[sel].max())}
    return out


def tile_diffs(run, arms):
    """V0/W0 tile stock-vs-ridge: rel Frobenius + max-element rel."""
    with h5py.File(restart_path(os.path.join(run, arms["stock"])), "r") as f:
        V0s = np.asarray(f["V_qmunu"][0])
        W0s = np.asarray(f["W0_qmunu"][0])
    out = {}
    for eps, arm in arms.items():
        if eps == "stock":
            continue
        with h5py.File(restart_path(os.path.join(run, arm)), "r") as f:
            V0 = np.asarray(f["V_qmunu"][0])
            W0 = np.asarray(f["W0_qmunu"][0])
        out[eps] = {
            "V0_relF": float(np.linalg.norm(V0 - V0s) / np.linalg.norm(V0s)),
            "V0_max_rel": float(np.abs(V0 - V0s).max() / np.abs(V0s).max()),
            "W0_relF": float(np.linalg.norm(W0 - W0s) / np.linalg.norm(W0s)),
            "W0_max_rel": float(np.abs(W0 - W0s).max() / np.abs(W0s).max()),
        }
    return out


def sigma_table(run, arms, fname, nb=80, gap_lo=23, gap_hi=30):
    stock = parse_sigma_diag(os.path.join(run, arms["stock"], fname))
    band_1b = (np.arange(stock.shape[0]) % nb) + 1
    sel = (band_1b >= gap_lo) & (band_1b <= gap_hi)
    out = {}
    for eps, arm in arms.items():
        if eps == "stock":
            continue
        cur = parse_sigma_diag(os.path.join(run, arm, fname))
        assert cur.shape == stock.shape
        dX = np.abs(cur[:, 0] - stock[:, 0]) * 1e3
        dC = np.abs(cur[:, 1] - stock[:, 1]) * 1e3
        out[eps] = {
            "max_dsigX_meV": float(dX.max()),
            "max_dResigC_meV": float(dC.max()),
            "gap_max_dsigX_meV": float(dX[sel].max()),
            "gap_max_dResigC_meV": float(dC[sel].max()),
            "mae_dsigX_meV": float(dX.mean()),
        }
    return out


def main():
    res = {}

    # ---------- MoS2 ----------
    print("=== MoS2 gnppm fixture ===", flush=True)
    # PPM-clean mask from the STOCK sigma_diag: pole-adjacent bands
    # (|Im Σc(Edft)| ≥ 1 eV) have divergent dEqp/dΣ — their eqp deltas
    # are pole-crossing noise, not a ridge observable.
    sd_stock = parse_sigma_diag(
        os.path.join(MOS2_RUN, MOS2_ARMS["stock"], "sigma_diag.dat"))
    clean = np.abs(sd_stock[:, 2]) < 1.0
    print(f"  PPM-clean bands: {clean.sum()}/{len(clean)}")
    # nval=26 → gap window bands 23..30 (1-based; VBM=26, CBM=27)
    res["mos2_eqp"] = eqp_table(MOS2_RUN, MOS2_ARMS,
                                ["eqp0.dat", "eqp1.dat"], nb=80,
                                gap_lo=23, gap_hi=30, clean_mask=clean)
    res["mos2_eqp_pairwise"] = eqp_pairwise(
        MOS2_RUN, MOS2_ARMS, "eqp0.dat", nb=80, gap_lo=23, gap_hi=30)
    res["mos2_sigma"] = sigma_table(MOS2_RUN, MOS2_ARMS, "sigma_diag.dat")
    res["mos2_tiles"] = tile_diffs(MOS2_RUN, MOS2_ARMS)
    # Baseline chaos floor: this worktree's stock arm vs the ORIGINAL
    # 00_lorrax_gnppm run (different code era AND device count 4→1) —
    # upper bound on the fixture's non-ridge sensitivity.
    orig = parse_eqp(f"{MOS2_RUN}/00_lorrax_gnppm/eqp0.dat")
    st = parse_eqp(os.path.join(MOS2_RUN, MOS2_ARMS["stock"], "eqp0.dat"))
    if orig.shape == st.shape:
        d0 = np.abs(orig[:, 1] - st[:, 1]) * 1e3
        band_1b = (np.arange(len(d0)) % 80) + 1
        sel = (band_1b >= 23) & (band_1b <= 30)
        res["mos2_baseline_vs_orig"] = {
            "max_abs_meV": float(d0.max()),
            "gap_window_max_meV": float(d0[sel].max()),
            "ppm_clean_max_meV": float(d0[clean].max()),
        }
    print(json.dumps(res["mos2_eqp"], indent=1))
    print(json.dumps(res["mos2_eqp_pairwise"], indent=1))
    print(json.dumps(res["mos2_sigma"], indent=1))
    print(json.dumps(res["mos2_tiles"], indent=1))
    print(json.dumps(res.get("mos2_baseline_vs_orig", {}), indent=1))

    alpha_m, ntran_m = centroid_alpha(
        os.path.join(MOS2_RUN, MOS2_ARMS["stock"], "WFN.h5"),
        os.path.join(MOS2_RUN, MOS2_ARMS["stock"], "centroids_frac_642.txt"),
        np.array([24, 24, 80]))
    res["mos2_cov"] = {}
    for eps, arm in MOS2_ARMS.items():
        res["mos2_cov"][eps] = cov_ladder(os.path.join(MOS2_RUN, arm), alpha_m)
        print(f"  [cov mos2 {eps}] " + " ".join(
            f"{k}={v['max_rel']:.3e}" for k, v in res['mos2_cov'][eps].items()),
            flush=True)

    lam = fit_c0_eig(os.path.join(MOS2_RUN, MOS2_ARMS["stock"]))
    res["mos2_cond"] = {"lam_max": float(lam[-1]), "lam_min": float(lam[0]),
                        "cond_C": float(lam[-1] / max(lam[0], 1e-300))}
    for eps_s in ("1e-4", "1e-5", "1e-6"):
        e = float(eps_s) * lam[-1]
        res["mos2_cond"][f"cond_B_{eps_s}"] = float(
            (lam[-1] ** 2 + e ** 2) / (lam[0] ** 2 + e ** 2))
        res["mos2_cond"][f"amp_cap_{eps_s}"] = float(1.0 / (2 * e))
        b = ridge_banner(os.path.join(MOS2_RUN, MOS2_ARMS[eps_s]))
        res["mos2_cond"][f"banner_{eps_s}"] = b
    print(json.dumps(res["mos2_cond"], indent=1))

    # ---------- Si ----------
    print("=== Si COHSEX fixture ===", flush=True)
    # nval=8 → gap window bands 5..12 (1-based); static COHSEX (no PPM
    # poles → no clean mask needed).  eqp_si_test.dat is sigma_diag
    # format (sigSX/sigCOH/sigTOT) — use parse_eqp_rows-style totals
    # via the sigma parser instead of the eqp parser.
    res["si_sigma"] = si_sigma_table(SI_RUN, SI_ARMS, "eqp_si_test.dat",
                                     nb=None, gap_lo=5, gap_hi=12)
    res["si_tiles"] = tile_diffs(SI_RUN, SI_ARMS)
    print(json.dumps(res["si_sigma"], indent=1))
    print(json.dumps(res["si_tiles"], indent=1))

    alpha_s, ntran_s = centroid_alpha(
        os.path.join(SI_RUN, SI_ARMS["stock"], "WFN.h5"),
        os.path.join(SI_RUN, SI_ARMS["stock"], "centroids_frac_792.txt"),
        np.array([24, 24, 24]))
    res["si_cov"] = {}
    for eps, arm in SI_ARMS.items():
        res["si_cov"][eps] = cov_ladder(os.path.join(SI_RUN, arm), alpha_s)
        print(f"  [cov si {eps}] " + " ".join(
            f"{k}={v['max_rel']:.3e}" for k, v in res['si_cov'][eps].items()),
            flush=True)

    lam = fit_c0_eig(os.path.join(SI_RUN, SI_ARMS["stock"]))
    res["si_cond"] = {"lam_max": float(lam[-1]), "lam_min": float(lam[0]),
                      "cond_C": float(lam[-1] / max(lam[0], 1e-300))}
    for eps_s in ("1e-4", "1e-5", "1e-6"):
        e = float(eps_s) * lam[-1]
        res["si_cond"][f"cond_B_{eps_s}"] = float(
            (lam[-1] ** 2 + e ** 2) / (lam[0] ** 2 + e ** 2))
        b = ridge_banner(os.path.join(SI_RUN, SI_ARMS[eps_s]))
        res["si_cond"][f"banner_{eps_s}"] = b
    print(json.dumps(res["si_cond"], indent=1))

    out = f"{SB}/reports/zeta_ridge_ab_2026-07-17/ab_results.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
