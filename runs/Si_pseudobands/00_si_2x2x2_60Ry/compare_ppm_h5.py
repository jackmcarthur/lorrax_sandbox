#!/usr/bin/env python3
"""GN-PPM comparison from sigma_mnk.h5 directly — bypasses broken eqp_g0w0 interp.

Adapted from runs/Si/02_si_4x4x4_nosym/D_lorrax_gnppm_overlay_1440c_noavg/compare_gnppm_h5.py.
Reports MAE for nb=60 PROT and nb={108..1208} parabands runs vs the matching BGW PPM run.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry")
SRC = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
sys.path.insert(0, str(SRC))


def parse_kih(path):
    out = {}
    cur = None
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        p = s.split()
        if len(p) == 5:
            try:
                cur = (float(p[0]), float(p[1]), float(p[2]))
                out[cur] = {}
            except ValueError:
                pass
            continue
        if len(p) >= 4 and cur is not None:
            try:
                n = int(p[1])
                out[cur][n] = float(p[2])
            except ValueError:
                pass
    return out


def parse_bgw(path):
    blocks = []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)", lines[i])
        if not m:
            i += 1
            continue
        kc = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        i += 1
        while i < len(lines) and not re.match(r"\s+n\s+Emf", lines[i]):
            i += 1
        i += 1
        bands = {}
        while i < len(lines):
            t = lines[i].strip()
            if not t:
                break
            p = t.split()
            if len(p) >= 14 and p[0].isdigit():
                n = int(p[0])
                bands[n] = {"Emf": float(p[1]),
                            "X": float(p[3]), "SXX": float(p[4]),
                            "CH": float(p[5]), "Sig": float(p[6]),
                            "CHp": float(p[10]), "Sigp": float(p[11])}
                i += 1
            else:
                break
        if bands:
            blocks.append({"kc": kc, "bands": bands})
        i += 1
    return blocks


def get_lorrax_unfolded_kpts(wfn_path):
    from file_io import WFNReader
    from common import symmetry_maps
    w = WFNReader(str(wfn_path))
    sym = symmetry_maps.SymMaps(w)
    return np.asarray(sym.unfolded_kpts, dtype=np.float64)


def match_k(kc, lx_kpts, tol=1e-4):
    target = np.array(kc)
    for i, k in enumerate(lx_kpts):
        d = target - k
        d -= np.round(d)
        if np.max(np.abs(d)) < tol:
            return i
    return None


def evaluate(label, lx_dir, h5_name, bgw_dir, ref_col="Sig"):
    h5_path = lx_dir / h5_name
    if not h5_path.exists():
        return {"label": label, "status": f"missing {h5_name}"}
    bgw = parse_bgw(bgw_dir / "sigma_hp.log")
    kih = parse_kih(lx_dir / "kih.dat")
    lx_kpts = get_lorrax_unfolded_kpts(lx_dir / "WFN.h5")

    with h5py.File(h5_path, "r") as h:
        omega = h["omega_ev"][:]
        sig_c = h["sigma_c_kij_ev"][:]      # (Nω, nk, nb, nb)
        sig_sx = h["sigma_sx_kij_ev"][:]    # (nk, nb, nb)
        sig_c_diag = np.diagonal(sig_c, axis1=2, axis2=3)
        sig_sx_diag = np.diagonal(sig_sx, axis1=1, axis2=2)
    nk_h5 = sig_sx_diag.shape[0]
    nb_h5 = sig_sx_diag.shape[1]

    n_occ = 8
    all_emf = [(n, b["Emf"]) for blk in bgw for n, b in blk["bands"].items()]
    vbm = max(e for n, e in all_emf if n <= n_occ)
    cbm = min(e for n, e in all_emf if n > n_occ)
    ef = 0.5 * (vbm + cbm)

    om_min, om_max = float(omega.min()), float(omega.max())
    diffs = []
    n_nan = 0
    for blk in bgw:
        ki = match_k(blk["kc"], lx_kpts)
        if ki is None or ki >= nk_h5:
            continue
        for n, bd in blk["bands"].items():
            n0 = n - 1
            if n0 >= nb_h5:
                continue
            omega_rel = bd["Emf"] - ef
            if not (om_min <= omega_rel <= om_max):
                # Out of grid — emit NaN, don't clamp.
                n_nan += 1
                continue
            re_c = float(np.interp(omega_rel, omega, sig_c_diag[:, ki, n0].real))
            sigma_xc = float(sig_sx_diag[ki, n0].real) + re_c
            diffs.append(sigma_xc - bd[ref_col])

    arr = np.array(diffs) if diffs else np.array([np.nan])
    return {"label": label, "n": len(diffs), "n_nan": n_nan,
            "MAE_meV": (np.mean(np.abs(arr)) * 1000) if diffs else float("nan"),
            "max_meV": (np.max(np.abs(arr)) * 1000) if diffs else float("nan"),
            "median_meV": (np.median(np.abs(arr)) * 1000) if diffs else float("nan"),
            "ref": ref_col, "om_range": (om_min, om_max)}


def main():
    cases = [
        ("nb=60 PROT (sym 4kpt)", ROOT/"D_lorrax_canonical_gnppm_test", "sigma_mnk.h5",     ROOT/"D_bgw_canonical_gnppm_noavg",   "Sig"),
        ("nb=60 PROT (nosym 8k)", ROOT/"D_lorrax_canonical_gnppm_test", "sigma_mnk.h5",     ROOT/"D_bgw_canonical_gnppm_nosym",   "Sig"),
        ("nb=108  vs Sig",        ROOT/"D_lorrax_para_108_gnppm",       "sigma_mnk_w10.h5", ROOT/"D_bgw_para_108_gnppm_noavg",   "Sig"),
        ("nb=108  vs Sig'",       ROOT/"D_lorrax_para_108_gnppm",       "sigma_mnk_w10.h5", ROOT/"D_bgw_para_108_gnppm_noavg",   "Sigp"),
        ("nb=208  vs Sig",        ROOT/"D_lorrax_para_208_gnppm",       "sigma_mnk_3264.h5",ROOT/"D_bgw_para_208_gnppm_noavg",   "Sig"),
        ("nb=208  vs Sig'",       ROOT/"D_lorrax_para_208_gnppm",       "sigma_mnk_3264.h5",ROOT/"D_bgw_para_208_gnppm_noavg",   "Sigp"),
        ("nb=408  vs Sig",        ROOT/"D_lorrax_para_408_gnppm",       "sigma_mnk_w10.h5", ROOT/"D_bgw_para_408_gnppm_noavg",   "Sig"),
        ("nb=408  vs Sig'",       ROOT/"D_lorrax_para_408_gnppm",       "sigma_mnk_w10.h5", ROOT/"D_bgw_para_408_gnppm_noavg",   "Sigp"),
        ("nb=808  vs Sig",        ROOT/"D_lorrax_para_808_gnppm",       "sigma_mnk_3264.h5",ROOT/"D_bgw_para_808_gnppm_noavg",   "Sig"),
        ("nb=808  vs Sig'",       ROOT/"D_lorrax_para_808_gnppm",       "sigma_mnk_3264.h5",ROOT/"D_bgw_para_808_gnppm_noavg",   "Sigp"),
        ("nb=1208 vs Sig",        ROOT/"D_lorrax_para_1208_gnppm",      "sigma_mnk_w10.h5", ROOT/"D_bgw_para_1208_gnppm_noavg",  "Sig"),
        ("nb=1208 vs Sig'",       ROOT/"D_lorrax_para_1208_gnppm",      "sigma_mnk_w10.h5", ROOT/"D_bgw_para_1208_gnppm_noavg",  "Sigp"),
    ]
    print(f"{'case':<25} {'in_grid':>7} {'NaN':>5} {'med (meV)':>10} {'MAE (meV)':>10} {'max (meV)':>10}")
    print("-" * 80)
    for c in cases:
        r = evaluate(*c)
        n = r.get("n", 0)
        if n == 0:
            print(f"{r['label']:<25} {'-':>7} {r.get('n_nan','-'):>5} {'-':>10} {'-':>10} {'-':>10}  ({r.get('status','no match')})")
            continue
        print(f"{r['label']:<25} {n:>7} {r.get('n_nan',0):>5} {r['median_meV']:>10.2f} {r['MAE_meV']:>10.2f} {r['max_meV']:>10.2f}")


if __name__ == "__main__":
    main()
