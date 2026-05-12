#!/usr/bin/env python3
"""GN-PPM comparison: LORRAX eqp_g0w0.dat vs BGW sigma_hp.log.

Per skill compare/SKILL.md §4g:
  LORRAX Σ_xc(E_DFT) = Re[eqp_g0w0.dat] - KIH
  BGW    Σ_xc(E_DFT) = Sig (col 6 in sigma_hp.log)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LORRAX_DIR = HERE
BGW_LOG = HERE.parent / "03_bgw_gnppm_noavg" / "sigma_hp.log"


def parse_kih(path: Path) -> dict[tuple[float, float, float], dict[int, float]]:
    """Returns {(kx,ky,kz): {band1ix: KIH_eV}}."""
    out: dict = {}
    cur_k = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        p = s.split()
        if len(p) == 5:
            try:
                kx, ky, kz = float(p[0]), float(p[1]), float(p[2])
                cur_k = (kx, ky, kz)
                out[cur_k] = {}
            except ValueError:
                pass
            continue
        if len(p) >= 4 and cur_k is not None:
            try:
                ik_dummy = int(p[0])
                n = int(p[1])
                kih_re = float(p[2])
                out[cur_k][n] = kih_re
            except ValueError:
                pass
    return out


def parse_eqp_g0w0(path: Path) -> dict[int, dict[int, complex]]:
    """eqp_g0w0.dat → {kpt_idx_0based: {band_0based: complex Re[H0+Σ_xc]}}."""
    out: dict = {}
    cur_k = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        m = re.match(r"k-point\s+(\d+):", s)
        if m:
            cur_k = int(m.group(1))
            out[cur_k] = {}
            continue
        m2 = re.match(r"n=(\d+)\s+E_DFT=\s*([-\d.]+)\s+Re=\s*([-\d.]+)\s+Im=\s*([-\d.]+)", s)
        if m2 and cur_k is not None:
            n0 = int(m2.group(1))
            re_v = float(m2.group(3))
            im_v = float(m2.group(4))
            out[cur_k][n0] = complex(re_v, im_v)
    return out


def parse_bgw_sigma_hp(path: Path):
    blocks = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)", s)
        if not m:
            i += 1
            continue
        kc = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        ik_bgw = int(m.group(4))
        i += 1
        while i < len(lines) and not re.match(r"\s+n\s+Emf", lines[i]):
            i += 1
        i += 1
        bands = {}
        while i < len(lines):
            s2 = lines[i].strip()
            if not s2:
                break
            p = s2.split()
            if len(p) >= 14 and p[0].isdigit():
                n = int(p[0])
                bands[n] = {
                    "Emf": float(p[1]), "Eo": float(p[2]),
                    "X": float(p[3]), "SXmX": float(p[4]), "CH": float(p[5]),
                    "Sig": float(p[6]), "KIH": float(p[7]),
                    "Eqp0": float(p[8]), "Eqp1": float(p[9]),
                    "CHp": float(p[10]), "Sigp": float(p[11]),
                }
                i += 1
                continue
            break
        if bands:
            blocks.append({"kcrys": kc, "ik_bgw": ik_bgw, "bands": bands})
        i += 1
    return blocks


def get_lorrax_kpts(wfn_path: Path) -> np.ndarray:
    src = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
    sys.path.insert(0, str(src))
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--omega-window", type=float, default=14.5,
                   help="|E_DFT - E_F| max in eV (skip bands outside grid coverage)")
    args = p.parse_args()
    bgw = parse_bgw_sigma_hp(BGW_LOG)
    lr = parse_eqp_g0w0(LORRAX_DIR / "eqp_g0w0.dat")
    kih = parse_kih(LORRAX_DIR / "kih.dat")
    lx_kpts = get_lorrax_kpts(LORRAX_DIR / "WFN.h5")

    # Determine E_F (midgap on BGW side as proxy)
    all_emf = []
    for blk in bgw:
        for n_b, bd in blk["bands"].items():
            all_emf.append((n_b, bd["Emf"]))
    # Si has 8 occupied bands per spinor
    n_occ = 8
    valence = [e for n, e in all_emf if n <= n_occ]
    conduction = [e for n, e in all_emf if n > n_occ]
    vbm = max(valence)
    cbm = min(conduction)
    efermi = 0.5 * (vbm + cbm)
    print(f"VBM={vbm:.4f} CBM={cbm:.4f} E_F(midgap)={efermi:.4f} eV")
    print(f"Filtering: |E_DFT - E_F| <= {args.omega_window:.1f} eV\n")

    rows = []
    for blk in bgw:
        ki = match_k(blk["kcrys"], lx_kpts)
        if ki is None or ki not in lr:
            continue
        kih_for_k = kih.get(blk["kcrys"])
        if kih_for_k is None:
            for kk, vv in kih.items():
                if max(abs(a - b - round(a - b)) for a, b in zip(kk, blk["kcrys"])) < 1e-4:
                    kih_for_k = vv
                    break
        for n_b, bd in sorted(blk["bands"].items()):
            n0 = n_b - 1
            if n0 not in lr[ki]:
                continue
            re_h0_plus_sxc = lr[ki][n0].real
            im_h0_plus_sxc = lr[ki][n0].imag
            kih_v = kih_for_k.get(n_b) if kih_for_k else None
            if kih_v is None:
                continue
            # Filter bands outside the LORRAX omega grid window
            omega_rel = bd["Emf"] - efermi
            if abs(omega_rel) > args.omega_window:
                continue
            if not np.isfinite(re_h0_plus_sxc) or not np.isfinite(im_h0_plus_sxc):
                continue
            lorrax_sig = re_h0_plus_sxc - kih_v
            bgw_sig = bd["Sig"]
            bgw_sigp = bd["Sigp"]
            rows.append({
                "ki": ki, "n": n_b, "kc": blk["kcrys"],
                "omega_rel": omega_rel,
                "lorrax_sig": lorrax_sig,
                "lorrax_im": im_h0_plus_sxc,
                "bgw_sig": bgw_sig,
                "bgw_sigp": bgw_sigp,
                "diff": lorrax_sig - bgw_sig,
                "diff_p": lorrax_sig - bgw_sigp,
            })

    finite = [r for r in rows if np.isfinite(r["diff"])]
    n_total = len(finite)
    diffs = np.array([r["diff"] for r in finite])
    diffs_p = np.array([r["diff_p"] for r in finite])
    print(f"Matched {n_total} (k, n) pairs")
    print(f"MAE  vs BGW Sig  (unprimed) = {np.mean(np.abs(diffs))*1000:.2f} meV")
    print(f"MAE  vs BGW Sig' (primed)   = {np.mean(np.abs(diffs_p))*1000:.2f} meV")
    print(f"max  vs BGW Sig  (unprimed) = {np.max(np.abs(diffs))*1000:.2f} meV")
    print(f"max  vs BGW Sig' (primed)   = {np.max(np.abs(diffs_p))*1000:.2f} meV")

    # Show worst-offending bands
    print("\nLargest |diff| (vs Sig'):")
    sorted_rows = sorted(finite, key=lambda r: -abs(r["diff_p"]))
    sigp_label = "BGW_Sigp"
    print(f"  {'ki':>3} {'n':>3} {'kc':>22}  {'LORRAX':>10} {'BGW_Sig':>10} {sigp_label:>10}  {'D':>9}  {'Im_LX':>9}")
    for r in sorted_rows[:15]:
        kstr = f"({r['kc'][0]:+.3f},{r['kc'][1]:+.3f},{r['kc'][2]:+.3f})"
        print(f"  {r['ki']:3d} {r['n']:3d} {kstr:>22}  "
              f"{r['lorrax_sig']:10.4f} {r['bgw_sig']:10.4f} {r['bgw_sigp']:10.4f}  "
              f"{r['diff_p']:+9.4f}  {r['lorrax_im']:+9.4f}")

    # Im at occupied vs empty
    print("\nMax |Im[H0+Σ_xc(E_DFT)]| from LORRAX:")
    by_occ = {"valence_n<=8": [], "conduction_n>=9": []}
    for r in finite:
        if r["n"] <= 8:
            by_occ["valence_n<=8"].append(abs(r["lorrax_im"]))
        else:
            by_occ["conduction_n>=9"].append(abs(r["lorrax_im"]))
    for k, vs in by_occ.items():
        if vs:
            print(f"  {k}: max={max(vs):.3f} eV, mean={np.mean(vs):.3f} eV")


if __name__ == "__main__":
    main()
