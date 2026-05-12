#!/usr/bin/env python3
"""GN-PPM comparison from sigma_mnk.h5 directly.

Bypasses eqp_g0w0.dat (which is broken: np.interp clamps out-of-grid).
Extracts diag(Σ_x + Σ_c(ω=E_DFT-E_F)) per (k, n) from H5, only for
bands inside the omega grid window. Compares to BGW Sig from sigma_hp.log.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
LORRAX_DIR = HERE
BGW_LOG = HERE.parent / "03_bgw_gnppm_noavg" / "sigma_hp.log"


def parse_kih(path: Path):
    out = {}
    cur = None
    for line in path.read_text().splitlines():
        s = line.strip()
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


def parse_bgw(path: Path):
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
                bands[n] = {"Emf": float(p[1]), "Sig": float(p[6]), "Sigp": float(p[11])}
                i += 1
                continue
            break
        if bands:
            blocks.append({"kc": kc, "bands": bands})
        i += 1
    return blocks


def get_lorrax_kpts():
    src = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
    sys.path.insert(0, str(src))
    from file_io import WFNReader
    from common import symmetry_maps
    w = WFNReader(str(LORRAX_DIR / "WFN.h5"))
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
    bgw = parse_bgw(BGW_LOG)
    kih = parse_kih(LORRAX_DIR / "kih.dat")
    lx_kpts = get_lorrax_kpts()

    with h5py.File(LORRAX_DIR / "sigma_mnk.h5", "r") as h:
        omega = h["omega_ev"][:]
        sig_c = h["sigma_c_kij_ev"][:]      # (Nω, nk, nb, nb)
        sig_sx = h["sigma_sx_kij_ev"][:]    # (nk, nb, nb)
        # Diagonals
        sig_c_diag = np.diagonal(sig_c, axis1=2, axis2=3)   # (Nω, nk, nb)
        sig_sx_diag = np.diagonal(sig_sx, axis1=1, axis2=2) # (nk, nb)

    nk = sig_sx_diag.shape[0]
    nb = sig_sx_diag.shape[1]
    print(f"omega grid: [{omega.min():.1f}, {omega.max():.1f}] eV, Nω={len(omega)}")

    # Find midgap from BGW Emf
    all_emf = [(n, bd["Emf"]) for blk in bgw for n, bd in blk["bands"].items()]
    n_occ = 8
    vbm = max(e for n, e in all_emf if n <= n_occ)
    cbm = min(e for n, e in all_emf if n > n_occ)
    efermi = 0.5 * (vbm + cbm)
    print(f"E_F(midgap) from BGW = {efermi:.4f} eV (VBM={vbm:.4f}, CBM={cbm:.4f})")

    # Build comparison per (lorrax_ki, n) 0-indexed
    om_min, om_max = omega.min(), omega.max()
    margin = 0.5  # safety margin
    rows = []
    for blk in bgw:
        ki = match_k(blk["kc"], lx_kpts)
        if ki is None or ki >= nk:
            continue
        kih_k = kih.get(blk["kc"]) or next(
            (v for k, v in kih.items()
             if max(abs(a - b - round(a - b)) for a, b in zip(k, blk["kc"])) < 1e-4),
            None,
        )
        for n_b, bd in sorted(blk["bands"].items()):
            n0 = n_b - 1
            if n0 >= nb:
                continue
            omega_rel = bd["Emf"] - efermi
            if not (om_min + margin <= omega_rel <= om_max - margin):
                continue
            re_c = np.interp(omega_rel, omega, sig_c_diag[:, ki, n0].real)
            im_c = np.interp(omega_rel, omega, sig_c_diag[:, ki, n0].imag)
            sigma_xc = sig_sx_diag[ki, n0].real + re_c
            diff = sigma_xc - bd["Sig"]
            rows.append({"ki": ki, "n": n_b, "kc": blk["kc"],
                         "omega_rel": omega_rel,
                         "lorrax_sig": sigma_xc,
                         "lorrax_im": im_c,
                         "bgw_sig": bd["Sig"],
                         "diff": diff})

    rows = [r for r in rows if np.isfinite(r["diff"])]
    print(f"\nMatched {len(rows)} (k, n) pairs (filtered to ω in [{om_min+margin:.1f}, {om_max-margin:.1f}])")
    diffs = np.array([r["diff"] for r in rows])
    print(f"MAE  vs BGW Sig = {np.mean(np.abs(diffs))*1000:.2f} meV")
    print(f"max  vs BGW Sig = {np.max(np.abs(diffs))*1000:.2f} meV")
    print(f"mean Δ          = {np.mean(diffs)*1000:+.2f} meV")
    print(f"median |Δ|      = {np.median(np.abs(diffs))*1000:.2f} meV")

    # Per band-bin breakdown
    occ_diffs = [r["diff"] for r in rows if r["n"] <= 8]
    cond_diffs = [r["diff"] for r in rows if r["n"] > 8]
    if occ_diffs:
        print(f"  occupied (n<=8): MAE={np.mean(np.abs(occ_diffs))*1000:.2f} meV, N={len(occ_diffs)}")
    if cond_diffs:
        print(f"  conduction (n>8): MAE={np.mean(np.abs(cond_diffs))*1000:.2f} meV, N={len(cond_diffs)}")

    # Top 10 largest |diff|
    print("\nTop |diff|:")
    for r in sorted(rows, key=lambda r: -abs(r["diff"]))[:10]:
        kstr = f"({r['kc'][0]:+.3f},{r['kc'][1]:+.3f},{r['kc'][2]:+.3f})"
        print(f"  ki={r['ki']:3d} n={r['n']:2d} kc={kstr} ω={r['omega_rel']:+.2f} "
              f"LX={r['lorrax_sig']:8.4f} BGW={r['bgw_sig']:8.4f} Δ={r['diff']*1000:+8.2f} meV "
              f"Im={r['lorrax_im']:+.3f}")


if __name__ == "__main__":
    main()
