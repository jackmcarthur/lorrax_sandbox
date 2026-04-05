#!/usr/bin/env python3
"""Compare BGW Sigma_cor vs GWJAX sigC_EDFT for MoS2.

Reads BGW sigma_hp.log (primed columns) and GWJAX eqp0.dat, matching k-points
by crystal coordinates. BGW k-points are a symmetry-reduced subset; GWJAX
k-points are always the full grid in (0,0,0), (0,1/3,0), (0,2/3,0), ... order.

Usage:
    python compare_bgw_gwjax.py \\
        --bgw-hp ../qe_mos2/sigma_hp.log \\
        --gw-eqp eqp0.dat \\
        --wfn WFN.h5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_bgw_sigma_hp(path: Path) -> list[dict]:
    """Parse sigma_hp.log: extract k-coords, band data with primed columns.

    Returns list of dicts, one per k-point block:
        {'kcrys': (kx, ky, kz), 'ik_bgw': int,
         'bands': {n: {'SXmX': float, 'CH': float, 'CHp': float,
                        'Sig': float, 'Sigp': float, ...}}}

    Column layout (frequency_dependence 3):
      n  Emf  Eo  X  SX-X  CH  Sig  Vxc  Eqp0  Eqp1  CH'  Sig'  Eqp0'  Eqp1'  Znk
      0   1    2  3    4    5    6    7     8     9     10    11    12     13     14
    """
    lines = path.read_text().splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(
            r"k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)",
            s,
        )
        if not m:
            i += 1
            continue
        kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        ik_bgw = int(m.group(4))
        # Skip to data lines (next line with "n" header, then data)
        i += 1
        while i < len(lines) and not re.match(r"\s+n\s+Emf", lines[i]):
            i += 1
        i += 1  # skip header
        bands: dict[int, dict] = {}
        while i < len(lines):
            s2 = lines[i].strip()
            if not s2:
                break
            p = s2.split()
            if len(p) >= 14 and p[0].isdigit():
                n = int(p[0])
                bands[n] = {
                    "X": float(p[3]),
                    "SXmX": float(p[4]),
                    "CH": float(p[5]),
                    "Sig": float(p[6]),
                    "CHp": float(p[10]),
                    "Sigp": float(p[11]),
                    "Cor": float(p[4]) + float(p[5]),
                    "Corp": float(p[4]) + float(p[10]),
                }
                i += 1
                continue
            break
        if bands:
            blocks.append({"kcrys": kcrys, "ik_bgw": ik_bgw, "bands": bands})
        i += 1
    return blocks


def parse_gwjax_sigc(path: Path) -> dict[tuple[int, int], float]:
    """Parse sigC_EDFT from GWJAX eqp0 output.

    Returns dict of (k_index, band_1indexed) -> Re(sigC_EDFT) in eV.
    """
    out: dict[tuple[int, int], float] = {}
    k = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        m = re.match(r"^k-point\s+(\d+):", s)
        if m:
            k = int(m.group(1))
            continue
        m = re.match(r"^n=(\d+)\s+.*sigC_EDFT=", s)
        if not m or k is None:
            continue
        n0 = int(m.group(1))
        try:
            frag = s.split("sigC_EDFT=", 1)[1].split("sigXC_EDFT=", 1)[0]
        except Exception:
            continue
        tok = frag.replace(" ", "").replace("i", "j")
        tok = tok.replace("+-", "-").replace("-+", "-")
        tok = tok.replace("--", "+").replace("++", "+")
        try:
            val = complex(tok)
        except Exception:
            continue
        if np.isnan(val.real):
            continue
        out[(k, n0 + 1)] = float(val.real)
    return out


def get_gwjax_kpoints(wfn_path: Path) -> np.ndarray:
    """Read k-points from WFN h5 file. Returns (nk, 3) in crystal coords."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lorrax" / "src"))
    try:
        from common.wfnreader import WFNReader
        w = WFNReader(str(wfn_path))
        return np.array(w.kpoints, dtype=np.float64)
    finally:
        pass


def match_kpoint(kcrys_bgw: tuple[float, ...], gwjax_kpts: np.ndarray, tol: float = 1e-4) -> int | None:
    """Find GWJAX k-index matching a BGW k-point (mod G)."""
    target = np.array(kcrys_bgw, dtype=np.float64)
    for i, k in enumerate(gwjax_kpts):
        diff = target - k
        # Fold to [-0.5, 0.5]
        diff = diff - np.round(diff)
        if np.max(np.abs(diff)) < tol:
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare BGW sigma_hp vs GWJAX sigC")
    parser.add_argument("--bgw-hp", required=True, help="Path to BGW sigma_hp.log")
    parser.add_argument("--gw-eqp", required=True, help="Path to GWJAX eqp0.dat")
    parser.add_argument("--wfn", required=True, help="Path to WFN.h5 (for k-point matching)")
    parser.add_argument("--use-primed", action="store_true", default=True,
                        help="Use CH' (primed) columns from sigma_hp (default: True)")
    parser.add_argument("--use-unprimed", action="store_true",
                        help="Use CH (unprimed) columns from sigma_hp")
    parser.add_argument("--out-dat", default="compare_cor.dat")
    parser.add_argument("--out-png", default="compare_cor.png")
    args = parser.parse_args()

    use_primed = not args.use_unprimed

    bgw_blocks = parse_bgw_sigma_hp(Path(args.bgw_hp))
    gw = parse_gwjax_sigc(Path(args.gw_eqp))
    gwjax_kpts = get_gwjax_kpoints(Path(args.wfn))

    # Match BGW k-points to GWJAX indices
    rows = []
    matched_k = []
    for blk in bgw_blocks:
        gw_ki = match_kpoint(blk["kcrys"], gwjax_kpts)
        if gw_ki is None:
            kstr = f"({blk['kcrys'][0]:.4f}, {blk['kcrys'][1]:.4f}, {blk['kcrys'][2]:.4f})"
            print(f"WARNING: BGW k={kstr} (ik={blk['ik_bgw']}) not found in GWJAX grid")
            continue
        matched_k.append((blk, gw_ki))
        kstr = f"({blk['kcrys'][0]:.4f}, {blk['kcrys'][1]:.4f}, {blk['kcrys'][2]:.4f})"
        for n, bdata in sorted(blk["bands"].items()):
            bgw_cor = bdata["Corp"] if use_primed else bdata["Cor"]
            gw_val = gw.get((gw_ki, n), np.nan)
            rows.append({
                "bgw_ik": blk["ik_bgw"],
                "gw_ki": gw_ki,
                "kcrys": blk["kcrys"],
                "n": n,
                "bgw_cor": bgw_cor,
                "gw_sigc": gw_val,
                "diff": gw_val - bgw_cor,
            })

    cor_label = "Cor'" if use_primed else "Cor"

    # Write table
    out_dat = Path(args.out_dat)
    with out_dat.open("w") as f:
        f.write(f"# bgw_ik  gw_ki  kx       ky       kz       band  BGW_{cor_label}     GW_sigC       diff\n")
        for r in rows:
            kx, ky, kz = r["kcrys"]
            f.write(
                f"  {r['bgw_ik']:5d}  {r['gw_ki']:5d}  {kx:8.5f} {ky:8.5f} {kz:8.5f}"
                f"  {r['n']:4d}  {r['bgw_cor']:11.6f}  {r['gw_sigc']:11.6f}  {r['diff']:+10.6f}\n"
            )

    finite = [r for r in rows if not np.isnan(r["diff"])]
    mae = np.mean([abs(r["diff"]) for r in finite]) if finite else np.nan
    print(f"Matched {len(matched_k)} k-points, {len(finite)} band comparisons")
    print(f"MAE = {mae:.4f} eV")
    print(f"Wrote {out_dat}")

    # Collect all bands present
    all_bands = sorted(set(r["n"] for r in rows))

    # Plot
    nk_plot = len(matched_k)
    fig, axes = plt.subplots(nk_plot, 2, figsize=(12, 3.5 * nk_plot), dpi=180,
                              constrained_layout=True, squeeze=False)

    for ki, (blk, gw_ki) in enumerate(matched_k):
        kstr = (f"k=({blk['kcrys'][0]:.3f}, {blk['kcrys'][1]:.3f}, {blk['kcrys'][2]:.3f})"
                f"  [BGW ik={blk['ik_bgw']}, GW k={gw_ki}]")
        rr = [r for r in rows if r["gw_ki"] == gw_ki and not np.isnan(r["diff"])]
        if not rr:
            continue
        x = np.array([r["n"] for r in rr])
        yb = np.array([r["bgw_cor"] for r in rr])
        yg = np.array([r["gw_sigc"] for r in rr])
        dd = np.array([r["diff"] for r in rr])

        ax0 = axes[ki, 0]
        ax0.plot(x, yb, "o-", color="C0", lw=1.8, ms=4.5, label=f"BGW {cor_label}")
        ax0.plot(x, yg, "s--", color="C1", lw=1.5, ms=4.5, label="GWJAX sigC")
        ax0.set_title(kstr, fontsize=9)
        ax0.set_xlabel("Band index")
        ax0.set_ylabel(f"Sigma_cor (eV)")
        ax0.set_xticks(all_bands[::2])
        ax0.grid(alpha=0.25)
        ax0.legend(fontsize=7)

        ax1 = axes[ki, 1]
        ax1.plot(x, dd, "o-", color="C2", lw=1.5, ms=4.5)
        ax1.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        local_mae = np.mean(np.abs(dd))
        ax1.set_title(f"Residual, MAE={local_mae:.3f} eV", fontsize=9)
        ax1.set_xlabel("Band index")
        ax1.set_ylabel("GWJAX - BGW (eV)")
        ax1.set_xticks(all_bands[::2])
        ax1.grid(alpha=0.25)

    fig.suptitle(f"BGW {cor_label} vs GWJAX sigC_EDFT — MAE={mae:.3f} eV", fontsize=11)
    out_png = Path(args.out_png)
    fig.savefig(out_png)
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
