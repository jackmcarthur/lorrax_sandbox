#!/usr/bin/env python3
"""Si 2x2x2 60Ry — extract major-section timings from each LORRAX run.

Pulls the bottom-of-output timing block from each gw_*.out and reports:
  isdf+zeta : load_centroid_wfns + zeta_fit_chunked
  V_q       : V_q_compute
  chi0/W    : chi0_W
  sigma     : sigma + ppm_sigma + cohsex_sigma (whichever applies)
  total     : Total recorded
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry")


def parse_timing(path: Path):
    if not path.exists():
        return None
    txt = path.read_text()
    out = {}
    m = re.search(r"Total recorded:\s+([\d.]+)\s*s", txt)
    if not m:
        return None
    out["total"] = float(m.group(1))
    for sec in [
        "gw_jax.load_centroid_wfns",
        "gw_jax.zeta_fit_chunked",
        "gw_jax.V_q_compute",
        "gw_jax.wavefunction_setup",
        "gw_jax.chi0_W",
        "gw_jax.sigma",
        "gw_jax.ppm_sigma",
        "gw_jax.cohsex_sigma",
    ]:
        # match "<sec>  <count>  <total>  <self>  <pct>" — total is the 2nd float column
        m = re.search(rf"^\s*{re.escape(sec)}\s+\d+\s+([\d.]+)\s+", txt, re.MULTILINE)
        if m:
            out[sec] = float(m.group(1))
    return out


def main():
    cases = []
    # PROT
    cases.append(("nb=60 PROT xonly",   "1440", ROOT/"D_lorrax_canonical_xonly/gw_xonly.out"))
    cases.append(("nb=60 PROT COHSEX",  "1440", ROOT/"D_lorrax_canonical_noavg/gw1440.out"))
    cases.append(("nb=60 PROT PPM",     "1440", ROOT/"D_lorrax_canonical_gnppm_test/gw_test_w10.out"))
    # PB COHSEX
    for nb in [108, 208, 408, 808, 1208]:
        for nc in [1464, 2448, 3264]:
            d = ROOT / f"D_lorrax_para_{nb}"
            f = d / f"gw_{nc}.out"
            if not f.exists():
                f = d / f"gw_c{nc}.out"
            cases.append((f"nb={nb} COHSEX", str(nc), f))
    # PB PPM (single Nc each from sweep)
    for nb in [108, 208, 408, 808, 1208]:
        d = ROOT / f"D_lorrax_para_{nb}_gnppm"
        for fn in ["gw_w10.out", "gw_w10_3264.out", "gw_w10_1464.out"]:
            f = d / fn
            if f.exists():
                cases.append((f"nb={nb} PPM", fn.replace("gw_w10","").strip("_.out") or "default", f))
                break

    print(f"{'case':<22} {'N_c':>5}  {'isdf+zeta':>10}  {'V_q':>8}  {'chi0/W':>8}  {'sigma':>9}  {'total':>9}  notes")
    print("-" * 100)
    for label, nc, path in cases:
        t = parse_timing(path)
        if t is None:
            print(f"{label:<22} {nc:>5}  {'-':>10}  {'-':>8}  {'-':>8}  {'-':>9}  {'-':>9}  missing ({path.name})")
            continue
        isdf = (t.get("gw_jax.load_centroid_wfns", 0.0) + t.get("gw_jax.zeta_fit_chunked", 0.0))
        vq = t.get("gw_jax.V_q_compute", 0.0)
        chi0 = t.get("gw_jax.chi0_W", 0.0)
        sigma = t.get("gw_jax.sigma", 0.0) + t.get("gw_jax.ppm_sigma", 0.0) + t.get("gw_jax.cohsex_sigma", 0.0)
        print(f"{label:<22} {nc:>5}  {isdf:>10.2f}  {vq:>8.2f}  {chi0:>8.2f}  {sigma:>9.2f}  {t['total']:>9.2f}")


if __name__ == "__main__":
    main()
