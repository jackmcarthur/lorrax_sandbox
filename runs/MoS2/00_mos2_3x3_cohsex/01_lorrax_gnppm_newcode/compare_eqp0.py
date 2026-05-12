#!/usr/bin/env python3
"""Compare LORRAX eqp0.dat sigC directly to BGW sigma_hp.log Sig.
sigC in eqp0.dat (PPM mode, post-2026-04-25 fix) is dynamic Σ_c at E_DFT."""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BGW_LOG = HERE.parent / "01_bgw_gnppm" / "sigma_hp.log"
LORRAX_EQP = HERE / "eqp0.dat"

def parse_bgw(path):
    blocks = []; lines = path.read_text().splitlines(); i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)", s)
        if not m: i += 1; continue
        kc = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        i += 1
        while i < len(lines) and not re.match(r"\s+n\s+Emf", lines[i]): i += 1
        i += 1
        bands = {}
        while i < len(lines):
            t = lines[i].strip()
            if not t: break
            p = t.split()
            if len(p) >= 11 and p[0].isdigit():
                n = int(p[0])
                bands[n] = {"Emf": float(p[1]), "X": float(p[3]), "SXmX": float(p[4]),
                            "CH": float(p[5]), "Sig": float(p[6])}
                i += 1; continue
            break
        if bands: blocks.append({"kc": kc, "bands": bands})
        i += 1
    return blocks

def parse_lorrax(path):
    out = {}; cur_k = None
    for line in path.read_text().splitlines():
        s = line.strip()
        m = re.match(r"k-point\s+(\d+):", s)
        if m: cur_k = int(m.group(1)); out[cur_k] = {}; continue
        m2 = re.match(r"n=(\d+)\s+sigX=\s*([-\d.]+)\s+sigC=\s*([^V]+?)\s+sigXC=\s*([^V]+?)\s+VH=", s)
        if m2 and cur_k is not None:
            n0 = int(m2.group(1))
            sigX = float(m2.group(2))
            sigC_str = m2.group(3).strip()
            tok = sigC_str.replace(" ", "").replace("i", "j").replace("+-", "-").replace("-+", "-")
            try: sigC = complex(tok)
            except Exception: sigC = complex("nan")
            out[cur_k][n0] = (sigX, sigC)
    return out

def get_lx_kpts():
    sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
    from file_io import WFNReader
    from common import symmetry_maps
    w = WFNReader(str(HERE / "WFN.h5"))
    return np.asarray(symmetry_maps.SymMaps(w).unfolded_kpts, dtype=np.float64)

def match_k(kc, lx_kpts, tol=1e-4):
    for i, k in enumerate(lx_kpts):
        d = np.array(kc) - k; d -= np.round(d)
        if np.max(np.abs(d)) < tol: return i
    return None

bgw = parse_bgw(BGW_LOG)
lr = parse_lorrax(LORRAX_EQP)
lx_kpts = get_lx_kpts()

rows = []
for blk in bgw:
    ki = match_k(blk["kc"], lx_kpts)
    if ki is None or ki not in lr: continue
    for n_b, bd in sorted(blk["bands"].items()):
        n0 = n_b - 1
        if n0 not in lr[ki]: continue
        lx_sigX, lx_sigC = lr[ki][n0]
        # head-invariant comparison: (LX sigC) vs BGW (SX-X + CH)
        lx_cor = lx_sigC.real
        bgw_cor = bd["SXmX"] + bd["CH"]
        # also full sigma comparison
        lx_sig = lx_sigX + lx_sigC.real
        bgw_sig = bd["Sig"]
        rows.append({"ki": ki, "n": n_b, "lx_cor": lx_cor, "bgw_cor": bgw_cor,
                     "lx_sigX": lx_sigX, "bgw_X": bd["X"],
                     "lx_sig": lx_sig, "bgw_sig": bgw_sig,
                     "lx_sigC_im": lx_sigC.imag, "kc": blk["kc"]})

n = len(rows)
diffs_cor = np.array([r["lx_cor"] - r["bgw_cor"] for r in rows])
diffs_sig = np.array([r["lx_sig"] - r["bgw_sig"] for r in rows])
diffs_x = np.array([r["lx_sigX"] - r["bgw_X"] for r in rows])
print(f"N={n}")
print(f"  bare X  MAE: {np.mean(np.abs(diffs_x))*1000:.2f} meV")
print(f"  Σ_c (sigC vs BGW SX-X+CH) MAE: {np.mean(np.abs(diffs_cor))*1000:.2f} meV  max={np.max(np.abs(diffs_cor))*1000:.2f}")
print(f"  full Σ_xc (sigX+sigC vs BGW Sig) MAE: {np.mean(np.abs(diffs_sig))*1000:.2f} meV  max={np.max(np.abs(diffs_sig))*1000:.2f}")
print(f"  median |Δ_cor|: {np.median(np.abs(diffs_cor))*1000:.2f} meV")
print(f"  Im[sigC] max: {np.max(np.abs([r['lx_sigC_im'] for r in rows])):.2f} eV")
