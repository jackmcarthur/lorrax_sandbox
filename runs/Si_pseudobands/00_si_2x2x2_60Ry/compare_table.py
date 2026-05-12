#!/usr/bin/env python3
"""Si 2x2x2 60Ry — comprehensive comparison table (LORRAX vs BGW).

All LORRAX runs use a fully-explicit (parabands) WFN file (4200 bands explicit,
no stochastic compression). The "nb" axis is the cohsex.in `nband` parameter
(how many DFT bands LORRAX uses to build χ₀ and Σ_c). The "N_c" axis is the
number of ISDF centroids.

Per (nb, N_c) cell, three quantities are compared:
  bare_X  : LORRAX `sigSX` (x_only run, sigCOH=0)  or  PPM `sigX`
            vs BGW col 4 (X)                     [bare exchange]
  COHSEX  : LORRAX `sigTOT`  vs  BGW Sig' = X+SX-X+CH'  (col 4+5+10)
            (BGW col 6 `Sig` uses exact-static-CH; col 11 `Sig'` uses the
             partial-sum CH that LORRAX computes — they only match in the
             G→∞ limit. Reference: skills/compare/SKILL.md.)
  PPM     : LORRAX `sigXC` (PPM) vs BGW Sig' (col 11) on the GN-PPM BGW run
            (frequency_dependence=3); for fd=3 BGW primed = unprimed.

For Si 2×2×2 the IBZ has 4 kpts {Γ, X, M, L} mapping to the 8-kpt unfolded
BZ at lex positions [0, 1, 3, 7]. LORRAX `sigma_diag.dat` writes 8 kidx in
unfolded order; we remap to IBZ for comparison vs BGW (4 IBZ).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry")
SIGMA_WINDOW_EV = 2.5  # σ-window radius around E_F (tight, just band-edge)

# IBZ kidx → unfolded BZ kidx in lex order, for 2×2×2 Si
IBZ_TO_UNF = {0: 0, 1: 1, 2: 3, 3: 7}


def kr(k, tol=1e-5):
    return tuple(round((float(c) % 1.0) % 1.0, 6) if abs(float(c)) > tol else 0.0 for c in k)


def parse_sigma_diag(path):
    """Parse a sigma_diag-style file (legacy or PPM format).

    Returns dict keyed by (kidx, n0). kidx is in the file's native ordering
    (8 unfolded BZ kpts in lex order for 2×2×2 Si parabands).
    """
    out = {}
    cur_k = None
    pat_legacy = re.compile(r"n=\s*(\d+)\s+sigSX=\s*([-\d.eE+]+(?=\s))\s+sigCOH=\s*([-\d.eE+]+(?=\s))\s+sigTOT=\s*([-+]?[\d.eE]+)")
    pat_ppm = re.compile(r"n=\s*(\d+)\s+sigX=\s*([-+]?[\d.eE]+)\s+sigC=\s*([-+]?[\d.eE]+)\+\s*([-+]?[\d.eE]+)i\s+sigXC=\s*([-+]?[\d.eE]+)\+\s*([-+]?[\d.eE]+)i")
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        m = re.match(r"k-point\s+(\d+)", raw)
        if m:
            cur_k = int(m.group(1))
            continue
        if cur_k is None:
            continue
        m = pat_ppm.search(raw)
        if m:
            out[(cur_k, int(m.group(1)))] = {
                "sigX": float(m.group(2)),
                "sigC_re": float(m.group(3)),
                "sigC_im": float(m.group(4)),
                "sigxc_tot": float(m.group(5)),
                "sigxc_im": float(m.group(6)),
                "format": "ppm",
            }
            continue
        m = pat_legacy.search(raw)
        if m:
            sigSX, sigCOH, sigTOT = float(m.group(2)), float(m.group(3)), float(m.group(4))
            out[(cur_k, int(m.group(1)))] = {
                "sigSX": sigSX,
                "sigCOH": sigCOH,
                "sigxc_tot": sigTOT,
                "sigX_if_xonly": sigSX if abs(sigCOH) < 1e-9 else None,
                "format": "legacy_xonly" if abs(sigCOH) < 1e-9 else "legacy_cohsex",
            }
    return out


def parse_bgw_log(path):
    """BGW sigma_hp.log → list of {kcrys, ik, bands={n: {Emf, X, SXX, CH, Sig, CHp, Sigp}}}."""
    blocks = []
    cur, cur_bands = None, None
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        m = re.match(r"k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)", s)
        if m:
            if cur is not None:
                cur["bands"] = cur_bands
                blocks.append(cur)
            cur = {"kcrys": (float(m.group(1)), float(m.group(2)), float(m.group(3)))}
            cur_bands = {}
            continue
        if cur is not None:
            p = s.split()
            if len(p) >= 14 and p[0].isdigit():
                n = int(p[0])
                cur_bands[n] = {
                    "Emf": float(p[1]),
                    "X": float(p[3]),
                    "SXX": float(p[4]),
                    "CH": float(p[5]),
                    "Sig": float(p[6]),
                    "CHp": float(p[10]),
                    "Sigp": float(p[11]),
                }
    if cur is not None:
        cur["bands"] = cur_bands
        blocks.append(cur)
    return blocks


def evaluate(label, lx_path, bgw_dir, kind, kpt_remap=True, sigma_window_ev=SIGMA_WINDOW_EV):
    """kind ∈ {'X', 'Sig'}.
       lx_path: path to sigma_diag-style file from a LORRAX run.
       bgw_dir: directory containing sigma_hp.log."""
    bgw_path = bgw_dir / "sigma_hp.log"
    if not lx_path.exists() or not bgw_path.exists():
        return {"label": label, "status": "missing", "n": 0}
    lx = parse_sigma_diag(lx_path)
    bgw = parse_bgw_log(bgw_path)

    # Determine if remap needed: count distinct kidx in lx and BGW
    n_lx_k = len({k[0] for k in lx})
    n_bgw_k = len(bgw)
    if n_lx_k == 8 and n_bgw_k == 4 and kpt_remap:
        ibz_to_lx = IBZ_TO_UNF
    else:
        ibz_to_lx = {i: i for i in range(min(n_lx_k, n_bgw_k))}

    n_occ = 8
    all_emf = [(n, b["Emf"]) for blk in bgw for n, b in blk["bands"].items()]
    vbm = max(e for n, e in all_emf if n <= n_occ)
    cbm = min(e for n, e in all_emf if n > n_occ)
    ef = 0.5 * (vbm + cbm)

    diffs = []
    for ibz_kidx, blk in enumerate(bgw):
        lx_kidx = ibz_to_lx.get(ibz_kidx)
        if lx_kidx is None:
            continue
        for nb, bd in blk["bands"].items():
            n0 = nb - 1
            if (lx_kidx, n0) not in lx:
                continue
            entry = lx[(lx_kidx, n0)]
            if abs(bd["Emf"] - ef) > sigma_window_ev:
                continue
            if kind == "X":
                v = entry.get("sigX") or entry.get("sigX_if_xonly")
                ref = bd["X"]
            elif kind == "sigC":
                # Compare LORRAX sigC directly to BGW (SX-X + CH')
                v = entry.get("sigC_re")
                ref = bd["SXX"] + bd["CHp"]
                if v is None:
                    continue
            else:
                v = entry.get("sigxc_tot")
                ref = bd["Sigp"]
            if v is None:
                continue
            diffs.append(v - ref)
    if not diffs:
        return {"label": label, "n": 0}
    arr = np.array(diffs)
    return {"label": label, "n": len(arr),
            "MAE_meV": np.mean(np.abs(arr)) * 1000,
            "max_meV": np.max(np.abs(arr)) * 1000,
            "median_meV": np.median(np.abs(arr)) * 1000}


def main():
    rows = []  # (nb, N_c, kind, lx_path, bgw_dir)

    # nb=60 (canonical, no parabands): N_c=1440
    rows.append((60,   1440, "bare_X", ROOT/"D_lorrax_canonical_xonly/eqp0.dat",       ROOT/"D_bgw_canonical_noavg"))
    rows.append((60,   1440, "COHSEX", ROOT/"D_lorrax_canonical_noavg/eqp0.dat",       ROOT/"D_bgw_canonical_noavg"))
    rows.append((60,   1440, "PPM_tot",  ROOT/"D_lorrax_canonical_gnppm_test/sigma_diag.dat", ROOT/"D_bgw_canonical_gnppm_noavg"))
    rows.append((60,   1440, "PPM_sigC", ROOT/"D_lorrax_canonical_gnppm_test/sigma_diag.dat", ROOT/"D_bgw_canonical_gnppm_noavg"))

    # nb=108..1208 with N_c sweep — COHSEX (sigdiag_NC.dat)
    for nb in [108, 208, 408, 808, 1208]:
        for nc in [1464, 2448, 3264]:
            base = ROOT / f"D_lorrax_para_{nb}"
            sd = base / f"sigdiag_{nc}.dat"
            bgw_cohsex = ROOT / f"D_bgw_para_{nb}_noavg"
            rows.append((nb, nc, "COHSEX", sd, bgw_cohsex))
        # bare_X from PPM run (sigX field is bare X)
        base_ppm = ROOT / f"D_lorrax_para_{nb}_gnppm"
        bgw_ppm  = ROOT / f"D_bgw_para_{nb}_gnppm_noavg"
        rows.append((nb, "PPM_one", "bare_X",   base_ppm/"sigma_diag.dat", bgw_ppm))
        rows.append((nb, "PPM_one", "PPM_tot",  base_ppm/"sigma_diag.dat", bgw_ppm))
        rows.append((nb, "PPM_one", "PPM_sigC", base_ppm/"sigma_diag.dat", bgw_ppm))

    print(f"{'nb':>5} {'N_c':>8}  {'kind':<7}  {'kept':>4} {'med (meV)':>10} {'MAE (meV)':>10} {'max (meV)':>10}")
    print("-" * 70)
    for nb, nc, kind, lx_path, bgw_dir in rows:
        # For PPM-mode bare_X we also need to use sigX field, not legacy
        if kind == "bare_X":
            cmp_kind = "X"
        elif kind == "PPM_sigC":
            cmp_kind = "sigC"
        else:
            cmp_kind = "Sig"
        r = evaluate(f"nb={nb} N_c={nc} {kind}", lx_path, bgw_dir, cmp_kind)
        if r.get("n", 0) == 0:
            print(f"{nb:>5} {nc:>8}  {kind:<7}  {'-':>4} {'-':>10} {'-':>10} {'-':>10}  ({r.get('status','no match')})")
            continue
        print(f"{nb:>5} {nc:>8}  {kind:<7}  {r['n']:>4} {r['median_meV']:>10.2f} {r['MAE_meV']:>10.2f} {r['max_meV']:>10.2f}")


if __name__ == "__main__":
    main()
