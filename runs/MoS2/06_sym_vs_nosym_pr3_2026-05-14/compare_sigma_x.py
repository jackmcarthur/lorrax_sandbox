#!/usr/bin/env python3
"""Compare bare Σ_X across run_sym and run_nosym sigma_freq_debug.dat files.

Uses skills/compare-style parsing of sigma_freq_debug.dat. The two runs
have identical k-point orderings (both 9 k-pts in the same crystal
order), so we can match on (k, n) directly.

Output: per-(k, n) |Δx_bare|, max/mean/stddev, per-k breakdown.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


def parse_sigma_freq_debug_allk(path: Path):
    """Return list of dicts per k-point: {'ik', 'bands': {n: {field: float}}}.

    Columns (sigma_freq_debug.dat after Generated/Sigma/legend header):
      0=k 1=n 2=E_dft 3=Edft-Ef 4=kin_ion 5=V_H 6=x_bare 7=x_head 8=sex_0 9=coh_0 10=eqp0 11=eqp1
    """
    blocks = {}
    cur_k = None
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('k-point'):
            try:
                cur_k = int(s.split()[1].rstrip(':'))
            except (ValueError, IndexError):
                cur_k = None
            blocks.setdefault(cur_k, {'ik': cur_k, 'bands': {}})
            continue
        p = s.split()
        if len(p) >= 12 and cur_k is not None:
            try:
                k_col = int(p[0]); n = int(p[1])
                blocks[cur_k]['bands'][n] = {
                    'E_dft':   float(p[2]),
                    'Edft_Ef': float(p[3]),
                    'kin_ion': float(p[4]),
                    'V_H':     float(p[5]),
                    'x_bare':  float(p[6]),
                    'x_head':  float(p[7]),
                    'sex_0':   float(p[8]),
                    'coh_0':   float(p[9]),
                    'eqp0':    float(p[10]),
                    'eqp1':    float(p[11]),
                }
            except (ValueError, IndexError):
                pass
    return [blocks[k] for k in sorted(blocks)]


def main():
    base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14')
    sym_path   = base / 'run_sym/sigma_freq_debug.dat'
    nosym_path = base / 'run_nosym/sigma_freq_debug.dat'

    sym   = parse_sigma_freq_debug_allk(sym_path)
    nosym = parse_sigma_freq_debug_allk(nosym_path)
    assert len(sym) == len(nosym), f"k-count mismatch: sym={len(sym)} nosym={len(nosym)}"

    trs_k = {1, 3, 4, 5}  # PR3 testbed: TRS-fold k-points in the sym WFN
    n_k = len(sym)

    # Build per-(k, n) deltas.
    rows = []
    for s, n in zip(sym, nosym):
        ik = s['ik']
        assert s['ik'] == n['ik']
        bands = sorted(set(s['bands']) & set(n['bands']))
        for b in bands:
            sb = s['bands'][b]; nb = n['bands'][b]
            d_xb   = sb['x_bare']  - nb['x_bare']
            d_sex  = sb['sex_0']   - nb['sex_0']
            d_coh  = sb['coh_0']   - nb['coh_0']
            d_xh   = sb['x_head']  - nb['x_head']
            d_edft = sb['E_dft']   - nb['E_dft']
            rows.append({
                'ik': ik, 'n': b,
                'x_bare_sym': sb['x_bare'], 'x_bare_nosym': nb['x_bare'],
                'd_x_bare_eV':  d_xb,
                'd_sex0_eV':    d_sex,
                'd_coh0_eV':    d_coh,
                'd_x_head_eV':  d_xh,
                'd_Edft_eV':    d_edft,
                'd_SigX_total': d_xb + d_sex + d_coh,  # total Σ_X = x_bare + sex_0 + coh_0
                'is_trs_k':     ik in trs_k,
            })

    arr_d_xb_mev    = np.array([r['d_x_bare_eV']     * 1000.0 for r in rows])
    arr_d_sex_mev   = np.array([r['d_sex0_eV']       * 1000.0 for r in rows])
    arr_d_coh_mev   = np.array([r['d_coh0_eV']       * 1000.0 for r in rows])
    arr_d_total_mev = np.array([r['d_SigX_total']    * 1000.0 for r in rows])
    arr_d_edft_mev  = np.array([r['d_Edft_eV']       * 1000.0 for r in rows])
    is_trs          = np.array([r['is_trs_k'] for r in rows])

    abs_d_total = np.abs(arr_d_total_mev)
    abs_d_xb    = np.abs(arr_d_xb_mev)
    abs_d_sex   = np.abs(arr_d_sex_mev)

    print("===== sym vs nosym PR3 Σ_X comparison =====")
    print(f"  total (k, n) pairs: {len(rows)} = {n_k} k-pts × {len(rows)//n_k} bands")
    print()
    print("--- DFT eigenvalue offset (orientation check, should be ~ULP) ---")
    print(f"   max |ΔE_dft|        = {np.max(np.abs(arr_d_edft_mev)):10.4f} meV")
    print(f"   mean ΔE_dft         = {np.mean(arr_d_edft_mev):10.4f} meV")
    print()
    print("--- Σ_X components ---")
    print(f"   max |Δx_bare|       = {np.max(abs_d_xb):10.4f} meV")
    print(f"   max |Δsex_0|        = {np.max(abs_d_sex):10.4f} meV")
    print(f"   max |Δcoh_0|        = {np.max(np.abs(arr_d_coh_mev)):10.4f} meV")
    print(f"   max |ΔΣ_X (total)|  = {np.max(abs_d_total):10.4f} meV")
    print(f"   mean ΔΣ_X (total)   = {np.mean(arr_d_total_mev):10.4f} meV")
    print(f"   stddev ΔΣ_X (total) = {np.std(arr_d_total_mev):10.4f} meV")
    print()

    # Per-k breakdown.
    print("--- Per-k breakdown (max/mean |ΔΣ_X|, meV) ---")
    print("   ik   trs?    max|ΔΣ_X|     mean|ΔΣ_X|   max|Δx_bare|  max|Δsex_0|   N_bands")
    for ik in range(n_k):
        mask = np.array([r['ik'] == ik for r in rows])
        nb = mask.sum()
        if nb == 0:
            continue
        flag = "TRS" if ik in trs_k else "  -"
        print(f"   {ik:2d}    {flag}   {np.max(abs_d_total[mask]):10.4f}   "
              f"{np.mean(abs_d_total[mask]):10.4f}   {np.max(abs_d_xb[mask]):10.4f}   "
              f"{np.max(abs_d_sex[mask]):10.4f}   {nb}")

    # TRS vs non-TRS group means.
    print()
    print("--- TRS vs non-TRS group statistics (meV) ---")
    for grp_name, grp_mask in [("TRS k-points {1,3,4,5}", is_trs),
                                ("non-TRS k-points {0,2,6,7,8}", ~is_trs)]:
        gn = grp_mask.sum()
        print(f"   {grp_name:32s}: N={gn}  max|ΔΣ_X|={np.max(abs_d_total[grp_mask]):.4f}  "
              f"mean|ΔΣ_X|={np.mean(abs_d_total[grp_mask]):.4f}  stddev={np.std(arr_d_total_mev[grp_mask]):.4f}")

    # Pass/fail verdict.
    max_dtotal = float(np.max(abs_d_total))
    print()
    print("==========================================")
    if max_dtotal <= 1.0:
        verdict = "PASS"
        msg = f"max |ΔΣ_X| = {max_dtotal:.4f} meV ≤ 1.0 meV pass gate"
    elif max_dtotal <= 10.0:
        verdict = "MARGINAL"
        msg = f"max |ΔΣ_X| = {max_dtotal:.4f} meV (between 1 meV and 10 meV — investigate)"
    else:
        verdict = "FAIL"
        msg = f"max |ΔΣ_X| = {max_dtotal:.4f} meV ≥ 10 meV — real bug"
    print(f"  Verdict: {verdict}")
    print(f"  {msg}")
    print("==========================================")

    # Top-10 worst rows.
    print()
    print("--- Top-10 |ΔΣ_X| rows ---")
    print("   ik   n   x_bare_sym    x_bare_nosym  Δx_bare      Δsex_0       ΔΣ_X(total)    trs?")
    idx = np.argsort(-abs_d_total)
    for i in idx[:10]:
        r = rows[i]
        flag = "TRS" if r['is_trs_k'] else "  -"
        print(f"   {r['ik']:2d}  {r['n']:3d}  "
              f"{r['x_bare_sym']:+12.6f}  {r['x_bare_nosym']:+12.6f}  "
              f"{r['d_x_bare_eV']*1000:+10.4f}   {r['d_sex0_eV']*1000:+10.4f}   "
              f"{r['d_SigX_total']*1000:+12.4f}   {flag}")

    return verdict, max_dtotal


if __name__ == '__main__':
    verdict, mx = main()
    # Exit 0 PASS, 1 MARGINAL, 2 FAIL — for CI-style use.
    sys.exit(0 if verdict == "PASS" else (1 if verdict == "MARGINAL" else 2))
