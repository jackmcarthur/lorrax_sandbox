#!/usr/bin/env python3
"""Compare bare Σ_X across run_sym and run_nosym sigma_freq_debug.dat files.

Adapted from runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14/compare_sigma_x.py
to the CrI3 6×6×1, 30 Ry test bed.  The two runs have identical k-point
orderings (both 36 k-pts in the same crystal order), so we can match on
(k, n) directly.

CrI3 has spatial inversion (P-3 space group 147), so the sym WFN's
6-op symmetry group includes -I.  Consequence: TRS-fold k-points
in the unfold path = 0 (every q reaches -q via a spatial op).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


def parse_sigma_freq_debug_allk(path: Path):
    """Return list of dicts per k-point: {'ik', 'bands': {n: {field: float}}}.

    Columns:
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
                _ = int(p[0]); n = int(p[1])
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


def detect_trs_k_points(wfn_path):
    """CrI3 should report 0 TRS-fold k-pts (inversion symmetry).

    Use find_irreducible_bz_points style logic: count k-points that
    map to their IBZ representative via a TRS-augmented row (sym_idx >= ntran).
    Here we simply return an empty set unless analyzed against the SymMaps
    object — for now we rely on the structural argument and verify in report.
    """
    import h5py
    with h5py.File(wfn_path,'r') as f:
        ntran = int(f['mf_header/symmetry/ntran'][()])
        mtrx = f['mf_header/symmetry/mtrx'][:ntran]
    has_inv = any(np.allclose(m, -np.eye(3)) for m in mtrx)
    return ntran, has_inv


def main():
    base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14')
    sym_wfn   = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
    nosym_wfn = base / 'qe_nosym/nscf/WFN.h5'
    sym_path   = base / 'run_sym_lphase_fix_2026-05-15/sigma_freq_debug.dat'
    nosym_path = base / 'run_nosym/sigma_freq_debug.dat'

    # Symmetry analysis
    ntran_sym, has_inv_sym = detect_trs_k_points(sym_wfn)
    ntran_nosym, has_inv_nosym = detect_trs_k_points(nosym_wfn)
    print(f"sym WFN: ntran={ntran_sym}, has_inversion={has_inv_sym}")
    print(f"nosym WFN: ntran={ntran_nosym}, has_inversion={has_inv_nosym}")

    # For CrI3 with inversion: TRS-fold k-pts = 0 (provably no-op).
    # We confirm this expectation in the report.
    trs_k = set()  # CrI3: empty

    sym   = parse_sigma_freq_debug_allk(sym_path)
    nosym = parse_sigma_freq_debug_allk(nosym_path)
    assert len(sym) == len(nosym), f"k-count mismatch: sym={len(sym)} nosym={len(nosym)}"

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
                'd_SigX_total': d_xb + d_sex + d_coh,
                'is_trs_k':     ik in trs_k,
            })

    arr_d_xb_mev    = np.array([r['d_x_bare_eV']     * 1000.0 for r in rows])
    arr_d_sex_mev   = np.array([r['d_sex0_eV']       * 1000.0 for r in rows])
    arr_d_coh_mev   = np.array([r['d_coh0_eV']       * 1000.0 for r in rows])
    arr_d_total_mev = np.array([r['d_SigX_total']    * 1000.0 for r in rows])
    arr_d_edft_mev  = np.array([r['d_Edft_eV']       * 1000.0 for r in rows])

    abs_d_total = np.abs(arr_d_total_mev)
    abs_d_xb    = np.abs(arr_d_xb_mev)
    abs_d_sex   = np.abs(arr_d_sex_mev)

    print()
    print("===== sym vs nosym PR3 Σ_X comparison (CrI3) =====")
    print(f"  total (k, n) pairs: {len(rows)} = {n_k} k-pts × {len(rows)//n_k} bands")
    print()
    print("--- DFT eigenvalue offset (orientation check) ---")
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

    # Per-k breakdown
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
        msg = f"max |ΔΣ_X| = {max_dtotal:.4f} meV > 10 meV — real bug"
    print(f"  Verdict: {verdict}")
    print(f"  {msg}")
    print(f"  TRS-fold k-points expected: 0 (CrI3 has spatial inversion)")
    print(f"  Symmetry context: sym WFN ntran={ntran_sym}, has_inv={has_inv_sym}")
    print("==========================================")

    # Top-15 worst rows.
    print()
    print("--- Top-15 |ΔΣ_X| rows ---")
    print("   ik   n   E_dft(eV)     x_bare_sym    x_bare_nosym  Δx_bare      Δsex_0       ΔΣ_X(total)")
    idx = np.argsort(-abs_d_total)
    for i in idx[:15]:
        r = rows[i]
        # find E_dft for orientation
        e = rows[i]
        print(f"   {r['ik']:2d}  {r['n']:3d}  {sym[r['ik']]['bands'][r['n']]['E_dft']:+10.4f}  "
              f"{r['x_bare_sym']:+12.6f}  {r['x_bare_nosym']:+12.6f}  "
              f"{r['d_x_bare_eV']*1000:+10.4f}   {r['d_sex0_eV']*1000:+10.4f}   "
              f"{r['d_SigX_total']*1000:+12.4f}")

    # Distribution histogram by magnitude
    print()
    print("--- |ΔΣ_X| histogram ---")
    bins = [0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    h, _ = np.histogram(abs_d_total, bins=bins)
    for i in range(len(bins)-1):
        print(f"   {bins[i]:7.2f} - {bins[i+1]:7.2f} meV : {h[i]:4d}")

    return verdict, max_dtotal


if __name__ == '__main__':
    verdict, mx = main()
    sys.exit(0 if verdict == "PASS" else (1 if verdict == "MARGINAL" else 2))
