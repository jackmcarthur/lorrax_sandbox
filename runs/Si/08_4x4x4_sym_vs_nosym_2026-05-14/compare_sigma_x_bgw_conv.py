#!/usr/bin/env python3
"""Compare bare Σ_X across run_sym and run_nosym sigma_freq_debug.dat files
on Si 4x4x4 SOC. The Si sym WFN has 36/48 non-symmorphic τ-phase operations
(diamond Fd-3m glides/screws); the nosym WFN has ntran=1.  This is a mirror
of the MoS₂ test (sym_vs_nosym_pr3_validation.md) but on a system that
exercises the τ-phase code path in unfold_psi.

E_dft, kin_ion, V_H all match exactly across runs at every (k, n) — only
x_bare can differ if Σ_X has a symmetry-unfold bug.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


def parse_sigma_freq_debug_allk(path: Path):
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
    base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14')
    sym_path   = base / 'run_sym_bgw_conv_2026-05-15/sigma_freq_debug.dat'
    nosym_path = base / 'run_nosym/sigma_freq_debug.dat'

    sym   = parse_sigma_freq_debug_allk(sym_path)
    nosym = parse_sigma_freq_debug_allk(nosym_path)
    assert len(sym) == len(nosym), f"k-count mismatch: sym={len(sym)} nosym={len(nosym)}"

    # Si has inversion → no TRS-fold rows fire.  All 8 IBZ k-pts unfold to
    # 64 BZ k-pts via spatial sym ops alone.  But 36/48 sym ops are
    # non-symmorphic → τ-phase code path in unfold_psi is EXERCISED at
    # most kpts.
    n_k = len(sym)

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
            })

    arr_d_xb_mev    = np.array([r['d_x_bare_eV']     * 1000.0 for r in rows])
    arr_d_sex_mev   = np.array([r['d_sex0_eV']       * 1000.0 for r in rows])
    arr_d_coh_mev   = np.array([r['d_coh0_eV']       * 1000.0 for r in rows])
    arr_d_total_mev = np.array([r['d_SigX_total']    * 1000.0 for r in rows])
    arr_d_edft_mev  = np.array([r['d_Edft_eV']       * 1000.0 for r in rows])

    abs_d_total = np.abs(arr_d_total_mev)
    abs_d_xb    = np.abs(arr_d_xb_mev)
    abs_d_sex   = np.abs(arr_d_sex_mev)

    print("===== sym vs nosym PR3 Σ_X comparison — Si 4×4×4 SOC =====")
    print(f"  total (k, n) pairs: {len(rows)} = {n_k} k-pts × {len(rows)//n_k} bands")
    print()
    print("--- DFT eigenvalue offset (orientation check, should be 0) ---")
    print(f"   max |ΔE_dft|        = {np.max(np.abs(arr_d_edft_mev)):10.4f} meV")
    print()
    print("--- Σ_X components ---")
    print(f"   max |Δx_bare|       = {np.max(abs_d_xb):10.4f} meV")
    print(f"   max |Δsex_0|        = {np.max(abs_d_sex):10.4f} meV")
    print(f"   max |Δcoh_0|        = {np.max(np.abs(arr_d_coh_mev)):10.4f} meV")
    print(f"   max |ΔΣ_X (total)|  = {np.max(abs_d_total):10.4f} meV")
    print(f"   mean ΔΣ_X (total)   = {np.mean(arr_d_total_mev):10.4f} meV")
    print(f"   stddev ΔΣ_X (total) = {np.std(arr_d_total_mev):10.4f} meV")
    print()

    print("--- Per-k breakdown (max/mean |ΔΣ_X|, meV) ---")
    print("   ik     max|ΔΣ_X|     mean|ΔΣ_X|   max|Δx_bare|  max|Δsex_0|   N_bands")
    for ik in range(n_k):
        mask = np.array([r['ik'] == ik for r in rows])
        nb = mask.sum()
        if nb == 0:
            continue
        print(f"   {ik:2d}    {np.max(abs_d_total[mask]):10.4f}   "
              f"{np.mean(abs_d_total[mask]):10.4f}   {np.max(abs_d_xb[mask]):10.4f}   "
              f"{np.max(abs_d_sex[mask]):10.4f}   {nb}")

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

    print()
    print("--- Top-10 |ΔΣ_X| rows ---")
    print("   ik   n   x_bare_sym    x_bare_nosym  Δx_bare       Δsex_0        ΔΣ_X(total)")
    idx = np.argsort(-abs_d_total)
    for i in idx[:10]:
        r = rows[i]
        print(f"   {r['ik']:2d}  {r['n']:3d}  "
              f"{r['x_bare_sym']:+12.6f}  {r['x_bare_nosym']:+12.6f}  "
              f"{r['d_x_bare_eV']*1000:+11.4f}   {r['d_sex0_eV']*1000:+11.4f}   "
              f"{r['d_SigX_total']*1000:+13.4f}")

    return verdict, max_dtotal


if __name__ == '__main__':
    verdict, mx = main()
    sys.exit(0 if verdict == "PASS" else (1 if verdict == "MARGINAL" else 2))
