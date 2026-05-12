"""Compare LORRAX val_cond COHSEX vs BGW COHSEX.

For each (wfn_label, n_cond, N_mu) triple:
  • Read LORRAX eqp0.dat → {band: {sigSX, sigCOH, sigTOT}}
  • Read BGW sigma_hp.log → {k: {band: {X, SX_X, CH, CHp, Corp=SX_X+CHp}}}
  • Compare sigTOT = sigSX + sigCOH  vs  BGW Sig = X + SX_X + CH
    and      sigTOT  vs  BGW Sig' = X + SX_X + CH'  (primed, preferred)

Reports mean absolute error (MAE) per config, and emits a multi-panel
figure:
  - one subplot per (wfn, n_cond) combination (2 × 3 = 6 panels)
  - in each panel: MAE vs N_μ with one line per reference (Sig, Sigp)

Plus a summary bar chart of MAE at N_μ=1200 across all 6 configs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_valcond_2026-04-18')

N_CONDS = [60, 140, 208]
N_MUS = [400, 800, 1200]
WFNS = [('parabands', 'parabands_4200'), ('pseudo', 'pseudo_100sl_216')]


def parse_eqp0_cohsex(path: Path) -> dict:
    """LORRAX COHSEX eqp0: {k_idx: {band_1idx: {sigSX, sigCOH, sigTOT}}}.

    LORRAX format (eqp0_noqsym.dat):
        k-point <N>:
        -------
        n=<0idx>  sigSX=...  sigCOH=...  sigTOT=...  VH=...
    k-point indices are LORRAX 0-indexed (0..nk-1). We map to 1-indexed to
    match BGW.
    """
    result: dict[int, dict[int, dict]] = {}
    current_k = None
    for line in path.read_text().splitlines():
        ks = line.strip()
        mk = re.match(r'k-point\s+(\d+):', ks)
        if mk:
            current_k = int(mk.group(1)) + 1  # 0-idx → 1-idx
            result.setdefault(current_k, {})
            continue
        mb = re.match(r'n=(\d+)\s+', ks)
        if not mb:
            continue
        n1 = int(mb.group(1)) + 1  # band 0-idx → 1-idx
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', ks)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', ks)
        mt = re.search(r'sigTOT=\s*([-\d.Ee]+)', ks)
        if ms and mc and mt:
            if current_k is None:
                current_k = 1
                result.setdefault(1, {})
            result[current_k][n1] = {
                'sigSX': float(ms.group(1)),
                'sigCOH': float(mc.group(1)),
                'sigTOT': float(mt.group(1)),
            }
    return result


def parse_bgw_sigma_hp(path: Path) -> tuple[dict, dict]:
    """BGW sigma_hp.log: ({k_idx: {band_1idx: data}}, {k_idx: (kx,ky,kz)}).

    Uses the primed columns too, same as in parser in PARSE_OUTPUTS.md.
    """
    result: dict[int, dict[int, dict]] = {}
    kcoords: dict[int, tuple] = {}
    current_k = None
    for line in path.read_text().splitlines():
        s = line.strip()
        mk = re.match(
            r'k\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+ik\s*=\s*(\d+)', s
        )
        if mk:
            current_k = int(mk.group(4))
            kcoords[current_k] = tuple(float(mk.group(i)) for i in (1, 2, 3))
            result.setdefault(current_k, {})
            continue
        p = s.split()
        if len(p) >= 14 and p[0].isdigit():
            if current_k is None:
                continue
            n = int(p[0])
            X = float(p[3]); SXmX = float(p[4])
            CH = float(p[5]); CHp = float(p[10])
            Sig = X + SXmX + CH
            Sigp = X + SXmX + CHp
            result[current_k][n] = {
                'X': X, 'SXmX': SXmX, 'CH': CH, 'CHp': CHp,
                'Sig': Sig, 'Sigp': Sigp,
                'Cor': SXmX + CH, 'Corp': SXmX + CHp,
            }
    return result, kcoords


def lorrax_kvecs_from_wfn(wfn_path: Path) -> np.ndarray:
    """Read k-vectors (fractional) from an NSCF WFN.h5 in the *same order*
    LORRAX's full-BZ unfold lists them. For a 2×2×2 grid:

        (0,0,0), (0,0,0.5), (0,0.5,0), (0,0.5,0.5),
        (0.5,0,0), (0.5,0,0.5), (0.5,0.5,0), (0.5,0.5,0.5)

    We synthesize them directly from kgrid here (no HDF5 read required):
    LORRAX's ``load_wfns`` full-BZ unfold enumerates kz fastest, then ky,
    then kx (see common/load_wfns.py).
    """
    import h5py
    with h5py.File(wfn_path, 'r') as h:
        kgrid = [int(k) for k in h['mf_header/kpoints/kgrid'][:]]
    nkx, nky, nkz = kgrid
    kvecs = []
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                kvecs.append([ix / nkx, iy / nky, iz / nkz])
    return np.asarray(kvecs, dtype=np.float64)


def match_bgw_to_lorrax(kcoords_bgw, kvecs_lorrax, tol=1e-3):
    """For each BGW IBZ k-point, find the matching LORRAX full-BZ k-index
    by fractional-coord equality mod 1. Returns {bgw_ik: lorrax_k_1idx}.
    """
    result = {}
    for ik_bgw, kbgw in kcoords_bgw.items():
        q = np.asarray(kbgw, dtype=np.float64)
        diff = np.mod(kvecs_lorrax - q + 0.5, 1.0) - 0.5
        dist = np.linalg.norm(diff, axis=1)
        imin = int(np.argmin(dist))
        if dist[imin] > tol:
            print(f"[match_kpt] BGW ik={ik_bgw} k={kbgw} no match within tol (closest LORRAX k={imin} dist={dist[imin]:.4f})")
            continue
        result[ik_bgw] = imin + 1  # 0-idx → 1-idx
    return result


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float('nan')


def compare_one(lrx_path: Path, bgw_path: Path,
                lorrax_kvecs: np.ndarray) -> dict:
    lrx = parse_eqp0_cohsex(lrx_path)
    bgw, bgw_kcoords = parse_bgw_sigma_hp(bgw_path)
    kmap = match_bgw_to_lorrax(bgw_kcoords, lorrax_kvecs)
    # Build per-band comparison arrays (matched k-points, bands in common).
    lrx_totals, bgw_sigs, bgw_sigps = [], [], []
    band_info = []
    for ik_bgw, bgw_bands in bgw.items():
        lrx_ik = kmap.get(ik_bgw)
        if lrx_ik is None: continue
        lrx_k = lrx.get(lrx_ik, {})
        if not lrx_k: continue
        for n, bgw_row in bgw_bands.items():
            lrx_row = lrx_k.get(n)
            if lrx_row is None: continue
            lrx_totals.append(lrx_row['sigTOT'])
            bgw_sigs.append(bgw_row['Sig'])
            bgw_sigps.append(bgw_row['Sigp'])
            band_info.append((ik_bgw, n))
    lrx_totals = np.asarray(lrx_totals)
    bgw_sigs = np.asarray(bgw_sigs)
    bgw_sigps = np.asarray(bgw_sigps)
    return {
        'n_compared': len(lrx_totals),
        'kmap_bgw_to_lorrax': kmap,
        'mae_vs_sig': mae(lrx_totals, bgw_sigs),
        'mae_vs_sigp': mae(lrx_totals, bgw_sigps),
        'max_abs_vs_sigp': float(np.nanmax(np.abs(lrx_totals - bgw_sigps)))
            if bgw_sigps.size else float('nan'),
        'band_info': band_info,
        'lrx_sigTOT': lrx_totals.tolist(),
        'bgw_Sig': bgw_sigs.tolist(),
        'bgw_Sigp': bgw_sigps.tolist(),
    }


def main():
    # Read LORRAX's full-BZ k-ordering from the pseudo WFN (same kgrid for all runs).
    pseudo_wfn = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5')
    kvecs = lorrax_kvecs_from_wfn(pseudo_wfn)
    print(f"[compare] LORRAX full-BZ k-ordering: {kvecs.tolist()}")

    results = defaultdict(dict)  # results[(wfn, ncond)][Nmu] = compare dict
    for wfn_short, wfn_label in WFNS:
        for nc in N_CONDS:
            bgw_log = ROOT / 'bgw' / f'{wfn_short}_ncond{nc}' / 'sigma_hp.log'
            if not bgw_log.exists():
                print(f"[compare] missing BGW: {bgw_log}")
                continue
            for nm in N_MUS:
                lrx_dir = ROOT / 'lorrax' / f'{wfn_short}_ncond{nc}_Nmu{nm}'
                # LORRAX eqp file could be eqp0_noqsym.dat or eqp0.dat
                lrx_path = lrx_dir / 'eqp0_noqsym.dat'
                if not lrx_path.exists():
                    lrx_path = lrx_dir / 'eqp0.dat'
                if not lrx_path.exists():
                    print(f"[compare] missing LORRAX: {lrx_dir}")
                    continue
                cmp_ = compare_one(lrx_path, bgw_log, kvecs)
                results[(wfn_label, nc)][nm] = cmp_

    # Dump full JSON.
    (ROOT / 'comparison.json').write_text(
        json.dumps(
            {f"{w}_{n}": {nm: v for nm, v in d.items()}
             for (w, n), d in results.items()},
            indent=2,
        )
    )
    print(f"[compare] wrote {ROOT / 'comparison.json'}")

    # Print summary table.
    print("\n=== MAE vs BGW Sig' (primed, SX-X + CH') ===")
    print(f"{'WFN':>20} {'n_cond':>6} {'N_mu':>5}  {'n_cmp':>5}  {'MAE_Sig_eV':>12}  {'MAE_Sigp_eV':>12}")
    for (wfn, nc), by_nm in sorted(results.items()):
        for nm in N_MUS:
            r = by_nm.get(nm)
            if r is None: continue
            print(f"{wfn:>20} {nc:>6} {nm:>5}  {r['n_compared']:>5}  "
                  f"{r['mae_vs_sig']:>12.4f}  {r['mae_vs_sigp']:>12.4f}")

    # Multi-panel: MAE vs N_μ, one panel per (wfn, ncond).
    cmap = plt.get_cmap('tab10')
    n_wfn = len(WFNS)
    n_nc = len(N_CONDS)
    fig, axes = plt.subplots(n_wfn, n_nc, figsize=(15, 8), squeeze=False, sharey='row')
    for i, (_, wfn_label) in enumerate(WFNS):
        for j, nc in enumerate(N_CONDS):
            ax = axes[i][j]
            nm_xs = []
            mae_sig = []; mae_sigp = []
            for nm in N_MUS:
                r = results.get((wfn_label, nc), {}).get(nm)
                if r is None: continue
                nm_xs.append(nm)
                mae_sig.append(r['mae_vs_sig'])
                mae_sigp.append(r['mae_vs_sigp'])
            if nm_xs:
                ax.plot(nm_xs, mae_sigp, 'o-', color=cmap(0), label="MAE vs BGW Sig' (CH')", lw=2, ms=8)
                ax.plot(nm_xs, mae_sig, 's--', color=cmap(1), label="MAE vs BGW Sig (CH)", lw=1.5, ms=6)
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.2)
            ax.set_title(f'{wfn_label}  n_cond={nc}', fontsize=10)
            if i == n_wfn - 1: ax.set_xlabel(r'$N_\mu$')
            if j == 0: ax.set_ylabel('MAE (eV)')
            if i == 0 and j == 0: ax.legend(fontsize=8)
    fig.suptitle('LORRAX val_cond COHSEX vs BGW: MAE vs N_μ  (Si 2×2×2, use_bgw_vcoul=true)',
                 y=1.00, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(ROOT / 'mae_vs_Nmu.png', dpi=150, bbox_inches='tight')
    print(f"[compare] wrote {ROOT / 'mae_vs_Nmu.png'}")

    # Per-band scatter at max N_μ, best reference (Sig').
    fig2, axes2 = plt.subplots(n_wfn, n_nc, figsize=(15, 8), squeeze=False)
    for i, (_, wfn_label) in enumerate(WFNS):
        for j, nc in enumerate(N_CONDS):
            ax = axes2[i][j]
            for nm, marker, color in [(400, 'v', cmap(2)),
                                      (800, 's', cmap(1)),
                                      (1200, 'o', cmap(0))]:
                r = results.get((wfn_label, nc), {}).get(nm)
                if r is None: continue
                x = np.asarray(r['bgw_Sigp'])
                y = np.asarray(r['lrx_sigTOT'])
                ax.scatter(x, y, s=20, marker=marker, color=color, alpha=0.7,
                           label=f'N_μ={nm}')
            # Diagonal y=x reference.
            all_v = []
            for nm in N_MUS:
                r = results.get((wfn_label, nc), {}).get(nm)
                if r: all_v.extend(r['bgw_Sigp'] + r['lrx_sigTOT'])
            if all_v:
                lo, hi = min(all_v) - 0.5, max(all_v) + 0.5
                ax.plot([lo, hi], [lo, hi], 'k-', lw=0.5, alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.set_title(f'{wfn_label}  n_cond={nc}', fontsize=10)
            if i == n_wfn - 1: ax.set_xlabel("BGW Sig' (eV)")
            if j == 0: ax.set_ylabel("LORRAX sigTOT (eV)")
            if i == 0 and j == 0: ax.legend(fontsize=8)
    fig2.suptitle('Per-band parity: LORRAX sigTOT vs BGW Sig′', y=1.00,
                  fontsize=12, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(ROOT / 'parity_scatter.png', dpi=150, bbox_inches='tight')
    print(f"[compare] wrote {ROOT / 'parity_scatter.png'}")


if __name__ == '__main__':
    main()
