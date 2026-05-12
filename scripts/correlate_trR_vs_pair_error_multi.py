#!/usr/bin/env python3
"""Multi-series version of ``correlate_trR_vs_pair_error.py``: scan every
(n_cond, band_pool, N_μ) combination under a sweep dir, pair with
``tr(R_k)/tr(G)`` from a pchol assay JSON, and plot them all on one
log-log axis. Fits a single slope across all points to answer: does
``pair_err ∝ sqrt(trR/trG)`` hold universally, or does the slope
depend on n_cond / band class?

Directory convention (matching /tmp/isdf_big_sweep/ naming):
    <sweep_dir>/ncond<NC>_Nmu<NMU>_<POOL>/aggregate.json

Usage
-----
    $ python3 correlate_trR_vs_pair_error_multi.py \\
        --pchol-json /path/to/pchol_assay.json \\
        --sweep-dir /tmp/isdf_big_sweep \\
        --out multi_correlation.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pchol-json', required=True, type=Path)
    ap.add_argument('--sweep-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('multi_correlation.png'))
    args = ap.parse_args()

    # Load trR/trG traces, keyed by n_cond.
    assay = json.loads(args.pchol_json.read_text())

    # Walk sweep dir, collect (n_cond, Nmu, pool, median_rel_err).
    rx = re.compile(r'ncond(\d+)_Nmu(\d+)_(\w+)$')
    records = []
    for d in sorted(args.sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        m = rx.match(d.name)
        if m is None:
            continue
        nc, Nmu, pool = int(m.group(1)), int(m.group(2)), m.group(3)
        agg_path = d / 'aggregate.json'
        if not agg_path.exists():
            continue
        agg = json.loads(agg_path.read_text())
        med = float(agg['overall']['median'])
        if str(nc) not in assay['runs']:
            continue
        trace = np.asarray(assay['runs'][str(nc)]['trR_over_trG'])
        if Nmu >= trace.size:
            continue
        records.append(dict(n_cond=nc, Nmu=Nmu, pool=pool,
                            trR_over_trG=float(trace[Nmu]), median_err=med,
                            by_class=agg.get('by_class', {})))

    if not records:
        raise RuntimeError("no records found; check --sweep-dir pattern")

    # Plot: series per (n_cond, pool); log-log; fit all points.
    fig, (ax, ax_by_class) = plt.subplots(1, 2, figsize=(16, 6))

    markers = {300: 'o', 600: '^', 1000: 's'}
    colors = {'mixed': '#1f77b4', 'cross': '#d62728', 'val_only': '#2ca02c',
              'cond_only': '#ff7f0e'}

    by_series = defaultdict(list)
    for r in records:
        by_series[(r['n_cond'], r['pool'])].append(r)
    for (nc, pool), pts in sorted(by_series.items()):
        pts.sort(key=lambda p: p['Nmu'])
        xs = [p['trR_over_trG'] for p in pts]
        ys = [p['median_err'] for p in pts]
        ax.loglog(xs, ys, marker=markers.get(nc, 'x'), ms=9, lw=1.6,
                  color=colors.get(pool, 'k'),
                  label=f'n_cond={nc}, {pool}')
        for p, x, y in zip(pts, xs, ys):
            ax.annotate(f"  {p['Nmu']}", (x, y), fontsize=7, alpha=0.7)

    # Overall fit across all points.
    xs_all = np.log10([r['trR_over_trG'] for r in records])
    ys_all = np.log10([r['median_err'] for r in records])
    slope, intercept = np.polyfit(xs_all, ys_all, 1)
    r_all = np.corrcoef(xs_all, ys_all)[0, 1]
    tr_fit = np.logspace(xs_all.min() - 0.3, xs_all.max() + 0.3, 40)
    ax.loglog(tr_fit, 10 ** (slope * np.log10(tr_fit) + intercept),
              'k--', lw=1.3, alpha=0.7,
              label=f'global fit: slope={slope:.2f}, r={r_all:.3f}')

    ax.set_xlabel(r'$\mathrm{tr}(R_k)/\mathrm{tr}(G)$ at $k = N_\mu$', fontsize=12)
    ax.set_ylabel(r'median rel. $L_2$ pair-product error', fontsize=12)
    ax.set_title('Selection metric vs actual reconstruction error\n'
                 '(Si 2×2×2 parabands; labels = N_μ)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95, ncol=2)

    # Right panel: per-band-class errors at fixed N_μ, one marker per class per sweep point
    # — answers "does cross (vc) have different reconstruction quality than
    # mixed (mostly cc)?"
    ax2 = ax_by_class
    for r in records:
        x = r['trR_over_trG']
        for cls, st in r.get('by_class', {}).items():
            if 'median' not in st or st.get('count', 0) < 3:
                continue
            ax2.loglog(x, st['median'],
                       marker={'vv': 'o', 'cc': 's', 'vc': 'D', 'cv': 'v'}.get(cls, 'x'),
                       color={'vv': 'tab:orange', 'vc': 'tab:red',
                              'cv': 'tab:purple', 'cc': 'tab:blue'}.get(cls, 'k'),
                       ms=7, alpha=0.55)
    # Legend stub
    for cls, c in [('vv', 'tab:orange'), ('cc', 'tab:blue'),
                   ('vc', 'tab:red'), ('cv', 'tab:purple')]:
        ax2.plot([], [], marker={'vv': 'o', 'cc': 's', 'vc': 'D', 'cv': 'v'}[cls],
                 color=c, ls='', ms=8, label=cls)
    ax2.set_xlabel(r'$\mathrm{tr}(R_k)/\mathrm{tr}(G)$ at $k = N_\mu$', fontsize=12)
    ax2.set_ylabel(r'median rel. $L_2$ pair-product error (per class)', fontsize=12)
    ax2.set_title('Per-band-class reconstruction error', fontsize=11,
                  fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"wrote {args.out}")
    print(f"Global fit: slope={slope:.3f}, Pearson r={r_all:.4f}, "
          f"N_points={len(records)}")
    print()
    # Summary table
    print(f"{'n_cond':>6}  {'N_μ':>5}  {'pool':>9}  {'trR/trG':>10}  "
          f"{'median err':>12}")
    for r in sorted(records, key=lambda r: (r['n_cond'], r['pool'], r['Nmu'])):
        print(f"{r['n_cond']:>6}  {r['Nmu']:>5}  {r['pool']:>9}  "
              f"{r['trR_over_trG']:>10.2e}  {r['median_err']:>12.3e}")


if __name__ == '__main__':
    main()
