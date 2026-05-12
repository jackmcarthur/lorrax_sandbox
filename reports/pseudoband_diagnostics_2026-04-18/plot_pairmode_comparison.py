"""Plot all pair-mode × density-mode × pool-size diagnostics side-by-side.

Two figures:

  ``pchol_pairmode_curves.png`` — 3 rows (WFN type) × 2 cols (isdf_asym vs
      val_cond). Shows trR(k)/trG decay for every n_cond in each panel,
      colored by n_cond. Demonstrates val_cond gives sharp exponential
      decay even for pseudobands.

  ``pchol_diagnostic_summary.png`` — bar chart: at matched n_cond=64,
      trR/trG at k=1300 across every experimental axis:
      [valence M=2400 isdf_asym, M=4000 isdf_asym, M=4000 no-norms,
       uniform k-means, val_cond].
      One group per WFN type.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_runs(root, tag):
    d = json.loads((root / tag / 'pchol_assay.json').read_text())
    runs = sorted(
        [r for r in d['runs'].values() if not r.get('skipped', False)],
        key=lambda r: int(r['n_cond']),
    )
    return d, runs


def make_pairmode_curves(root, out_path):
    """3 × 2 grid: rows = WFN, cols = pair_mode {isdf_asym, val_cond}."""
    wfns = [
        ('parabands_4200', 'parabands_4200_valcond', 'parabands  (4200 bands, deterministic)'),
        ('pseudo_50sl_116', 'pseudo_50sl_116_valcond', 'pseudo 50-slab  (116 bands, 100 pseudo)'),
        ('pseudo_100sl_216', 'pseudo_100sl_216_valcond', 'pseudo 100-slab  (216 bands, 198 pseudo)'),
    ]
    fig, axes = plt.subplots(len(wfns), 2, figsize=(14, 4.2 * len(wfns)),
                             squeeze=False)
    cmap = plt.get_cmap('viridis')

    axes[0][0].set_title('isdf_asym  (sigma-like: |P|² full window)',
                         fontsize=12, fontweight='bold', pad=14)
    axes[0][1].set_title('val_cond  (chi-like: P_v · P_c cross term)',
                         fontsize=12, fontweight='bold', pad=14)

    for i, (isdf_tag, vc_tag, row_label) in enumerate(wfns):
        _, runs_isdf = _load_runs(root, isdf_tag)
        _, runs_vc = _load_runs(root, vc_tag)
        for col, runs in [(0, runs_isdf), (1, runs_vc)]:
            ax = axes[i][col]
            n = len(runs)
            for idx, r in enumerate(runs):
                trace = np.asarray(r['trR_over_trG'])
                k_acc = int(r['rank'])
                color = cmap(idx / max(n - 1, 1))
                ax.semilogy(np.arange(k_acc + 1), trace[:k_acc + 1],
                            label=f"n_cond = {r['n_cond']}",
                            color=color, lw=1.6)
            for h in [1e-3, 1e-6, 1e-9]:
                ax.axhline(h, color='k', lw=0.4, ls=':', alpha=0.4)
            ax.grid(True, which='both', alpha=0.2)
            ax.set_ylabel(r'$\mathrm{tr}(R_k)/\mathrm{tr}(G)$')
            if col == 1:
                ax.legend(loc='upper right', fontsize=8)
            if i == len(wfns) - 1:
                ax.set_xlabel('pivot count $k$')
        axes[i][0].text(
            -0.20, 0.5, row_label,
            transform=axes[i][0].transAxes, rotation=90,
            ha='center', va='center', fontsize=11, fontweight='bold',
        )

    fig.suptitle('Pair-mode comparison: isdf_asym (sigma-like) vs val_cond (chi-like)\n'
                 'Si 2×2×2, M = 2400, n_keep = 1300, seed = 42, ISDF band-norm clamp on',
                 y=1.00, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0.02, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"wrote {out_path}")
    plt.close(fig)


def make_diagnostic_summary(root, out_path, n_cond=64):
    """Grouped bar chart of trR/trG at k=1300 across every axis."""
    # (axis-label, data-tag for each WFN)
    axes_defs = [
        ('valence  M=2400\nisdf_asym', {
            'parabands_4200': 'parabands_4200',
            'pseudo_50sl_116': 'pseudo_50sl_116',
            'pseudo_100sl_216': 'pseudo_100sl_216',
        }),
        ('valence  M=4000\nisdf_asym', {
            'parabands_4200': 'parabands_4200_M4000',
            'pseudo_50sl_116': 'pseudo_50sl_116_M4000',
            'pseudo_100sl_216': 'pseudo_100sl_216_M4000',
        }),
        ('uniform  M=2400\nisdf_asym', {
            'parabands_4200': 'parabands_4200_uniform',
            'pseudo_50sl_116': 'pseudo_50sl_116_uniform',
            'pseudo_100sl_216': 'pseudo_100sl_216_uniform',
        }),
        ('M=4000\nno band_norms\nisdf_asym', {
            'pseudo_100sl_216': 'pseudo_100sl_216_M4000_nonorms',
        }),
        ('valence  M=2400\nval_cond', {
            'parabands_4200': 'parabands_4200_valcond',
            'pseudo_50sl_116': 'pseudo_50sl_116_valcond',
            'pseudo_100sl_216': 'pseudo_100sl_216_valcond',
        }),
    ]

    wfns = ['parabands_4200', 'pseudo_50sl_116', 'pseudo_100sl_216']

    # Collect trR/trG@k=1300 at n_cond=n_cond for every axis × wfn.
    data = {}
    for axis_label, mapping in axes_defs:
        for wfn in wfns:
            tag = mapping.get(wfn)
            if tag is None:
                continue
            d = json.loads((root / tag / 'pchol_assay.json').read_text())
            r = d['runs'].get(str(int(n_cond)))
            if r is None or r.get('skipped'):
                continue
            trace = np.asarray(r['trR_over_trG'])
            rank = int(r['rank'])
            data[(axis_label, wfn)] = float(trace[min(1300, rank)])

    # Build grouped bar chart.
    fig, ax = plt.subplots(figsize=(14, 6))
    axis_labels = [ad[0] for ad in axes_defs]
    x = np.arange(len(axis_labels))
    width = 0.25
    wfn_labels = {
        'parabands_4200':  'parabands (4200)',
        'pseudo_50sl_116': 'pseudo 50sl (116)',
        'pseudo_100sl_216':'pseudo 100sl (216)',
    }
    colors = {
        'parabands_4200': 'tab:blue',
        'pseudo_50sl_116': 'tab:orange',
        'pseudo_100sl_216': 'tab:green',
    }
    for j, wfn in enumerate(wfns):
        vals = [data.get((al, wfn), np.nan) for al in axis_labels]
        offs = (j - 1) * width
        ax.bar(x + offs, vals, width, label=wfn_labels[wfn], color=colors[wfn])

    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{tr}(R_k)/\mathrm{tr}(G)$  at  $k=1300$   (lower = better)',
                  fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.axhline(1e-6, color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(1e-3, color='k', lw=0.5, ls=':', alpha=0.5)
    ax.grid(True, which='both', axis='y', alpha=0.2)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.set_title(f'Diagnostic summary: trR/trG at k=1300, all tweaks  '
                 f'(shared n_cond = {n_cond})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"wrote {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_B_assay/pseudobands_sweep')
    ap.add_argument('--out-curves', default='pchol_pairmode_curves.png')
    ap.add_argument('--out-summary', default='pchol_diagnostic_summary.png')
    ap.add_argument('--summary-n-cond', type=int, default=64)
    args = ap.parse_args()
    root = Path(args.root)

    make_pairmode_curves(root, root / args.out_curves)
    make_diagnostic_summary(root, root / args.out_summary,
                            n_cond=args.summary_n_cond)


if __name__ == '__main__':
    main()
