#!/usr/bin/env python3
"""Correlate the centroid-selection metric ``tr(R_k)/tr(G)`` (from
``pchol_pseudoband_assay.py``) with the actual pair-product
reconstruction error measured by ``validate_isdf_reconstruction.py``.

The pivoted-Cholesky pruning stage produces a convergence proxy:
``tr(R_k)/tr(G)`` after k pivots, which is the fraction of the
candidate-Gram's Frobenius mass left uncovered. It is cheap to compute
(runs in seconds) but only gives information about the q=0 Gram span
— not about how faithfully the downstream ζ_q reconstructs arbitrary
(n, m, k, q) pair products.

This helper joins both sides: for a list of (n_cond, N_μ) points it
loads

  * ``tr(R_k)/tr(G)[k = N_μ]`` from a pchol assay JSON
  * ``median rel-L2 pair error`` from a validate_isdf_reconstruction
    aggregate JSON

and plots them on log-log axes with a linear fit. A Pearson r close
to 1 and a slope close to 0.5 means "cheap selection metric
faithfully tracks actual fit error", which is the interesting
empirical result for this question.

Usage
-----
Point it at the assay JSON and a list of ``validate_isdf_reconstruction``
output directories (one per N_μ, same n_cond)::

    $ python3 correlate_trR_vs_pair_error.py \\
        --pchol-json /path/to/pchol_assay.json \\
        --ncond 300 \\
        --validate-dirs  /tmp/isdf_v_Nmu400  /tmp/isdf_v_Nmu1200  \\
                         /tmp/isdf_v_Nmu3200  /tmp/isdf_v_Nmu6000 \\
        --Nmu 400 1200 3200 6000 \\
        --out trR_vs_pair_err.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pchol-json', required=True, type=Path,
                    help='Output of pchol_pseudoband_assay.py '
                         '(contains trR_over_trG arrays per n_cond).')
    ap.add_argument('--ncond', type=int, required=True,
                    help='Which n_cond row of the assay to use.')
    ap.add_argument('--validate-dirs', nargs='+', required=True, type=Path,
                    help='One validate_isdf_reconstruction output dir per N_μ.')
    ap.add_argument('--Nmu', nargs='+', type=int, required=True,
                    help='N_μ for each --validate-dirs entry, in matching order.')
    ap.add_argument('--out', type=Path, default=Path('trR_vs_pair_err.png'))
    args = ap.parse_args()

    if len(args.validate_dirs) != len(args.Nmu):
        raise ValueError("--validate-dirs and --Nmu must have same length")

    assay = json.loads(args.pchol_json.read_text())
    trace = np.asarray(assay['runs'][str(args.ncond)]['trR_over_trG'])

    # Gather (trR/trG, median rel L2) pairs.
    tr_vals, pair_vals = [], []
    for dir_, Nmu in zip(args.validate_dirs, args.Nmu):
        agg = json.loads((dir_ / 'aggregate.json').read_text())
        tr_vals.append(float(trace[Nmu]))
        pair_vals.append(float(agg['overall']['median']))
    tr = np.asarray(tr_vals)
    pe = np.asarray(pair_vals)

    # Log-log fit + Pearson.
    logtr, logpe = np.log10(tr), np.log10(pe)
    r = np.corrcoef(logtr, logpe)[0, 1]
    slope, intercept = np.polyfit(logtr, logpe, 1)

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(tr, pe, 'o', ms=11, color='tab:blue', zorder=3,
              label=f'data (n_cond = {args.ncond})')
    for t, p, N in zip(tr, pe, args.Nmu):
        ax.annotate(f'  N_μ={N}', (t, p), fontsize=9, color='k',
                    verticalalignment='center')
    tr_fit = np.logspace(np.log10(tr.min()) - 0.5,
                         np.log10(tr.max()) + 0.5, 40)
    ax.loglog(tr_fit, 10 ** (slope * np.log10(tr_fit) + intercept),
              'r--', lw=1.5,
              label=f'fit: slope = {slope:.2f},  r = {r:.3f}')
    ax.set_xlabel(r'$\mathrm{tr}(R_k)/\mathrm{tr}(G)$ at $k = N_\mu$',
                  fontsize=13)
    ax.set_ylabel(r'median rel. $L_2$ pair-product reconstruction error',
                  fontsize=13)
    ax.set_title(f'Centroid-selection metric vs actual ISDF reconstruction error\n'
                 f'(Si 2×2×2 parabands, n_cond = {args.ncond})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"wrote {args.out}")
    print(f"  Pearson r (log-log) = {r:.4f}")
    print(f"  slope = {slope:.3f}   (≈ 0.5 means pair_err ~ sqrt(trR/trG))")
    for N, t, p in zip(args.Nmu, tr, pe):
        print(f"  N_μ={N:>5}  trR/trG={t:.3e}  median_pair_err={p:.3e}")


if __name__ == '__main__':
    main()
