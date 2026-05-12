"""Parse + compare BGW vs LORRAX COHSEX, plot N_μ convergence.

Assumes this layout:

    cohsex_highncond_2026-04-19/
      bgw/
        parabands_ncond60/sigma_hp.log
        parabands_ncond140/sigma_hp.log
        parabands_ncond208/sigma_hp.log
        pseudo_ncond60/sigma_hp.log
        pseudo_ncond140/sigma_hp.log
        pseudo_ncond208/sigma_hp.log       (symlinked from 03_bgw run; nbnd=215)
      lorrax/
        {parabands,pseudo}_ncond{60,140,208}_Nmu{400,800,1200}/eqp0_noqsym.dat

Pair-mode for centroid selection is val_cond throughout (see centroids/).

Comparison metric: for each band n ∈ [1, 16] and each k-point,
    LORRAX Σ_COHSEX = sigSX + sigCOH   (from eqp0_noqsym.dat)
    BGW    Σ_COHSEX = X + (SX-X) + CH' (col 4 + 5 + 11 of sigma_hp.log) ≡ Sig'
Plots: (a) per-band Σ scatter, colored by n_cond; (b) MAE vs N_μ for each
(WFN, n_cond) combination.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/'
            '00_si_2x2x2_60Ry/cohsex_highncond_2026-04-19')


def parse_sigma_hp(path: Path):
    """BGW sigma_hp.log → [{kcrys, ik, bands: {n: {X, SX_X, CH, CHp, Sig_prime}}}]"""
    blocks = []
    ik = None
    kcrys = None
    for line in path.read_text().splitlines():
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = tuple(float(m.group(i)) for i in (1, 2, 3))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            n = int(p[0])
            X, SX_X, CH = float(p[3]), float(p[4]), float(p[5])
            CHp = float(p[10])
            blocks[-1]['bands'][n] = {
                'X': X, 'SX_X': SX_X, 'CH': CH, 'CHp': CHp,
                'Sig': X + SX_X + CH,
                'Sig_prime': X + SX_X + CHp,
                'Cor_prime': SX_X + CHp,
            }
    return blocks


def parse_eqp0_cohsex(path: Path):
    """LORRAX COHSEX eqp0.dat → {(k_full_index, n_one_indexed): {sigSX, sigCOH, sigTOT}}

    Parses the ``k-point N:`` header markers — the file is split into
    one block per full-BZ k-point. k_full_index is 0-indexed matching
    the order of ``qp_wfn_rotations.h5/kpoints_crys``.
    """
    result = {}
    k_full = None
    for line in path.read_text().splitlines():
        mk = re.match(r'k-point\s+(\d+)\s*:', line.strip())
        if mk:
            k_full = int(mk.group(1))
            continue
        mn = re.match(r'n=(\d+)\s+', line.strip())
        if not mn or k_full is None:
            continue
        n1 = int(mn.group(1)) + 1
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', line)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', line)
        mt = re.search(r'sigTOT=\s*([-\d.Ee]+)', line)
        if ms and mc and mt:
            result[(k_full, n1)] = {
                'sigSX': float(ms.group(1)),
                'sigCOH': float(mc.group(1)),
                'sigTOT': float(mt.group(1)),
            }
    return result


def load_kpoints_full(run_dir: Path):
    """Load LORRAX's full-BZ k-point coords from qp_wfn_rotations.h5."""
    import h5py
    with h5py.File(run_dir / 'qp_wfn_rotations.h5', 'r') as f:
        return np.asarray(f['kpoints_crys'][:], dtype=np.float64)


def match_kfull_to_bgw_ibz(k_full, bgw_ibz_coords, tol=1e-4):
    """For each full-BZ k, find the BGW IBZ k it maps to via umklapp + sym.

    Uses simple Oh point-group of Si (permutations × sign flips) + integer
    umklapp shift. Returns dict: k_full_index → bgw_ik (1-indexed).
    """
    from itertools import permutations, product
    signs = list(product([1, -1], repeat=3))
    perms = list(permutations(range(3)))
    sym_mats = []
    for p in perms:
        for s in signs:
            M = np.zeros((3, 3))
            for i, pi in enumerate(p):
                M[i, pi] = s[i]
            sym_mats.append(M)

    mapping = {}
    for i, k in enumerate(k_full):
        found = False
        for bgw_idx, kb in enumerate(bgw_ibz_coords):
            for S in sym_mats:
                diff = k - S @ np.asarray(kb)
                diff_int = np.rint(diff)
                if np.all(np.abs(diff - diff_int) < tol):
                    mapping[i] = bgw_idx + 1  # BGW ik is 1-indexed
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"LORRAX k_full[{i}]={k} has no BGW IBZ match")
    return mapping


def collect():
    """Walk the tree and return a list of per-run records."""
    records = []
    # BGW: {(wfn, ncond): {ik → {n → bands dict}}}
    bgw_cache = {}
    for wfn in ('parabands',):
        for nc in (100, 300, 600, 1000):
            p = ROOT / 'bgw' / f'{wfn}_ncond{nc}' / 'sigma_hp.log'
            if not p.exists():
                print(f'[warn] missing BGW: {p}')
                continue
            blocks = parse_sigma_hp(p)
            by_ik = {b['ik']: b['bands'] for b in blocks}
            bgw_cache[(wfn, nc)] = by_ik

    # LORRAX: each run dir has eqp0_noqsym.dat or eqp0.dat.
    lorrax_runs = []
    for d in sorted((ROOT / 'lorrax').iterdir()):
        if not d.is_dir():
            continue
        name = d.name  # e.g. pseudo_ncond140_Nmu800
        m = re.match(r'(parabands|pseudo)_ncond(\d+)_Nmu(\d+)', name)
        if m is None:
            continue
        wfn = m.group(1)
        nc = int(m.group(2))
        Nm = int(m.group(3))
        eqp = d / 'eqp0_noqsym.dat'
        if not eqp.exists():
            eqp = d / 'eqp0.dat'
        if not eqp.exists():
            print(f'[warn] missing LORRAX: {d}')
            continue
        lorrax_bands = parse_eqp0_cohsex(eqp)
        lorrax_runs.append({
            'wfn': wfn, 'ncond': nc, 'Nmu': Nm,
            'dir': d, 'bands': lorrax_bands,
        })
    return bgw_cache, lorrax_runs


def compute_errors(bgw_cache, lorrax_runs):
    """For each LORRAX run, compute per-(k_full,n) LORRAX sigTOT − BGW Sig'.

    LORRAX writes all 8 full-BZ k-points; BGW writes the 4 IBZ k-points.
    We map each LORRAX k_full to its BGW IBZ partner via symmetry, so
    every LORRAX (k_full, n) gets compared against the BGW value at its
    IBZ representative.
    """
    records = []
    for r in lorrax_runs:
        key = (r['wfn'], r['ncond'])
        if key not in bgw_cache:
            continue
        bgw_by_ik = bgw_cache[key]
        # BGW IBZ k coords from the sigma_hp.log blocks (kcrys field).
        bgw_blocks_list = list(bgw_by_ik.items())  # ((ik, bands_dict), ...)
        # Need kcrys → re-parse to get them. Use a separate pass.
        bgw_ik_to_kcrys = {}
        for wfn_name, nc in [(key[0], key[1])]:
            sigma_path = ROOT / 'bgw' / f'{wfn_name}_ncond{nc}' / 'sigma_hp.log'
            for b in parse_sigma_hp(sigma_path):
                bgw_ik_to_kcrys[b['ik']] = np.asarray(b['kcrys'])
        # LORRAX full-BZ k coords.
        k_full = load_kpoints_full(r['dir'])
        bgw_ibz_coords = [bgw_ik_to_kcrys[ik] for ik in sorted(bgw_ik_to_kcrys)]
        full_to_bgw = match_kfull_to_bgw_ibz(k_full, bgw_ibz_coords)

        diffs = {}
        for (k_full_idx, n), lor in r['bands'].items():
            bgw_ik = full_to_bgw.get(k_full_idx)
            if bgw_ik is None or bgw_ik not in bgw_by_ik:
                continue
            if n not in bgw_by_ik[bgw_ik]:
                continue
            bg = bgw_by_ik[bgw_ik][n]
            diffs[(k_full_idx, n)] = {
                'lorrax_sigTOT': lor['sigTOT'],
                'bgw_Sig_prime': bg['Sig_prime'],
                'bgw_Sig_unprimed': bg['Sig'],
                'diff_prime': lor['sigTOT'] - bg['Sig_prime'],
                'diff_unprimed': lor['sigTOT'] - bg['Sig'],
                'bgw_ik': bgw_ik,
            }
        if not diffs:
            continue
        arr_prime = np.array([d['diff_prime'] for d in diffs.values()])
        arr_unpr = np.array([d['diff_unprimed'] for d in diffs.values()])
        records.append({
            **r,
            'n_points': len(diffs),
            'mae_prime': float(np.mean(np.abs(arr_prime))),
            'rmse_prime': float(np.sqrt(np.mean(arr_prime ** 2))),
            'max_prime': float(np.max(np.abs(arr_prime))),
            'mae_unpr': float(np.mean(np.abs(arr_unpr))),
            'diffs': diffs,
        })
    return records


def plot_mae_vs_Nmu(records, out_path):
    """MAE(Σ) vs N_μ for each n_cond. Log-log so power-law scaling shows."""
    from collections import defaultdict
    by_group = defaultdict(list)  # ncond → list of (Nmu, mae)
    for r in records:
        by_group[r['ncond']].append((r['Nmu'], r['mae_prime']))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    cmap = plt.get_cmap('viridis')
    ncond_values = sorted(by_group.keys())
    for idx, nc in enumerate(ncond_values):
        pts = sorted(by_group[nc])
        Ns, maes = zip(*pts)
        color = cmap(idx / max(len(ncond_values) - 1, 1))
        ax.loglog(Ns, maes, marker='o', color=color, lw=2.0, ms=9,
                  label=f'n_cond = {nc}')
    ax.set_xlabel(r'$N_\mu$  (centroids)', fontsize=13)
    ax.set_ylabel(r'MAE$(\Sigma_{\mathrm{COHSEX}})$ vs BGW Sig$^\prime$  (eV)',
                  fontsize=13)
    ax.set_title('LORRAX vs BGW COHSEX: MAE vs centroid count at increasing n_cond\n'
                 '(Si 2×2×2 parabands, val_cond centroids, use_bgw_vcoul=true, 16 bands × 8 k)',
                 fontsize=11, fontweight='bold')
    ax.axhline(1e-1, color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(1e-2, color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(3e-2, color='r', lw=0.7, ls='--', alpha=0.6,
               label='30 meV (target)')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"wrote {out_path}")
    plt.close(fig)


def plot_Nmu_per_band(records, out_path, target_mae_eV=0.03):
    """For each n_cond, find N_μ needed to hit target MAE; plot N_μ/n_cond vs n_cond.

    User's hypothesis: at high n_cond, N_μ per band (= N_μ / n_cond) should
    drop as the pair-product effective rank saturates.
    """
    from collections import defaultdict
    import scipy.interpolate as si
    by_group = defaultdict(list)
    for r in records:
        by_group[r['ncond']].append((r['Nmu'], r['mae_prime']))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: N_μ needed for target MAE (log interp in (Nmu, log_mae)).
    ncond_vals = sorted(by_group.keys())
    Nmu_needed = []
    nc_plotted = []
    for nc in ncond_vals:
        pts = sorted(by_group[nc])
        Ns = np.array([p[0] for p in pts], dtype=float)
        maes = np.array([p[1] for p in pts], dtype=float)
        if maes.min() > target_mae_eV:
            print(f"[warn] n_cond={nc}: min MAE={maes.min():.3f} > target={target_mae_eV}, "
                  f"extrapolating linearly in log-log")
            # Linear fit in log(Nmu) vs log(mae), extrapolate
            lnNs, lnMAEs = np.log(Ns), np.log(maes)
            slope, intercept = np.polyfit(lnNs, lnMAEs, 1)
            # log(target) = slope * log(Nmu_needed) + intercept
            lnNmu_need = (np.log(target_mae_eV) - intercept) / slope
            Nmu_need = float(np.exp(lnNmu_need))
        else:
            # Monotonic interp on log(mae) vs log(Nmu). Reverse so MAE is increasing -> use lin interp on log.
            lnNs = np.log(Ns)
            lnMAEs = np.log(maes)
            # interp: given lnMAEs (may not be monotone), sort by lnMAEs.
            srt = np.argsort(lnMAEs)[::-1]   # decreasing MAE
            f = si.interp1d(lnMAEs[srt], lnNs[srt], kind='linear',
                            fill_value='extrapolate')
            Nmu_need = float(np.exp(f(np.log(target_mae_eV))))
        Nmu_needed.append(Nmu_need)
        nc_plotted.append(nc)

    ax = axes[0]
    ax.plot(nc_plotted, Nmu_needed, marker='o', lw=2.0, ms=10, color='#1f77b4')
    ax.set_xlabel(r'$n_{\mathrm{cond}}$', fontsize=12)
    ax.set_ylabel(rf'$N_\mu$ needed for  MAE $\leq$ {target_mae_eV*1000:.0f} meV',
                  fontsize=12)
    ax.set_title(f'Centroids needed vs n_cond', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: ratio = N_μ / n_cond.
    ratio = [n / nc for n, nc in zip(Nmu_needed, nc_plotted)]
    ax2 = axes[1]
    ax2.plot(nc_plotted, ratio, marker='s', lw=2.0, ms=10, color='#d62728')
    ax2.set_xlabel(r'$n_{\mathrm{cond}}$', fontsize=12)
    ax2.set_ylabel(r'$N_\mu$ / $n_{\mathrm{cond}}$  (centroids per band)',
                   fontsize=12)
    ax2.set_title('Centroids per band vs n_cond\n'
                  '(decreasing → pair-product rank saturates; paper hypothesis)',
                  fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', lw=0.3)

    for ax in axes:
        ax.tick_params(labelsize=11)

    fig.suptitle(f'N_μ scaling with n_cond at fixed target MAE = {target_mae_eV*1000:.0f} meV  '
                 '(Si 2×2×2 parabands, val_cond centroids)',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"wrote {out_path}")
    plt.close(fig)
    return dict(zip(nc_plotted, zip(Nmu_needed, ratio)))


def plot_scatter(records, out_path):
    """Per-band scatter: BGW Σ vs LORRAX Σ, split by (WFN, n_cond, N_μ)."""
    from collections import defaultdict
    by_wfn_nc = defaultdict(list)
    for r in records:
        by_wfn_nc[(r['wfn'], r['ncond'])].append(r)

    n_rows = 1   # parabands only
    n_cols = 4   # ncond = 100, 300, 600, 1000
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5),
                             sharex=False, sharey=False, squeeze=False)
    Nmu_colors = {400: '#1f77b4', 1200: '#ff7f0e', 3200: '#2ca02c', 6000: '#d62728'}
    for i, wfn in enumerate(['parabands']):
        for j, nc in enumerate([100, 300, 600, 1000]):
            ax = axes[i][j]
            runs = sorted(by_wfn_nc.get((wfn, nc), []), key=lambda r: r['Nmu'])
            if not runs:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center')
                continue
            for r in runs:
                diffs = r['diffs']
                bgw_vals = np.array([d['bgw_Sig_prime'] for d in diffs.values()])
                lorrax_vals = np.array([d['lorrax_sigTOT'] for d in diffs.values()])
                ax.scatter(bgw_vals, lorrax_vals, s=15, alpha=0.7,
                           color=Nmu_colors[r['Nmu']],
                           label=f"N_μ={r['Nmu']}  MAE={r['mae_prime']:.3f}eV")
            all_vals = np.concatenate(
                [np.array([d['bgw_Sig_prime']
                           for d in r['diffs'].values()])
                 for r in runs])
            lo, hi = all_vals.min(), all_vals.max()
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.5, alpha=0.5)
            ax.set_title(f'{wfn}  n_cond={nc}', fontsize=10)
            ax.set_xlabel('BGW Sig′  (eV)', fontsize=9)
            ax.set_ylabel('LORRAX sigTOT  (eV)', fontsize=9)
            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.2)
    fig.suptitle('Per-band COHSEX self-energy: LORRAX (val_cond centroids) vs '
                 'BGW (nbands = 8 + n_cond)',
                 fontsize=12, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"wrote {out_path}")
    plt.close(fig)


def main():
    bgw_cache, lorrax_runs = collect()
    print(f"[analyze] BGW references: {sorted(bgw_cache.keys())}")
    print(f"[analyze] LORRAX runs: {len(lorrax_runs)}")
    records = compute_errors(bgw_cache, lorrax_runs)
    print(f"[analyze] Computed errors for {len(records)} LORRAX runs")

    # Dump summary JSON + markdown table.
    summary = []
    for r in sorted(records, key=lambda x: (x['wfn'], x['ncond'], x['Nmu'])):
        summary.append({
            'wfn': r['wfn'], 'ncond': r['ncond'], 'Nmu': r['Nmu'],
            'mae_eV': r['mae_prime'],
            'rmse_eV': r['rmse_prime'],
            'max_abs_eV': r['max_prime'],
            'n_points': r['n_points'],
        })
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))

    # Markdown table
    md = ["# COHSEX LORRAX vs BGW — val_cond centroids",
          "",
          f"Same (WFN, n_cond) pair; BGW nbands = n_cond + 8. "
          f"LORRAX uses val_cond-selected N_μ centroids + use_bgw_vcoul=true.",
          "",
          "| WFN | n_cond | N_μ | MAE (eV) | RMSE (eV) | max abs (eV) | points |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        md.append(f"| {s['wfn']} | {s['ncond']} | {s['Nmu']} | "
                  f"{s['mae_eV']:.4f} | {s['rmse_eV']:.4f} | "
                  f"{s['max_abs_eV']:.4f} | {s['n_points']} |")
    (ROOT / 'summary.md').write_text('\n'.join(md) + '\n')
    print(f"[analyze] wrote summary.json + summary.md")

    # Plots
    if records:
        plot_mae_vs_Nmu(records, ROOT / 'mae_vs_Nmu.png')
        plot_scatter(records, ROOT / 'scatter_bgw_vs_lorrax.png')
        # Key plot for the paper's hypothesis — N_μ per band saturation.
        # 30 meV is below the noise floor (k-symmetry residuals etc) so the
        # extrapolation is noisy. Use 100 meV to measure the real N_μ
        # requirement; also emit 50 meV for comparison.
        for tag, target in [('100meV', 0.10), ('50meV', 0.05), ('30meV', 0.03)]:
            try:
                ratios = plot_Nmu_per_band(records,
                                           ROOT / f'Nmu_per_band_{tag}.png',
                                           target_mae_eV=target)
                print(f"[analyze] N_μ needed (for MAE ≤ {target*1000:.0f} meV):")
                for nc, (Nmu, ratio) in sorted(ratios.items()):
                    print(f"  n_cond={nc:5d}  N_μ≈{Nmu:7.0f}  "
                          f"N_μ/n_cond={ratio:6.2f}")
            except Exception as e:
                print(f'[warn] Nmu_per_band {tag} plot failed: {e}')


if __name__ == '__main__':
    main()
