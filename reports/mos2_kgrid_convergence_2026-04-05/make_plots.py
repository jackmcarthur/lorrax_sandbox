import re, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'lines.linewidth': 1.8, 'lines.markersize': 6,
    'axes.grid': False, 'figure.dpi': 150,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

BASE3 = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex'
BASE4 = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm'

def parse_bgw(path):
    blocks = {}; ik = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            ik = int(m.group(4))
            blocks[ik] = {'k': (float(m.group(1)),float(m.group(2)),float(m.group(3))), 'bands': {}}
            continue
        if ik is None: continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            blocks[ik]['bands'][int(p[0])] = {'Sigp': float(p[11]), 'Corp': float(p[4])+float(p[10])}
    return blocks

def parse_lrx_tot(path):
    r = {}; k = -1
    for line in open(path):
        if 'k-point' in line: k += 1; r[k] = {}
        if 'n=' in line and 'sigSX=' in line:
            parts = line.strip().replace('sigSX=','|').replace('sigCOH=','|').replace('sigTOT=','|').replace('VH=','|').replace('n=','|').split('|')
            r[k][int(parts[1].strip())+1] = float(parts[4].strip())
    return r

def parse_lrx_sigc(path):
    result = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k-point') or s.startswith('k ') or not s: continue
        p = s.split()
        if len(p) >= 13:
            try:
                k = int(p[0]); n = int(p[1]) + 1
                if k not in result: result[k] = {}
                result[k][n] = float(p[12]) if p[12] != 'nan' else float('nan')
            except: pass
    return result

# ============================================================
# Fig 1: COHSEX Sig' residuals, 3x3 vs 4x4, first 3 k-points each
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row')

for row, (label, base) in enumerate([('3x3', BASE3), ('4x4', BASE4)]):
    bgw = parse_bgw(f'{base}/00_bgw_cohsex/sigma_hp.log')
    lrx = parse_lrx_tot(f'{base}/00_lorrax_cohsex/eqp0.dat')
    iks = sorted(bgw.keys())[:3]
    for col, ik in enumerate(iks):
        ax = axes[row, col]
        ki = ik - 1
        kl = bgw[ik]['k']
        bands = sorted(bgw[ik]['bands'].keys())
        nb = min(len(bands), len(lrx.get(ki, {})))
        if nb == 0: continue
        diffs = [(lrx[ki][bands[i]] - bgw[ik]['bands'][bands[i]]['Sigp'])*1000 for i in range(nb)]
        ax.bar(range(nb), diffs, color='C0' if row==0 else 'C2', alpha=0.85)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Band')
        ax.set_xticks(range(0, nb, 4))
        ax.set_xticklabels([str(bands[i]) for i in range(0, nb, 4)])
        kstr = f'({kl[0]:.2f}, {kl[1]:.2f})'
        mae = np.mean(np.abs(diffs))
        ax.set_title(f'{label} ik={ik} k={kstr}\nMAE={mae:.0f} meV')
        if col == 0:
            ax.set_ylabel('LRX - BGW (meV)')

fig.suptitle("MoS2 COHSEX Sig' residuals: 3x3 vs 4x4", fontsize=14)
plt.tight_layout()
plt.savefig('mos2_cohsex_3x3_vs_4x4.png', bbox_inches='tight')
print('Saved mos2_cohsex_3x3_vs_4x4.png')

# ============================================================
# Fig 2: GN-PPM Corp residuals, 3x3 vs 4x4
# ============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 6), sharey='row')

for row, (label, base, ldir) in enumerate([
    ('3x3', BASE3, '01_lorrax_gnppm_newcode'),
    ('4x4', BASE4, '01_lorrax_gnppm'),
]):
    bgw = parse_bgw(f'{base}/01_bgw_gnppm/sigma_hp.log')
    lrx = parse_lrx_sigc(f'{base}/{ldir}/sigma_freq_debug.dat')
    iks = sorted(bgw.keys())[:3]
    for col, ik in enumerate(iks):
        ax = axes2[row, col]
        ki = ik - 1
        kl = bgw[ik]['k']
        bands = sorted(bgw[ik]['bands'].keys())
        diffs = []
        blist = []
        for n in bands:
            if ki in lrx and n in lrx[ki] and not np.isnan(lrx[ki][n]):
                diffs.append(lrx[ki][n] - bgw[ik]['bands'][n]['Corp'])
                blist.append(n)
        if not diffs: continue
        ax.bar(range(len(diffs)), diffs, color='C1' if row==0 else 'C3', alpha=0.85)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Band')
        ax.set_xticks(range(0, len(blist), 4))
        ax.set_xticklabels([str(blist[i]) for i in range(0, len(blist), 4)])
        kstr = f'({kl[0]:.2f}, {kl[1]:.2f})'
        mae = np.mean(np.abs(diffs))*1000
        ax.set_title(f'{label} ik={ik} k={kstr}\nMAE={mae:.0f} meV')
        if col == 0:
            ax.set_ylabel('LRX - BGW Corp (eV)')

fig2.suptitle('MoS2 GN-PPM Corp residuals: 3x3 vs 4x4', fontsize=14)
plt.tight_layout()
plt.savefig('mos2_gnppm_3x3_vs_4x4.png', bbox_inches='tight')
print('Saved mos2_gnppm_3x3_vs_4x4.png')
