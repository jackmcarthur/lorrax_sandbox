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
    'xtick.minor.visible': False, 'ytick.minor.visible': False,
})

SIRUN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic'

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
            blocks[ik]['bands'][int(p[0])] = {
                'Corp': float(p[4]) + float(p[10]),
                'Sigp': float(p[11]),
                'X': float(p[3]),
            }
    return blocks

def parse_lrx_sigc(path):
    result = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k-point') or s.startswith('k ') or not s: continue
        p = s.split()
        if len(p) >= 13:
            try:
                k = int(p[0]); n_phys = int(p[1]) + 1
                if k not in result: result[k] = {}
                sigc = float(p[12]) if p[12] != 'nan' else float('nan')
                result[k][n_phys] = sigc
            except: pass
    return result

bgw = parse_bgw(f'{SIRUN}/01_bgw_gnppm/sigma_hp.log')
lrx = parse_lrx_sigc(f'{SIRUN}/08_lorrax_gnppm/sigma_freq_debug.dat')

# Pick 6 k-points: 3 good, 3 bad
good_iks = [1, 2, 12]  # Gamma, (0,0,0.25), (0.5,0.5,0.25)
bad_iks = [4, 7, 9]    # (0,0.25,0.25), (0,0.5,0.5), (0.25,0.25,0.25)

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row')

for col, ik in enumerate(good_iks):
    ax = axes[0, col]
    ki = ik - 1
    kl = bgw[ik]['k']
    bands = sorted(bgw[ik]['bands'].keys())
    diffs = []
    blist = []
    for n in bands:
        if ki in lrx and n in lrx[ki] and not np.isnan(lrx[ki][n]):
            diffs.append(lrx[ki][n] - bgw[ik]['bands'][n]['Corp'])
            blist.append(n)
    ax.bar(range(len(diffs)), [d*1000 for d in diffs], color='C0', alpha=0.85)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Band')
    ax.set_xticks(range(0, len(blist), 4))
    ax.set_xticklabels([str(blist[i]) for i in range(0, len(blist), 4)])
    kstr = f'({kl[0]:.2f}, {kl[1]:.2f}, {kl[2]:.2f})'
    mae = np.mean(np.abs(diffs))*1000 if diffs else 0
    ax.set_title(f'ik={ik} {kstr}\nMAE={mae:.0f} meV')
    if col == 0:
        ax.set_ylabel('LRX - BGW Corp (meV)')

for col, ik in enumerate(bad_iks):
    ax = axes[1, col]
    ki = ik - 1
    kl = bgw[ik]['k']
    bands = sorted(bgw[ik]['bands'].keys())
    diffs = []
    blist = []
    for n in bands:
        if ki in lrx and n in lrx[ki] and not np.isnan(lrx[ki][n]):
            diffs.append(lrx[ki][n] - bgw[ik]['bands'][n]['Corp'])
            blist.append(n)
    ax.bar(range(len(diffs)), [d*1000 for d in diffs], color='C3', alpha=0.85)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Band')
    ax.set_xticks(range(0, len(blist), 4))
    ax.set_xticklabels([str(blist[i]) for i in range(0, len(blist), 4)])
    kstr = f'({kl[0]:.2f}, {kl[1]:.2f}, {kl[2]:.2f})'
    mae = np.mean(np.abs(diffs))*1000 if diffs else 0
    ax.set_title(f'ik={ik} {kstr}\nMAE={mae:.0f} meV')
    if col == 0:
        ax.set_ylabel('LRX - BGW Corp (meV)')

axes[0, 0].annotate('Good k-points', xy=(0.02, 0.98), xycoords='axes fraction',
    fontsize=11, fontweight='bold', va='top', color='C0')
axes[1, 0].annotate('Bad k-points', xy=(0.02, 0.98), xycoords='axes fraction',
    fontsize=11, fontweight='bold', va='top', color='C3')

fig.suptitle('Si 4x4x4 GN-PPM: BGW Corp vs LORRAX sigC(Edft)', fontsize=14)
plt.tight_layout()
plt.savefig('si_gnppm_per_kpoint.png', bbox_inches='tight')
print('Saved si_gnppm_per_kpoint.png')

# Also: per-k MAE summary bar chart
fig2, ax2 = plt.subplots(figsize=(8, 3.5))
iks = sorted(bgw.keys())
maes = []
klabels = []
colors = []
for ik in iks:
    ki = ik - 1
    kl = bgw[ik]['k']
    diffs = []
    for n in sorted(bgw[ik]['bands'].keys()):
        if ki in lrx and n in lrx[ki] and not np.isnan(lrx[ki][n]):
            diffs.append(abs(lrx[ki][n] - bgw[ik]['bands'][n]['Corp']))
    mae = np.mean(diffs)*1000 if diffs else 0
    maes.append(mae)
    klabels.append(f'({kl[0]:.1f},{kl[1]:.1f},{kl[2]:.1f})')
    colors.append('C0' if mae < 50 else 'C1' if mae < 200 else 'C3')

ax2.bar(range(len(iks)), maes, color=colors, alpha=0.85)
ax2.set_xticks(range(len(iks)))
ax2.set_xticklabels(klabels, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('MAE (meV)')
ax2.set_title('Si GN-PPM Corp MAE per k-point')
ax2.axhline(50, color='k', lw=0.5, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('si_gnppm_mae_per_k.png', bbox_inches='tight')
print('Saved si_gnppm_mae_per_k.png')
