import re, numpy as np, os
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

SIRUN_ORIG = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band'
SIRUN_SYM = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic'
MOS2RUN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex'

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
            blocks[ik]['bands'][int(p[0])] = {'X': float(p[3]), 'Sigp': float(p[11]), 'Eo': float(p[2])}
    return blocks

def parse_lrx(path):
    r = {}; k = -1
    for line in open(path):
        if 'k-point' in line: k += 1; r[k] = {}
        if 'n=' in line and 'sigSX=' in line:
            parts = line.strip().replace('sigSX=','|').replace('sigCOH=','|').replace('sigTOT=','|').replace('VH=','|').replace('n=','|').split('|')
            n0 = int(parts[1].strip())
            r[k][n0+1] = {'sigSX': float(parts[2].strip()), 'sigTOT': float(parts[4].strip())}
    return r

# ============================================================
# Fig 1: Si force_symmorphic COHSEX residuals, 3 k-points
# ============================================================
bgw_sym = parse_bgw(f'{SIRUN_SYM}/00_bgw_cohsex/sigma_hp.log')
lrx_480 = parse_lrx(f'{SIRUN_SYM}/00_lorrax_cohsex/eqp0.dat')
lrx_960 = parse_lrx(f'{SIRUN_SYM}/06_lorrax_cohsex_960c/eqp0.dat')
lrx_bh = parse_lrx(f'{SIRUN_SYM}/05_lorrax_cohsex_bgwhead/eqp0.dat')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for pi, ik in enumerate([1, 2, 3]):
    ax = axes[pi]; ki = ik - 1
    kl = bgw_sym[ik]['k']
    bands = sorted(bgw_sym[ik]['bands'].keys())
    nb = min(len(bands), len(lrx_480.get(ki, {})))
    if nb == 0: continue
    d480 = [lrx_480[ki][bands[i]]['sigTOT'] - bgw_sym[ik]['bands'][bands[i]]['Sigp'] for i in range(nb)]
    d960 = [lrx_960[ki][bands[i]]['sigTOT'] - bgw_sym[ik]['bands'][bands[i]]['Sigp'] for i in range(nb)]
    dbh  = [lrx_bh[ki][bands[i]]['sigTOT']  - bgw_sym[ik]['bands'][bands[i]]['Sigp'] for i in range(nb)]
    x = np.arange(nb)
    ax.bar(x-0.22, d480, 0.2, label='480c S-head', color='C0', alpha=0.85)
    ax.bar(x, d960, 0.2, label='960c S-head', color='C1', alpha=0.85)
    ax.bar(x+0.22, dbh, 0.2, label='480c BGW head', color='C2', alpha=0.85)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Band index')
    ax.set_xticks(range(0, nb, 4))
    ax.set_xticklabels([str(bands[i]) for i in range(0, nb, 4)])
    ax.set_title(f'k=({kl[0]:.2f}, {kl[1]:.2f}, {kl[2]:.2f})')
    if pi == 0:
        ax.set_ylabel('LRX - BGW (eV)')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
fig.suptitle('Si 4x4x4 COHSEX (symmorphic only)', fontsize=14)
plt.tight_layout()
plt.savefig('si_symmorphic_cohsex_residuals.png', bbox_inches='tight')
print('Saved si_symmorphic_cohsex_residuals.png')

# ============================================================
# Fig 2: Bare exchange — original vs force_symmorphic
# ============================================================
bgw_orig = parse_bgw(f'{SIRUN_ORIG}/00_bgw_cohsex/sigma_hp.log')
lrx_orig_x = parse_lrx(f'{SIRUN_ORIG}/04_lorrax_xonly/eqp0.dat')
lrx_sym_x = parse_lrx(f'{SIRUN_SYM}/04_lorrax_xonly/eqp0.dat')

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 3.5))
ax = axes2[0]
bands_g = sorted(bgw_orig[1]['bands'].keys())[:16]
bgw_xo = [bgw_orig[1]['bands'][n]['X'] for n in bands_g]
lrx_xo = [lrx_orig_x[0][n]['sigSX'] for n in bands_g]
bands_s = sorted(bgw_sym[1]['bands'].keys())[:16]
bgw_xs = [bgw_sym[1]['bands'][n]['X'] for n in bands_s]
lrx_xs = [lrx_sym_x[0][n]['sigSX'] for n in bands_s]

ax.plot(bgw_xo, lrx_xo, 'o', color='C3', label='Original (Fd-3m)', alpha=0.8)
ax.plot(bgw_xs, lrx_xs, 's', color='C0', label='Symmorphic only', alpha=0.8)
ax.plot([-20, -3], [-20, -3], 'k-', lw=0.6)
ax.set_xlim(-20, -3); ax.set_ylim(-20, -3)
ax.set_xlabel('BGW X (eV)'); ax.set_ylabel('LORRAX X (eV)')
ax.legend(fontsize=10, frameon=False)
ax.set_title('Bare exchange at Gamma')

ax = axes2[1]
do = [lrx_xo[i]-bgw_xo[i] for i in range(len(bands_g))]
ds = [lrx_xs[i]-bgw_xs[i] for i in range(len(bands_s))]
x = np.arange(len(bands_g))
ax.bar(x-0.15, do, 0.28, label='Original', color='C3', alpha=0.8)
ax.bar(x+0.15, ds, 0.28, label='Symmorphic', color='C0', alpha=0.8)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('Band index'); ax.set_ylabel('LRX - BGW (eV)')
ax.legend(fontsize=10, frameon=False)
ax.set_title('Exchange residuals at Gamma')
ax.set_xticks(range(0, len(bands_g), 4))
fig2.suptitle('Non-symmorphic symmetry effect on bare exchange', fontsize=14)
plt.tight_layout()
plt.savefig('si_xonly_nonsymmorphic.png', bbox_inches='tight')
print('Saved si_xonly_nonsymmorphic.png')

# ============================================================
# Fig 3: MoS2 3x3 COHSEX residuals, 3 k-points
# ============================================================
bgw_m = parse_bgw(f'{MOS2RUN}/00_bgw_cohsex/sigma_hp.log')
lrx_m = parse_lrx(f'{MOS2RUN}/00_lorrax_cohsex/eqp0.dat')

fig3, axes3 = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for pi, ik in enumerate(sorted(bgw_m.keys())[:3]):
    ax = axes3[pi]; ki = ik - 1
    kl = bgw_m[ik]['k']
    bands = sorted(bgw_m[ik]['bands'].keys())
    nb = min(len(bands), len(lrx_m.get(ki, {})))
    if nb == 0: continue
    d = [lrx_m[ki][bands[i]]['sigTOT'] - bgw_m[ik]['bands'][bands[i]]['Sigp'] for i in range(nb)]
    ax.bar(range(nb), d, color='C4', alpha=0.85)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Band index')
    ax.set_xticks(range(0, nb, 4))
    ax.set_xticklabels([str(bands[i]) for i in range(0, nb, 4)])
    ax.set_title(f'k=({kl[0]:.2f}, {kl[1]:.2f}, {kl[2]:.2f})')
    if pi == 0: ax.set_ylabel('LRX - BGW (eV)')
fig3.suptitle('MoS2 3x3x1 COHSEX: 67 meV MAE', fontsize=14)
plt.tight_layout()
plt.savefig('mos2_cohsex_residuals.png', bbox_inches='tight')
print('Saved mos2_cohsex_residuals.png')

# Summary
print('\n=== Summary ===')
all_sym = []
for ik in sorted(bgw_sym.keys()):
    ki = ik-1
    if ki not in lrx_480: continue
    for n in sorted(bgw_sym[ik]['bands'].keys()):
        if n not in lrx_480[ki]: continue
        all_sym.append(abs(lrx_480[ki][n]['sigTOT'] - bgw_sym[ik]['bands'][n]['Sigp']))
print(f'Si symmorphic 480c: all-k MAE={np.mean(all_sym)*1000:.1f} meV, max={np.max(all_sym)*1000:.1f} meV, N={len(all_sym)}')

all_m = []
for ik in sorted(bgw_m.keys()):
    ki = ik-1
    if ki not in lrx_m: continue
    for n in sorted(bgw_m[ik]['bands'].keys()):
        if n not in lrx_m[ki]: continue
        all_m.append(abs(lrx_m[ki][n]['sigTOT'] - bgw_m[ik]['bands'][n]['Sigp']))
print(f'MoS2 3x3: all-k MAE={np.mean(all_m)*1000:.1f} meV, max={np.max(all_m)*1000:.1f} meV, N={len(all_m)}')
