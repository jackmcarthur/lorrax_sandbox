#!/usr/bin/env python3
"""Generate plots for the COHSEX head investigation report."""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNDIR = '/home/jackm/projects/lorrax_sandbox/runs/MoS2_1x1_full_workflow'

# ── Parsers ──

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {'X': float(p[3]), 'SX_X': float(p[4]),
                        'CH': float(p[5]), 'CHp': float(p[10]),
                        'Corp': float(p[4]) + float(p[10])}
    return bands

def parse_eqp0_sigc(path):
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        mc = re.search(r'sigC_EDFT=\s*([-\d.Ee]+)', line)
        if mc:
            v = float(mc.group(1))
            if not np.isnan(v): bands[n1] = v
    return bands

def parse_cohsex(path, x_ref):
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', line)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', line)
        if ms and mc and n1 in x_ref:
            bands[n1] = {'SX_X': float(ms.group(1)) - x_ref[n1],
                         'COH': float(mc.group(1)),
                         'Cor': (float(ms.group(1)) - x_ref[n1]) + float(mc.group(1))}
    return bands

def parse_sigx(path):
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        mx = re.search(r'sigX=\s*([-\d.Ee]+)', line)
        if mx: bands[n1] = float(mx.group(1))
    return bands

def parse_sigma_freq_debug(path):
    bands = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k') or not s: continue
        p = s.split()
        if len(p) >= 13:
            try:
                n1 = int(p[1]) + 1
                sigc = float(p[12]) if p[12] != 'nan' else np.nan
                head = float(p[13]) if len(p) >= 14 and p[13] != 'nan' else 0.0
                if not np.isnan(sigc):
                    bands[n1] = {'sigc': sigc, 'head': head}
            except (ValueError, IndexError):
                pass
    return bands

# ── Load data ──

bgw_gn = parse_sigma_hp(f'{RUNDIR}/00_bgw/sigma_hp.log')
bgw_coh = parse_sigma_hp(f'{RUNDIR}/01_bgw_cohsex/sigma_hp.log')
gw_x = parse_sigx(f'{RUNDIR}/00_lorrax/eqp0.dat')

gw_coh_stensor = parse_cohsex(f'{RUNDIR}/01_lorrax_cohsex/eqp0.dat', gw_x)
gw_coh_800c = parse_cohsex(f'{RUNDIR}/02_lorrax_cohsex_800c/eqp0.dat', gw_x)
gw_coh_bgwhead = parse_cohsex(f'{RUNDIR}/03_lorrax_cohsex_bgwhead/eqp0.dat', gw_x)

gw_gn_body = parse_sigma_freq_debug(f'{RUNDIR}/00_lorrax/sigma_freq_debug.dat')
gw_gn_applied = parse_eqp0_sigc(f'{RUNDIR}/06_lorrax_gn_bgwhead_applied/eqp0.dat')

# ── Common bands ──
coh_bands = sorted(set(bgw_coh) & set(gw_coh_stensor) & set(gw_coh_800c) & set(gw_coh_bgwhead))
gn_bands = sorted(set(bgw_gn) & set(gw_gn_body) & set(gw_gn_applied))

# ── Figure 1: COHSEX variants ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x = np.arange(len(coh_bands))
bgw_cor = np.array([bgw_coh[n]['Corp'] for n in coh_bands])

variants = [
    ('S-tensor (640c)', [gw_coh_stensor[n]['Cor'] for n in coh_bands], 'steelblue'),
    ('800 centroids', [gw_coh_800c[n]['Cor'] for n in coh_bands], 'coral'),
    ('BGW head override', [gw_coh_bgwhead[n]['Cor'] for n in coh_bands], 'seagreen'),
]

# Panel 1: absolute values
w = 0.2
axes[0].bar(x - 1.5*w, bgw_cor, w, label='BGW Cor\'', color='gray', alpha=0.8)
for i, (label, vals, color) in enumerate(variants):
    axes[0].bar(x + (i - 0.5)*w, vals, w, label=label, color=color, alpha=0.8)
axes[0].set_xlabel('Band index')
axes[0].set_ylabel('Cor\' (eV)')
axes[0].set_title('Static COHSEX: absolute values')
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(n) for n in coh_bands], fontsize=7, rotation=45)
axes[0].legend(fontsize=7)
axes[0].grid(axis='y', alpha=0.3)

# Panel 2: differences
for label, vals, color in variants:
    diff = np.array(vals) - bgw_cor
    axes[1].bar(x + variants.index((label, vals, color))*w - w, diff, w, label=label, color=color, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('Band index')
axes[1].set_ylabel('GWJAX - BGW (eV)')
axes[1].set_title('Static COHSEX: difference from BGW')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(n) for n in coh_bands], fontsize=7, rotation=45)
axes[1].legend(fontsize=7)
axes[1].grid(axis='y', alpha=0.3)

# Panel 3: component breakdown for BGW-head variant
bgw_sxx = np.array([bgw_coh[n]['SX_X'] for n in coh_bands])
bgw_chp = np.array([bgw_coh[n]['CHp'] for n in coh_bands])
gw_sxx = np.array([gw_coh_bgwhead[n]['SX_X'] for n in coh_bands])
gw_coh_vals = np.array([gw_coh_bgwhead[n]['COH'] for n in coh_bands])
axes[2].bar(x - w/2, gw_sxx - bgw_sxx, w, label='Δ(SX-X)', color='steelblue', alpha=0.8)
axes[2].bar(x + w/2, gw_coh_vals - bgw_chp, w, label='Δ(COH vs CH\')', color='coral', alpha=0.8)
axes[2].axhline(0, color='black', linewidth=0.5)
axes[2].set_xlabel('Band index')
axes[2].set_ylabel('Component Δ (eV)')
axes[2].set_title('BGW-head variant: component errors')
axes[2].set_xticks(x)
axes[2].set_xticklabels([str(n) for n in coh_bands], fontsize=7, rotation=45)
axes[2].legend(fontsize=7)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('MoS₂ 1×1 Γ-only: COHSEX Head Investigation', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('cohsex_comparison.png', dpi=150, bbox_inches='tight')
print('Saved cohsex_comparison.png')

# ── Figure 2: GN-PPM head effect ──
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

x2 = np.arange(len(gn_bands))
bgw_gn_cor = np.array([bgw_gn[n]['Corp'] for n in gn_bands])
gw_body = np.array([gw_gn_body[n]['sigc'] for n in gn_bands])
gw_body_head = np.array([gw_gn_body[n]['sigc'] + gw_gn_body[n]['head'] for n in gn_bands])
gw_applied = np.array([gw_gn_applied[n] for n in gn_bands])

w2 = 0.2
# Panel 1: absolute
axes2[0].bar(x2 - 1.5*w2, bgw_gn_cor, w2, label='BGW Cor\'', color='gray', alpha=0.8)
axes2[0].bar(x2 - 0.5*w2, gw_body, w2, label='GWJAX body (no head)', color='steelblue', alpha=0.8)
axes2[0].bar(x2 + 0.5*w2, gw_applied, w2, label='+ BGW head applied', color='coral', alpha=0.8)
axes2[0].set_xlabel('Band index')
axes2[0].set_ylabel('Cor\' (eV)')
axes2[0].set_title('GN-GPP: absolute values')
axes2[0].set_xticks(x2)
axes2[0].set_xticklabels([str(n) for n in gn_bands], fontsize=7, rotation=45)
axes2[0].legend(fontsize=7)
axes2[0].grid(axis='y', alpha=0.3)

# Panel 2: differences
axes2[1].bar(x2 - w2/2, gw_body - bgw_gn_cor, w2, label='Body only', color='steelblue', alpha=0.8)
axes2[1].bar(x2 + w2/2, gw_applied - bgw_gn_cor, w2, label='+ BGW head applied', color='coral', alpha=0.8)
axes2[1].axhline(0, color='black', linewidth=0.5)
axes2[1].set_xlabel('Band index')
axes2[1].set_ylabel('GWJAX - BGW (eV)')
axes2[1].set_title('GN-GPP: head correction makes it worse')
axes2[1].set_xticks(x2)
axes2[1].set_xticklabels([str(n) for n in gn_bands], fontsize=7, rotation=45)
axes2[1].legend(fontsize=7)
axes2[1].grid(axis='y', alpha=0.3)

plt.suptitle('MoS₂ 1×1 Γ-only: GN-GPP Head Correction Effect', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('gn_gpp_head_effect.png', dpi=150, bbox_inches='tight')
print('Saved gn_gpp_head_effect.png')

# ── Figure 3: MAE summary bar chart ──
fig3, ax3 = plt.subplots(figsize=(8, 5))

labels = [
    'COHSEX\nS-tensor',
    'COHSEX\n800c',
    'COHSEX\nBGW head',
    'GN-PPM\nbody only',
    'GN-PPM\n+ BGW head',
]
maes = [
    np.mean(np.abs(np.array([gw_coh_stensor[n]['Cor'] for n in coh_bands]) - bgw_cor)),
    np.mean(np.abs(np.array([gw_coh_800c[n]['Cor'] for n in coh_bands]) - bgw_cor)),
    np.mean(np.abs(np.array([gw_coh_bgwhead[n]['Cor'] for n in coh_bands]) - bgw_cor)),
    np.mean(np.abs(gw_body - bgw_gn_cor)),
    np.mean(np.abs(gw_applied - bgw_gn_cor)),
]
colors = ['steelblue', 'coral', 'seagreen', 'steelblue', 'coral']
hatches = ['', '', '', '///', '///']

bars = ax3.bar(range(len(labels)), maes, color=colors, alpha=0.8)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel('MAE vs BGW (eV)')
ax3.set_title('MoS₂ 1×1: Summary of all variants')
ax3.axhline(0.050, color='green', linestyle='--', linewidth=1, label='50 meV target')
ax3.axhline(0.100, color='orange', linestyle='--', linewidth=1, label='100 meV target')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

for i, v in enumerate(maes):
    ax3.text(i, v + 0.05, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('mae_summary.png', dpi=150, bbox_inches='tight')
print('Saved mae_summary.png')
