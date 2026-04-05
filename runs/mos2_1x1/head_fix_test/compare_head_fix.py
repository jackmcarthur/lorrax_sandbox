#!/usr/bin/env python3
"""Compare GWJAX sigma_freq_debug (with/without head) vs BGW sigma_hp.log.

Produces a comparison table and a matplotlib bar chart.
Uses parsers from PARSE_OUTPUTS.md.
"""
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parsers (from PARSE_OUTPUTS.md) ──

def parse_sigma_hp(path):
    blocks = []
    ik = None
    kcrys = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            blocks[-1]['bands'][n] = {
                'Corp': float(p[4]) + float(p[10]),
            }
    return blocks


def parse_sigma_freq_debug(path):
    bands = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k') or not s:
            continue
        p = s.split()
        if len(p) >= 13:
            try:
                n_phys = int(p[1]) + 1
                sigc = float(p[12]) if p[12] != 'nan' else np.nan
                head = float(p[13]) if len(p) >= 14 and p[13] != 'nan' else 0.0
                bands[n_phys] = {'sigc_edft': sigc, 'sigc_head': head}
            except (ValueError, IndexError):
                pass
    return bands


# ── Main ──

bgw_path = sys.argv[1] if len(sys.argv) > 1 else '/home/jackm/projects/tests_isdf/mos2_debug_template_1x1/sigma_hp.log'
gw_path = sys.argv[2] if len(sys.argv) > 2 else 'gwjax/sigma_freq_debug.dat'

bgw = parse_sigma_hp(bgw_path)
gw = parse_sigma_freq_debug(gw_path)
bgw_bands = bgw[0]['bands']

bands_common = sorted(set(bgw_bands) & set(gw))
bgw_cor = np.array([bgw_bands[n]['Corp'] for n in bands_common])
gw_sigc = np.array([gw[n]['sigc_edft'] for n in bands_common])
gw_head = np.array([gw[n]['sigc_head'] for n in bands_common])

valid = ~np.isnan(gw_sigc)
n_v = np.array(bands_common)[valid]
bgw_v = bgw_cor[valid]
sigc_v = gw_sigc[valid]
head_v = gw_head[valid]

# "Body only" = sig_c(Edft) as written (head NOT in sig_c when apply_head_diagonal=false)
# "Body + head" = sig_c(Edft) + sig_c_head
body_v = sigc_v            # body only (head excluded from ISDF W)
total_v = sigc_v + head_v  # body + separate head correction

diff_body = body_v - bgw_v
diff_total = total_v - bgw_v

print(f"{'Band':>5} {'BGW_Cor':>10} {'body':>10} {'body+head':>10} {'head':>10} {'body-BGW':>10} {'(b+h)-BGW':>10}")
print("-" * 75)
for i, n in enumerate(n_v):
    print(f"{n:5d} {bgw_v[i]:10.3f} {body_v[i]:10.3f} {total_v[i]:10.3f} "
          f"{head_v[i]:10.3f} {diff_body[i]:+10.3f} {diff_total[i]:+10.3f}")

print(f"\nBody only (head removed from ISDF W):  MAE = {np.mean(np.abs(diff_body)):.3f} eV, max|Δ| = {np.max(np.abs(diff_body)):.3f} eV")
print(f"Body + separate head diagonal:         MAE = {np.mean(np.abs(diff_total)):.3f} eV, max|Δ| = {np.max(np.abs(diff_total)):.3f} eV")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

x = np.arange(len(n_v))
w = 0.35

# Panel 1: differences
axes[0].bar(x - w/2, diff_body, w, label='Body only − BGW', color='steelblue')
axes[0].bar(x + w/2, diff_total, w, label='Body+Head − BGW', color='coral')
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_xlabel('Band index')
axes[0].set_ylabel('Δ Cor (eV)')
axes[0].set_title('GWJAX − BGW  (difference)')
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(n) for n in n_v], fontsize=7, rotation=45)
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.3)

# Panel 2: absolute values body vs BGW
axes[1].bar(x - w/2, bgw_v, w, label='BGW Cor\'', color='gray')
axes[1].bar(x + w/2, body_v, w, label='GWJAX body (no head in W)', color='steelblue')
axes[1].set_xlabel('Band index')
axes[1].set_ylabel('Cor (eV)')
axes[1].set_title('Body only vs BGW')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(n) for n in n_v], fontsize=7, rotation=45)
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', alpha=0.3)

# Panel 3: absolute values body+head vs BGW
axes[2].bar(x - w/2, bgw_v, w, label='BGW Cor\'', color='gray')
axes[2].bar(x + w/2, total_v, w, label='GWJAX body + head', color='coral')
axes[2].set_xlabel('Band index')
axes[2].set_ylabel('Cor (eV)')
axes[2].set_title('Body + Head vs BGW')
axes[2].set_xticks(x)
axes[2].set_xticklabels([str(n) for n in n_v], fontsize=7, rotation=45)
axes[2].legend(fontsize=8)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('MoS₂ 1×1 (Γ-only): Head Correction Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('head_fix_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: head_fix_comparison.png")
