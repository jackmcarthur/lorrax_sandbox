#!/usr/bin/env python3
"""Compare BGW vs GWJAX for both GN-GPP and static COHSEX variants."""
import re
import numpy as np

def parse_sigma_hp(path):
    """Parse BGW sigma_hp.log -> dict of {band: {'Corp': float, 'X': float, 'Sigp': float}}."""
    bands = {}
    for line in open(path):
        s = line.strip()
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            X = float(p[3])
            sx_x = float(p[4])
            ch_prime = float(p[10])
            sigp = float(p[11])
            bands[n] = {'Corp': sx_x + ch_prime, 'X': X, 'Sigp': sigp}
    return bands

def parse_eqp0_gn(path):
    """Parse GWJAX GN-PPM eqp0.dat -> dict of {band_1idx: {'sigC': float, 'sigX': float}}."""
    bands = {}
    for line in open(path):
        s = line.strip()
        m = re.match(r'n=(\d+)\s+', s)
        if not m:
            continue
        n1 = int(m.group(1)) + 1  # 1-indexed
        mx = re.search(r'sigX=\s*([-\d.Ee]+)', s)
        mc = re.search(r'sigC_EDFT=\s*([-\d.Ee]+)', s)
        if mx and mc:
            sigx = float(mx.group(1))
            sigc = float(mc.group(1))
            if not np.isnan(sigc):
                bands[n1] = {'sigC': sigc, 'sigX': sigx}
    return bands

def parse_eqp0_cohsex(path):
    """Parse GWJAX COHSEX eqp0.dat -> dict of {band_1idx: {'sigSX': float, 'sigCOH': float, 'sigTOT': float}}."""
    bands = {}
    for line in open(path):
        s = line.strip()
        m = re.match(r'n=(\d+)\s+', s)
        if not m:
            continue
        n1 = int(m.group(1)) + 1
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', s)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', s)
        mt = re.search(r'sigTOT=\s*([-\d.Ee]+)', s)
        if ms and mc and mt:
            bands[n1] = {
                'sigSX': float(ms.group(1)),
                'sigCOH': float(mc.group(1)),
                'sigTOT': float(mt.group(1)),
            }
    return bands

# -- GN-GPP comparison --
print("=" * 80)
print("GN-GPP: BGW (freq_dep 3) vs GWJAX (use_ppm_sigma = true)")
print("  BGW: Cor' = SX-X + CH'")
print("  GWJAX: sigC_EDFT (correlation at DFT energies)")
print("=" * 80)
bgw_gn = parse_sigma_hp('00_bgw/sigma_hp.log')
gw_gn = parse_eqp0_gn('00_lorrax/eqp0.dat')

common = sorted(set(bgw_gn) & set(gw_gn))
print(f"\n{'Band':>5} {'BGW X':>10} {'GW X':>10} {'dX':>8} | {'BGW Cor':>10} {'GW sigC':>10} {'dCor':>8}")
print("-" * 75)
diffs_x, diffs_c = [], []
for n in common:
    dx = gw_gn[n]['sigX'] - bgw_gn[n]['X']
    dc = gw_gn[n]['sigC'] - bgw_gn[n]['Corp']
    diffs_x.append(dx)
    diffs_c.append(dc)
    print(f"{n:5d} {bgw_gn[n]['X']:10.3f} {gw_gn[n]['sigX']:10.3f} {dx:+8.3f}"
          f" | {bgw_gn[n]['Corp']:10.3f} {gw_gn[n]['sigC']:10.3f} {dc:+8.3f}")
diffs_x = np.array(diffs_x)
diffs_c = np.array(diffs_c)
print(f"\nExchange: MAE = {np.mean(np.abs(diffs_x)):.3f} eV")
print(f"Correlation: MAE = {np.mean(np.abs(diffs_c)):.3f} eV, max|Δ| = {np.max(np.abs(diffs_c)):.3f} eV")

# -- COHSEX comparison --
print("\n" + "=" * 80)
print("Static COHSEX: BGW (freq_dep 0) vs GWJAX (use_ppm_sigma = false)")
print("  BGW: SX-X, CH', Cor' = SX-X + CH'")
print("  GWJAX: sigSX-X, sigCOH, (sigSX-X)+sigCOH")
print("  (X from GN-PPM GWJAX run, same wavefunctions)")
print("=" * 80)
bgw_coh = parse_sigma_hp('01_bgw_cohsex/sigma_hp.log')
gw_coh = parse_eqp0_cohsex('01_lorrax_cohsex/eqp0.dat')

# Get bare exchange from the GN-PPM GWJAX run
gw_x = {n: gw_gn[n]['sigX'] for n in gw_gn}

common_c = sorted(set(bgw_coh) & set(gw_coh) & set(gw_x))
print(f"\n{'Band':>5} {'BGW SX-X':>10} {'GW SX-X':>10} {'dSXX':>8} | {'BGW CH':>10} {'GW COH':>10} {'dCH':>8} | {'BGW Cor':>10} {'GW Cor':>10} {'dCor':>8}")
print("-" * 110)
diffs_sxx, diffs_ch, diffs_cor = [], [], []
for n in common_c:
    bgw_sxx = bgw_coh[n]['Corp'] - (bgw_coh[n]['Sigp'] - bgw_coh[n]['X'] - bgw_coh[n]['Corp'])
    # Simpler: BGW has SX-X and CH' directly in the raw output. Let me re-parse.
    pass

# Actually, let me re-read sigma_hp to get SX-X and CH' separately
def parse_sigma_hp_full(path):
    bands = {}
    for line in open(path):
        s = line.strip()
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {
                'X': float(p[3]),
                'SX_X': float(p[4]),
                'CH': float(p[5]),
                'CHp': float(p[10]),
                'Corp': float(p[4]) + float(p[10]),
            }
    return bands

bgw_coh_full = parse_sigma_hp_full('01_bgw_cohsex/sigma_hp.log')

print(f"\n{'Band':>5} {'BGW SX-X':>10} {'GW SX-X':>10} {'dSXX':>8} | {'BGW CH':>10} {'GW COH':>10} {'dCH':>8} | {'BGW Cor':>10} {'GW Cor':>10} {'dCor':>8}")
print("-" * 110)
diffs_cor = []
for n in common_c:
    x = gw_x[n]  # bare exchange from GWJAX
    gw_sxx = gw_coh[n]['sigSX'] - x
    gw_cor = gw_sxx + gw_coh[n]['sigCOH']

    bgw_sxx = bgw_coh_full[n]['SX_X']
    bgw_chp = bgw_coh_full[n]['CHp']
    bgw_cor = bgw_coh_full[n]['Corp']

    d_sxx = gw_sxx - bgw_sxx
    d_ch = gw_coh[n]['sigCOH'] - bgw_chp
    d_cor = gw_cor - bgw_cor
    diffs_cor.append(d_cor)
    print(f"{n:5d} {bgw_sxx:10.3f} {gw_sxx:10.3f} {d_sxx:+8.3f}"
          f" | {bgw_chp:10.3f} {gw_coh[n]['sigCOH']:10.3f} {d_ch:+8.3f}"
          f" | {bgw_cor:10.3f} {gw_cor:10.3f} {d_cor:+8.3f}")
diffs_cor = np.array(diffs_cor)
if len(diffs_cor) > 0:
    print(f"\nCOHSEX Cor': MAE = {np.mean(np.abs(diffs_cor)):.3f} eV, max|Δ| = {np.max(np.abs(diffs_cor)):.3f} eV")
else:
    print("\nNo matching bands found for COHSEX comparison.")
