#!/usr/bin/env python3
"""Compare GWJAX COHSEX against BGW primed vs unprimed Cor."""
import re
import numpy as np

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {
                'X': float(p[3]),
                'SX_X': float(p[4]),
                'CH': float(p[5]),     # unprimed
                'CHp': float(p[10]),   # primed
                'Cor': float(p[4]) + float(p[5]),   # unprimed
                'Corp': float(p[4]) + float(p[10]), # primed
            }
    return bands

def parse_cohsex_eqp0(path, x_ref):
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        ms = re.search(r'sigSX=\s*([-\d.Ee]+)', line)
        mc = re.search(r'sigCOH=\s*([-\d.Ee]+)', line)
        if ms and mc and n1 in x_ref:
            sx = float(ms.group(1))
            coh = float(mc.group(1))
            bands[n1] = {'sigSX': sx, 'sigCOH': coh, 'SX_X': sx - x_ref[n1], 'Cor': (sx - x_ref[n1]) + coh}
    return bands

def parse_gn_sigx(path):
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        mx = re.search(r'sigX=\s*([-\d.Ee]+)', line)
        if mx: bands[n1] = float(mx.group(1))
    return bands

bgw = parse_sigma_hp('01_bgw_cohsex/sigma_hp.log')
gw_x = parse_gn_sigx('00_lorrax/eqp0.dat')
gw = parse_cohsex_eqp0('01_lorrax_cohsex/eqp0.dat', gw_x)

common = sorted(set(bgw) & set(gw))

print("CH' vs CH (BGW primed vs unprimed):")
print(f"{'Band':>5} {'CH':>10} {'CHp':>10} {'ΔCH':>10}")
for n in common:
    print(f"{n:5d} {bgw[n]['CH']:10.3f} {bgw[n]['CHp']:10.3f} {bgw[n]['CHp'] - bgw[n]['CH']:+10.3f}")

print(f"\n\nCor comparison: GWJAX 640c vs BGW primed (Corp) and unprimed (Cor):")
print(f"{'Band':>5} {'GW Cor':>10} {'BGW Corp':>10} {'Δ_primed':>10} | {'BGW Cor':>10} {'Δ_unprimed':>10}")
print("-" * 70)
d_primed, d_unprimed = [], []
for n in common:
    dp = gw[n]['Cor'] - bgw[n]['Corp']
    du = gw[n]['Cor'] - bgw[n]['Cor']
    d_primed.append(dp)
    d_unprimed.append(du)
    print(f"{n:5d} {gw[n]['Cor']:10.3f} {bgw[n]['Corp']:10.3f} {dp:+10.3f}"
          f" | {bgw[n]['Cor']:10.3f} {du:+10.3f}")

print(f"\nvs primed   (Corp): MAE = {np.mean(np.abs(d_primed)):.3f} eV, max|Δ| = {np.max(np.abs(d_primed)):.3f} eV")
print(f"vs unprimed (Cor):  MAE = {np.mean(np.abs(d_unprimed)):.3f} eV, max|Δ| = {np.max(np.abs(d_unprimed)):.3f} eV")

# Also show component breakdown for SX-X and COH vs CH/CHp
print(f"\n\nComponent breakdown:")
print(f"{'Band':>5} {'BGW SX-X':>9} {'GW SX-X':>9} {'dSXX':>8} | {'BGW CH':>9} {'BGW CHp':>9} {'GW COH':>9} {'dCH':>8} {'dCHp':>8}")
print("-" * 95)
for n in common:
    dsxx = gw[n]['SX_X'] - bgw[n]['SX_X']
    dch = gw[n]['sigCOH'] - bgw[n]['CH']
    dchp = gw[n]['sigCOH'] - bgw[n]['CHp']
    print(f"{n:5d} {bgw[n]['SX_X']:9.3f} {gw[n]['SX_X']:9.3f} {dsxx:+8.3f}"
          f" | {bgw[n]['CH']:9.3f} {bgw[n]['CHp']:9.3f} {gw[n]['sigCOH']:9.3f} {dch:+8.3f} {dchp:+8.3f}")
