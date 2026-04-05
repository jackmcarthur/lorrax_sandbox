#!/usr/bin/env python3
"""Compare COHSEX convergence: 640 vs 800 centroids vs BGW."""
import re
import numpy as np

def parse_sigma_hp_full(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {'X': float(p[3]), 'SX_X': float(p[4]), 'CHp': float(p[10]),
                        'Corp': float(p[4]) + float(p[10])}
    return bands

def parse_cohsex_eqp0(path, x_ref):
    """Parse COHSEX eqp0 and compute Cor = (sigSX - X) + sigCOH using external X."""
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
            bands[n1] = {'sigSX': sx, 'sigCOH': coh, 'Cor': (sx - x_ref[n1]) + coh}
    return bands

def parse_gn_sigx(path):
    """Get bare exchange from GN-PPM eqp0."""
    bands = {}
    for line in open(path):
        m = re.match(r'n=(\d+)\s+', line.strip())
        if not m: continue
        n1 = int(m.group(1)) + 1
        mx = re.search(r'sigX=\s*([-\d.Ee]+)', line)
        if mx:
            bands[n1] = float(mx.group(1))
    return bands

bgw = parse_sigma_hp_full('01_bgw_cohsex/sigma_hp.log')
gw_x = parse_gn_sigx('00_lorrax/eqp0.dat')
gw_640 = parse_cohsex_eqp0('01_lorrax_cohsex/eqp0.dat', gw_x)
gw_800 = parse_cohsex_eqp0('02_lorrax_cohsex_800c/eqp0.dat', gw_x)

common = sorted(set(bgw) & set(gw_640) & set(gw_800))

print(f"{'Band':>5} {'BGW Cor':>9} {'640c Cor':>9} {'800c Cor':>9} | {'Δ640':>8} {'Δ800':>8} | {'640→800':>8}")
print("-" * 80)
d640, d800 = [], []
for n in common:
    b = bgw[n]['Corp']
    g6 = gw_640[n]['Cor']
    g8 = gw_800[n]['Cor']
    delta6 = g6 - b
    delta8 = g8 - b
    d640.append(delta6)
    d800.append(delta8)
    print(f"{n:5d} {b:9.3f} {g6:9.3f} {g8:9.3f} | {delta6:+8.3f} {delta8:+8.3f} | {g8-g6:+8.3f}")

d640 = np.array(d640)
d800 = np.array(d800)
print(f"\n640 centroids: MAE = {np.mean(np.abs(d640)):.3f} eV, max|Δ| = {np.max(np.abs(d640)):.3f} eV")
print(f"800 centroids: MAE = {np.mean(np.abs(d800)):.3f} eV, max|Δ| = {np.max(np.abs(d800)):.3f} eV")
print(f"\nImprovement:   ΔMAE = {np.mean(np.abs(d640)) - np.mean(np.abs(d800)):+.3f} eV")
