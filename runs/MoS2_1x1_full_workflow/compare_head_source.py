import re, numpy as np

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {'X': float(p[3]), 'SX_X': float(p[4]), 'CHp': float(p[10]),
                        'Corp': float(p[4]) + float(p[10])}
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
            bands[n1] = (float(ms.group(1)) - x_ref[n1]) + float(mc.group(1))
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

bgw = parse_sigma_hp('01_bgw_cohsex/sigma_hp.log')
gw_x = parse_sigx('00_lorrax/eqp0.dat')
runs = {
    'S-tensor (640c)': parse_cohsex('01_lorrax_cohsex/eqp0.dat', gw_x),
    '800 centroids':   parse_cohsex('02_lorrax_cohsex_800c/eqp0.dat', gw_x),
    'BGW head override': parse_cohsex('03_lorrax_cohsex_bgwhead/eqp0.dat', gw_x),
}

common = sorted(set(bgw) & set.intersection(*[set(r) for r in runs.values()]))
print(f"{'Band':>5} {'BGW Cor':>9}", end="")
for label in runs: print(f" {label:>18}", end="")
print()
print("-" * (14 + 19*len(runs)))

results = {l: [] for l in runs}
for n in common:
    b = bgw[n]['Corp']
    print(f"{n:5d} {b:9.3f}", end="")
    for label, data in runs.items():
        d = data[n] - b
        results[label].append(d)
        print(f" {d:+18.3f}", end="")
    print()

print()
for label, diffs in results.items():
    d = np.array(diffs)
    print(f"{label:25s}: MAE = {np.mean(np.abs(d)):.3f} eV, max|Δ| = {np.max(np.abs(d)):.3f} eV")
