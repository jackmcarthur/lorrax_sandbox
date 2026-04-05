import re, numpy as np

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {'Corp': float(p[4]) + float(p[10]), 'X': float(p[3])}
    return bands

def parse_sigma_freq_debug(path):
    """Parse sigma_freq_debug.dat -> {band_1idx: {'sigc_edft': float, 'sigc_head': float}}"""
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
                    bands[n1] = {'sigc_edft': sigc, 'sigc_head': head}
            except (ValueError, IndexError):
                pass
    return bands

bgw = parse_sigma_hp('00_bgw/sigma_hp.log')
gw = parse_sigma_freq_debug('00_lorrax/sigma_freq_debug.dat')

common = sorted(set(bgw) & set(gw))

print("GN-GPP: body-only vs body+head vs BGW Cor'")
print(f"{'Band':>5} {'BGW Cor':>9} {'body':>9} {'head':>7} {'body+h':>9} | {'Δbody':>8} {'Δ(b+h)':>8}")
print("-" * 75)
d_body, d_total = [], []
for n in common:
    b = bgw[n]['Corp']
    body = gw[n]['sigc_edft']
    h = gw[n]['sigc_head']
    total = body + h
    db = body - b
    dt = total - b
    d_body.append(db)
    d_total.append(dt)
    print(f"{n:5d} {b:9.3f} {body:9.3f} {h:7.3f} {total:9.3f} | {db:+8.3f} {dt:+8.3f}")

d_body = np.array(d_body)
d_total = np.array(d_total)
print(f"\nBody only:  MAE = {np.mean(np.abs(d_body)):.3f} eV, max|Δ| = {np.max(np.abs(d_body)):.3f} eV")
print(f"Body+head:  MAE = {np.mean(np.abs(d_total)):.3f} eV, max|Δ| = {np.max(np.abs(d_total)):.3f} eV")
