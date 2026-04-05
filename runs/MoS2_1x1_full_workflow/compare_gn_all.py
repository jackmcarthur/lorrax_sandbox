import re, numpy as np

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = float(p[4]) + float(p[10])  # Corp
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

bgw = parse_sigma_hp('00_bgw/sigma_hp.log')
gw_stensor = parse_sigma_freq_debug('00_lorrax/sigma_freq_debug.dat')
gw_bgwhead = parse_sigma_freq_debug('05_lorrax_gn_bgwhead/sigma_freq_debug.dat')

common = sorted(set(bgw) & set(gw_stensor) & set(gw_bgwhead))

print("GN-GPP: S-tensor body vs BGW-head body vs body+head variants")
print(f"{'Band':>5} {'BGW':>9} | {'S body':>9} {'S b+h':>9} | {'B body':>9} {'B head':>7} {'B b+h':>9} | {'ΔS_b':>7} {'ΔB_b':>7} {'ΔB_bh':>7}")
print("-" * 100)
ds_b, db_b, db_bh = [], [], []
for n in common:
    b = bgw[n]
    sb = gw_stensor[n]['sigc']
    sh = gw_stensor[n]['head']
    bb = gw_bgwhead[n]['sigc']
    bh = gw_bgwhead[n]['head']
    ds_b.append(sb - b)
    db_b.append(bb - b)
    db_bh.append(bb + bh - b)
    print(f"{n:5d} {b:9.3f} | {sb:9.3f} {sb+sh:9.3f} | {bb:9.3f} {bh:7.3f} {bb+bh:9.3f} | {sb-b:+7.3f} {bb-b:+7.3f} {bb+bh-b:+7.3f}")

for label, d in [("S-tensor body", ds_b), ("BGW-head body", db_b), ("BGW-head body+head", db_bh)]:
    d = np.array(d)
    print(f"{label:25s}: MAE = {np.mean(np.abs(d)):.3f} eV, max|Δ| = {np.max(np.abs(d)):.3f} eV")
