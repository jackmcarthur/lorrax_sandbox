import re, numpy as np

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            bands[int(p[0])] = float(p[4]) + float(p[10])
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

bgw = parse_sigma_hp('00_bgw/sigma_hp.log')
body_only = parse_eqp0_sigc('00_lorrax/eqp0.dat')          # no head override, body only
applied = parse_eqp0_sigc('06_lorrax_gn_bgwhead_applied/eqp0.dat')  # BGW heads, applied

common = sorted(set(bgw) & set(body_only) & set(applied))

print(f"{'Band':>5} {'BGW Cor':>9} {'body':>9} {'applied':>9} | {'Δbody':>8} {'Δapplied':>8}")
print("-" * 65)
db, da = [], []
for n in common:
    b = bgw[n]
    d1 = body_only[n] - b
    d2 = applied[n] - b
    db.append(d1); da.append(d2)
    print(f"{n:5d} {b:9.3f} {body_only[n]:9.3f} {applied[n]:9.3f} | {d1:+8.3f} {d2:+8.3f}")

db, da = np.array(db), np.array(da)
print(f"\nBody only (S-tensor, no head):   MAE = {np.mean(np.abs(db)):.3f} eV, max|Δ| = {np.max(np.abs(db)):.3f} eV")
print(f"BGW heads applied:               MAE = {np.mean(np.abs(da)):.3f} eV, max|Δ| = {np.max(np.abs(da)):.3f} eV")
print(f"Change:                          ΔMAE = {np.mean(np.abs(da)) - np.mean(np.abs(db)):+.3f} eV")
