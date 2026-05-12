import re, numpy as np, h5py
def kfold(x):
    v = round(x, 6) % 1.0
    return round(v, 6) if abs(round(v,6)) > 1e-5 else 0.0
def kr(k):
    return tuple(kfold(c) for c in k)
def parse_bgw(path):
    rows = []
    cur_k = None
    with open(path) as f:
        for line in f:
            m = re.match(r'\s+k\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+ik', line)
            if m:
                cur_k = kr([float(m.group(i)) for i in (1,2,3)])
            else:
                p = line.split()
                if len(p) >= 14 and p[0].isdigit() and cur_k is not None:
                    rows.append({'k':cur_k,'n':int(p[0]),'X':float(p[3]),'SXX':float(p[4]),'CH':float(p[5]),'Sig':float(p[6]),'CHp':float(p[10]),'SigP':float(p[11])})
    return rows
def parse_lor(path):
    rows = []
    cur_k = None
    with open(path) as f:
        for line in f:
            m = re.match(r'k-point (\d+)', line)
            if m: cur_k=int(m.group(1)); continue
            mm = re.match(r'n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)', line)
            if mm: rows.append({'kidx':cur_k,'n':int(mm.group(1)),'sigSX':float(mm.group(2)),'sigCOH':float(mm.group(3)),'sigTOT':float(mm.group(4))})
    return rows

f = h5py.File('WFN.h5','r')
kpts_round = [kr(k) for k in f['mf_header/kpoints/rk'][:]]
bgw = parse_bgw('../D_bgw_canonical_noavg/sigma_hp.log')
lr = parse_lor('eqp0.dat')
lr_idx = {(L['kidx'], L['n']):L for L in lr}

dx, dsxx, dch, dchp, dsigp = [], [], [], [], []
for r in bgw:
    if r['k'] not in kpts_round: continue
    kidx = kpts_round.index(r['k'])
    L = lr_idx.get((kidx, r['n']-1))
    if L is None: continue
    bgw_x_full = r['X'] + r['SXX']  # full exchange (X+SX-X)
    dx.append(L['sigSX'] - bgw_x_full)         # SX residual
    dch.append(L['sigCOH'] - r['CH'])          # COH residual (BGW unprimed)
    dchp.append(L['sigCOH'] - r['CHp'])        # COH residual (BGW head-corrected)
    dsigp.append(L['sigTOT'] - r['SigP'])      # total
dx = np.array(dx); dch = np.array(dch); dchp = np.array(dchp); dsigp = np.array(dsigp)
def stats(name, a):
    print(f"  {name:18s} mean={np.mean(a)*1000:+8.3f}  MAE={np.mean(np.abs(a))*1000:7.3f}  max|Δ|={np.max(np.abs(a))*1000:7.3f}  std={np.std(a)*1000:7.3f}  meV")
print(f"matched {len(dx)} (k,n) pairs")
stats("sigSX vs (X+SX-X)", dx)
stats("sigCOH vs CH",      dch)
stats("sigCOH vs CH'",     dchp)
stats("sigTOT vs Sig'",    dsigp)
