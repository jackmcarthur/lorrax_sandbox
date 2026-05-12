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

# Print residuals per kpoint
for kidx in range(8):
    diffs_sx = []
    diffs_chp = []
    for r in bgw:
        if r['k'] != kpts_round[kidx]: continue
        L = lr_idx.get((kidx, r['n']-1))
        if L is None: continue
        diffs_sx.append(L['sigSX'] - (r['X']+r['SXX']))
        diffs_chp.append(L['sigCOH'] - r['CHp'])
    arrSX = np.array(diffs_sx)
    arrCH = np.array(diffs_chp)
    print(f"kidx={kidx} {kpts_round[kidx]}: nb={len(arrSX):2d}  meanSX={np.mean(arrSX)*1000:+7.3f}  meanCH={np.mean(arrCH)*1000:+7.3f}  meV  maxSX={np.max(np.abs(arrSX))*1000:6.2f}  maxCH={np.max(np.abs(arrCH))*1000:6.2f}")
