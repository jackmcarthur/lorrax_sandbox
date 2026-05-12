import re, numpy as np
def kfold(x):
    v = round(x, 6) % 1.0
    return round(v, 6) if abs(round(v,6)) > 1e-5 else 0.0
def kr(k): return tuple(kfold(c) for c in k)
def parse_bgw(path):
    rows=[]; cur_k=None
    for line in open(path):
        m = re.match(r'\s+k\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+ik', line)
        if m: cur_k = kr([float(m.group(i)) for i in (1,2,3)])
        else:
            p = line.split()
            if len(p)>=14 and p[0].isdigit() and cur_k is not None:
                rows.append({'k':cur_k,'n':int(p[0]),'X':float(p[3]),'SXX':float(p[4]),'CHp':float(p[10]),'SigP':float(p[11])})
    return rows
def parse_sigdiag(path):
    rows=[]; cur_k=None
    for line in open(path):
        m = re.match(r'k-point (\d+)', line)
        if m: cur_k=int(m.group(1)); continue
        mm = re.match(r'n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)', line)
        if mm: rows.append({'kidx':cur_k,'n':int(mm.group(1)),'sigSX':float(mm.group(2)),'sigCOH':float(mm.group(3)),'sigTOT':float(mm.group(4))})
    return rows
si_unfolded = [(0,0,0),(0,0,0.5),(0,0.5,0),(0,0.5,0.5),(0.5,0,0),(0.5,0,0.5),(0.5,0.5,0),(0.5,0.5,0.5)]

def compare(label, lr_path, bgw_path):
    bgw = parse_bgw(bgw_path); lr = parse_sigdiag(lr_path)
    lr_idx = {(L['kidx'], L['n']): L for L in lr}
    diffs_x, diffs_chp, diffs_sigp = [], [], []
    for r in bgw:
        if r['k'] not in si_unfolded: continue
        kidx = si_unfolded.index(r['k'])
        L = lr_idx.get((kidx, r['n']-1))
        if L is None: continue
        diffs_x.append(L['sigSX'] - (r['X']+r['SXX']))
        diffs_chp.append(L['sigCOH'] - r['CHp'])
        diffs_sigp.append(L['sigTOT'] - r['SigP'])
    print(f"  {label}: matched={len(diffs_sigp)}")
    print(f"    sigSX(X+SX-X) mean={np.mean(diffs_x)*1000:+8.3f} MAE={np.mean(np.abs(diffs_x))*1000:8.3f} max={np.max(np.abs(diffs_x))*1000:8.3f}")
    print(f"    sigCOH(CH')   mean={np.mean(diffs_chp)*1000:+8.3f} MAE={np.mean(np.abs(diffs_chp))*1000:8.3f} max={np.max(np.abs(diffs_chp))*1000:8.3f}")
    print(f"    sigTOT(Sig')  mean={np.mean(diffs_sigp)*1000:+8.3f} MAE={np.mean(np.abs(diffs_sigp))*1000:8.3f} max={np.max(np.abs(diffs_sigp))*1000:8.3f}  meV")

print("\n========== ncond=8 (FIXED) — PB-100sl ==========")
print("\n--- FULL nband=215 ---")
for nc in [1464,2448,3264]:
    compare(f"N_c={nc}", f"D_lorrax_pb100sl_full/sigdiag_nc8_{nc}.dat",
                         "D_bgw_pb100sl_noavg/sigma_hp.log")
print("\n--- HALF nband=115 ---")
for nc in [1464,2448,3264]:
    compare(f"N_c={nc}", f"D_lorrax_pb100sl_half/sigdiag_nc8_{nc}.dat",
                         "D_bgw_pb100sl_half_noavg/sigma_hp.log")

print("\n========== for comparison: ncond=207 (BROKEN) at N_c=3264 ==========")
compare("FULL ncond=207 N_c=3264", "D_lorrax_pb100sl_full/sigdiag_3264.dat",
                                   "D_bgw_pb100sl_noavg/sigma_hp.log")
compare("HALF ncond=107 N_c=3264", "D_lorrax_pb100sl_half/sigdiag_3264.dat",
                                   "D_bgw_pb100sl_half_noavg/sigma_hp.log")
