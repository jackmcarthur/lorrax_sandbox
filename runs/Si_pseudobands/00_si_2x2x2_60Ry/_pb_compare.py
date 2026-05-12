import re, numpy as np, h5py, os
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
                rows.append({'k':cur_k,'n':int(p[0]),'X':float(p[3]),'SXX':float(p[4]),'CHp':float(p[10]),'SigP':float(p[11]),'Eo':float(p[2])})
    return rows
def parse_lor(path):
    rows=[]; cur_k=None
    for line in open(path):
        m = re.match(r'k-point (\d+)', line)
        if m: cur_k=int(m.group(1)); continue
        mm = re.match(r'n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)', line)
        if mm: rows.append({'kidx':cur_k,'n':int(mm.group(1)),'sigSX':float(mm.group(2)),'sigCOH':float(mm.group(3)),'sigTOT':float(mm.group(4))})
    return rows

def compare(lr_path, bgw_path, wfn_path):
    f = h5py.File(wfn_path,'r')
    lor_kpts = [kr(k) for k in f['mf_header/kpoints/rk'][:]]
    # lor uses unfolded kpts; bgw uses IBZ kpts; match via crystal coord
    bgw = parse_bgw(bgw_path)
    lr  = parse_lor(lr_path)
    lr_idx = {(L['kidx'], L['n']):L for L in lr}
    # Need to map BGW IBZ k-coord -> LORRAX k-index. With sym path, LORRAX outputs at unfolded kpts.
    # WFN.h5 used here is the IBZ WFN (4 kpts) — LORRAX may unfold. Let's see what kpts LORRAX has.
    # Read LORRAX's actual kpt count from rows
    n_lr_k = max(L['kidx'] for L in lr) + 1 if lr else 0
    return bgw, lr, lor_kpts, n_lr_k

# Quick eqp0 inspection
print("=== Full PB 1464 first 5 rows ===")
with open('D_lorrax_pb100sl_full/eqp0_1464.dat') as f:
    print(''.join(f.readlines()[:8]))
print("LORRAX k-pts in eqp0:")
seen_k = set()
for L in parse_lor('D_lorrax_pb100sl_full/eqp0_1464.dat'):
    seen_k.add(L['kidx'])
print(f"  num kpoints: {len(seen_k)}")

print("\n=== BGW PB 215 first rows ===")
with open('D_bgw_pb100sl_noavg/sigma_hp.log') as f:
    for ln in f.readlines()[:25]:
        print(ln,end='')
