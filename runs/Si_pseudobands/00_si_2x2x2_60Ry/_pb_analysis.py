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
def parse_sigdiag(path):
    """Parse new sigma_diag.dat format with sigSX/sigCOH/sigTOT."""
    rows=[]; cur_k=None
    for line in open(path):
        m = re.match(r'k-point (\d+)', line)
        if m: cur_k=int(m.group(1)); continue
        mm = re.match(r'n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)', line)
        if mm: rows.append({'kidx':cur_k,'n':int(mm.group(1)),'sigSX':float(mm.group(2)),'sigCOH':float(mm.group(3)),'sigTOT':float(mm.group(4))})
    return rows

# Build BGW IBZ kpts list
def get_bgw_kpts(path):
    ks = []
    seen = set()
    for line in open(path):
        m = re.match(r'\s+k\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+ik\s*=\s*(\d+)', line)
        if m:
            k = kr([float(m.group(i)) for i in (1,2,3)])
            ik = int(m.group(4))
            if ik not in seen:
                seen.add(ik); ks.append(k)
    return ks

def compare(lr_path, bgw_path, label):
    bgw = parse_bgw(bgw_path)
    lr = parse_sigdiag(lr_path)
    bgw_kpts = get_bgw_kpts(bgw_path)
    lr_idx = {(L['kidx'], L['n']):L for L in lr}

    # LORRAX kpt index assumes WFN unfolded order. For SYM PB WFN, LORRAX outputs at unfolded kpts.
    # LORRAX kidx 0..7 may be unfolded. Match BGW IBZ kpts to LORRAX unfolded kpts by coord.
    # Read LORRAX WFN.h5 (which has IBZ kpts) — but LORRAX may unfold internally.
    # Find which LORRAX kidx matches each BGW IBZ kpt.
    all_lr_kidx = sorted(set(L['kidx'] for L in lr))
    # If LORRAX outputs at IBZ same as BGW (4 kidx), kidx 0=k0, 1=k1, etc match BGW
    # If LORRAX unfolded to 8, need to match by k-coord
    # Try IBZ assumption first: kidx i matches BGW kpt i (both in IBZ order)
    diffs_x, diffs_chp, diffs_sigp = [], [], []
    for r in bgw:
        ik = bgw_kpts.index(r['k']) if r['k'] in bgw_kpts else None
        if ik is None: continue
        L = lr_idx.get((ik, r['n']-1))
        if L is None: continue
        diffs_x.append(L['sigSX'] - (r['X']+r['SXX']))
        diffs_chp.append(L['sigCOH'] - r['CHp'])
        diffs_sigp.append(L['sigTOT'] - r['SigP'])
    a = np.array(diffs_sigp)
    b = np.array(diffs_x)
    c = np.array(diffs_chp)
    print(f"  {label}: matched={len(a)} pairs")
    print(f"    sigSX vs (X+SX-X):  MAE={np.mean(np.abs(b))*1000:8.3f}  max|Δ|={np.max(np.abs(b))*1000:8.3f}  meV")
    print(f"    sigCOH vs CH':      MAE={np.mean(np.abs(c))*1000:8.3f}  max|Δ|={np.max(np.abs(c))*1000:8.3f}  meV")
    print(f"    sigTOT vs Sig':     MAE={np.mean(np.abs(a))*1000:8.3f}  max|Δ|={np.max(np.abs(a))*1000:8.3f}  meV")
    return a, b, c

print("\n=== FULL PB-100sl (215 bands) ===")
for nc in [1464, 2448, 3264]:
    compare(f'D_lorrax_pb100sl_full/sigdiag_{nc}.dat', 'D_bgw_pb100sl_noavg/sigma_hp.log', f'N_c={nc}')

print("\n=== HALF PB-100sl (115 bands) ===")
for nc in [1464, 2448, 3264]:
    compare(f'D_lorrax_pb100sl_half/sigdiag_{nc}.dat', 'D_bgw_pb100sl_half_noavg/sigma_hp.log', f'N_c={nc}')
