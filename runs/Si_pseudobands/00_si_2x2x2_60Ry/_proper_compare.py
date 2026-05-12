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
                rows.append({'k':cur_k,'n':int(p[0]),'X':float(p[3]),'SXX':float(p[4]),'CHp':float(p[10]),'SigP':float(p[11]),'Eo':float(p[2])})
    return rows

def parse_sigdiag(path):
    """sigma_diag.dat — kidx is 0..N-1, no k-coords inline."""
    rows=[]; cur_k=None
    for line in open(path):
        m = re.match(r'k-point (\d+)', line)
        if m: cur_k=int(m.group(1)); continue
        mm = re.match(r'n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)', line)
        if mm: rows.append({'kidx':cur_k,'n':int(mm.group(1)),'sigSX':float(mm.group(2)),'sigCOH':float(mm.group(3)),'sigTOT':float(mm.group(4))})
    return rows

def parse_eqp0_kpts(path):
    """LORRAX eqp0.dat k-block headers give exact k-coords in IBZ order (sym) or full BZ (nosym)."""
    ks=[]
    for line in open(path):
        m = re.match(r'^\s+(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\s+(-?[0-9]+\.[0-9]+)\s+\d+\s*$', line)
        if m:
            ks.append(kr([float(m.group(i)) for i in (1,2,3)]))
    return ks

# For sym-mode runs, sigma_diag has 8 unfolded kpts in lex sort order
# (verified by symmetry partition of sigSX values: Γ × M3 × X3 × R1 = 1+3+3+1 = 8).
# Build the lex-sorted unfolded kpt list explicitly for Si 2x2x2:
si_2x2x2_unfolded_lex = [
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0), (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.5, 0.5, 0.5)
]

def compare_with_kcoords(label, lr_path, lr_kpts, bgw_path, channels=('X',)):
    """Match BGW IBZ kpt → exact-coord LORRAX kidx."""
    bgw = parse_bgw(bgw_path)
    lr = parse_sigdiag(lr_path)
    lr_idx = {(L['kidx'], L['n']): L for L in lr}

    out = {}
    for r in bgw:
        if r['k'] not in lr_kpts:
            continue
        kidx = lr_kpts.index(r['k'])  # exact coord match
        L = lr_idx.get((kidx, r['n']-1))
        if L is None: continue
        for ch in channels:
            if ch == 'X':
                out.setdefault(ch, []).append(L['sigSX'] - r['X'])
            elif ch == 'XfullSX':
                out.setdefault(ch, []).append(L['sigSX'] - (r['X']+r['SXX']))
            elif ch == 'CHp':
                out.setdefault(ch, []).append(L['sigCOH'] - r['CHp'])
            elif ch == 'SigP':
                out.setdefault(ch, []).append(L['sigTOT'] - r['SigP'])
    print(f"=== {label} ===")
    for ch, diffs in out.items():
        a = np.array(diffs)
        print(f"  {ch}:  N={len(a)}  mean={np.mean(a)*1000:+8.3f}  MAE={np.mean(np.abs(a))*1000:8.3f}  max|Δ|={np.max(np.abs(a))*1000:8.3f}  meV")

# 1) PB nb=16 x_only — bare X (sym path, 8 kidx in sigma_diag)
compare_with_kcoords(
    "PB-100sl SYM nb=16 x_only N_c=408",
    "D_lorrax_pb100sl_nb16_xonly/sigdiag_408.dat",
    si_2x2x2_unfolded_lex,
    "D_bgw_pb100sl_noavg/sigma_hp.log",
    channels=('X',))

# 2) PB nb=215 x_only — bare X (sym path)
for nc in [1464, 2448, 3264]:
    compare_with_kcoords(
        f"PB-100sl SYM nb=215 x_only N_c={nc}",
        f"D_lorrax_pb100sl_full_xonly/sigdiag_{nc}.dat",
        si_2x2x2_unfolded_lex,
        "D_bgw_pb100sl_noavg/sigma_hp.log",
        channels=('X',))

# 3) PB nb=215 cohsex — full Σ (sym path)
for nc in [1464, 2448, 3264]:
    compare_with_kcoords(
        f"PB-100sl SYM nb=215 cohsex N_c={nc}",
        f"D_lorrax_pb100sl_full/sigdiag_{nc}.dat",
        si_2x2x2_unfolded_lex,
        "D_bgw_pb100sl_noavg/sigma_hp.log",
        channels=('XfullSX','CHp','SigP'))

# 4) PB nb=115 cohsex — half-PB (sym path)
for nc in [1464, 2448, 3264]:
    compare_with_kcoords(
        f"PB-100sl SYM nb=115 (half) cohsex N_c={nc}",
        f"D_lorrax_pb100sl_half/sigdiag_{nc}.dat",
        si_2x2x2_unfolded_lex,
        "D_bgw_pb100sl_half_noavg/sigma_hp.log",
        channels=('XfullSX','CHp','SigP'))

# 5) Sanity check: explicit nb=16 nosym N_c=400 (should be ~0.004 meV)
compare_with_kcoords(
    "Explicit Si 2x2x2 NOSYM nb=16 x_only N_c=400 (sanity)",
    "D_lorrax_explicit_nb16_xonly/sigma_diag.dat",
    si_2x2x2_unfolded_lex,
    "D_bgw_canonical_noavg/sigma_hp.log",
    channels=('X',))
