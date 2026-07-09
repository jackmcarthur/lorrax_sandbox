#!/usr/bin/env python3
"""Aligned LORRAX-vs-BGW self-energy cross-check (gate-0), using the new Eo column.

Band alignment is DERIVED, not guessed: LORRAX's 30-band window is a contiguous
sub-window of BGW's 34, and the two use different energy zeros (constant offset).
Fit both from the DFT-energy (Eo) ladders: pick the band-shift s and constant c
minimizing |Eo_L(k,i) - (Eo_B(k,i+s) + c)|, then compare sigTOT to BGW (X+Cor)
on the matched bands. Frozen ref reuses the SAME band map (same code, same order).
"""
import re, sys
import numpy as np
from pathlib import Path

D = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/e2e_gate0_verify")
FROZEN = D / "eqp_ref.dat"
CURRENT = D / "eqp_test.dat"
BGW = D / "sigma_static_ref.out"


def parse_lorrax(path, want_eo):
    fl = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    tot = re.compile(rf"n=\s*(\d+)\s.*?sigTOT=\s*{fl}")
    eo = re.compile(rf"Eo=\s*{fl}")
    kre = re.compile(r"k-point\s+(\d+)\s*:")
    sig, eos, k = {}, {}, -1
    for line in Path(path).read_text().splitlines():
        m = kre.search(line)
        if m:
            k = int(m.group(1)); continue
        mt = tot.search(line)
        if mt:
            n = int(mt.group(1)); sig[(k, n)] = float(mt.group(2))
            if want_eo:
                me = eo.search(line)
                if me:
                    eos[(k, n)] = float(me.group(1))
    return sig, eos


def parse_bgw(path):
    sig, eo, ik = {}, {}, None
    kre = re.compile(r"k =\s*[\d.Ee+-]+\s+[\d.Ee+-]+\s+[\d.Ee+-]+\s+ik =\s*(\d+)")
    for line in Path(path).read_text().splitlines():
        m = kre.search(line)
        if m:
            ik = int(m.group(1)) - 1; continue
        if ik is None:
            continue
        p = line.split()
        if len(p) == 9 and p[0].isdigit():
            n = int(p[0]) - 1               # -> 0-indexed BGW band
            eo[(ik, n)] = float(p[2])
            sig[(ik, n)] = float(p[4]) + float(p[5])   # X + Cor
    return sig, eo


cur_sig, cur_eo = parse_lorrax(CURRENT, want_eo=True)
frz_sig, _ = parse_lorrax(FROZEN, want_eo=False)
bgw_sig, bgw_eo = parse_bgw(BGW)

if not cur_eo:
    sys.exit("ERROR: current eqp_test.dat has no Eo column — re-run with the Eo patch.")

nk = 1 + max(k for k, _ in cur_sig)
nL = len(cur_sig) // nk
nB = len(bgw_sig) // nk
print(f"k-pts {nk} | LORRAX bands {nL} | BGW bands {nB}")

# --- fit global band-shift s and energy offset c from the Eo ladders ---
best = None
for s in range(0, nB - nL + 1):
    res = []
    for k in range(nk):
        for i in range(nL):
            if (k, i) in cur_eo and (k, i + s) in bgw_eo:
                res.append(cur_eo[(k, i)] - bgw_eo[(k, i + s)])
    if not res:
        continue
    res = np.array(res)
    c = np.median(res)
    spread = np.median(np.abs(res - c))          # MAD, robust
    if best is None or spread < best[2]:
        best = (s, c, spread)
s, c, spread = best
print(f"band-shift s={s} | Eo offset c={c:+.3f} eV | match MAD={1000*spread:.1f} meV "
      f"({'OK' if spread < 0.05 else 'LOOSE — alignment suspect'})")

# --- compare sigTOT vs BGW (X+Cor) on matched bands ---
rows = []
for k in range(nk):
    for i in range(nL):
        if (k, i) in cur_sig and (k, i) in frz_sig and (k, i + s) in bgw_sig:
            rows.append((bgw_sig[(k, i + s)], frz_sig[(k, i)], cur_sig[(k, i)]))
r = np.array(rows)
d_frz = np.abs(r[:, 1] - r[:, 0])
d_cur = np.abs(r[:, 2] - r[:, 0])
print(f"\nmatched bands: {len(r)}")
print(f"FROZEN  sigTOT vs BGW : MAE {1000*d_frz.mean():7.2f} meV | max {1000*d_frz.max():7.1f} meV")
print(f"CURRENT sigTOT vs BGW : MAE {1000*d_cur.mean():7.2f} meV | max {1000*d_cur.max():7.1f} meV")
print(f"closer to BGW: {'CURRENT' if d_cur.mean() < d_frz.mean() else 'FROZEN'} "
      f"(ΔMAE {1000*(d_cur.mean()-d_frz.mean()):+.2f} meV) — "
      f"{'RE-FREEZE justified' if d_cur.mean() <= d_frz.mean()+0.5 else 'REGRESSION: current is worse'}")
print(f"bands where current closer: {(d_cur<d_frz).sum()}/{len(r)}")
