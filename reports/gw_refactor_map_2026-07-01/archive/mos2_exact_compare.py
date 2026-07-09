#!/usr/bin/env python3
"""MoS2 3x3 COHSEX: LORRAX total Sigma vs BGW-noavg, matched WFN + k by coords.

Reuses the sanctioned compare script's BGW parser + WFN k-matching. Compares the
TOTAL static self-energy: LORRAX sigTOT (=sigSX+sigCOH) vs BGW Sig' (=X+(SX-X)+CH',
primed, col 11) AND vs BGW Sig (unprimed, col 6) to see which convention matches.
Reference-free (self-energy, not Eqp0). Alignment: LORRAX n = BGW band - 1.
"""
import re, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts")
from compare_bgw_gwjax import parse_bgw_sigma_hp, get_gwjax_kpoints, match_kpoint

RUN = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex")
BGW = RUN / "01_bgw_cohsex_noavg/sigma_hp.log"
WFN = RUN / "qe/nscf/WFN.h5"

def parse_lorrax_sigtot(path):
    fl = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    row = re.compile(rf"n=\s*(\d+)\s.*?sigTOT=\s*{fl}")
    kre = re.compile(r"k-point\s+(\d+)\s*:")
    out, k = {}, -1
    for line in Path(path).read_text().splitlines():
        m = kre.search(line)
        if m: k = int(m.group(1)); continue
        m = row.search(line)
        if m: out[(k, int(m.group(1)))] = float(m.group(2))
    return out

def run(lorrax_eqp, label):
    lor = parse_lorrax_sigtot(lorrax_eqp)
    bgw = parse_bgw_sigma_hp(BGW)
    kpts = get_gwjax_kpoints(WFN)
    dprime, dunpr = [], []
    nmatch = 0
    for blk in bgw:
        ki = match_kpoint(blk["kcrys"], kpts)
        if ki is None:
            print(f"  unmatched BGW k={blk['kcrys']}"); continue
        for n, b in blk["bands"].items():
            key = (ki, n - 1)                      # LORRAX n = BGW band - 1
            if key not in lor: continue
            nmatch += 1
            sig_prime = b["X"] + b["SXmX"] + b["CHp"]   # Sig' = X+(SX-X)+CH'
            sig_unpr  = b["Sig"]                          # unprimed total (col 6)
            dprime.append(lor[key] - sig_prime)
            dunpr.append(lor[key] - sig_unpr)
    dprime, dunpr = np.array(dprime), np.array(dunpr)
    print(f"\n=== {label} ({nmatch} bands x k) ===")
    print(f"  LORRAX sigTOT vs BGW Sig'(primed)  : MAE {1000*np.abs(dprime).mean():7.3f} meV | max {1000*np.abs(dprime).max():7.2f} meV | mean {1000*dprime.mean():+7.3f}")
    print(f"  LORRAX sigTOT vs BGW Sig (unprimed): MAE {1000*np.abs(dunpr).mean():7.3f} meV | max {1000*np.abs(dunpr).max():7.2f} meV | mean {1000*dunpr.mean():+7.3f}")
    return min(np.abs(dprime).mean(), np.abs(dunpr).mean())

if __name__ == "__main__":
    eqp = sys.argv[1] if len(sys.argv) > 1 else str(RUN / "D_lorrax_cohsex_overlay/eqp0.dat")
    run(eqp, Path(eqp).parent.name)
