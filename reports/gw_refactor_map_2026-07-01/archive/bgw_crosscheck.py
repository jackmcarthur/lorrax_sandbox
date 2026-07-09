#!/usr/bin/env python3
"""Gate-0 drift adjudication: shape of the current-vs-frozen drift + BGW scale.

Primary (alignment-free, definitive): characterize the drift current-main vs the
frozen reference. Both are LORRAX on the SAME 9 path k-points with the SAME band
indexing, so alignment is trivial. A benign convention change from fc1602a shows
as a UNIFORM plateau (tight std, no band/k structure); a real unfold bug shows as
localized k/band-dependent outliers (cf. the TRS bug's multi-eV spikes).

Secondary (BGW scale, robustness-checked): the frozen ref was validated <100 meV
vs BGW. Compare LORRAX sigTOT to BGW (X+Cor) under both plausible band offsets and
report whether the closer/further verdict is robust to the offset choice.
"""
import re
import numpy as np
from pathlib import Path

FIX = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/tests/regression/cohsex_debug")
RERUN = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/e2e_rerun_nowfnqp")


def parse_lorrax_sigma(path):
    fl = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    row = re.compile(rf"n=\s*(\d+)\s+sigSX=\s*{fl}\s+sigCOH=\s*{fl}\s+sigTOT=\s*{fl}")
    kre = re.compile(r"k-point\s+(\d+)\s*:")
    out, k = {}, -1
    for line in Path(path).read_text().splitlines():
        m = kre.search(line)
        if m:
            k = int(m.group(1)); continue
        m = row.search(line)
        if m:
            out[(k, int(m.group(1)))] = float(m.group(4))   # sigTOT
    return out


def parse_bgw(path):
    """-> {(ik0, n1): Sig=X+Cor}. Cols: n Emf Eo Vxc X Cor Eqp0 Eqp1 Znk."""
    out, ik = {}, None
    kre = re.compile(r"k =\s*[\d.Ee+-]+\s+[\d.Ee+-]+\s+[\d.Ee+-]+\s+ik =\s*(\d+)")
    for line in Path(path).read_text().splitlines():
        m = kre.search(line)
        if m:
            ik = int(m.group(1)) - 1; continue
        if ik is None:
            continue
        p = line.split()
        if len(p) == 9 and p[0].isdigit():
            out[(ik, int(p[0]))] = float(p[4]) + float(p[5])
    return out


frozen = parse_lorrax_sigma(FIX / "eqp_ref.dat")
current = parse_lorrax_sigma(RERUN / "eqp_test.dat")
bgw = parse_bgw(FIX / "sigma_static_ref.out")

keys = sorted(set(frozen) & set(current))
nk = 1 + max(k for k, _ in keys)
drift = np.array([current[key] - frozen[key] for key in keys])   # eV
kidx = np.array([k for k, _ in keys])

print(f"=== DRIFT SHAPE (current-main vs frozen ref, {len(keys)} bands, {nk} path k-pts) ===")
print(f"  mean {1000*drift.mean():+6.2f} meV | std {1000*drift.std():5.2f} meV | "
      f"min {1000*drift.min():+6.1f} | max {1000*drift.max():+6.1f} meV")
print(f"  |drift - mean|: 95th pct {1000*np.percentile(np.abs(drift-drift.mean()),95):.2f} meV | "
      f"max {1000*np.abs(drift-drift.mean()).max():.2f} meV")
print("  per-k mean drift (meV): " +
      " ".join(f"k{k}:{1000*drift[kidx==k].mean():+5.1f}" for k in range(nk)))
uniform = drift.std() < 0.005 and np.abs(drift - drift.mean()).max() < 0.02
print(f"  -> {'UNIFORM PLATEAU (benign convention signature)' if uniform else 'STRUCTURED (investigate)'}")

print(f"\n=== BGW SCALE (LORRAX sigTOT vs BGW X+Cor), robustness over band offset ===")
print(f"  BGW bands/k: {len(bgw)//nk} | LORRAX bands/k: {len(current)//nk}")
for off in (1, 5):   # LORRAX n(0-idx) -> BGW n = n + off
    pairs = [(k, n) for (k, n) in keys if (k, n + off) in bgw]
    if not pairs:
        continue
    a = np.array([bgw[(k, n + off)] for k, n in pairs])
    f = np.array([frozen[(k, n)] for k, n in pairs])
    c = np.array([current[(k, n)] for k, n in pairs])
    df, dc = np.abs(f - a).mean(), np.abs(c - a).mean()
    tag = "re-freeze" if dc < df else "regression"
    print(f"  offset +{off} ({len(pairs)} bands): frozen-BGW MAE {df:6.3f} eV | "
          f"current-BGW MAE {dc:6.3f} eV | Δ {1000*(dc-df):+5.2f} meV -> {tag}")
