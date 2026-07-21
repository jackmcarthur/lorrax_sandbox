"""One table for every stage-4 variant: gaps + per-band QP drift vs the baseline.

Baseline is the stage-3 production run 00b (rank-truncate, xi floor on,
+/-10 eV omega grid, one_shot_dft).  Every variant restarts from ITS ISDF
tensors, so the only difference is the Sigma-stage knob named in the tag.

usage: python3 stage4_summary.py <baseline_rundir> <variant_rundir> [...]
"""
import os
import sys

import numpy as np

NVAL = int(os.environ.get("S4_NVAL", "26"))


def parse_eqp(path):
    ks, rows, cur = [], [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) == 4 and '.' in p[0]:
            ks.append([float(v) for v in p[:3]])
            cur = {}
            rows.append(cur)
        elif len(p) == 4 and cur is not None:
            cur[int(p[1])] = (float(p[2]), float(p[3]))
    nb = max(max(r) for r in rows)
    ed = np.full((len(ks), nb), np.nan)
    eq = np.full((len(ks), nb), np.nan)
    for k in range(len(ks)):
        for b, (a, c) in rows[k].items():
            ed[k, b - 1] = a
            eq[k, b - 1] = c
    return np.asarray(ks), ed, eq


def _wrap(d):
    return (d + 0.5) % 1.0 - 0.5


def summarize(rd):
    p = os.path.join(rd, "eqp1.dat")
    if not os.path.exists(p):
        return None
    ks, ed, eq = parse_eqp(p)
    ikK = int(np.argmin(np.linalg.norm(
        _wrap(ks[:, :2] - np.array([1 / 3, 1 / 3])), axis=1)))
    return dict(
        tag=os.path.basename(rd), ks=ks, ed=ed, eq=eq,
        direct_K=float(eq[ikK, NVAL] - eq[ikK, NVAL - 1]),
        indirect=float(np.nanmin(eq[:, NVAL]) - np.nanmax(eq[:, NVAL - 1])),
        dE=eq - ed, nb=eq.shape[1])


base = summarize(sys.argv[1])
if base is None:
    raise SystemExit(f"baseline {sys.argv[1]} has no eqp1.dat")

print(f"baseline = {base['tag']}   direct@K {base['direct_K']:.4f} eV   "
      f"indirect {base['indirect']:.4f} eV   ({base['nb']} QP bands)")
print()
print(f"{'variant':>20} {'direct@K':>10} {'d(gap)':>9} {'indirect':>10} "
      f"{'d(ind)':>9} | mean |d<dE>| (meV): {'val':>6} {'c27-60':>7} {'c61-80':>7} {'max':>7}")
print("-" * 122)
print(f"{base['tag'][:20]:>20} {base['direct_K']:10.4f} {0.0:9.1f} "
      f"{base['indirect']:10.4f} {0.0:9.1f} |{'':>22} {'--':>6} {'--':>7} {'--':>7} {'--':>7}")

for rd in sys.argv[2:]:
    v = summarize(rd)
    if v is None:
        print(f"{os.path.basename(rd)[:20]:>20}   (no eqp1.dat -- did not complete)")
        continue
    nb = min(v['nb'], base['nb'])
    d = (np.nanmean(v['dE'][:, :nb], axis=0)
         - np.nanmean(base['dE'][:, :nb], axis=0)) * 1e3
    seg = lambda lo, hi: (np.nanmean(np.abs(d[lo:hi])) if hi > lo else np.nan)
    print(f"{v['tag'][:20]:>20} {v['direct_K']:10.4f} "
          f"{(v['direct_K']-base['direct_K'])*1e3:9.1f} {v['indirect']:10.4f} "
          f"{(v['indirect']-base['indirect'])*1e3:9.1f} |{'':>22} "
          f"{seg(0,NVAL):6.1f} {seg(NVAL,min(60,nb)):7.1f} "
          f"{seg(min(60,nb),nb):7.1f} {np.nanmax(np.abs(d)):7.1f}")
print()
print("d(gap)/d(ind) in meV vs baseline; mean |d<dE>| averages |E_QP-E_DFT| "
      "differences over all 144 k, per band segment.")
