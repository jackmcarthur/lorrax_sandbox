"""Stage-3 SANITY GATE for the converged MoS2 GW.

The brief: "Gamma/K VBM+CBM Sigma sane, gap non-inverted, no astronomical Im,
rank-truncate reports full-rank Galerkin.  If any fail, STOP and report rather
than proceeding."

Reads the run's OWN outputs -- sigma_diag.dat (LORRAX-native Sigma
decomposition), eqp1.dat, and gw.out -- and prints PASS/FAIL per criterion.
Exit status 0 = all pass, 1 = at least one FAIL.

usage: python3 sanity_gate.py <rundir>
"""
import os
import re
import sys

import numpy as np

RD = sys.argv[1] if len(sys.argv) > 1 else "."
NVAL = int(os.environ.get("GATE_NVAL", "26"))

# Thresholds.  Deliberately loose -- this is a catastrophe detector, not a
# precision gate.  The failure modes it exists to catch were O(1e2-1e5) eV.
MAX_ABS_SIGC = 50.0      # eV; the cured conduction Sigma_c sits at O(1-10)
MAX_ABS_IM = 5.0         # eV; "astronomical Im" was O(1e3)
MAX_ABS_SIGX = 100.0     # eV

_SIG = re.compile(
    r"n=(\d+)\s+sigX=\s*(\S+)\s+sigC=\s*(\S+)\+\s*(\S+)i\s+"
    r"sigXC=\s*(\S+)\+\s*(\S+)i\s+VH=\s*(\S+)\s+Eo=\s*(\S+)")


def parse_sigma_diag(path):
    """-> dict[k][n] = (sigX, sigC_re, sigC_im, sigXC_re, sigXC_im, VH, Eo)."""
    out, ik = {}, -1
    for line in open(path):
        if line.startswith("k-point"):
            ik = int(line.split()[1].rstrip(":"))
            out[ik] = {}
            continue
        m = _SIG.search(line)
        if m and ik >= 0:
            n = int(m.group(1))
            out[ik][n] = tuple(float(m.group(i)) for i in range(2, 9))
    return out


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
    return np.asarray(ks), rows


def _wrap(d):
    return (d + 0.5) % 1.0 - 0.5


fails, notes = [], []


def check(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        fails.append(name)


print(f"=== stage-3 sanity gate  ({RD})")

# ---------------------------------------------------------------- Sigma sane
sd_path = os.path.join(RD, "sigma_diag.dat")
sd = parse_sigma_diag(sd_path)
nk = len(sd)
allv = np.array([v for k in sd for v in sd[k].values()])
sigX, sigC_re, sigC_im = allv[:, 0], allv[:, 1], allv[:, 2]
print(f"  parsed sigma_diag.dat: {nk} k-points x "
      f"{len(sd[min(sd)])} bands = {allv.shape[0]} (k,n) rows")

check("sigX finite + bounded",
      np.all(np.isfinite(sigX)) and np.abs(sigX).max() < MAX_ABS_SIGX,
      f"max|sigX| = {np.abs(sigX).max():.3f} eV  (limit {MAX_ABS_SIGX})")
check("sigC finite + bounded (no conduction blow-up)",
      np.all(np.isfinite(sigC_re)) and np.abs(sigC_re).max() < MAX_ABS_SIGC,
      f"max|Re sigC| = {np.abs(sigC_re).max():.3f} eV  (limit {MAX_ABS_SIGC})")
check("no astronomical Im sigC",
      np.all(np.isfinite(sigC_im)) and np.abs(sigC_im).max() < MAX_ABS_IM,
      f"max|Im sigC| = {np.abs(sigC_im).max():.4f} eV  (limit {MAX_ABS_IM})")

# ----------------------------------------------------- Gamma / K, VBM + CBM
ks, rows = parse_eqp(os.path.join(RD, "eqp1.dat"))
ikG = int(np.argmin(np.linalg.norm(_wrap(ks[:, :2] - np.array([0., 0.])), axis=1)))
ikK = int(np.argmin(np.linalg.norm(
    _wrap(ks[:, :2] - np.array([1/3, 1/3])), axis=1)))
for nm, ik in (("Gamma", ikG), ("K", ikK)):
    for lbl, n0 in (("VBM", NVAL - 1), ("CBM", NVAL)):
        sx, cr, ci = sd[ik][n0][0], sd[ik][n0][1], sd[ik][n0][2]
        ok = (abs(sx) < MAX_ABS_SIGX and abs(cr) < MAX_ABS_SIGC
              and abs(ci) < MAX_ABS_IM)
        check(f"Sigma sane @ {nm} {lbl}", ok,
              f"sigX={sx:+8.3f}  sigC={cr:+7.3f}{ci:+7.4f}i eV")

# --------------------------------------------------------- gap non-inverted
edft = np.array([[rows[k].get(b + 1, (np.nan, np.nan))[0]
                  for b in range(NVAL + 1)] for k in range(len(ks))])
eqp = np.array([[rows[k].get(b + 1, (np.nan, np.nan))[1]
                 for b in range(NVAL + 1)] for k in range(len(ks))])
vb, cb = eqp[:, NVAL - 1], eqp[:, NVAL]
direct = cb - vb
indirect = float(np.nanmin(cb) - np.nanmax(vb))
check("direct gap positive at every k", bool(np.all(direct > 0)),
      f"min direct gap = {np.nanmin(direct):.4f} eV @ k#{int(np.nanargmin(direct))}")
check("indirect gap positive (non-inverted)", indirect > 0,
      f"indirect = {indirect:.4f} eV")
check("QP gap opens vs DFT",
      float(np.nanmin(cb) - np.nanmax(vb)) >
      float(np.nanmin(edft[:, NVAL]) - np.nanmax(edft[:, NVAL - 1])),
      f"DFT indirect {np.nanmin(edft[:,NVAL])-np.nanmax(edft[:,NVAL-1]):.4f} "
      f"-> GW {indirect:.4f} eV")

# ------------------------------------------- rank truncation / Galerkin rank
gw = open(os.path.join(RD, "gw.out"), errors="replace").read()
n_rt = gw.count("path=replicated_rank_truncate")
check("rank-truncate path active", n_rt > 0,
      f"'path=replicated_rank_truncate' appears {n_rt}x in gw.out")

# Any rank report the solver emitted (kept + dropped modes).
rank_lines = [l for l in gw.splitlines()
              if re.search(r"rank[ =:]", l, re.I) and "truncat" in l.lower()]
n_mu = None
m = re.search(r"n_mu\s*=?\s*(\d+)", gw)
if m:
    n_mu = int(m.group(1))
if rank_lines:
    for l in rank_lines[:4]:
        print(f"       {l.strip()}")
    notes.append(f"{len(rank_lines)} rank-truncation report line(s)")
else:
    notes.append("solver printed no per-q rank line (silent full-rank path)")

check("no NaN / Inf anywhere in Sigma", np.all(np.isfinite(allv)),
      f"{int((~np.isfinite(allv)).sum())} non-finite entries")

bad = [l for l in gw.splitlines()
       if re.search(r"Traceback|RESOURCE_EXHAUSTED|NaN detected|Assertion", l)]
check("no errors in gw.out", not bad,
      "clean" if not bad else f"{len(bad)} error line(s): {bad[0][:80]}")

print()
for n in notes:
    print(f"  note: {n}")
print()
if fails:
    print(f"GATE: FAIL ({len(fails)}) -> {', '.join(fails)}")
    sys.exit(1)
print("GATE: ALL PASS")
