"""Merge the two sweep halves into one markdown table (pure JSON -> text).

Also re-reads the saved interpolated paths so the smoothness column can be
reported for the LOWEST conduction band alone.  ``d2`` over all 8 sorted bands
is dominated by the sorting kinks where bands cross — real, but not a statement
about the interpolant.  The lowest band of the window has no band below it to
swap with, so its |2nd difference| is a clean ringing measure.
"""
import json
import os
import sys

import numpy as np

REP = os.path.dirname(os.path.abspath(__file__))
rows = []
paths = {}
for d in (REP, os.path.join(REP, "sweep_hi")):
    p = os.path.join(d, "window_sweep.json")
    if os.path.exists(p):
        rows += json.load(open(p))
    z = os.path.join(d, "window_sweep_paths.npz")
    if os.path.exists(z):
        paths.update({k: v for k, v in np.load(z).items() if k.endswith("_path")
                      and k[0].isdigit()})
rows = {int(r["nb"]): r for r in rows}
rows = [rows[k] for k in sorted(rows)]
for r in rows:
    for tag in ("DFT", "QP"):
        E = paths.get(f"{r['nb']}_{tag}_path")
        if E is not None and E.ndim == 2 and E.shape[0] > 2:
            d2b = np.abs(E[2:, 0] - 2 * E[1:-1, 0] + E[:-2, 0]).max() * 1e3
            r[f"{tag}_d2b0_meV"] = float(d2b)

NK = 144
NMU = 2412
NS = 2


def f(r, k, fmt="{:.3f}", dash="—"):
    v = r.get(k)
    if v is None:
        return dash
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


hdr = ("| nb | window | nk·nb | SVD rank | guards | ortho | fH_R GiB/dev | "
       "DFT max\\|Δε_c\\| | DFT min-sval | QP max\\|Δε_c\\| | QP min-sval | "
       "recon (QP) | max\\|2nd-diff\\| CB1 | verdict |")
sep = "|" + "---|" * 14
out = [hdr, sep]
for r in rows:
    if r.get("status") != "ok":
        out.append(f"| {r['nb']} | | | {r.get('rank','')} | | | | | | | | | | "
                   f"**{r.get('status','error')}** |")
        continue
    # PASS = the driver's own two gate metrics on DFT energies, plus the
    # ORDER-INSENSITIVE recon on QP energies.  The QP gate value itself is not
    # used as a criterion: it compares the htransform's ascending-QP order
    # against the restart's DFT band order, so a QP level crossing inside the
    # 8-band window shows up as a fixed 57.902 meV permutation offset at EVERY
    # window (identical at nb = 16, 20, 24) while recon, which sorts both sides,
    # is 0.000 meV.
    both = (bool(r.get("DFT_passed"))
            and float(r.get("QP_recon_meV", 1e9)) < 50.0)
    out.append(
        f"| **{r['nb']}** | [{r['b_start']}, {r['b_end']}) | {NK*r['nb']} | "
        f"{r['rank']} | {r.get('n_guard','—')} | {f(r,'ortho','{:.1e}')} | "
        f"{f(r,'fH_R_GiB','{:.2f}')} | {f(r,'DFT_gate_meV','{:.3f}')} | "
        f"{f(r,'DFT_min_sval','{:.4f}')} | {f(r,'QP_gate_meV','{:.3f}')} | "
        f"{f(r,'QP_min_sval','{:.4f}')} | {f(r,'QP_recon_meV','{:.3f}')} | "
        f"{f(r,'QP_d2b0_meV','{:.1f}') if r.get('QP_d2b0_meV') else f(r,'QP_d2_meV','{:.0f}')} | {'**PASS**' if both else 'FAIL'} |")

print(f"capacity (nominal): nb < nspinor*n_mu/nk = {NS*NMU/NK:.2f}")
print("\n".join(out))
print()
for r in rows:
    if r.get("status") == "ok" and r.get("rank"):
        eff = r["rank"] / NK
        print(f"  nb={r['nb']:>3}  rank={r['rank']:>5}  "
              f"{'RANK-DEFICIENT' if r['rank'] < NK*r['nb'] else 'full rank'}"
              f"  (rank/nk = {eff:.2f})")

if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(out) + "\n")
    print(f"wrote {sys.argv[1]}")
