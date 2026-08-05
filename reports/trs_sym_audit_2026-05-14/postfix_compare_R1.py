#!/usr/bin/env python3
"""R1 verification: compare run_A_ibz_postfix vs run_B_fullbz, per-k.

Computes max |Δx_bare| and max |Δ(sex_0+coh_0)| per k-point and globally.
Bit-equality gate: max |Δ| ≤ 1e-10 eV.

Header layout (current LORRAX, post-Vh column addition):
k, n, E_dft, Edft-Ef, kin_ion, V_H, x_bare, x_head, sex_0, coh_0, sex_head, coh_head, eqp0, eqp1
"""
from __future__ import annotations
import numpy as np

# Column indices (0-based) for the current header layout. Per
# skills/compare/SKILL.md §2c, the canonical example has sex_0/coh_0 at
# 5/6 and x_bare at 7. The CURRENT file has an extra V_H column inserted,
# so the offsets shift by +1: x_bare=6, x_head=7, sex_0=8, coh_0=9.
# We parse from header names explicitly to be robust.
def load_sigma_freq_debug(path):
    """Returns dict of {(k, n): {field: value}}. Parses header to locate columns."""
    rows = []
    cols = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("# k") or (s.startswith("#") and "x_bare" in s):
                # Header line, parse columns
                # Strip the leading "# " then split by tab
                hdr = s.lstrip("#").strip()
                cols = [c.strip() for c in hdr.split("\t") if c.strip()]
                continue
            if s.startswith("#") or s.startswith("k-point"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(p) for p in parts]
                rows.append(vals)
            except ValueError:
                continue
    assert cols is not None, "Could not find header"
    arr = np.array(rows)
    # Build per-(k, n) dict
    data = {}
    for r in arr:
        k = int(round(r[0]))
        n = int(round(r[1]))
        rec = {name: r[i] for i, name in enumerate(cols)}
        data[(k, n)] = rec
    return data, cols

def main():
    base = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14"
    pa, cols_a = load_sigma_freq_debug(f"{base}/run_A_ibz_postfix/sigma_freq_debug.dat")
    pb, cols_b = load_sigma_freq_debug(f"{base}/run_B_fullbz/sigma_freq_debug.dat")

    print(f"columns A: {cols_a}")
    print(f"columns B: {cols_b}")
    assert cols_a == cols_b
    keys = sorted(set(pa.keys()) & set(pb.keys()))
    print(f"Matched (k, n) records: {len(keys)}")
    ks = sorted({k for (k, n) in keys})
    print(f"k-points: {ks}")

    print("\nPer-k max |Δ| in eV:")
    print(f"  {'k':>3} {'|Δx_bare|':>12} {'|Δsex_0|':>12} {'|Δcoh_0|':>12} {'|Δ(sex_0+coh_0)|':>18}  (n at max x_bare)")
    fields = ["x_bare", "sex_0", "coh_0"]
    gmax = {"x_bare": (0.0, None), "sex_combined": (0.0, None), "sex_0": (0.0, None), "coh_0": (0.0, None)}
    for k in ks:
        rows = [(n, pa[(k, n)], pb[(k, n)]) for (kk, n) in keys if kk == k]
        per = {}
        for f in fields:
            diffs = np.array([abs(a[f] - b[f]) for n, a, b in rows])
            per[f] = (diffs.max(), int(rows[int(np.argmax(diffs))][0]))
            if per[f][0] > gmax[f][0]:
                gmax[f] = (per[f][0], (k, per[f][1]))
        # sex_0 + coh_0
        diffs_sc = np.array([abs((a["sex_0"] + a["coh_0"]) - (b["sex_0"] + b["coh_0"])) for n, a, b in rows])
        max_sc = diffs_sc.max()
        n_sc = int(rows[int(np.argmax(diffs_sc))][0])
        if max_sc > gmax["sex_combined"][0]:
            gmax["sex_combined"] = (max_sc, (k, n_sc))
        print(f"  {k:>3} {per['x_bare'][0]:>12.3e} {per['sex_0'][0]:>12.3e} {per['coh_0'][0]:>12.3e} {max_sc:>18.3e}  (n={per['x_bare'][1]})")

    print("\nGlobal max |Δ| in eV:")
    print(f"  x_bare         : {gmax['x_bare'][0]:.6e}  at (k,n)={gmax['x_bare'][1]}")
    print(f"  sex_0          : {gmax['sex_0'][0]:.6e}  at (k,n)={gmax['sex_0'][1]}")
    print(f"  coh_0          : {gmax['coh_0'][0]:.6e}  at (k,n)={gmax['coh_0'][1]}")
    print(f"  sex_0+coh_0    : {gmax['sex_combined'][0]:.6e}  at (k,n)={gmax['sex_combined'][1]}")

    GATE = 1e-10
    pass_x = gmax["x_bare"][0] <= GATE
    pass_sc = gmax["sex_combined"][0] <= GATE
    print(f"\nR1 GATE (≤ {GATE} eV):  x_bare {'PASS' if pass_x else 'FAIL'}   sex_0+coh_0 {'PASS' if pass_sc else 'FAIL'}")
    # Soft floor: 1e-6 eV (criterion in user message for the patch-issue threshold)
    SOFT = 1e-6
    pass_x_soft = gmax["x_bare"][0] <= SOFT
    print(f"R1 SOFT FLOOR (≤ {SOFT} eV):  x_bare {'PASS' if pass_x_soft else 'FAIL'}")

if __name__ == "__main__":
    main()
