"""Compare Σ_X(LORRAX) vs Σ_X(BGW) for the no-SOC Si 4x4x4 run, side-by-side
with the original SOC run."""
import re, sys
import numpy as np

def parse_sigma_hp(path, n_per_k=16):
    """Yield (ik, kvec, list of (n, X, sx_minus_x, ch, sig, eqp0)) per k."""
    blocks = []
    with open(path) as f:
        text = f.read()
    # Split on "k = " block headers
    for m in re.finditer(
        r"k\s*=\s*([-\d. ]+?)\s*ik\s*=\s*(\d+)\s*spin\s*=\s*1\s*\n\s*n\s+Emf.+\n((?:\s*\d+\s+[-\d. ]+\n)+)",
        text):
        kvec = tuple(float(x) for x in m.group(1).split())
        ik = int(m.group(2))
        rows = []
        for line in m.group(3).strip().split("\n"):
            cols = line.split()
            n = int(cols[0])
            Emf = float(cols[1])
            X = float(cols[3])
            rows.append((n, Emf, X))
        blocks.append((ik, kvec, rows))
    return blocks

def parse_lorrax_eqp(path):
    """Return dict ik -> dict n -> sigma_x (eV).  eqp0.dat columns:
    spin n  E_DFT(eV)  E_QP0(eV).  Sig_X is implicitly E_QP0 - E_DFT - V_xc + KIH"""
    # eqp0 doesn't directly give Sigma_X — read sigma debug file or stdout.
    pass

def parse_lorrax_stdout(path):
    """Get bare Σ_X values printed by LORRAX (k=0 only)."""
    with open(path) as f:
        for line in f:
            if "Bare Σ_X diagonal (eV), k=0" in line:
                parts = line.split(":", 1)[1]
                vals = [float(x) for x in parts.split()]
                return vals
    return None

def parse_lorrax_eqp0(path):
    """eqp0.dat: header line (kvec, nb), then nb lines: spin band E_DFT E_QP0."""
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    out = {}
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if len(head) >= 4:
            kvec = tuple(float(x) for x in head[:3])
            nb = int(head[3])
            rows = []
            for j in range(1, nb + 1):
                cols = lines[i + j].split()
                rows.append((int(cols[1]), float(cols[2]), float(cols[3])))
            out[kvec] = rows
            i += 1 + nb
        else:
            i += 1
    return out

# Load both runs
runs = {
    "soc_on  (00)": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band",
    "soc_off (06)": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc",
}

soc_paths = {
    "soc_on  (00)": (
        "00_bgw_cohsex/sigma_hp.log",
        "D_lorrax_xonly_overlay/run.out",
    ),
    "soc_off (06)": (
        "D_bgw_cohsex/sigma_hp.log",
        "D_lorrax_xonly_overlay/gw.out",
    ),
}

import os.path
print(f"{'run':18s} {'k':36s} {'n':>3s} {'X_BGW':>10s} {'X_LOR':>10s} {'Δ(meV)':>10s}")
print("-" * 90)
for tag, root in runs.items():
    bgw_path = os.path.join(root, soc_paths[tag][0])
    lor_path = os.path.join(root, soc_paths[tag][1])
    blocks = parse_sigma_hp(bgw_path)
    lor_vals = parse_lorrax_stdout(lor_path)
    if lor_vals is None:
        print(f"  no LORRAX stdout in {lor_path}")
        continue
    # k=0 block
    ik0, kvec0, rows0 = blocks[0]
    print()
    for i, (n, Emf, X_bgw) in enumerate(rows0):
        if i >= len(lor_vals):
            break
        X_lor = lor_vals[i]
        diff_meV = (X_lor - X_bgw) * 1000.0
        print(f"{tag:18s} {str(kvec0):36s} {n:>3d} {X_bgw:>10.4f} {X_lor:>10.4f} {diff_meV:>+10.2f}")
