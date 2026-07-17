#!/usr/bin/env python3
"""Predict screening-window degeneracy-fix impact on each regression fixture.

For each fixture: read the WFN energies (all spins/k, Ry), the .in band config,
compute band edges (b1,b2,b3,b4), and the largest degeneracy-closed boundary
<= nband (tol=1e-6 Ry). Report whether the fix drops bands and whether it stays
>= b3 (sigma top) — i.e. whether the golden gate is exposed.
"""
import os, glob, re
import numpy as np, h5py

LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
REG = os.path.join(LROOT, "tests/regression")
RY = 13.6056980659
TOL = 1e-6


def parse_in(path):
    cfg = {}
    for ln in open(path):
        m = re.match(r"\s*(nval|ncond|nband)\s*=\s*(\d+)", ln)
        if m:
            cfg[m.group(1)] = int(m.group(2))
    return cfg


def closed_down(e, b_hi, tol=TOL):
    nb = e.shape[1]
    b = int(b_hi)
    while b > 0:
        if b >= nb:
            b -= 1
            continue
        if float(np.min(e[:, b] - e[:, b - 1])) > tol:
            return b
        b -= 1
    return b


FIX = {
    "cohsex_debug": "cohsex_test_ctsp_compare.in",
    "si_cohsex_debug": "cohsex_si_test.in",
    "gnppm_debug": "cohsex_ibz_test.in",
    "bispinor_debug": "bispinor_test.in",
}

for fx, inp in FIX.items():
    d = os.path.join(REG, fx)
    ipath = os.path.join(d, inp)
    if not os.path.exists(ipath):
        print(f"### {fx}: input {inp} MISSING"); continue
    cfg = parse_in(ipath)
    # WFN path
    wfnp = None
    for cand in ("WFN.h5",):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            wfnp = os.path.realpath(p); break
    if wfnp is None:
        gg = glob.glob(os.path.join(d, "*WFN*.h5"))
        wfnp = os.path.realpath(gg[0]) if gg else None
    if wfnp is None:
        print(f"### {fx}: no WFN.h5"); continue
    with h5py.File(wfnp, "r") as f:
        el = np.asarray(f["mf_header/kpoints/el"][:])   # (nspin, nk, nb)
        ifmax = np.asarray(f["mf_header/kpoints/ifmax"][:])
    nspin, nk, nbf = el.shape
    nelec = int(np.max(ifmax))
    e = el.reshape(-1, nbf)   # (nspin*nk, nb)
    nval, ncond, nband = cfg["nval"], cfg["ncond"], cfg["nband"]
    b1 = nelec - nval; b2 = nelec; b3 = nelec + ncond; b4 = nband
    bc = closed_down(e, b4)
    gap_at_b4 = float(np.min(e[:, b4] - e[:, b4 - 1])) if b4 < nbf else float("nan")
    print(f"### {fx}: nelec={nelec} nbands_file={nbf}  nval={nval} ncond={ncond} nband={nband}")
    print(f"    b1={b1} b2={b2} b3(sigma_top)={b3} b4(screen_top)={b4}  b4>b3? {b4>b3}")
    print(f"    gap at boundary b4={b4}: {gap_at_b4*RY*1e3 if gap_at_b4==gap_at_b4 else float('nan'):.3f} meV"
          f"  -> boundary {'CLOSED' if (gap_at_b4==gap_at_b4 and gap_at_b4>TOL) else 'CUT/edge'}")
    print(f"    closed_down(b4)={bc}   dropped_if_unclamped={b4-bc}")
    if bc < b3:
        print(f"    => closed boundary {bc} < b3 {b3}: CLAMP at b3 -> b4 stays {max(bc,b3)} (NO screening change; warn b3 exposure)")
    elif bc < b4:
        print(f"    => FIX ACTIVATES: b4 {b4} -> {bc} (drops {b4-bc} bands); GOLDEN GATE EXPOSED")
    else:
        print(f"    => boundary already closed; NO CHANGE")
    print()
