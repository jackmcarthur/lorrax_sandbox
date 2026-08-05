#!/usr/bin/env python3
"""Σ_xc Breit comparison: standard vs +Breit(screened/unscreened) for CrI3.

Reads a run's ``breit_comparison.dat`` (per k, band: sigSX, sigCOH, std,
SigB_scr, SigB_unscr, ...) + the WFN DFT energies, identifies the band-edge
"key values" (VBM/CBM), and emits a table + a per-band plot.  The physically
interesting number is the Breit *gap* correction ΔΣ_xc(CBM) − ΔΣ_xc(VBM).

Usage: python3 analyze_breit.py <run_dir>   (run_dir holds breit_comparison.dat,
WFN.h5, cohsex.in).  Self-checks at the bottom.
"""
import os, sys, re
import numpy as np

EV = 1.0


def read_cohsex_in(run):
    cfg = {}
    with open(os.path.join(run, "cohsex.in")) as f:
        for ln in f:
            ln = ln.split("#")[0].strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def load(run):
    dat = np.loadtxt(os.path.join(run, "breit_comparison.dat"))
    # cols: k n sigSX sigCOH std SigB_scr SigB_unscr std+Bscr std+Bunscr
    cfg = read_cohsex_in(run)
    nval = int(cfg.get("nval", 8))
    # nelec from WFN occupations (robust); band index of VBM = nelec-1.
    import h5py
    with h5py.File(os.path.join(run, "WFN.h5"), "r") as fh:
        occ = np.asarray(fh["/mf_header/kpoints/occ"][()])[0]  # (nk, nb)
        el = np.asarray(fh["/mf_header/kpoints/el"][()])[0] * 13.605693  # eV
    nelec = int(round(occ[0].sum()))
    band0 = nelec - nval                 # .dat n=0 → this absolute band
    vbm_abs, cbm_abs = nelec - 1, nelec   # 0-indexed absolute band of VBM/CBM
    return dat, band0, vbm_abs, cbm_abs, el, nelec


def table(run):
    dat, band0, vbm_abs, cbm_abs, el, nelec = load(run)
    k = dat[:, 0].astype(int); n = dat[:, 1].astype(int)
    std = dat[:, 4]; bscr = dat[:, 5]; bbar = dat[:, 6]
    vbm_n, cbm_n = vbm_abs - band0, cbm_abs - band0
    print(f"# nelec={nelec}, .dat n=0 → band {band0}; VBM=band{vbm_abs} (n={vbm_n}), "
          f"CBM=band{cbm_abs} (n={cbm_n})")
    print(f"# {'k':>3} {'n':>3} {'band':>4} {'Σxc_std':>10} {'+Breit_scr':>11} "
          f"{'+Breit_unscr':>12} {'ΔB_scr[meV]':>11} {'ΔB_unscr[meV]':>13} {'edge':>5}")
    # k=0 window around the gap
    m = (k == 0) & (n >= vbm_n - 3) & (n <= cbm_n + 3)
    for i in np.where(m)[0]:
        edge = "VBM" if n[i] == vbm_n else ("CBM" if n[i] == cbm_n else "")
        print(f"  {k[i]:3d} {n[i]:3d} {band0+n[i]:4d} {std[i]:10.4f} "
              f"{std[i]+bscr[i]:11.4f} {std[i]+bbar[i]:12.4f} "
              f"{1e3*bscr[i]:11.2f} {1e3*bbar[i]:13.2f} {edge:>5}")
    # Breit gap correction (k=0): ΔΣ_xc(CBM) − ΔΣ_xc(VBM)
    iv = np.where((k == 0) & (n == vbm_n))[0]
    ic = np.where((k == 0) & (n == cbm_n))[0]
    if len(iv) and len(ic):
        dgap_scr = 1e3 * (bscr[ic[0]] - bscr[iv[0]])
        dgap_bar = 1e3 * (bbar[ic[0]] - bbar[iv[0]])
        print(f"\n# Breit QP-gap correction @k=0 (ΔΣ_xc(CBM)−ΔΣ_xc(VBM)):")
        print(f"#   screened   : {dgap_scr:+.2f} meV")
        print(f"#   unscreened : {dgap_bar:+.2f} meV")
        print(f"#   screening effect on Breit: {dgap_scr - dgap_bar:+.3f} meV")
    return dat, band0, vbm_n, cbm_n


def plot(run, dat, band0, vbm_n, cbm_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = dat[:, 0].astype(int); n = dat[:, 1].astype(int)
    bscr = 1e3 * dat[:, 5]; bbar = 1e3 * dat[:, 6]
    m = k == 0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(n[m], bscr[m], "o-", label="Breit screened", ms=4)
    ax.plot(n[m], bbar[m], "x--", label="Breit unscreened", ms=5)
    ax.axvline(vbm_n, color="g", ls=":", alpha=.6, label="VBM")
    ax.axvline(cbm_n, color="r", ls=":", alpha=.6, label="CBM")
    ax.set_xlabel("σ-window band index n"); ax.set_ylabel("ΔΣ_xc Breit (meV)")
    ax.set_title("CrI₃ FM: Breit contribution to Σ_xc (k=0)")
    ax.legend(); ax.grid(alpha=.3)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breit_per_band.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\n# plot → {out}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    dat, band0, vbm_n, cbm_n = table(run)
    try:
        plot(run, dat, band0, vbm_n, cbm_n)
    except Exception as e:
        print(f"# plot skipped: {e}")
    # self-check: std + Breit columns are consistent with stored totals
    assert np.allclose(dat[:, 4] + dat[:, 5], dat[:, 7], atol=1e-4), "std+Bscr mismatch"
    assert np.allclose(dat[:, 4] + dat[:, 6], dat[:, 8], atol=1e-4), "std+Bunscr mismatch"
    print("# self-check OK (std+Breit == stored totals)")
