"""Stage 4(c): is the far-band clamp still doing harm out to band 80?

The Sigma_c(omega) tensor is tabulated on a finite grid
[sigma_omega_min_ev, sigma_omega_max_ev] measured from the Fermi reference.
Bands whose E_DFT falls outside that window cannot be evaluated on the grid;
``qsgw_utils`` clamps the evaluation energy to the grid edge
(``E_clamped = np.clip(E, omega_lo, omega_hi)``) and ``gw.scissor``
(``classify_bands_in_grid`` / ``fit_scissor``) offers an affine extrapolation
instead -- but only when ``sigma_at_dft_extrapolate`` is set, and it defaults
to False.

The clamp's fingerprint is unmistakable: every out-of-grid band receives the
SAME Sigma (the one at the grid edge), so the QP shift dE(n) = E_qp - E_dft
goes FLAT in n beyond the crossing band, and the QP dispersion within those
bands collapses onto the DFT dispersion shifted by a constant.

This script measures, from the run's own eqp1.dat + sigma_diag.dat:
  * where the omega-grid edge falls in band index,
  * how many of the QP-corrected bands (1..b_qp) are out of grid,
  * the QP shift vs band index, and the flatness statistic
    std_k(dE) within each band, which collapses for clamped bands.

usage: python3 farband.py <rundir> [<rundir2> ...]
env:   FB_NVAL (26), FB_WMIN (-10), FB_WMAX (+10)
"""
import os
import re
import sys

import numpy as np

NVAL = int(os.environ.get("FB_NVAL", "26"))
WMIN = float(os.environ.get("FB_WMIN", "-10.0"))
WMAX = float(os.environ.get("FB_WMAX", "10.0"))


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


def grid_ref(rd, edft):
    """Fermi reference (eV) the omega grid is measured from.

    ``fermi_reference = midgap`` -> midpoint of the DFT gap.  Read from
    gw.out when the driver printed it, else recomputed here the same way.
    """
    p = os.path.join(rd, "gw.out")
    if os.path.exists(p):
        txt = open(p, errors="replace").read()
        m = re.search(r"[Ee][_ ]?[Ff]ermi[^-\d]*(-?\d+\.\d+)", txt)
        if m:
            return float(m.group(1)), "gw.out"
    vbm = np.nanmax(edft[:, NVAL - 1])
    cbm = np.nanmin(edft[:, NVAL])
    return 0.5 * (vbm + cbm), "recomputed midgap"


def analyse(rd):
    ks, rows = parse_eqp(os.path.join(rd, "eqp1.dat"))
    nk = len(ks)
    nb = max(max(r) for r in rows)
    edft = np.full((nk, nb), np.nan)
    eqp = np.full((nk, nb), np.nan)
    for k in range(nk):
        for b, (ed, eq) in rows[k].items():
            edft[k, b - 1] = ed
            eqp[k, b - 1] = eq

    ef, src = grid_ref(rd, edft)
    rel = edft - ef
    dE = eqp - edft

    print(f"=== {rd}")
    print(f"    nk={nk}  QP bands 1..{nb}   Fermi ref {ef:+.4f} eV ({src})")
    print(f"    omega grid [{WMIN:+.1f}, {WMAX:+.1f}] eV relative to it "
          f"= [{ef+WMIN:+.3f}, {ef+WMAX:+.3f}] eV absolute")

    # A band is "in grid" iff E_DFT-Ef is inside [WMIN, WMAX] at EVERY k
    # (classify_bands_in_grid semantics).
    in_all = np.all((rel >= WMIN) & (rel <= WMAX), axis=0)
    in_any = np.any((rel >= WMIN) & (rel <= WMAX), axis=0)
    n_in = int(in_all.sum())
    # The interesting edge is the ABOVE-gap one: the deep semicore bands are
    # out of grid at the bottom too, but nobody cares about their QP energy.
    # Walk up from the conduction edge to the first band that leaves the grid
    # and never comes back.
    first_out = None
    for b in range(NVAL, len(in_all)):
        if not in_all[b]:
            first_out = b + 1
            break
    print(f"    bands fully in grid : {n_in}/{nb}"
          + (f"   first out-of-grid band = {first_out}" if first_out else ""))
    print(f"    bands partly in grid: {int((in_any & ~in_all).sum())}"
          "  (clipped at some k only -- these get the scissor too)")

    # Flatness: std over k of the QP shift within a band.  Clamped bands
    # inherit a single Sigma, so their k-dispersion of dE collapses.
    print()
    print(f"    {'band':>5} {'E_dft-Ef (min..max)':>24} {'in?':>4} "
          f"{'<dE>':>8} {'std_k dE':>9}")
    marks = sorted(set([1, NVAL - 1, NVAL, NVAL + 1, NVAL + 4]
                       + list(range(30, nb + 1, 5)) + [nb]))
    for b in marks:
        if b < 1 or b > nb:
            continue
        r = rel[:, b - 1]
        d = dE[:, b - 1]
        flag = "yes" if in_all[b - 1] else ("part" if in_any[b - 1] else "NO")
        print(f"    {b:5d} {r.min():+10.3f} .. {r.max():+9.3f} {flag:>4} "
              f"{np.nanmean(d):+8.3f} {np.nanstd(d):9.4f}")

    stdk = np.nanstd(dE, axis=0)
    if first_out:
        io = first_out - 1
        below = stdk[max(NVAL, io - 10):io]
        above = stdk[io:]
        print()
        print(f"    conduction bands {max(NVAL,io-10)+1}..{io} (IN grid), "
              f"mean std_k(dE) = {np.nanmean(below):.4f} eV")
        print(f"    conduction bands {io+1}..{nb} (OUT of grid), "
              f"mean std_k(dE) = {np.nanmean(above):.4f} eV")
        print(f"    NOTE: a flat std_k is NOT the clamp signature here -- "
              f"one-shot Sigma(E_DFT) clamps only the omega ARGUMENT, and "
              f"Sigma_x(k,n) is exact and band-varying, so dE keeps "
              f"dispersing.  The clamp is measured by the wide-omega variant, "
              f"not by this statistic.")
    print()
    return dict(rd=rd, nb=nb, n_in=n_in, first_out=first_out, ef=ef,
                dE=dE, rel=rel, stdk=stdk, in_all=in_all)


if __name__ == "__main__":
    outs = [analyse(rd) for rd in sys.argv[1:]]
    if len(outs) > 1:
        a, b = outs[0], outs[1]
        nb = min(a["nb"], b["nb"])
        d = np.nanmean(a["dE"][:, :nb], axis=0) - np.nanmean(b["dE"][:, :nb], axis=0)
        print(f"=== {a['rd']}  minus  {b['rd']}")
        print(f"    max |d<dE>| over bands 1..{nb} = {np.nanmax(np.abs(d)):.4f} eV"
              f"  @ band {int(np.nanargmax(np.abs(d)))+1}")
        for lo, hi, nm in ((0, NVAL, "valence 1..26"),
                           (NVAL, min(60, nb), "cond 27..60"),
                           (min(60, nb), nb, f"far cond 61..{nb}")):
            if hi > lo:
                print(f"    mean |d<dE>| {nm:>16s}: "
                      f"{np.nanmean(np.abs(d[lo:hi])):.4f} eV   "
                      f"max {np.nanmax(np.abs(d[lo:hi])):.4f} eV")
