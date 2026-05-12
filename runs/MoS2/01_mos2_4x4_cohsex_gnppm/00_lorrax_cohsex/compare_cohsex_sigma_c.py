#!/usr/bin/env python3
"""Compare LORRAX COHSEX sigma vs BGW sigma_hp.log for MoS2 4x4.

LORRAX eqp0.dat (legacy sigma_diag format) → sigSX, sigCOH per (k, n)
BGW sigma_hp.log → X, SX-X, CH per (k, n)

Comparison:
  LORRAX sigSX                ≡ BGW (X + SX-X)            [screened exchange total]
  LORRAX sigCOH               ≡ BGW CH                    [Coulomb hole]
  LORRAX sigC = sigSX + sigCOH − BGW X = BGW (SX-X + CH)  [correlation only]

Band convention: BGW is 1-indexed, LORRAX is 0-indexed.  Physical band n_phys = LORRAX n + 1 = BGW n.
"""
from __future__ import annotations
import re, sys
import numpy as np

LOR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm/00_lorrax_cohsex/eqp0.dat"
BGW = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm/00_bgw_cohsex/sigma_hp.log"


def parse_lorrax(path: str) -> dict:
    """LORRAX sigma_diag-formatted output.
    Returns out[(ik, n_phys)] = {sigSX, sigCOH, sigTOT, VH}, with n_phys = LORRAX_n + 1."""
    out = {}
    cur_k = -1
    re_k = re.compile(r"^k-point\s+(\d+):")
    re_n = re.compile(r"^n=\s*(\d+)\s+sigSX=\s*([-+0-9.eE]+)\s+sigCOH=\s*([-+0-9.eE]+)\s+sigTOT=\s*([-+0-9.eE]+)\s+VH=\s*([-+0-9.eE]+)")
    with open(path) as f:
        for line in f:
            mk = re_k.match(line.strip())
            if mk:
                cur_k = int(mk.group(1))
                continue
            mn = re_n.match(line.strip())
            if mn and cur_k >= 0:
                n_phys = int(mn.group(1)) + 1
                out[(cur_k, n_phys)] = dict(
                    sigSX=float(mn.group(2)),
                    sigCOH=float(mn.group(3)),
                    sigTOT=float(mn.group(4)),
                    VH=float(mn.group(5)),
                )
    return out


def parse_bgw_sigma_hp(path: str) -> dict:
    """BGW sigma_hp.log. Returns out[(ik, n_phys)] = {X, SXmX, CH, Sig, KIH, Eqp0, ...}.
    ik is BGW's ik (1-indexed). n_phys = BGW band index (1-indexed)."""
    out = {}
    ik = None
    kcrys = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            try:
                out[(ik, n)] = dict(
                    Emf=float(p[1]), Eo=float(p[2]),
                    X=float(p[3]), SXmX=float(p[4]), CH=float(p[5]),
                    Sig=float(p[6]), KIH=float(p[7]),
                    Eqp0=float(p[8]), Eqp1=float(p[9]),
                    CHp=float(p[10]), Sigp=float(p[11]),
                    Eqp0p=float(p[12]), Eqp1p=float(p[13]),
                    Znk=float(p[14]),
                    kcrys=kcrys,
                )
            except (ValueError, IndexError):
                pass
    return out


def main():
    lor = parse_lorrax(LOR)
    bgw = parse_bgw_sigma_hp(BGW)
    print(f"LORRAX: {len(lor)} (k, n) entries")
    print(f"BGW   : {len(bgw)} (k, n) entries")

    # LORRAX k is 0-indexed (full BZ), BGW ik is 1-indexed but only over irreducible k.
    # BGW sigma_hp lists 12 k blocks (irreducible) for 4x4 MoS2.
    # Match by k-crystal coords using BGW's kcrys.
    # LORRAX uses full BZ; we need a k-mapping (LORRAX_k → BGW_ik).
    # For comparison, use BGW's k order: each (BGW_ik, n) in BGW, find matching LORRAX (k_lor, n).
    # Build LORRAX k-index → k-crys map from WFN.h5 (use sym.unfolded_kpts).
    import h5py
    with h5py.File("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm/00_lorrax_cohsex/WFN.h5") as h5:
        # LORRAX prints k-point index in full-BZ order; build the map
        kpts_red = np.array(h5["mf_header/kpoints/rk"])  # (3, nk_red) — BGW reduced k
        ntran = int(h5["mf_header/symmetry/ntran"][()])
        symops = np.array(h5["mf_header/symmetry/mtrx"])  # (3, 3, 48) - sym ops
    # Build full-BZ k list by applying syms to reduced k (this is what LORRAX does internally)
    # Simpler: LORRAX 4x4 mesh enumerated in row-major (kx, ky) order:
    kvec_lor = []
    nkx = nky = 4; nkz = 1
    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                kvec_lor.append((ikx/nkx, iky/nky, ikz/nkz))
    kvec_lor = np.array(kvec_lor)
    print(f"\nLORRAX k order (first 4): {kvec_lor[:4].tolist()}")

    # Build BGW kcrys list (irreducible) in order of ik
    bgw_ks = {}
    for (ik, n), v in bgw.items():
        bgw_ks[ik] = v['kcrys']
    print(f"BGW irreducible k-list:")
    for ik in sorted(bgw_ks.keys()):
        print(f"  ik={ik}: k={bgw_ks[ik]}")

    # Map BGW ik → LORRAX k index by finding closest match in kvec_lor (mod 1)
    bgw_to_lor = {}
    for ik, kbgw in bgw_ks.items():
        kb = np.array(kbgw)
        dists = []
        for i, kl in enumerate(kvec_lor):
            d = min(np.linalg.norm(((kl - kb) + 0.5) % 1 - 0.5),
                    np.linalg.norm(((kb - kl) + 0.5) % 1 - 0.5))
            dists.append(d)
        ilor = int(np.argmin(dists))
        bgw_to_lor[ik] = (ilor, dists[ilor])
    print(f"\nBGW ik → LORRAX k_index mapping:")
    for ik in sorted(bgw_to_lor.keys()):
        ilor, dist = bgw_to_lor[ik]
        print(f"  BGW ik={ik} (k={bgw_ks[ik]}) → LORRAX k={ilor} (dist={dist:.4f})")

    # Now compare per (BGW_ik, band).
    # NOTE: BGW CH (col 5) is the FINITE-band sum up to number_bands.  BGW CH'
    # (col 10, ``CHp``) adds the analytic static-remainder over the un-summed
    # high-band tail (the ``exact_static_ch=0`` default still emits CH' via
    # the truncation formula — see BGW's Sigma/mtxel_cor.f90).  LORRAX's
    # ``sigCOH`` includes the same static-remainder physics by construction
    # (the G_RI build_G sum is over the full σ window and the resolvent
    # uses the right asymptotic).  So the cross-code comparable quantity
    # is ``sigCOH`` vs BGW **CH'**, not CH.  Comparing to the unprimed CH
    # systematically over-reports a several-eV "disagreement" that is
    # really the truncation correction.
    print(f"\n{'='*135}")
    print(f"COHSEX comparison: LORRAX vs BGW  (CH primed = static-remainder-corrected)")
    print(f"{'='*135}")
    print(f"{'ik':>3} {'n':>3} | {'LOR_sigSX':>10} {'BGW_X+SXmX':>11} {'Δ(SX)':>8} | {'LOR_sigCOH':>11} {'BGW_CHp':>10} {'Δ(CHp)':>8} | {'LOR_Cor':>10} {'BGW_Corp':>10} {'Δ(Corp)':>8}")
    print("-" * 135)

    rows = []
    for (ik, n) in sorted(bgw.keys()):
        if ik not in bgw_to_lor:
            continue
        ilor, _ = bgw_to_lor[ik]
        if (ilor, n) not in lor:
            continue
        L = lor[(ilor, n)]
        B = bgw[(ik, n)]
        # LORRAX sigSX = BGW (X + SX-X)
        bgw_sx_total = B['X'] + B['SXmX']
        d_sx = L['sigSX'] - bgw_sx_total
        # LORRAX sigCOH ≡ BGW CHp (primed: with static remainder)
        d_chp = L['sigCOH'] - B['CHp']
        lor_cor = L['sigSX'] - B['X'] + L['sigCOH']
        bgw_corp = B['SXmX'] + B['CHp']
        d_corp = lor_cor - bgw_corp
        rows.append((ik, n, L['sigSX'], bgw_sx_total, d_sx,
                     L['sigCOH'], B['CHp'], d_chp,
                     lor_cor, bgw_corp, d_corp))
        print(f"{ik:>3} {n:>3} | {L['sigSX']:>10.4f} {bgw_sx_total:>11.4f} {d_sx:>+8.4f} | {L['sigCOH']:>11.4f} {B['CHp']:>10.4f} {d_chp:>+8.4f} | {lor_cor:>10.4f} {bgw_corp:>10.4f} {d_corp:>+8.4f}")

    if rows:
        arr = np.array([r[2:] for r in rows])
        d_sx_arr = arr[:, 2]
        d_chp_arr = arr[:, 5]
        d_corp_arr = arr[:, 8]
        print("-" * 135)
        print(f"  MAE  Δ(SX)    = {np.abs(d_sx_arr).mean():>7.4f} eV     max|Δ| = {np.abs(d_sx_arr).max():>7.4f} eV")
        print(f"  MAE  Δ(CHp)   = {np.abs(d_chp_arr).mean():>7.4f} eV     max|Δ| = {np.abs(d_chp_arr).max():>7.4f} eV")
        print(f"  MAE  Δ(Corp)  = {np.abs(d_corp_arr).mean():>7.4f} eV     max|Δ| = {np.abs(d_corp_arr).max():>7.4f} eV")


if __name__ == "__main__":
    main()
