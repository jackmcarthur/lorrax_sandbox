#!/usr/bin/env python3
"""LORRAX ppm_invalid_mode (zero vs 2ry) delta table vs BGW invalid_gpp_mode (0 vs 2).

Companion to mode_diff_tables.py (BGW side). Parses the two LORRAX runs

    runs/Si/00_si_4x4x4_60band/03_lorrax_gnppm_invalidmode_zero
    runs/Si/00_si_4x4x4_60band/03_lorrax_gnppm_invalidmode_2ry

with the compare-skill sigma_freq_debug parser (header-driven current-format
variant, skills/compare/SKILL.md section 2c), builds the per-band
Delta sig_c(Edft) = (2ry - zero) table, and sets it against BGW's GN-line
(mode2 - mode0) deltas from mode_table_gn.dat.

Comparable quantity is the MODE-TO-MODE DELTA per band (not absolute Sigma_c):
LORRAX fits GN poles per ISDF centroid pair, BGW per plane-wave (G,G') pair, so
absolute Sigma_c differs; the invariant is how Sigma_c moves when the
invalid-pole treatment changes (see report.md section "How to validate").

k-point handling: LORRAX evaluates all 64 full-BZ k, BGW the 8 irreducible k.
LORRAX k are grouped into the 8 BGW stars by DFT-eigenvalue fingerprint
(LORRAX E_dft vs BGW Eo over the 16 window bands, both straight QE eigenvalues
in eV) — self-validating: every LORRAX k must match exactly one BGW ik and the
star sizes must sum to 64. Per-band deltas are then compared star-by-star
(all members of a star are symmetry-equivalent, so their diagonal Sigma deltas
should agree; the star spread is reported as a symmetry-consistency check).

Run on a compute node (container has numpy on PYTHONPATH):
    LORRAX_NGPU=1 lxrun python3 -u lorrax_mode_diff.py
"""
import re

import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band"
REP = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/bgw_invalid_mode_refs_2026-07-08"
RUN_ZERO = f"{BASE}/03_lorrax_gnppm_invalidmode_zero"
RUN_2RY = f"{BASE}/03_lorrax_gnppm_invalidmode_2ry"
BGW_MODE0_HP = f"{BASE}/01b_bgw_gn_mode0/sigma_hp.log"
NBAND_WIN = 16  # sigma window bands 1..16 (BGW band_index 1..16 = LORRAX nval=8/ncond=8)


# --- compare-skill parser (skills/compare/SKILL.md 2c, current format) -------
def parse_sigma_freq_debug_v2(path):
    """Header-driven parser for the current (2026-06+) named-column format.

    Returns (columns, data) where data maps (k, n_physical) -> {colname: float};
    n_physical = in-file 0-indexed n + 1. NaN values are kept as np.nan.
    """
    cols = None
    data = {}
    for line in open(path):
        s = line.strip()
        if s.startswith("#"):
            p = s.lstrip("#").split()
            if len(p) >= 3 and p[0] == "k" and p[1] == "n":
                cols = p[2:]
            continue
        if not s or cols is None:
            continue
        p = s.split()
        if len(p) != len(cols) + 2:
            continue
        try:
            k, n = int(p[0]), int(p[1])
        except ValueError:
            continue
        data[(k, n + 1)] = {
            c: (np.nan if v == "nan" else float(v)) for c, v in zip(cols, p[2:])
        }
    if cols is None:
        raise ValueError(f"{path}: no '# k n ...' header line found")
    return cols, data


def parse_sigma_hp_eo(path):
    """(ik) -> {n: Eo} from sigma_hp.log (col 2 of the 15-col band rows)."""
    out = {}
    ik = None
    for line in open(path):
        s = line.strip()
        m = re.match(
            r"k\s*=\s*[\d.Ee+-]+\s+[\d.Ee+-]+\s+[\d.Ee+-]+\s+ik\s*=\s*(\d+)", s)
        if m:
            ik = int(m.group(1))
            out.setdefault(ik, {})
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            out[ik][int(p[0])] = float(p[2])
    return out


def parse_mode_table_gn(path):
    """(ik, n) -> dict of Eqp0_m0/2/3, Corp_m0/2/3 from mode_table_gn.dat."""
    out = {}
    for line in open(path):
        if line.lstrip().startswith("#") or not line.strip():
            continue
        p = line.split()
        out[(int(p[0]), int(p[1]))] = {
            "Eqp0_m0": float(p[2]), "Eqp0_m2": float(p[3]), "Eqp0_m3": float(p[4]),
            "Corp_m0": float(p[5]), "Corp_m2": float(p[6]), "Corp_m3": float(p[7]),
        }
    return out


def grep_invalid_line(gw_out):
    for line in open(gw_out, errors="replace"):
        if "GN invalid modes:" in line:
            return line.strip()
    return "(no 'GN invalid modes' line found)"


def main():
    cols_z, dz = parse_sigma_freq_debug_v2(f"{RUN_ZERO}/sigma_freq_debug.dat")
    cols_t, dt = parse_sigma_freq_debug_v2(f"{RUN_2RY}/sigma_freq_debug.dat")
    assert cols_z == cols_t, (cols_z, cols_t)
    assert set(dz) == set(dt)
    print(f"columns: {cols_z}")
    nk = max(k for k, _ in dz) + 1
    nb = max(n for _, n in dz)
    print(f"parsed {len(dz)} (k,n) rows: nk={nk}, bands 1..{nb}")

    print("\n-- invalid-pole counts (gw.out) --")
    print(f"  zero: {grep_invalid_line(f'{RUN_ZERO}/gw.out')}")
    print(f"  2ry : {grep_invalid_line(f'{RUN_2RY}/gw.out')}")

    sc = "sig_c(Edft).Re"
    # Determinism guard: PPM-independent static columns must be bit-close,
    # otherwise the two runs' ISDF pipelines diverged and the delta is polluted.
    for col in ("x_bare", "kin_ion", "V_H"):
        if col in cols_z:
            d = max(abs(dz[key][col] - dt[key][col]) for key in dz)
            print(f"  determinism check max|d {col}| = {d:.3e} eV")

    # LORRAX deltas (meV), all k, window bands.
    dsig = np.full((nk, nb), np.nan)
    deqp0 = np.full((nk, nb), np.nan)
    edft = np.full((nk, nb), np.nan)
    for (k, n), row in dz.items():
        dsig[k, n - 1] = (dt[(k, n)][sc] - row[sc]) * 1000.0
        deqp0[k, n - 1] = (dt[(k, n)]["eqp0"] - row["eqp0"]) * 1000.0
        edft[k, n - 1] = row["E_dft"]
    n_nan = int(np.isnan(dsig[:, :NBAND_WIN]).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} NaN sig_c(Edft) entries in window bands "
              "(omega grid coverage)")

    # ---- star assignment by DFT-eigenvalue fingerprint ----------------------
    bgw_eo = parse_sigma_hp_eo(BGW_MODE0_HP)
    iks = sorted(bgw_eo)
    fp_bgw = np.array([[bgw_eo[ik][n] for n in range(1, NBAND_WIN + 1)]
                       for ik in iks])  # (8, 16)
    star_of_k = np.full(nk, -1, dtype=int)
    for k in range(nk):
        d = np.max(np.abs(fp_bgw - edft[k, :NBAND_WIN][None, :]), axis=1)
        j = int(np.argmin(d))
        if d[j] < 2e-3:  # 2 meV
            star_of_k[k] = j
    assert np.all(star_of_k >= 0), (
        f"unmatched LORRAX k: {np.where(star_of_k < 0)[0]}; "
        f"min residuals {[np.min(np.max(np.abs(fp_bgw - edft[k, :NBAND_WIN][None, :]), axis=1)) for k in np.where(star_of_k < 0)[0]]}")
    star_sizes = np.bincount(star_of_k, minlength=len(iks))
    print(f"\n  star sizes (BGW ik order {iks}): {star_sizes.tolist()} "
          f"(sum {star_sizes.sum()})")

    # per-star LORRAX delta mean + spread (symmetry consistency)
    lor_star = np.full((len(iks), NBAND_WIN), np.nan)
    lor_star_spread = np.full((len(iks), NBAND_WIN), np.nan)
    for j in range(len(iks)):
        m = star_of_k == j
        lor_star[j] = np.nanmean(dsig[m, :NBAND_WIN], axis=0)
        lor_star_spread[j] = (np.nanmax(dsig[m, :NBAND_WIN], axis=0)
                              - np.nanmin(dsig[m, :NBAND_WIN], axis=0))
    print(f"  max star spread of d_sigc (symmetry check): "
          f"{np.nanmax(lor_star_spread):.3f} meV")

    # BGW deltas (meV) at the 8 irreducible k.
    bgw_tab = parse_mode_table_gn(f"{REP}/mode_table_gn.dat")
    bgw_d = np.array([[(bgw_tab[(ik, n)]["Eqp0_m2"]
                        - bgw_tab[(ik, n)]["Eqp0_m0"]) * 1000.0
                       for n in range(1, NBAND_WIN + 1)] for ik in iks])

    # ---- Per-band delta-vs-delta table --------------------------------------
    hdr = (f"{'band':>4} | {'BGW mean':>9} {'BGW max|d|':>10} | "
           f"{'LORRAX mean':>11} {'LORRAX max|d|':>13} | {'ratio':>6}")
    print("\n-- per-band Delta (meV), mean over the 8 irreducible stars: "
          "BGW (mode2-mode0) vs LORRAX (2ry-zero) --")
    print(hdr)
    print("-" * len(hdr))
    lines = [f"# {hdr}"]
    for n in range(1, NBAND_WIN + 1):
        bm = np.mean(bgw_d[:, n - 1])
        bx = np.max(np.abs(bgw_d[:, n - 1]))
        lm = np.mean(lor_star[:, n - 1])
        lx = np.max(np.abs(lor_star[:, n - 1]))
        ratio = lm / bm if abs(bm) > 1e-12 else np.nan
        row = (f"{n:>4} | {bm:>9.3f} {bx:>10.3f} | "
               f"{lm:>11.3f} {lx:>13.3f} | {ratio:>6.2f}")
        print(row)
        lines.append(row)

    # ---- Summary stats -------------------------------------------------------
    print("\n-- summary --")
    print(f"  BGW    (m2-m0,   8 irr k x 16b): max|d|={np.max(np.abs(bgw_d)):7.3f}  "
          f"mean|d|={np.mean(np.abs(bgw_d)):6.3f} meV")
    print(f"  LORRAX (2ry-zero, 8 stars x 16b): max|d|={np.max(np.abs(lor_star)):7.3f}  "
          f"mean|d|={np.mean(np.abs(lor_star)):6.3f} meV")
    dd = np.nanmax(np.abs(deqp0[:, :NBAND_WIN] - dsig[:, :NBAND_WIN]))
    print(f"  LORRAX max|d_eqp0 - d_sigc| = {dd:.3e} meV "
          "(statics-cancel check; Z-factor makes eqp0 move slightly less)")

    bmean = bgw_d.mean(axis=0)
    lmean = lor_star.mean(axis=0)
    sign_ok = int(np.sum(np.sign(bmean) == np.sign(lmean)))
    cc = np.corrcoef(bmean, lmean)[0, 1]
    ccf = np.corrcoef(bgw_d.ravel(), lor_star.ravel())[0, 1]
    print(f"  per-band-mean sign agreement: {sign_ok}/{NBAND_WIN}")
    print(f"  per-band-mean Pearson r = {cc:.3f}; per-(star,band) r = {ccf:.3f}")

    # Gamma star: identify by fingerprint of BGW ik whose kcrys was Gamma =
    # the star with the deep-valence degenerate pair at min E_dft (ik=1 in refs).
    j_gamma = 0  # iks[0] == 1 == Gamma in the reference set
    bgw_gap = bgw_d[j_gamma, 8] - bgw_d[j_gamma, 2]
    lor_gap = lor_star[j_gamma, 8] - lor_star[j_gamma, 2]
    print(f"  Gamma direct-gap (b9-b3) delta shift: BGW {bgw_gap:+.3f} meV, "
          f"LORRAX {lor_gap:+.3f} meV")

    with open(f"{REP}/lorrax_mode_table.dat", "w") as f:
        f.write("# LORRAX (2ry-zero) vs BGW GN (mode2-mode0), Si 4x4x4, meV\n")
        f.write("# LORRAX values: star-mean of d sig_c(Edft).Re over the\n"
                "# symmetry-equivalent full-BZ k of each BGW irreducible k\n")
        f.write("\n".join(lines) + "\n")
        f.write("# per-(irr k, band): d_bgw = Eqp0_m2-m0; d_lorrax = star-mean "
                "2ry-zero; spread = star max-min\n")
        f.write("# ik  n  d_bgw_meV  d_lorrax_meV  star_spread_meV\n")
        for j, ik in enumerate(iks):
            for n in range(1, NBAND_WIN + 1):
                f.write(f"{ik:4d} {n:3d} {bgw_d[j, n-1]:12.3f} "
                        f"{lor_star[j, n-1]:12.3f} {lor_star_spread[j, n-1]:10.3f}\n")
    print(f"\nwrote {REP}/lorrax_mode_table.dat")


if __name__ == "__main__":
    main()
