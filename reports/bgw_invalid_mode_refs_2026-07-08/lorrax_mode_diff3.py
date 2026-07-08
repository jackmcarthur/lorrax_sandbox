#!/usr/bin/env python3
"""LORRAX ppm_invalid_mode THREE-WAY delta table vs BGW invalid_gpp_mode 0/2/3.

Extends lorrax_mode_diff.py (the zero-vs-2ry scaffold) to the full mode
triple, using the same-code rerun triple produced by the static_limit
implementation session (agent/memplanner-cleanup, 2026-07-08):

    runs/Si/00_si_4x4x4_60band/03b_lorrax_gnppm_invalidmode_zero    (BGW mode 0)
    runs/Si/00_si_4x4x4_60band/03b_lorrax_gnppm_invalidmode_2ry     (BGW mode 2)
    runs/Si/00_si_4x4x4_60band/03_lorrax_gnppm_invalidmode_static   (BGW mode 3)

(the original 03_ zero/2ry pair ran at a pre-qp_solver/pre-pad-refactor
commit; the triple keeps every delta same-code.)

Comparable quantity is the MODE-TO-MODE DELTA of sig_c(Edft) per band —
see report.md "How to validate" and lorrax_zero_2ry_validation.md.  BGW
anchors from mode_table_gn.dat: (m2-m0), (m3-m0), (m3-m2).

Run on a compute node:
    LORRAX_NGPU=1 lxrun python3 -u lorrax_mode_diff3.py
"""
import re

import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band"
REP = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/bgw_invalid_mode_refs_2026-07-08"
RUNS = {
    "zero": f"{BASE}/03b_lorrax_gnppm_invalidmode_zero",
    "2ry": f"{BASE}/03b_lorrax_gnppm_invalidmode_2ry",
    "static": f"{BASE}/03_lorrax_gnppm_invalidmode_static",
}
BGW_MODE0_HP = f"{BASE}/01b_bgw_gn_mode0/sigma_hp.log"
NBAND_WIN = 16
# (label, lorrax pair, bgw Eqp0 column pair)
PAIRS = [
    ("2ry-zero    vs m2-m0", ("2ry", "zero"), ("Eqp0_m2", "Eqp0_m0")),
    ("static-zero vs m3-m0", ("static", "zero"), ("Eqp0_m3", "Eqp0_m0")),
    ("static-2ry  vs m3-m2", ("static", "2ry"), ("Eqp0_m3", "Eqp0_m2")),
]


# --- compare-skill parser (skills/compare/SKILL.md 2c, current format) -------
def parse_sigma_freq_debug_v2(path):
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


def grep_lines(gw_out, needle):
    return [line.strip() for line in open(gw_out, errors="replace")
            if needle in line]


def main():
    parsed = {}
    cols_ref = None
    for tag, run in RUNS.items():
        cols, d = parse_sigma_freq_debug_v2(f"{run}/sigma_freq_debug.dat")
        if cols_ref is None:
            cols_ref = cols
        assert cols == cols_ref, (tag, cols)
        parsed[tag] = d
    keys = set(parsed["zero"])
    for tag in RUNS:
        assert set(parsed[tag]) == keys, tag
    nk = max(k for k, _ in keys) + 1
    nb = max(n for _, n in keys)
    print(f"columns: {cols_ref}")
    print(f"parsed {len(keys)} (k,n) rows per run: nk={nk}, bands 1..{nb}")

    print("\n-- invalid-pole / static-term lines (gw.out) --")
    for tag, run in RUNS.items():
        for ln in grep_lines(f"{run}/gw.out", "GN invalid modes"):
            print(f"  {tag:>6}: {ln}")
        for ln in grep_lines(f"{run}/gw.out", "static COHSEX: max"):
            print(f"  {tag:>6}: {ln}")

    # Determinism guard: statics must be bit-identical across the triple.
    sc = "sig_c(Edft).Re"
    for col in ("x_bare", "kin_ion", "V_H"):
        if col in cols_ref:
            for tag in ("2ry", "static"):
                d = max(abs(parsed["zero"][k][col] - parsed[tag][k][col])
                        for k in keys)
                print(f"  determinism check max|d {col}| ({tag}-zero) = {d:.3e} eV")

    edft = np.full((nk, nb), np.nan)
    sig = {tag: np.full((nk, nb), np.nan) for tag in RUNS}
    for (k, n), row in parsed["zero"].items():
        edft[k, n - 1] = row["E_dft"]
    for tag in RUNS:
        for (k, n), row in parsed[tag].items():
            sig[tag][k, n - 1] = row[sc]

    # ---- star assignment by DFT-eigenvalue fingerprint ----------------------
    bgw_eo = parse_sigma_hp_eo(BGW_MODE0_HP)
    iks = sorted(bgw_eo)
    fp_bgw = np.array([[bgw_eo[ik][n] for n in range(1, NBAND_WIN + 1)]
                       for ik in iks])
    star_of_k = np.full(nk, -1, dtype=int)
    for k in range(nk):
        d = np.max(np.abs(fp_bgw - edft[k, :NBAND_WIN][None, :]), axis=1)
        j = int(np.argmin(d))
        if d[j] < 2e-3:
            star_of_k[k] = j
    assert np.all(star_of_k >= 0), np.where(star_of_k < 0)[0]
    star_sizes = np.bincount(star_of_k, minlength=len(iks))
    print(f"\n  star sizes (BGW ik order {iks}): {star_sizes.tolist()} "
          f"(sum {star_sizes.sum()})")

    bgw_tab = parse_mode_table_gn(f"{REP}/mode_table_gn.dat")

    out_lines = [
        "# LORRAX three-way ppm_invalid_mode deltas vs BGW GN invalid_gpp_mode,",
        "# Si 4x4x4, meV. LORRAX = star-mean d sig_c(Edft).Re over full-BZ k;",
        "# BGW = d Eqp0 (== d Cor' within a line). Rerun triple 03b/03 (same-code).",
    ]
    summary = []
    for label, (ta, tb), (ca, cb) in PAIRS:
        dsig = (sig[ta] - sig[tb]) * 1000.0  # meV
        lor_star = np.full((len(iks), NBAND_WIN), np.nan)
        spread = np.full((len(iks), NBAND_WIN), np.nan)
        for j in range(len(iks)):
            m = star_of_k == j
            lor_star[j] = np.nanmean(dsig[m, :NBAND_WIN], axis=0)
            spread[j] = (np.nanmax(dsig[m, :NBAND_WIN], axis=0)
                         - np.nanmin(dsig[m, :NBAND_WIN], axis=0))
        bgw_d = np.array([[(bgw_tab[(ik, n)][ca] - bgw_tab[(ik, n)][cb]) * 1000.0
                           for n in range(1, NBAND_WIN + 1)] for ik in iks])

        hdr = (f"{'band':>4} | {'BGW mean':>9} {'BGW max|d|':>10} | "
               f"{'LORRAX mean':>11} {'LORRAX max|d|':>13} | {'ratio':>6}")
        print(f"\n== {label} ==   (max star spread "
              f"{np.nanmax(spread):.3f} meV)")
        print(hdr)
        print("-" * len(hdr))
        out_lines += ["", f"== {label} ==", f"# {hdr}"]
        for n in range(1, NBAND_WIN + 1):
            bm = np.mean(bgw_d[:, n - 1])
            bx = np.max(np.abs(bgw_d[:, n - 1]))
            lm = np.mean(lor_star[:, n - 1])
            lx = np.max(np.abs(lor_star[:, n - 1]))
            ratio = lm / bm if abs(bm) > 1e-12 else np.nan
            row = (f"{n:>4} | {bm:>9.3f} {bx:>10.3f} | "
                   f"{lm:>11.3f} {lx:>13.3f} | {ratio:>6.2f}")
            print(row)
            out_lines.append(row)

        bmean = bgw_d.mean(axis=0)
        lmean = lor_star.mean(axis=0)
        sign_ok = int(np.sum(np.sign(bmean) == np.sign(lmean)))
        cc = np.corrcoef(bmean, lmean)[0, 1]
        j_gamma = 0
        bgw_gap = bgw_d[j_gamma, 8] - bgw_d[j_gamma, 2]
        lor_gap = lor_star[j_gamma, 8] - lor_star[j_gamma, 2]
        s = (f"{label}: BGW max|d|={np.max(np.abs(bgw_d)):6.2f} "
             f"mean|d|={np.mean(np.abs(bgw_d)):5.2f} | "
             f"LORRAX max|d|={np.max(np.abs(lor_star)):6.2f} "
             f"mean|d|={np.mean(np.abs(lor_star)):5.2f} meV | "
             f"sign {sign_ok}/16 r={cc:+.2f} | "
             f"Gamma-gap d: BGW {bgw_gap:+.2f} LORRAX {lor_gap:+.2f} meV")
        summary.append(s)
        # deep-valence / VBM / window-top pattern check
        print(f"  bands 1-2 mean: BGW {bmean[:2].mean():+7.3f}  "
              f"LORRAX {lmean[:2].mean():+7.3f} meV")
        print(f"  bands 3-8 mean: BGW {bmean[2:8].mean():+7.3f}  "
              f"LORRAX {lmean[2:8].mean():+7.3f} meV")
        print(f"  bands 15-16 mean: BGW {bmean[14:].mean():+7.3f}  "
              f"LORRAX {lmean[14:].mean():+7.3f} meV")

    print("\n-- summary --")
    for s in summary:
        print("  " + s)
    out_lines += ["", "# summary"] + ["# " + s for s in summary]

    with open(f"{REP}/lorrax_mode_table3.dat", "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nwrote {REP}/lorrax_mode_table3.dat")


if __name__ == "__main__":
    main()
