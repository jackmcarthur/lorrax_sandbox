#!/usr/bin/env python3
"""
compare_all.py  --  BGW vs LORRAX comparison for Si 4x4x4 60-band run.

Compares:
  1. COHSEX: BGW (00_bgw_cohsex/sigma_hp.log) vs LORRAX (00_lorrax_cohsex/eqp0.dat)
  2. GN-PPM: BGW (01_bgw_gn_ppm/sigma_hp.log) vs LORRAX (01_lorrax_gn_ppm/sigma_freq_debug.dat)

Parsers taken from skills/compare/SKILL.md.
"""

import os
import sys
import re
import numpy as np

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BGW_COHSEX_LOG = os.path.join(SCRIPT_DIR, "00_bgw_cohsex", "sigma_hp.log")
BGW_GNPPM_LOG  = os.path.join(SCRIPT_DIR, "01_bgw_gn_ppm", "sigma_hp.log")
LRX_COHSEX_EQP = os.path.join(SCRIPT_DIR, "00_lorrax_cohsex", "eqp0.dat")
LRX_GNPPM_SFD  = os.path.join(SCRIPT_DIR, "01_lorrax_gn_ppm", "sigma_freq_debug.dat")

PLOT_PATH = os.path.join(SCRIPT_DIR, "compare_summary.png")

# BGW bands are 1..16; LORRAX 0-indexed so n_phys = n_lorrax + 1
BGW_BAND_MIN = 1
BGW_BAND_MAX = 16


# ===================================================================
# Parsers  (from skills/compare/SKILL.md)
# ===================================================================

def parse_sigma_hp(path):
    """Parse BerkeleyGW sigma_hp.log into a list of k-point blocks."""
    blocks = []
    ik = None
    kcrys = None
    for line in open(path):
        s = line.strip()
        m = re.match(
            r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)',
            s,
        )
        if m:
            kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            if not any(b.get("ik") == ik for b in blocks):
                blocks.append({"kcrys": kcrys, "ik": ik, "bands": {}})
            blk = next(b for b in blocks if b["ik"] == ik)
            blk["bands"][n] = {
                "Emf":  float(p[1]),
                "Eo":   float(p[2]),
                "X":    float(p[3]),
                "SXmX": float(p[4]),
                "CH":   float(p[5]),
                "Sig":  float(p[6]),
                "KIH":  float(p[7]),
                "Eqp0": float(p[8]),
                "Eqp1": float(p[9]),
                "CHp":  float(p[10]),
                "Sigp": float(p[11]),
                "Eqp0p": float(p[12]),
                "Eqp1p": float(p[13]),
                "Znk":  float(p[14]),
                "Cor":  float(p[4]) + float(p[5]),
                "Corp": float(p[4]) + float(p[10]),
            }
    return blocks


def parse_sigma_freq_debug(path):
    """Parse LORRAX sigma_freq_debug.dat."""
    result = {}  # {k_idx: {n_phys: {field: value}}}
    for line in open(path):
        s = line.strip()
        if s.startswith("#") or s.startswith("k-point") or s.startswith("k ") or not s:
            continue
        p = s.split()
        if len(p) >= 13:
            try:
                k = int(p[0])
                n = int(p[1])
                n_phys = n + 1  # BGW is 1-indexed
                if k not in result:
                    result[k] = {}
                # Columns (0-indexed):
                #  0: k, 1: n, 2: Edft-Ef, 3: E_dft, 4: kin_ion,
                #  5: sex_0, 6: coh_0, 7: x_bare, 8: sig_c(0),
                #  9: sig_c+(w), 10: sig_c-(w), 11: sig_c_invld(0),
                #  12: sig_c(Edft)
                result[k][n_phys] = {
                    "Edft_rel":   float(p[2]),
                    "Edft":       float(p[3]),
                    "kin_ion":    float(p[4]),
                    "sex_0":      float(p[5]),
                    "coh_0":      float(p[6]),
                    "x_bare":     float(p[7]),
                    "sigc_0":     float(p[8]),
                    "sigc_plus":  float(p[9])  if p[9]  != "nan" else np.nan,
                    "sigc_minus": float(p[10]) if p[10] != "nan" else np.nan,
                    "sigc_invld": float(p[11]) if p[11] != "nan" else np.nan,
                    "sigc_edft":  float(p[12]) if p[12] != "nan" else np.nan,
                }
            except (ValueError, IndexError):
                pass
    return result


def parse_lorrax_cohsex_eqp0(path):
    """Parse LORRAX COHSEX eqp0.dat.

    Format per line:
        n=X   sigSX=Y   sigCOH=Z   sigTOT=W   VH=V
    Grouped under 'k-point N:' headers.
    """
    result = {}  # {k_idx: {n_phys: {sigSX, sigCOH, sigTOT, VH}}}
    k_idx = None
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # k-point header
        m = re.match(r"k-point\s+(\d+):", s)
        if m:
            k_idx = int(m.group(1))
            if k_idx not in result:
                result[k_idx] = {}
            continue
        if s.startswith("-"):
            continue
        if k_idx is None:
            continue
        # Parse n=X  sigSX=Y  sigCOH=Z  sigTOT=W  VH=V
        m = re.match(
            r"n=(\d+)\s+sigSX=\s*([\d.Ee+-]+)\s+sigCOH=\s*([\d.Ee+-]+)"
            r"\s+sigTOT=\s*([\d.Ee+-]+)\s+VH=\s*([\d.Ee+-]+)",
            s,
        )
        if m:
            n_lorrax = int(m.group(1))
            n_phys = n_lorrax + 1  # map to BGW 1-indexed
            result[k_idx][n_phys] = {
                "sigSX":  float(m.group(2)),
                "sigCOH": float(m.group(3)),
                "sigTOT": float(m.group(4)),
                "VH":     float(m.group(5)),
            }
    return result


# ===================================================================
# Main comparison
# ===================================================================

def print_separator(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def compare_cohsex():
    """Compare COHSEX: BGW vs LORRAX at Gamma (first k-point)."""
    print_separator("COHSEX  COMPARISON  (BGW vs LORRAX)  --  Gamma point")

    bgw = parse_sigma_hp(BGW_COHSEX_LOG)
    lrx = parse_lorrax_cohsex_eqp0(LRX_COHSEX_EQP)

    # Also try to load LORRAX sigma_freq_debug for COHSEX if it exists
    lrx_sfd_path = os.path.join(SCRIPT_DIR, "00_lorrax_cohsex", "sigma_freq_debug.dat")
    lrx_sfd = None
    if os.path.isfile(lrx_sfd_path):
        lrx_sfd = parse_sigma_freq_debug(lrx_sfd_path)

    if not bgw:
        print("ERROR: No k-point blocks found in BGW COHSEX sigma_hp.log")
        return {}, {}

    bgw_gamma = bgw[0]  # ik=1 = Gamma
    lrx_gamma = lrx.get(0, {})  # k-point 0 = Gamma

    # Header
    # BGW COHSEX: frequency_dependence=0, so CH' == CH (static), Znk=1
    # BGW Cor = SXmX + CH
    # LORRAX sigCor = sigTOT - x_bare (if sigma_freq_debug available)
    #        or sigCor = sigSX + sigCOH - x_bare (sigSX = sex in COHSEX)
    # But in COHSEX eqp0.dat the decomposition is sigSX, sigCOH, sigTOT.
    # BGW decomposition: X (bare exchange), SX-X (screened - bare), CH (Coulomb hole)
    #   BGW Sig = X + SXmX + CH
    # LORRAX: sigTOT = sigSX + sigCOH
    #   where sigSX ~ BGW SX (screened exchange), sigCOH ~ BGW CH

    has_sfd = lrx_sfd is not None and 0 in lrx_sfd

    print()
    if has_sfd:
        print(f"  {'n':>3s}  {'BGW_Eo':>10s}  {'BGW_X':>10s}  {'BGW_SXmX':>10s}  "
              f"{'BGW_CH':>10s}  {'BGW_Sig':>10s}  "
              f"{'LRX_sex':>10s}  {'LRX_coh':>10s}  {'LRX_xbare':>10s}  "
              f"{'LRX_Sig':>10s}  {'dSig':>10s}")
    else:
        print(f"  {'n':>3s}  {'BGW_Eo':>10s}  {'BGW_X':>10s}  {'BGW_SXmX':>10s}  "
              f"{'BGW_CH':>10s}  {'BGW_Sig':>10s}  "
              f"{'LRX_sigSX':>10s}  {'LRX_sigCOH':>10s}  "
              f"{'LRX_sigTOT':>10s}  {'dSig':>10s}")
    print("  " + "-" * 120)

    deltas_sig = []
    deltas_cor = []
    bgw_cor_list = []
    lrx_cor_list = []

    for n in range(BGW_BAND_MIN, BGW_BAND_MAX + 1):
        if n not in bgw_gamma["bands"]:
            continue
        b = bgw_gamma["bands"][n]
        bgw_sig = b["Sig"]  # X + SXmX + CH

        if n in lrx_gamma:
            lr = lrx_gamma[n]
            # LORRAX total self-energy comparable to BGW Sig = X + SXmX + CH
            # But LORRAX sigTOT = sigSX + sigCOH (no bare exchange X separately in eqp0)
            # Actually LORRAX sigSX is the *screened* exchange, while BGW X is bare.
            # For full Sigma comparison we need LORRAX x_bare from sigma_freq_debug.

            if has_sfd and n in lrx_sfd[0]:
                sf = lrx_sfd[0][n]
                lrx_sig = sf["x_bare"] + sf["sex_0"] + sf["coh_0"]
                # BGW Sig = X + SXmX + CH, and X = bare exchange
                # sex_0 ~ SXmX (screened - bare), coh_0 ~ CH
                dsig = lrx_sig - bgw_sig
                deltas_sig.append(dsig)

                # Correlation: BGW Cor = SXmX + CH;  LORRAX Cor = sex_0 + coh_0
                bgw_cor = b["Cor"]
                lrx_cor = sf["sex_0"] + sf["coh_0"]
                deltas_cor.append(lrx_cor - bgw_cor)
                bgw_cor_list.append(bgw_cor)
                lrx_cor_list.append(lrx_cor)

                print(f"  {n:3d}  {b['Eo']:10.4f}  {b['X']:10.4f}  {b['SXmX']:10.4f}  "
                      f"{b['CH']:10.4f}  {bgw_sig:10.4f}  "
                      f"{sf['sex_0']:10.4f}  {sf['coh_0']:10.4f}  {sf['x_bare']:10.4f}  "
                      f"{lrx_sig:10.4f}  {dsig:10.4f}")
            else:
                lr = lrx_gamma[n]
                # Without sigma_freq_debug, compare what we can.
                # LORRAX sigTOT = sigSX + sigCOH
                # This is not directly comparable to BGW Sig = X + SXmX + CH
                # because sigSX is screened exchange (= BGW X + SXmX), sigCOH ~ CH
                # So LORRAX sigTOT ~ BGW X + SXmX + CH = BGW Sig
                lrx_sig = lr["sigTOT"]
                dsig = lrx_sig - bgw_sig
                deltas_sig.append(dsig)

                # Correlation comparison: BGW Cor = SXmX+CH, LRX sigCOH
                bgw_cor = b["Cor"]
                # LRX sigSX includes bare+screened, so Cor ~ sigTOT - X_bare
                # Without x_bare, use: sigSX ~ SX, sigCOH ~ CH
                # Actually for COHSEX eqp0.dat, sigSX = SEX component, sigCOH = COH.
                # BGW SXmX = SX - X, so LRX_cor = sigSX - x_bare + sigCOH
                # Without x_bare we can still compare: LRX sigCOH vs BGW CH
                lrx_cor = lr["sigCOH"]
                deltas_cor.append(lrx_cor - b["CH"])
                bgw_cor_list.append(b["CH"])
                lrx_cor_list.append(lrx_cor)

                print(f"  {n:3d}  {b['Eo']:10.4f}  {b['X']:10.4f}  {b['SXmX']:10.4f}  "
                      f"{b['CH']:10.4f}  {bgw_sig:10.4f}  "
                      f"{lr['sigSX']:10.4f}  {lr['sigCOH']:10.4f}  "
                      f"{lr['sigTOT']:10.4f}  {dsig:10.4f}")
        else:
            print(f"  {n:3d}  {b['Eo']:10.4f}  {b['X']:10.4f}  {b['SXmX']:10.4f}  "
                  f"{b['CH']:10.4f}  {bgw_sig:10.4f}  {'---':>10s}  {'---':>10s}  "
                  f"{'---':>10s}  {'---':>10s}")

    deltas_sig = np.array(deltas_sig)
    deltas_cor = np.array(deltas_cor)

    print()
    if len(deltas_sig) > 0:
        print(f"  Gamma-point  Sigma  MAE = {np.mean(np.abs(deltas_sig)):.6f} eV   "
              f"max|delta| = {np.max(np.abs(deltas_sig)):.6f} eV   "
              f"matched bands = {len(deltas_sig)}")
    if len(deltas_cor) > 0:
        print(f"  Gamma-point  Cor    MAE = {np.mean(np.abs(deltas_cor)):.6f} eV   "
              f"max|delta| = {np.max(np.abs(deltas_cor)):.6f} eV")

    return {"bgw": bgw, "lrx_eqp": lrx, "lrx_sfd": lrx_sfd,
            "bgw_cor": np.array(bgw_cor_list), "lrx_cor": np.array(lrx_cor_list),
            "deltas_sig": deltas_sig}


def compare_gnppm():
    """Compare GN-PPM: BGW Corp vs LORRAX sigc_edft across ALL k-points."""
    print_separator("GN-PPM  COMPARISON  (BGW vs LORRAX)")

    bgw = parse_sigma_hp(BGW_GNPPM_LOG)
    lrx = parse_sigma_freq_debug(LRX_GNPPM_SFD)

    if not bgw:
        print("ERROR: No k-point blocks found in BGW GN-PPM sigma_hp.log")
        return {}

    # ---- Gamma-point detailed table ----
    print()
    print("  --- Gamma point (ik=1) detailed table ---")
    print()
    print(f"  {'n':>3s}  {'BGW_Eo':>10s}  {'BGW_Corp':>10s}  {'BGW_SXmX':>10s}  "
          f"{'BGW_CHp':>10s}  {'LRX_Edft':>10s}  {'LRX_sigC_E':>10s}  "
          f"{'delta_C':>10s}")
    print("  " + "-" * 90)

    bgw_gamma = bgw[0]
    lrx_gamma = lrx.get(0, {})

    for n in range(BGW_BAND_MIN, BGW_BAND_MAX + 1):
        if n not in bgw_gamma["bands"]:
            continue
        b = bgw_gamma["bands"][n]
        bgw_corp = b["Corp"]  # SXmX + CH'

        if n in lrx_gamma:
            lr = lrx_gamma[n]
            lrx_sigc = lr["sigc_edft"]
            if np.isnan(lrx_sigc):
                delta = np.nan
                delta_str = "nan"
            else:
                delta = lrx_sigc - bgw_corp
                delta_str = f"{delta:10.4f}"
            print(f"  {n:3d}  {b['Eo']:10.4f}  {bgw_corp:10.4f}  {b['SXmX']:10.4f}  "
                  f"{b['CHp']:10.4f}  {lr['Edft']:10.4f}  "
                  f"{lrx_sigc:10.4f}  {delta_str:>10s}")
        else:
            print(f"  {n:3d}  {b['Eo']:10.4f}  {bgw_corp:10.4f}  {b['SXmX']:10.4f}  "
                  f"{b['CHp']:10.4f}  {'---':>10s}  {'---':>10s}  {'---':>10s}")

    # ---- All k-points statistics ----
    print()
    print("  --- All k-points statistics (BGW Corp vs LORRAX sigc_edft) ---")
    print()

    all_deltas = []
    per_k_mae = []

    # Map BGW ik to LORRAX k-index.
    # BGW ik is 1-indexed, LORRAX k is 0-indexed.
    # BGW has 9 k-points (from IBZ), LORRAX has 64 (full BZ).
    # We match by BGW ik -> LORRAX k = ik - 1.
    # This works if LORRAX k-point ordering matches BGW's first N k-points.
    # Since BGW only has the irreducible set, we match those.

    bgw_corp_all = []
    lrx_sigc_all = []

    for blk in bgw:
        ik = blk["ik"]
        k_lrx = ik - 1  # 0-indexed LORRAX k-index

        if k_lrx not in lrx:
            continue

        k_deltas = []
        for n in range(BGW_BAND_MIN, BGW_BAND_MAX + 1):
            if n not in blk["bands"]:
                continue
            if n not in lrx[k_lrx]:
                continue
            bgw_corp = blk["bands"][n]["Corp"]
            lrx_sigc = lrx[k_lrx][n]["sigc_edft"]
            if np.isnan(lrx_sigc):
                continue
            delta = lrx_sigc - bgw_corp
            all_deltas.append(delta)
            k_deltas.append(delta)
            bgw_corp_all.append(bgw_corp)
            lrx_sigc_all.append(lrx_sigc)

        if k_deltas:
            k_arr = np.array(k_deltas)
            per_k_mae.append((ik, blk["kcrys"], np.mean(np.abs(k_arr)),
                              np.max(np.abs(k_arr)), len(k_arr)))

    # Per-k summary
    print(f"  {'ik':>3s}  {'k-point':>30s}  {'MAE':>10s}  {'max|d|':>10s}  {'Nbands':>6s}")
    print("  " + "-" * 70)
    for ik, kc, mae, maxd, nb in per_k_mae:
        kstr = f"({kc[0]:.4f}, {kc[1]:.4f}, {kc[2]:.4f})"
        print(f"  {ik:3d}  {kstr:>30s}  {mae:10.6f}  {maxd:10.6f}  {nb:6d}")

    all_deltas = np.array(all_deltas)
    bgw_corp_all = np.array(bgw_corp_all)
    lrx_sigc_all = np.array(lrx_sigc_all)

    print()
    if len(all_deltas) > 0:
        print(f"  OVERALL  MAE = {np.mean(np.abs(all_deltas)):.6f} eV")
        print(f"  OVERALL  max|delta| = {np.max(np.abs(all_deltas)):.6f} eV")
        print(f"  OVERALL  matched (k,band) pairs = {len(all_deltas)}")
    else:
        print("  WARNING: No matching (k,band) pairs found.")

    return {"bgw": bgw, "lrx": lrx, "all_deltas": all_deltas,
            "bgw_corp_all": bgw_corp_all, "lrx_sigc_all": lrx_sigc_all}


def make_plot(cohsex_data, gnppm_data):
    """Save a summary comparison plot to compare_summary.png."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: COHSEX Sigma at Gamma ---
    ax = axes[0]
    if "deltas_sig" in cohsex_data and len(cohsex_data["deltas_sig"]) > 0:
        bands = np.arange(BGW_BAND_MIN, BGW_BAND_MIN + len(cohsex_data["deltas_sig"]))
        ax.bar(bands, cohsex_data["deltas_sig"] * 1000, color="steelblue", alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Band index")
        ax.set_ylabel("delta Sigma (meV)")
        ax.set_title("COHSEX: BGW - LORRAX Sigma\n(Gamma point)")
    else:
        ax.text(0.5, 0.5, "No COHSEX data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("COHSEX (no data)")

    # --- Panel 2: GN-PPM Corp scatter ---
    ax = axes[1]
    if "bgw_corp_all" in gnppm_data and len(gnppm_data["bgw_corp_all"]) > 0:
        ax.scatter(gnppm_data["bgw_corp_all"], gnppm_data["lrx_sigc_all"],
                   s=12, alpha=0.6, c="darkorange", edgecolors="none")
        mn = min(gnppm_data["bgw_corp_all"].min(), gnppm_data["lrx_sigc_all"].min()) - 0.5
        mx = max(gnppm_data["bgw_corp_all"].max(), gnppm_data["lrx_sigc_all"].max()) + 0.5
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="y=x")
        ax.set_xlabel("BGW Corp (SXmX + CH') [eV]")
        ax.set_ylabel("LORRAX sigC(Edft) [eV]")
        ax.set_title("GN-PPM: Correlation (all k)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GN-PPM data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("GN-PPM (no data)")

    # --- Panel 3: GN-PPM delta histogram ---
    ax = axes[2]
    if "all_deltas" in gnppm_data and len(gnppm_data["all_deltas"]) > 0:
        deltas_mev = gnppm_data["all_deltas"] * 1000
        ax.hist(deltas_mev, bins=30, color="seagreen", alpha=0.8, edgecolor="k", lw=0.3)
        ax.axvline(0, color="k", lw=0.8)
        mae_mev = np.mean(np.abs(deltas_mev))
        ax.axvline(mae_mev, color="red", ls="--", lw=1, label=f"MAE = {mae_mev:.1f} meV")
        ax.axvline(-mae_mev, color="red", ls="--", lw=1)
        ax.set_xlabel("delta Corp (meV)")
        ax.set_ylabel("Count")
        ax.set_title(f"GN-PPM: delta distribution\n(N={len(deltas_mev)})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GN-PPM data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("GN-PPM delta (no data)")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\n  Plot saved to: {PLOT_PATH}")


# ===================================================================
# Entry point
# ===================================================================

def main():
    print("=" * 80)
    print("  Si 4x4x4 60-band:  BGW vs LORRAX  comparison")
    print("=" * 80)

    # Check files exist
    missing = []
    for label, path in [("BGW COHSEX",  BGW_COHSEX_LOG),
                         ("BGW GN-PPM",  BGW_GNPPM_LOG),
                         ("LRX COHSEX",  LRX_COHSEX_EQP),
                         ("LRX GN-PPM",  LRX_GNPPM_SFD)]:
        if not os.path.isfile(path):
            missing.append(f"  MISSING: {label} -> {path}")
    if missing:
        print("\nWARNING: Some input files are missing:")
        for m in missing:
            print(m)
        print()

    # 1. COHSEX comparison
    cohsex_data = {}
    if os.path.isfile(BGW_COHSEX_LOG) and os.path.isfile(LRX_COHSEX_EQP):
        cohsex_data = compare_cohsex()
    else:
        print("\n  Skipping COHSEX comparison (missing files).")

    # 2. GN-PPM comparison
    gnppm_data = {}
    if os.path.isfile(BGW_GNPPM_LOG) and os.path.isfile(LRX_GNPPM_SFD):
        gnppm_data = compare_gnppm()
    else:
        print("\n  Skipping GN-PPM comparison (missing files).")

    # 3. Summary
    print_separator("SUMMARY")
    if "deltas_sig" in cohsex_data and len(cohsex_data["deltas_sig"]) > 0:
        d = cohsex_data["deltas_sig"]
        print(f"  COHSEX (Gamma):")
        print(f"    Sigma MAE      = {np.mean(np.abs(d)):.6f} eV  ({np.mean(np.abs(d))*1000:.2f} meV)")
        print(f"    Sigma max|d|   = {np.max(np.abs(d)):.6f} eV  ({np.max(np.abs(d))*1000:.2f} meV)")
        print(f"    Matched bands  = {len(d)}")
    if "all_deltas" in gnppm_data and len(gnppm_data["all_deltas"]) > 0:
        d = gnppm_data["all_deltas"]
        print(f"  GN-PPM (all k):")
        print(f"    Corp MAE       = {np.mean(np.abs(d)):.6f} eV  ({np.mean(np.abs(d))*1000:.2f} meV)")
        print(f"    Corp max|d|    = {np.max(np.abs(d)):.6f} eV  ({np.max(np.abs(d))*1000:.2f} meV)")
        print(f"    Matched (k,n)  = {len(d)}")

    # 4. Plot
    make_plot(cohsex_data, gnppm_data)


if __name__ == "__main__":
    main()
