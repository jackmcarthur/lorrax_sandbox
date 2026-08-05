"""Agent 4 — TRS-blind sym audit: extract pair-wise Σ_X and Σ comparisons.

Parses LORRAX sigma_freq_debug.dat (per-band x_bare, sex_0, coh_0, sigc_edft)
and BGW x.dat / sigma_hp.log for each (sym, nosym) or (cascade-on, cascade-off)
pair documented in STATUS.md. Reports max |Δ| per k-point.

Uses parsers consistent with skills/compare/SKILL.md, adapted to current LORRAX
sigma_freq_debug.dat header (includes V_H column between kin_ion and x_bare).
"""
from __future__ import annotations

import os
import re
import sys
import json
import numpy as np
from pathlib import Path


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------

def parse_sigma_freq_debug(path: str | os.PathLike) -> dict[int, dict[int, dict]]:
    """Parse LORRAX sigma_freq_debug.dat (current header w/ V_H column).

    Header: k  n  E_dft  Edft-Ef  kin_ion  V_H  x_bare  x_head  sex_0  coh_0  sex_head  coh_head  eqp0  eqp1

    Returns {k: {n: {field: float}}}.
    """
    out: dict[int, dict[int, dict]] = {}
    fields = ["k", "n", "E_dft", "Edft_rel", "kin_ion", "V_H", "x_bare",
              "x_head", "sex_0", "coh_0", "sex_head", "coh_head", "eqp0", "eqp1"]
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("k-point") or s.startswith("--"):
                continue
            parts = [p for p in re.split(r'\s+|\t+', s) if p]
            if len(parts) < 13:
                continue
            try:
                vals = [float(p) for p in parts[:len(fields)]]
                k = int(vals[0]); n = int(vals[1])
                row = dict(zip(fields, vals))
                out.setdefault(k, {})[n] = row
            except (ValueError, IndexError):
                continue
    return out


def parse_lorrax_eqp0(path: str | os.PathLike) -> dict[int, dict[int, dict]]:
    """Parse LORRAX eqp0.dat (sigSX/sigCOH/sigTOT/VH lines).

    Format:
        k-point 0:
        ----...
        n=0   sigSX=  -4.44   sigCOH=  -8.82   sigTOT= -13.26   VH=  3.32[+ ...i]

    Returns {k: {n: {'sigSX', 'sigCOH', 'sigTOT', 'VH'}}}
    """
    out: dict[int, dict[int, dict]] = {}
    cur_k = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            m = re.match(r'k-point\s+(\d+):', s)
            if m:
                cur_k = int(m.group(1))
                out.setdefault(cur_k, {})
                continue
            if cur_k is None:
                continue
            m2 = re.match(r'n=(\d+)\s+sigSX=\s*([+-]?\d+\.\d+)\s+sigCOH=\s*([+-]?\d+\.\d+)\s+sigTOT=\s*([+-]?\d+\.\d+)\s+VH=\s*([+-]?\d+\.\d+)', s)
                # VH may have complex part; relax
            if m2:
                n = int(m2.group(1))
                out[cur_k][n] = {
                    'sigSX': float(m2.group(2)),
                    'sigCOH': float(m2.group(3)),
                    'sigTOT': float(m2.group(4)),
                    'VH': float(m2.group(5)),
                }
            else:
                m3 = re.match(r'n=(\d+)\s+sigSX=\s*([+-]?\d+\.\d+)\s+sigCOH=\s*([+-]?\d+\.\d+)\s+sigTOT=\s*([+-]?\d+\.\d+)\s+VH=\s*([+-]?\d+\.\d+)', s)
                if m3:
                    n = int(m3.group(1))
                    out[cur_k][n] = {
                        'sigSX': float(m3.group(2)),
                        'sigCOH': float(m3.group(3)),
                        'sigTOT': float(m3.group(4)),
                        'VH': float(m3.group(5)),
                    }
    return out


def parse_bgw_xdat(path: str | os.PathLike):
    """Parse BGW x.dat → list of (kcrys, {band(1-idx): x_eV}).

    Format: header k-coord line (kx ky kz nbands ?), then 'spin band X 0.0'.
    """
    blocks = []
    cur = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('frequency') \
               or s.startswith('band_index') or s.startswith('sigma_matrix'):
                continue
            p = s.split()
            if len(p) >= 5 and all(re.match(r'[+-]?\d', q) or 'E' in q.upper() for q in p[:3]):
                # k-line
                try:
                    kx, ky, kz = (float(p[0]), float(p[1]), float(p[2]))
                    nb = int(p[3])
                    cur = {'kcrys': (kx, ky, kz), 'nb': nb, 'bands': {}}
                    blocks.append(cur)
                    continue
                except ValueError:
                    pass
            if cur is not None and len(p) >= 3:
                try:
                    spin = int(p[0]); band = int(p[1]); xeV = float(p[2])
                    cur['bands'][band] = xeV
                except ValueError:
                    pass
    return blocks


def parse_bgw_sigma_hp(path: str | os.PathLike):
    """Parse BGW sigma_hp.log; returns list of {'kcrys','ik','bands':{n:{'X','Cor','Corp',...}}}.

    Header:
      n  Emf  Eo  X  SX-X  CH  Sig  Vxc  Eqp0  Eqp1  CH'  Sig'  Eqp0'  Eqp1'  Znk
    Col indices: X=3, SX-X=4, CH=5, Sig=6, Vxc=7, Eqp0=8, Eqp1=9, CH'=10, Sig'=11.
    Cor = SX-X + CH; Corp = SX-X + CH'.
    """
    blocks = []
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
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            try:
                blocks[-1]['bands'][n] = {
                    'X': float(p[3]),
                    'SXmX': float(p[4]),
                    'CH': float(p[5]),
                    'Sig': float(p[6]),
                    'Vxc': float(p[7]),
                    'Eqp0': float(p[8]),
                    'Eqp1': float(p[9]),
                    'CHp': float(p[10]),
                    'Sigp': float(p[11]),
                    'Cor': float(p[4]) + float(p[5]),
                    'Corp': float(p[4]) + float(p[10]),
                }
            except (ValueError, IndexError):
                pass
    return blocks


# -----------------------------------------------------------------------------
# Pair helpers
# -----------------------------------------------------------------------------

def summarize_sigma_x_diff(A: dict, B: dict, *, field: str = 'x_bare'):
    """A, B are {k:{n: row}} dicts. Returns per-k (max|Δ|, argmax_n) and global."""
    ks = sorted(set(A) & set(B))
    rows = []
    glob_max = 0.0
    glob_loc = (None, None)
    for k in ks:
        ns = sorted(set(A[k]) & set(B[k]))
        diffs = np.array([A[k][n][field] - B[k][n][field] for n in ns])
        if diffs.size == 0:
            continue
        ami = int(np.argmax(np.abs(diffs)))
        mx = float(np.abs(diffs[ami]))
        rows.append({'k': k, 'n_bands': len(ns), 'max_abs_diff': mx,
                     'argmax_n': ns[ami], 'A_at_argmax': A[k][ns[ami]][field],
                     'B_at_argmax': B[k][ns[ami]][field], 'mae': float(np.mean(np.abs(diffs)))})
        if mx > glob_max:
            glob_max = mx
            glob_loc = (k, ns[ami])
    return rows, glob_max, glob_loc


def kcrys_match(k1, k2, tol=1e-4):
    """Match k-points modulo umklapp (k1 - k2 ∈ Z^3)."""
    d = np.array(k1) - np.array(k2)
    return np.allclose(d - np.round(d), 0.0, atol=tol)


def summarize_bgw_lorrax_x(bgw_blocks: list, lor: dict, *, band_offset_bgw=0,
                            band_offset_lor=0, ks_only: list[tuple] | None = None):
    """Compare BGW x.dat (1-indexed band) with LORRAX sigma_freq_debug 'x_bare' (0-indexed).

    band_offset_bgw: BGW physical band = idx + band_offset_bgw (typical = 0 if file already physical-1-idx)
    band_offset_lor: LORRAX physical band = idx + band_offset_lor + 1 (LORRAX n is 0-idx)
    """
    rows = []
    for b in bgw_blocks:
        kc = b['kcrys']
        # find matching LORRAX k by index (BGW ik is the physical k index in BGW symmetric layout)
        # We just use ik-1 → lor key. But lor keys are 0-indexed; assume ordered.
        # Simpler: compare for the first matching k by index.
        for kL in sorted(lor):
            # by row count match
            pass
    return rows


# -----------------------------------------------------------------------------
# Pair definitions
# -----------------------------------------------------------------------------

ROOT = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs')

PAIRS = {
    "MoS2_3x3_ibz_vs_fullbz_same_basis_POST_CASCADE": {
        "A": ROOT / "MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz/sigma_freq_debug.dat",
        "B": ROOT / "MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_B_fullbz/sigma_freq_debug.dat",
        "kind": "lorrax_sigma_freq_debug",
        "description": "MoS2 3x3, same 642-centroid orbit-closed basis. Run A: IBZ cascade active (5 IBZ q). Run B: LORRAX_FORCE_FULL_BZ=1 (9 q, full unfold). Both post-cascade c796420.",
        "expect_bug": True,
    },
    "MoS2_3x3_post_cascade_sym_LORRAX_vs_BGW": {
        "A": ROOT / "MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex_round8_baseline_2026-05-14/sigma_freq_debug.dat",
        "BGW_x": ROOT / "MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/x.dat",
        "BGW_hp": ROOT / "MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/sigma_hp.log",
        "kind": "lorrax_vs_bgw",
        "description": "MoS2 3x3 sym, LORRAX post-cascade (c796420, 2026-05-14) vs BGW reference. Bug should manifest as Σ_X(IBZ q's that fold via TRS) being wrong.",
        "expect_bug": True,
    },
    "MoS2_3x3_pre_cascade_sym_LORRAX_vs_BGW": {
        "A": ROOT / "MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/sigma_freq_debug.dat",
        "BGW_x": ROOT / "MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/x.dat",
        "kind": "lorrax_vs_bgw",
        "description": "MoS2 3x3 sym, LORRAX pre-cascade (2026-05-11 09:48, 8 min before cascade activation) vs BGW. Should NOT have the bug.",
        "expect_bug": False,
    },
    "MoS2_3x3_nosym_LORRAX_vs_BGW": {
        "A": None,
        "BGW_x": ROOT / "MoS2/02_mos2_3x3_nosym/00_bgw_cohsex/x.dat",
        "lorrax_eqp": ROOT / "MoS2/02_mos2_3x3_nosym/00_lorrax_cohsex/eqp0.dat",
        "kind": "lorrax_eqp_vs_bgw",
        "description": "MoS2 3x3 nosym, LORRAX pre-cascade (2026-05-04) vs BGW. ntran=1 so cascade is trivial — null check.",
        "expect_bug": False,
    },
    "Si_4x4x4_nosymmorphic_vs_nosym_LORRAX": {
        "A": ROOT / "Si/01_si_4x4x4_nosymmorphic/00_lorrax_cohsex/eqp0.dat",
        "B": ROOT / "Si/02_si_4x4x4_nosym/00_lorrax_cohsex/eqp0.dat",
        "kind": "lorrax_eqp_pair",
        "description": "Si 4x4x4 nosymmorphic (ntran=12) vs nosym (ntran=1). Both pre-cascade (2026-04). Null check: Si has inversion → no TRS bug.",
        "expect_bug": False,
    },
    "Si_2x2x2_sym_400c_vs_nosym_parity_LORRAX": {
        "A": ROOT / "Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_400c/eqp0_noqsym.dat",
        "B": ROOT / "Si_pseudobands/00_si_2x2x2_60Ry/21_lorrax_cohsex_nosym_parity/eqp0_noqsym.dat",
        "kind": "lorrax_eqp_pair",
        "description": "Si 2x2x2 60Ry: 400-centroid sym vs nosym 'parity' run. Both pre-cascade (2026-04-16/17). Null check.",
        "expect_bug": False,
    },
    "CrI3_6x6_80Ry_cascade_vs_round8_reference": {
        "A": ROOT / "CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_ibz_cascade_2026-05-14/gw.out",
        "B": ROOT / "CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14/gw.out",
        "kind": "cri3_sigma_x_eigenvalue",
        "description": "CrI3 6x6 80Ry: IBZ cascade (1508-orbit-closed, 8 IBZ q) vs reference (1504 not-orbit-closed, full-BZ unfold). BASIS-DIFFERENT but both post-cascade-code era; reported in cri3_ibz_cascade_validation.md as 'not a bug'.",
        "expect_bug": True,
    },
}


def main():
    out_dir = Path('/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/agent_4_data')
    results = {}

    for name, p in PAIRS.items():
        print(f"\n{'='*80}\n{name}\n  {p['description']}\n  expect_bug={p['expect_bug']}")
        kind = p['kind']
        try:
            if kind == 'lorrax_sigma_freq_debug':
                A = parse_sigma_freq_debug(p['A'])
                B = parse_sigma_freq_debug(p['B'])
                rows, gmax, gloc = summarize_sigma_x_diff(A, B, field='x_bare')
                results[name] = {
                    'kind': kind,
                    'fields_compared': ['x_bare', 'sex_0', 'coh_0'],
                    'per_k_x_bare': rows,
                    'global_max_abs_diff_x_bare': gmax,
                    'argmax_kn_x_bare': gloc,
                }
                print(f"  x_bare global max |Δ|={gmax:.4e} eV at (k,n)={gloc}")
                # also sex_0 and coh_0
                for fld in ('sex_0', 'coh_0'):
                    rs, gm, gl = summarize_sigma_x_diff(A, B, field=fld)
                    results[name][f'global_max_abs_diff_{fld}'] = gm
                    results[name][f'argmax_kn_{fld}'] = gl
                    results[name][f'per_k_{fld}'] = rs
                    print(f"  {fld} global max |Δ|={gm:.4e} eV at (k,n)={gl}")

            elif kind == 'lorrax_eqp_pair':
                A = parse_lorrax_eqp0(p['A'])
                B = parse_lorrax_eqp0(p['B'])
                rows, gmax, gloc = summarize_sigma_x_diff(A, B, field='sigSX')
                results[name] = {
                    'kind': kind,
                    'fields_compared': ['sigSX', 'sigCOH', 'sigTOT'],
                    'per_k_sigSX': rows,
                    'global_max_abs_diff_sigSX': gmax,
                    'argmax_kn_sigSX': gloc,
                }
                print(f"  sigSX global max |Δ|={gmax:.4e} eV at (k,n)={gloc}")
                for fld in ('sigCOH', 'sigTOT'):
                    rs, gm, gl = summarize_sigma_x_diff(A, B, field=fld)
                    results[name][f'global_max_abs_diff_{fld}'] = gm
                    results[name][f'argmax_kn_{fld}'] = gl
                    results[name][f'per_k_{fld}'] = rs
                    print(f"  {fld} global max |Δ|={gm:.4e} eV at (k,n)={gl}")

            elif kind == 'lorrax_vs_bgw':
                # Parse BGW x.dat → bare exchange per k-point (BGW symmetric layout).
                bgw_x_blocks = parse_bgw_xdat(p['BGW_x'])
                if p['A'] is not None:
                    lor = parse_sigma_freq_debug(p['A'])
                    field_lor = 'x_bare'
                else:
                    lor = parse_lorrax_eqp0(p.get('lorrax_eqp'))
                    field_lor = 'sigSX'
                # Per-k(BGW) comparison: match by index — BGW ik=1..N maps to LORRAX k (sym-reduced order)
                results[name] = {
                    'kind': kind,
                    'bgw_n_kpoints': len(bgw_x_blocks),
                    'lorrax_n_kpoints': len(lor),
                    'per_k_x_diff': [],
                }
                # If LORRAX is post-cascade sym MoS2, lorrax has 9 full-BZ k's; BGW has 4 IBZ k's.
                # Match BGW ik=1 → LORRAX k=0 (assumed Γ); the other BGW k's must be matched by crys-coord
                # via a WFN. For now, just match BGW band-by-band against LORRAX k=0 at minimum.
                glob_max = 0.0
                glob_loc = None
                # only compare BGW IK=1 (Γ) to LORRAX K=0 (also Γ in canonical ordering).
                if 0 in lor and bgw_x_blocks:
                    b0 = bgw_x_blocks[0]
                    # BGW bands are physical (1-indexed); LORRAX n is 0-indexed → +1 = physical.
                    # BGW reports only band_index_min..max range; LORRAX reports all bands.
                    diffs = []
                    for bgw_band, x_eV in b0['bands'].items():
                        lor_n = bgw_band - 1
                        if lor_n in lor[0]:
                            d = lor[0][lor_n][field_lor] - x_eV
                            diffs.append((bgw_band, x_eV, lor[0][lor_n][field_lor], d))
                            if abs(d) > glob_max:
                                glob_max = abs(d)
                                glob_loc = (0, bgw_band)
                    results[name]['gamma_band_diffs'] = diffs
                    results[name]['gamma_max_abs_x_diff'] = glob_max
                    results[name]['gamma_argmax_kn'] = glob_loc
                    print(f"  Γ-point bare-X max |Δ(LORRAX - BGW)|={glob_max:.4e} eV at (ik,n)={glob_loc}")
                    if diffs:
                        for tup in diffs[:6]:
                            print(f"    band {tup[0]:2d}  BGW={tup[1]:11.6f}  LORRAX={tup[2]:11.6f}  Δ={tup[3]:+.4e}")

            elif kind == 'lorrax_eqp_vs_bgw':
                bgw_x_blocks = parse_bgw_xdat(p['BGW_x'])
                lor = parse_lorrax_eqp0(p['lorrax_eqp'])
                # Match BGW IK=1 (Γ) ↔ LORRAX k=0 (Γ).
                glob_max = 0.0
                glob_loc = None
                if 0 in lor and bgw_x_blocks:
                    b0 = bgw_x_blocks[0]
                    diffs = []
                    for bgw_band, x_eV in b0['bands'].items():
                        lor_n = bgw_band - 1
                        if lor_n in lor[0]:
                            d = lor[0][lor_n]['sigSX'] - x_eV
                            diffs.append((bgw_band, x_eV, lor[0][lor_n]['sigSX'], d))
                            if abs(d) > glob_max:
                                glob_max = abs(d)
                                glob_loc = (0, bgw_band)
                    results[name] = {
                        'kind': kind,
                        'gamma_band_diffs': diffs,
                        'gamma_max_abs_x_diff': glob_max,
                        'gamma_argmax_kn': glob_loc,
                    }
                    print(f"  Γ-point sigSX max |Δ(LORRAX - BGW)|={glob_max:.4e} eV at (ik,n)={glob_loc}")
                    for tup in diffs[:6]:
                        print(f"    band {tup[0]:2d}  BGW_X={tup[1]:11.6f}  LORRAX_sigSX={tup[2]:11.6f}  Δ={tup[3]:+.4e}")

            elif kind == 'cri3_sigma_x_eigenvalue':
                # parse from gw.out 'Bare Σ_X diagonal' line
                def grab(p):
                    txt = open(p).read()
                    m = re.search(r"Bare Σ_X diagonal \(eV\), k=0:\s*([-\d.\s]+)", txt)
                    if not m:
                        return None
                    return [float(x) for x in m.group(1).split()]
                A = grab(p['A'])
                B = grab(p['B'])
                if A is None or B is None:
                    print(f"  CrI3 parse FAILED A={A} B={B}")
                else:
                    diff = np.array(A) - np.array(B)
                    results[name] = {
                        'kind': kind,
                        'A_vals': A, 'B_vals': B,
                        'diffs': diff.tolist(),
                        'max_abs_diff': float(np.max(np.abs(diff))),
                    }
                    print(f"  CrI3 k=0 Σ_X diagonal max |Δ|={np.max(np.abs(diff)):.4e} eV")
                    print(f"  A: {A}")
                    print(f"  B: {B}")
                    print(f"  Δ: {diff.tolist()}")
        except Exception as e:
            print(f"  ERROR parsing {name}: {e}")
            results[name] = {'kind': kind, 'error': repr(e)}

    (out_dir / 'audit_results.json').write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved results to {out_dir / 'audit_results.json'}")


if __name__ == '__main__':
    main()
