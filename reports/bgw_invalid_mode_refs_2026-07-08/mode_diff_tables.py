#!/usr/bin/env python3
"""Mode-diff tables for the BGW invalid_gpp_mode reference runs (Si 4x4x4, 60 bands).

Parses sigma_hp.log from each invalid_gpp_mode variant with the compare-skill
parser (skills/compare/SKILL.md section 2a, verbatim) and emits:
  - per-band Eqp0 / Cor' per mode at every k-point (mode_table_<line>.dat)
  - mode-to-mode deltas in meV, per band, plus summary stats (stdout + report tables)
"""
import re
import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band"

LINES = {
    "gn": {  # GN-GPP, frequency_dependence 3 (LORRAX GN-PPM analogue)
        0: f"{BASE}/01b_bgw_gn_mode0/sigma_hp.log",
        2: f"{BASE}/01_bgw_gn_ppm/sigma_hp.log",
        3: f"{BASE}/01c_bgw_gn_mode3/sigma_hp.log",
    },
    "hl": {  # HL-GPP, frequency_dependence 1
        0: f"{BASE}/02b_bgw_hl_mode0/sigma_hp.log",
        2: f"{BASE}/02d_bgw_hl_mode2/sigma_hp.log",
        3: f"{BASE}/02c_bgw_hl_mode3/sigma_hp.log",
    },
}


# --- compare-skill parser (skills/compare/SKILL.md 2a), verbatim -------------
def parse_sigma_hp(path):
    """Parse sigma_hp.log → list of dicts per k-point block.
    Each: {'kcrys': (kx,ky,kz), 'ik': int,
           'bands': {n: {'X','SXmX','CH','CHp','Cor','Corp','Sig','Vxc','Eqp0','Eqp1','Znk'}}}
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
            blocks[-1]['bands'][n] = {
                'X': float(p[3]), 'SXmX': float(p[4]), 'CH': float(p[5]),
                'Sig': float(p[6]), 'Vxc': float(p[7]),
                'Eqp0': float(p[8]), 'Eqp1': float(p[9]),
                'CHp': float(p[10]), 'Sigp': float(p[11]),
                'Znk': float(p[14]),
                'Cor': float(p[4]) + float(p[5]),
                'Corp': float(p[4]) + float(p[10]),
            }
        elif len(p) == 11 and p[0].isdigit():
            # freq_dep=1 (HL-GPP) layout: no primed columns; primed = unprimed.
            n = int(p[0])
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            blocks[-1]['bands'][n] = {
                'X': float(p[3]), 'SXmX': float(p[4]), 'CH': float(p[5]),
                'Sig': float(p[6]), 'Vxc': float(p[7]),
                'Eqp0': float(p[8]), 'Eqp1': float(p[9]),
                'CHp': float(p[5]), 'Sigp': float(p[6]),
                'Znk': float(p[10]),
                'Cor': float(p[4]) + float(p[5]),
                'Corp': float(p[4]) + float(p[5]),
            }
    return blocks
# -----------------------------------------------------------------------------


def load_line(files):
    data = {}
    for mode, path in files.items():
        blocks = parse_sigma_hp(path)
        data[mode] = {b['ik']: b for b in blocks}
    return data


def main():
    for line, files in LINES.items():
        data = load_line(files)
        modes = sorted(data)
        iks = sorted(data[modes[0]])
        bands = sorted(data[modes[0]][iks[0]]['bands'])

        # full per-band table -> .dat
        dat = f"/pscratch/sd/j/jackm/lorrax_sandbox/reports/bgw_invalid_mode_refs_2026-07-08/mode_table_{line}.dat"
        with open(dat, "w") as fh:
            fh.write(f"# BGW {line.upper()}-GPP invalid_gpp_mode comparison (Si 4x4x4)\n")
            fh.write("# ik  n   " + "  ".join(f"Eqp0_m{m:<9}" for m in modes)
                     + "  " + "  ".join(f"Corp_m{m:<9}" for m in modes) + "\n")
            for ik in iks:
                for n in bands:
                    row = [f"{ik:3d} {n:3d}"]
                    row += [f"{data[m][ik]['bands'][n]['Eqp0']:12.6f}" for m in modes]
                    row += [f"{data[m][ik]['bands'][n]['Corp']:12.6f}" for m in modes]
                    fh.write("  ".join(row) + "\n")

        # summary: mode-pair deltas in meV
        print(f"\n================ {line.upper()}-GPP line ================")
        pairs = [(0, 2), (0, 3), (2, 3)]
        for a, b in pairs:
            d_eqp = np.array([[data[b][ik]['bands'][n]['Eqp0'] - data[a][ik]['bands'][n]['Eqp0']
                               for n in bands] for ik in iks]) * 1e3
            d_cor = np.array([[data[b][ik]['bands'][n]['Corp'] - data[a][ik]['bands'][n]['Corp']
                               for n in bands] for ik in iks]) * 1e3
            print(f"\n-- mode {b} minus mode {a} (meV) --")
            print(f"  Eqp0: max|d|={np.abs(d_eqp).max():9.3f}  mean|d|={np.abs(d_eqp).mean():8.3f}")
            print(f"  Corp: max|d|={np.abs(d_cor).max():9.3f}  mean|d|={np.abs(d_cor).mean():8.3f}")
            print(f"  per-band mean d_Eqp0 (over {len(iks)} k):")
            for j, n in enumerate(bands):
                print(f"    band {n:3d}: dEqp0 mean {d_eqp[:, j].mean():9.3f}"
                      f"  max|{np.abs(d_eqp[:, j]).max():9.3f}|"
                      f"   dCorp mean {d_cor[:, j].mean():9.3f}"
                      f"  max|{np.abs(d_cor[:, j]).max():9.3f}|")

        # Gamma-point detail (ik=1)
        print(f"\n-- Gamma (ik=1) per-band detail, {line.upper()} line --")
        print("  n    Eqp0(m0)    Eqp0(m2)    Eqp0(m3)    Corp(m0)    Corp(m2)    Corp(m3)")
        for n in bands:
            g = [data[m][1]['bands'][n] for m in modes]
            print(f"{n:4d}" + "".join(f"{x['Eqp0']:12.6f}" for x in g)
                  + "".join(f"{x['Corp']:12.6f}" for x in g))


if __name__ == "__main__":
    main()
