#!/usr/bin/env python3
"""Extract planner predictions and mem_stats peaks from all gw_*.out files.

Outputs a CSV summary table for the agent_c report.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIGS = ['3x3x3_nb100', '3x3x3_nb200', '4x4x4_nb100', '4x4x4_nb200']
VARIANTS = ['platform_false', 'bfc_pre95']

# Regex patterns
RE_HWM = re.compile(r"HWM estimate\s*=\s*([\d.]+)\s*GB/dev.*bottleneck:\s*(\w+)")
RE_BANDCHUNK = re.compile(r"band_chunk\s*=\s*(\d+)")
RE_RCHUNK = re.compile(r"r_chunk\s*=\s*(\d+)\s*\((\d+)\s*chunks\)")
RE_CSCHUNK = re.compile(r"gflat_chunk_size\s*=\s*(\d+)")
RE_PEAK = re.compile(r"\[mem_probe\s+(\S+).*?peak=([\d.\-]+)\s*GB.*?nvsmi=([\d.]+)\s*GB.*?nvsmi_peak=([\d.]+)\s*GB")
RE_PEAK_COMPONENT = re.compile(r"^\s*\[([A-E])\]\s*$")
RE_TERM = re.compile(r"^\s*(\w+)\.{2,}\s*([\d.]+)\s*$")
RE_PEAK_TOTAL = re.compile(r"^\s+(\w+)\.{2,}\s*([\d.]+)\s*$")

def parse_one(out_path):
    """Parse one gw_*.out file → dict of metrics."""
    if not out_path.exists():
        return None
    txt = out_path.read_text()
    res = {'file': str(out_path), 'hwm_pred_gb': None, 'mem_stats_peak_gb': None,
           'nvsmi_peak_gb': None, 'bottleneck': None,
           'band_chunk': None, 'r_chunk': None, 'n_r_chunks': None,
           'gflat_chunk_size': None,
           'peak_totals': {}, 'peak_components': {}}
    # HWM
    m = RE_HWM.search(txt)
    if m:
        res['hwm_pred_gb'] = float(m.group(1))
        res['bottleneck'] = m.group(2)
    # Chunks
    m = RE_BANDCHUNK.search(txt);    res['band_chunk'] = int(m.group(1)) if m else None
    m = RE_RCHUNK.search(txt)
    if m:
        res['r_chunk'] = int(m.group(1))
        res['n_r_chunks'] = int(m.group(2))
    m = RE_CSCHUNK.search(txt);      res['gflat_chunk_size'] = int(m.group(1)) if m else None
    # mem_probe peaks (the max over all probes is the mem_stats peak)
    peaks = []
    nvsmi_peaks = []
    for m in RE_PEAK.finditer(txt):
        pk = float(m.group(2))
        nv = float(m.group(4))
        if pk > 0:
            peaks.append(pk)
        if nv > 0:
            nvsmi_peaks.append(nv)
    if peaks:
        res['mem_stats_peak_gb'] = max(peaks)
    if nvsmi_peaks:
        res['nvsmi_peak_gb'] = max(nvsmi_peaks)
    # Per-peak component breakdown: extract first occurrence
    # Find "per-peak components" section
    pp_idx = txt.find('per-peak components')
    if pp_idx >= 0:
        # parse the following ~30 lines
        block = txt[pp_idx:pp_idx + 5000]
        peak_letter = None
        for line in block.splitlines():
            m = RE_PEAK_COMPONENT.match(line)
            if m:
                peak_letter = m.group(1)
                continue
            if peak_letter is None:
                continue
            ms = re.match(r"^\s*([A-Za-z_]\w*)[\._\s]+([\d.]+)\s*$", line)
            if ms:
                term = ms.group(1).rstrip('.')
                val = float(ms.group(2))
                res['peak_components'][f"{peak_letter}.{term}"] = val
            else:
                # blank line or new section
                if line.strip() == '' and peak_letter == 'E':
                    break
    # Peak totals (the 5 lines under "peak totals (GB/dev):")
    pt_idx = txt.find('peak totals')
    if pt_idx >= 0:
        block = txt[pt_idx:pt_idx + 1000]
        for line in block.splitlines()[1:8]:
            ms = re.match(r"^\s*(\w+)\.+\s*([\d.]+)\s*$", line)
            if ms:
                res['peak_totals'][ms.group(1)] = float(ms.group(2))
    return res

def main():
    print("# Si non-bispinor band-count sensitivity — summary table")
    print()
    print("| config | variant | band_chunk | r_chunk | n_chunks | gflat_cs | HWM_pred (GB/dev) | mem_stats peak | nvsmi peak | %-err |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for d in CONFIGS:
        for v in VARIANTS:
            out_path = ROOT / d / f"gw_{v}.out"
            r = parse_one(out_path)
            if r is None:
                continue
            rows[(d, v)] = r
            hwm = r['hwm_pred_gb']
            mst = r['mem_stats_peak_gb']
            nvp = r['nvsmi_peak_gb']
            pe = ((hwm - mst) / mst * 100) if (hwm and mst and mst > 0) else None
            pe_s = f"{pe:+.1f}%" if pe is not None else "—"
            print(f"| {d} | {v} | {r['band_chunk']} | {r['r_chunk']} | {r['n_r_chunks']} | {r['gflat_chunk_size']} | "
                  f"{hwm if hwm else '—'} | {mst if mst else '—'} | {nvp if nvp else '—'} | {pe_s} |")
    print()
    print("# Per-peak components — bfc_pre95 (true peak detection)")
    print()
    print("| config | Peak A | Peak B | Peak C | Peak D | Peak E | bottleneck |")
    print("|---|---|---|---|---|---|---|")
    for d in CONFIGS:
        v = 'bfc_pre95'
        r = rows.get((d, v))
        if r is None:
            continue
        pt = r['peak_totals']
        a = pt.get('A_centroid', '—')
        b = pt.get('B_CCT_chol', '—')
        c = pt.get('C_fit_one_rchunk', '—')
        D = pt.get('D_accumulate', '—')
        e = pt.get('E_v_q', '—')
        bn = r['bottleneck']
        print(f"| {d} | {a} | {b} | {c} | {D} | {e} | {bn} |")
    print()
    print("# Peak C component breakdown — bfc_pre95")
    print()
    print("| config | P_pair | zeta_out | centroids_persist | gflat_acc | L_q | sphere_idx |")
    print("|---|---|---|---|---|---|---|")
    for d in CONFIGS:
        v = 'bfc_pre95'
        r = rows.get((d, v))
        if r is None:
            continue
        pc = r['peak_components']
        pp = pc.get('C.P_pair_concurrent_slots', '—')
        zo = pc.get('C.zeta_out', '—')
        cp = pc.get('C.centroids_persist', '—')
        ga = pc.get('C.gflat_acc', '—')
        lq = pc.get('C.L_q', '—')
        si = pc.get('C.sphere_idx_replicated', '—')
        print(f"| {d} | {pp} | {zo} | {cp} | {ga} | {lq} | {si} |")

if __name__ == '__main__':
    main()
