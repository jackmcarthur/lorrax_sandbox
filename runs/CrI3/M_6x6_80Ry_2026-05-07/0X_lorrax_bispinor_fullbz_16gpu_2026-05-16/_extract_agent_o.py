"""Extract Agent-O per-variant peak HBM stats from a gw.out log.

Reads mem_probe lines, returns dict with:
  - nvsmi_peak_gb  (max nvsmi_peak across all probe points; per-rank true HBM)
  - mem_stats_peak_gb (max peak_bytes_in_use across all probes; -1 if None throughout)
  - mem_stats_avail (True if any positive bytes_in_use seen — implies memory_stats works)
  - live_total_gb_global (max live_total reported)
  - planner_pred_gb (parsed from cohsex.in or planner WARNING; X3 reference 66.41)

Usage: python _extract_agent_o.py <gw.out>
"""
import re
import sys


def parse(path):
    rx = re.compile(
        r"\[mem_probe ([^\]]+)\] in_use=(\S+) GB\s+peak=(\S+) GB\s+"
        r"live_count=(\d+)\s+live_total=(\S+) GB\s+"
        r"nvsmi=(\S+) GB nvsmi_peak=(\S+) GB"
    )
    in_use_peak = -1.0
    mem_stats_peak = -1.0
    mem_stats_avail = False
    live_max = -1.0
    nvsmi_peak_max = -1.0
    n_probes = 0
    for line in open(path):
        m = rx.search(line)
        if not m:
            continue
        n_probes += 1
        label, in_use, peak, lc, live, nvsmi, nvsmi_peak = m.groups()
        try:
            inu = float(in_use)
            pk = float(peak)
            lv = float(live)
            np_ = float(nvsmi_peak)
            if inu > 0:
                mem_stats_avail = True
                if inu > in_use_peak:
                    in_use_peak = inu
            if pk > 0:
                mem_stats_avail = True
                if pk > mem_stats_peak:
                    mem_stats_peak = pk
            if lv > live_max:
                live_max = lv
            if np_ > nvsmi_peak_max:
                nvsmi_peak_max = np_
        except ValueError:
            pass
    return {
        "n_probes": n_probes,
        "nvsmi_peak_gb": nvsmi_peak_max,
        "mem_stats_peak_gb": mem_stats_peak,
        "mem_stats_bytes_in_use_gb": in_use_peak,
        "mem_stats_avail": mem_stats_avail,
        "live_total_gb_global": live_max,
    }


if __name__ == "__main__":
    path = sys.argv[1]
    out = parse(path)
    print(f"file: {path}")
    for k, v in out.items():
        print(f"  {k}: {v}")
