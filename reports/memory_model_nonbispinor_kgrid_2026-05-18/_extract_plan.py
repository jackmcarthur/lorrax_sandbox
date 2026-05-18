"""Extract G-flat memory model breakdown from gw.out files."""
import os, re, json, sys

def parse_block(path):
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"G-flat memory model — chunk plan \+ HWM estimate(.+?)(?=\n[^\s]|\Z)", text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    result = {}
    # chunk plan
    bcm = re.search(r"band_chunk\s+=\s+(\d+)", block)
    rcm = re.search(r"r_chunk\s+=\s+(\d+)\s+\((\d+) chunks\)", block)
    cs = re.search(r"gflat_chunk_size\s+=\s+(\d+)", block)
    hwm = re.search(r"HWM estimate\s+=\s+([\d.]+) GB/dev.*?bottleneck:\s+(\w+)", block)
    if bcm: result["band_chunk"] = int(bcm.group(1))
    if rcm:
        result["r_chunk"] = int(rcm.group(1))
        result["n_chunks"] = int(rcm.group(2))
    if cs: result["gflat_chunk_size"] = int(cs.group(1))
    if hwm:
        result["HWM_pred_gb"] = float(hwm.group(1))
        result["bottleneck"] = hwm.group(2)
    # peak totals
    peak_totals = {}
    for m in re.finditer(r"^\s+(\w+_\w+)\.+\s+([\d.]+)", block, re.MULTILINE):
        name = m.group(1)
        val = float(m.group(2))
        if "." in m.group(0).strip().split()[0]:  # peak total line
            peak_totals[name] = val
    # this regex catches both peak_totals and components. Distinguish by line indent depth
    # Re-parse with line-by-line awareness
    section = None
    peak_totals = {}
    per_peak = {}
    for line in block.split("\n"):
        if re.match(r"\s+peak totals", line):
            section = "totals"
            continue
        if re.match(r"\s+per-peak components", line):
            section = "components"
            continue
        # group header [A] [B] etc
        gm = re.match(r"\s+\[(\w)\]\s*$", line)
        if gm and section == "components":
            cur_group = gm.group(1)
            continue
        # peak totals: name........  X.XX  (6 dots)
        if section == "totals":
            tm = re.match(r"\s+(\w+)\.+\s+([\d.]+)", line)
            if tm:
                peak_totals[tm.group(1)] = float(tm.group(2))
        if section == "components":
            cm = re.match(r"\s+([A-Za-z_][\w]*)\.*\s+([\d.]+)\s*$", line)
            if cm:
                per_peak[f"{cur_group}.{cm.group(1)}"] = float(cm.group(2))
    result["peak_totals_gb"] = peak_totals
    result["per_peak_gb"] = per_peak
    # mem_stats peak
    mp = re.findall(r"\[mem_probe\s+\w+(?:\s+chunk=\d+)?\]\s+in_use=[\d.\-]+\s+GB\s+peak=([\d.\-]+)\s+GB", text)
    if mp:
        peaks = [float(p) for p in mp]
        result["mem_stats_peak_gb"] = max(peaks)
    nvs = re.findall(r"nvsmi_peak=([\d.\-]+)\s+GB", text)
    if nvs:
        result["nvsmi_peak_gb"] = max(float(p) for p in nvs)
    return result


base = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/KGRID_nonbispinor_2026-05-18"
out = {}
for kg in ["2x2x2", "3x3x3", "4x4x4", "6x6x6"]:
    for variant in ["platform_false", "bfc_pre95"]:
        p = f"{base}/{kg}/mu_run/gw_{variant}.out"
        if not os.path.exists(p):
            continue
        r = parse_block(p)
        if r:
            out[f"{kg}_{variant}"] = r

print(json.dumps(out, indent=2))
