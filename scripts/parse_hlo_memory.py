"""Parse the memory-usage-report from an XLA HLO dump and cross-match
every allocation against the primitive shapes in my fit_one_rchunk
model.  Answer: what actually gets allocated at peak, which primitive
(if any) predicts each allocation, and is there misattribution?
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

OUT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/aot_memory_model_poc_2026-04-20/hlo_dumps")
fn = OUT / "module_0001.jit__apply.cpu_after_optimizations-memory-usage-report.txt"
text = fn.read_text()

# DoE baseline config at which this was compiled (from hlo_dump_fit_rchunk.py):
sys_ = dict(n_k=9, n_rmu=320, n_s=1, n_b=40, n_b_sum=80,
            n_r=80 * 72 * 8, cr=10000, bc=20, p_x=2, p_y=2)
P = sys_["p_x"] * sys_["p_y"]

# Primitive bytes-per-device at this config
B = 16
def _p(fn):
    return fn(sys_) * B

prims = {
    "pair":       lambda s: s["n_k"] * s["n_rmu"] * s["cr"] / P,
    "psiG_cache": lambda s: s["n_k"] * s["n_b"] * s["n_s"] * s["n_r"] / P,
    "psiG_bc":    lambda s: s["n_k"] * s["bc"] * s["n_s"] * s["n_r"] / P,
    "psiY_bc":    lambda s: s["n_k"] * s["bc"] * s["n_s"] * s["cr"] / s["p_y"],
    "psiXY_bc":   lambda s: s["n_k"] * s["bc"] * s["n_s"] * s["cr"] / P,
    "centroid":   lambda s: s["n_k"] * s["n_rmu"] * s["n_b_sum"] * s["n_s"] / s["p_x"],
    "centroidY":  lambda s: s["n_k"] * s["n_rmu"] * s["n_b_sum"] * s["n_s"] / s["p_y"],
    "Lq_shard":   lambda s: s["n_k"] * s["n_rmu"] ** 2 / P,
    "Lq_rep":     lambda s: s["n_k"] * s["n_rmu"] ** 2,
    "mu2_shard":  lambda s: s["n_k"] * s["n_rmu"] ** 2 / P,
    # pair-density-μ² (cct_lr's T_Pq shape): n_k * μ²
    "pair_mumu":  lambda s: s["n_k"] * s["n_rmu"] ** 2,
    # bc-shaped pair intermediate (from pair_density_traced)
    "pair_bc":    lambda s: s["n_k"] * s["n_rmu"] * s["bc"] * s["n_s"] / s["p_x"],
    "pair_bc_P":  lambda s: s["n_k"] * s["n_rmu"] * s["bc"] * s["n_s"] / P,
}
prim_bytes = {k: _p(v) for k, v in prims.items()}

print(f"Config: {sys_}")
print(f"\nPrimitive sizes (bytes per device):")
for k, v in sorted(prim_bytes.items(), key=lambda kv: -kv[1]):
    print(f"  {k:14s} = {v/1e6:>8.2f} MB")

# Find the top memory pools — lines with "cumulative_size; size; offset;"
# The first block is the DEFAULT color space peak.
print(f"\n{'='*78}")
print(f"Top memory pools from HLO report")
print(f"{'='*78}")
pool_re = re.compile(
    r'^\s*([\d\.]+)\s*([KMG]?i?B)\s*\(\s*(\d+)%\s*\);\s*'
    r'([\d\.]+)\s*([KMG]?i?B);\s*(\d+);\s*(\d+);\s*(.+?)$',
    re.MULTILINE,
)

def to_bytes(val: str, unit: str) -> float:
    v = float(val)
    mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
            "KB": 1e3, "MB": 1e6, "GB": 1e9, "": 1}
    return v * mult.get(unit, 1)

# First block is the main arena
first_block_end = text.find("The rest")
first_block = text[:first_block_end] if first_block_end > 0 else text

entries = []
for m in pool_re.finditer(first_block):
    cum, cum_u, pct, size, size_u, n_used, offset, shapes = m.groups()
    entries.append({
        "cum": to_bytes(cum, cum_u),
        "size": to_bytes(size, size_u),
        "pct": int(pct),
        "n_values": int(n_used),
        "offset": int(offset),
        "shapes": shapes.strip(),
    })

# Print top 30 pools
print(f"\n{'size':>10} {'n_vals':>6} {'≈primitive':>14}  shapes")
total_seen = 0.0
for e in entries[:30]:
    size_bytes = e["size"]
    # Find the closest-matching primitive
    best_prim, best_ratio = None, 1e30
    for pname, pbytes in prim_bytes.items():
        if pbytes == 0:
            continue
        ratio = size_bytes / pbytes
        if 0.4 <= ratio <= 2.5:
            # within-2× candidate
            dist = abs(ratio - round(ratio))
            if dist < best_ratio:
                best_prim, best_ratio = f"{pname}×{ratio:.2f}", dist
    tag = best_prim if best_prim else "unknown"
    total_seen += size_bytes
    shapes_short = (e["shapes"][:60] + "…") if len(e["shapes"]) > 60 else e["shapes"]
    print(f"  {size_bytes/1e6:>7.2f}MB  {e['n_values']:>5}  {tag:>14}  {shapes_short}")
print(f"\nSum of top pools = {total_seen/1e6:.1f} MB")

# Total from report
m = re.search(r"Total bytes:\s*(\d+)", text)
if m:
    total = int(m.group(1))
    print(f"\nTotal peak from HLO:  {total/1e6:.1f} MB")
