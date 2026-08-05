"""Extract planner prediction + probe values from a gw.out file.

Usage: python3 _extract_config.py <out_file>

Prints:
  config: <name>
  band_chunk, r_chunk, gflat_chunk_size (planner-resolved)
  HWM_pred_GB_per_dev, bottleneck
  Peak A/B/C/D/E totals (GB/dev)
  visible_persistent_pred_GB_global (sum of live-arrays-visible persistent terms × p_xy_factor)
  pre_rchunk_loop live_total (global GB, mean over chunks listed)
  after_fit_one_rchunk live_total (max across chunks)
  after_accumulate live_total
  post_v_q live_total (if present)
  zeta_chunk visible bytes (global)
  per-signature counts: centroids/sphere/gflat_acc/L_q/zeta_chunk
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

P_XY = 16  # 4x4 mesh
# CrI3 production constants
NK = 36
NS = 2
MU = 1520
NB_L = 150
NB_R = 160
NGKMAX = 59990
NX, NY, NZ = 75, 75, 200
NQ = 36
NQ_DISK = 36

# global bytes for each persistent term
def gb(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 16 / 1e9

def gb_i32(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 4 / 1e9

# Centroid buffers — count is dynamic based on which probe.  At after_fit:
# live_arrays shows two of each (rmuT_X + Y-form) for L AND R.
# At pre_rchunk_loop (refit_verify_natural V1) we see:
#   complex128 (36, 1520, 160, 4) x 2 = 1.12 GB  (R: rmuT_X + Y)
#   complex128 (36, 160, 4, 1520) x 2 = 1.12 GB  (R reshape) — actually agent already showed this as ×2
#   complex128 (36, 1520, 150, 4) x 2 = 1.05 GB  (L)
#   complex128 (36, 150, 4, 1520) x 2 = 1.05 GB  (L reshape)
# Total = 4.34 GB global.  This is "4 buffers per L/R" = 4 logical buffers total
# (each appearing in two memory layouts).

def parse_planner(text: str):
    """Pick out the chunk plan + HWM + per-peak totals."""
    m = re.search(r"G-flat memory model — chunk plan \+ HWM estimate", text)
    if not m:
        return None
    block = text[m.start():m.start()+6000]
    out = {}
    m2 = re.search(r"band_chunk\s*=\s*(\d+)", block)
    out["band_chunk"] = int(m2.group(1)) if m2 else None
    m2 = re.search(r"r_chunk\s*=\s*(\d+)\s*\((\d+) chunks\)", block)
    out["r_chunk"] = int(m2.group(1)) if m2 else None
    out["n_r_chunks"] = int(m2.group(2)) if m2 else None
    m2 = re.search(r"gflat_chunk_size\s*=\s*(\d+)", block)
    out["gflat_chunk_size"] = int(m2.group(1)) if m2 else None
    m2 = re.search(r"HWM estimate\s*=\s*([\d.]+)\s*GB/dev\s*\((\d+)% of budget\)\s*\[bottleneck:\s*(\S+)\]", block)
    if m2:
        out["hwm_pred_gb_per_dev"] = float(m2.group(1))
        out["budget_pct"] = int(m2.group(2))
        out["bottleneck"] = m2.group(3).rstrip("]")
    # Peak totals
    peaks = {}
    for pl, name in [("A_centroid","A"), ("B_CCT_chol","B"), ("C_fit_one_rchunk","C"),
                     ("D_accumulate","D"), ("E_v_q","E")]:
        m2 = re.search(rf"{pl}\.+\s+([\d.]+)", block)
        if m2:
            peaks[name] = float(m2.group(1))
    out["peaks_gb_per_dev"] = peaks
    return out

PROBE_RE = re.compile(
    r"\[mem_probe ([^\]]+)\] in_use=[\-\d.]+ GB\s+peak=[\-\d.]+ GB\s+"
    r"live_count=(\d+) live_total=([\d.]+) GB"
)
SIG_RE = re.compile(
    r"\[mem_probe ([^\]]+)\]\s+(\S+)\s+\((.+?)\) x (\d+) = ([\d.]+) GB"
)


def parse_probes(text: str):
    """Return list of (label_key, live_count, live_total, list[(dtype,shape,count,gb)])."""
    probes = []
    current = None
    for ln in text.splitlines():
        m = PROBE_RE.match(ln)
        if m:
            if current:
                probes.append(current)
            label = m.group(1)
            current = (label, int(m.group(2)), float(m.group(3)), [])
            continue
        m = SIG_RE.match(ln)
        if m and current and m.group(1) == current[0]:
            dtype = m.group(2)
            shape = tuple(int(x) for x in m.group(3).split(",") if x.strip())
            cnt = int(m.group(4))
            sz = float(m.group(5))
            current[3].append((dtype, shape, cnt, sz))
    if current:
        probes.append(current)
    return probes


def aggregate_signature(probes, sig_pattern, label_substr):
    """For each probe whose label contains `label_substr`, sum bytes of arrays whose shape
    matches sig_pattern.  Returns dict label -> (count, total_gb)."""
    out = {}
    for label, lc, lt, sigs in probes:
        if label_substr not in label:
            continue
        cnt = 0
        gb = 0.0
        for dtype, shape, c, sz in sigs:
            if sig_pattern(dtype, shape):
                cnt += c
                gb += sz
        out[label] = (cnt, gb)
    return out


def main(path):
    text = Path(path).read_text()
    plan = parse_planner(text)
    probes = parse_probes(text)

    print(f"=== {path} ===")
    if plan:
        print(f"  PLAN: band={plan['band_chunk']}  r={plan['r_chunk']} "
              f"({plan['n_r_chunks']} chunks)  cs={plan['gflat_chunk_size']}")
        print(f"  HWM_pred = {plan.get('hwm_pred_gb_per_dev','?')} GB/dev  "
              f"bottleneck={plan.get('bottleneck','?')}")
        peaks = plan.get("peaks_gb_per_dev", {})
        for k in "ABCDE":
            if k in peaks:
                print(f"    Peak {k}: {peaks[k]:.2f} GB/dev")

    # Probe summaries (labels of interest)
    labels_of_interest = [
        "zeta_fit_start", "pre_rchunk_loop", "rchunk_start", "after_fit_one_rchunk",
        "after_accumulate", "zeta_fit_end", "pre_v_q", "post_v_q",
    ]
    by_label = {}
    for label, lc, lt, sigs in probes:
        # collapse "after_fit_one_rchunk chunk=0" -> base
        base = label.split(" ")[0]
        by_label.setdefault(base, []).append((label, lc, lt, sigs))

    for base in labels_of_interest:
        if base not in by_label:
            continue
        entries = by_label[base]
        totals = [lt for (_l, _lc, lt, _s) in entries]
        if not totals:
            continue
        print(f"  probe {base}: n={len(totals)}  "
              f"live_total min={min(totals):.2f} max={max(totals):.2f} mean={sum(totals)/len(totals):.2f} GB (global)")
        # show top signatures from first occurrence
        _l, _lc, _lt, sigs = entries[0]
        # rank by total bytes
        topsigs = sorted(sigs, key=lambda s: -s[3])[:6]
        for dtype, shape, cnt, sz in topsigs:
            print(f"      {dtype}{shape} x{cnt} = {sz:.2f} GB")

    # ---- Signature counts as user requested
    print()
    print("  Signature counts:")
    # centroids: c128(nk, mu, nb, ns) — appears as (nk, mu, nb, 4) and (nk, nb, 4, mu)
    def is_centroid(dtype, shape):
        if dtype != "complex128":
            return False
        # match (nk, mu, nb, ns) or (nk, nb, ns, mu) forms with nk=36, mu=1520, ns=2 or 4
        if len(shape) != 4:
            return False
        if 1520 not in shape or shape[0] != 36:
            return False
        # nb in {150, 160}
        if not any(s in (150, 160) for s in shape):
            return False
        return True

    def is_sphere(dtype, shape):
        return dtype == "int32" and shape == (36, 75, 75, 200)

    def is_gflat_acc(dtype, shape):
        return dtype == "complex128" and shape == (36, 1520, 59990)

    def is_L_q(dtype, shape):
        return dtype == "complex128" and shape == (36, 1520, 1520)

    def is_zeta_chunk(dtype, shape):
        # (36, 1520, r_chunk) — r_chunk varies but exclude L_q (1520, 1520) and gflat_acc (ngkmax)
        if dtype != "complex128":
            return False
        if len(shape) != 3:
            return False
        if shape[0] != 36 or shape[1] != 1520:
            return False
        if shape[2] in (1520, 59990):
            return False
        return True

    for sig_name, pred in [
        ("centroids c128(nk,mu,nb,ns)", is_centroid),
        ("sphere-idx i32(nk,nx,ny,nz)", is_sphere),
        ("gflat_acc c128(nq_disk,mu,ngkmax)", is_gflat_acc),
        ("L_q c128(nq,mu,mu)", is_L_q),
        ("zeta_chunk c128(nq_disk,mu,r_chunk)", is_zeta_chunk),
    ]:
        # by-label counts at after_fit_one_rchunk and post_v_q etc.
        for base in ("pre_rchunk_loop", "after_fit_one_rchunk", "after_accumulate", "post_v_q"):
            if base not in by_label:
                continue
            entries = by_label[base]
            l, lc, lt, sigs = entries[0]
            cnt = sum(c for (dt, sh, c, sz) in sigs if pred(dt, sh))
            gb = sum(sz for (dt, sh, c, sz) in sigs if pred(dt, sh))
            print(f"    {sig_name:42s} at {base:25s}: count={cnt}  total={gb:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1])
