"""Build the predicted-vs-observed comparison table for the Round-3 audit.

For each config, parse:
  * planner pick (band_chunk, r_chunk, gflat_chunk_size, HWM_pred, peaks)
  * probe live_total at after_fit_one_rchunk (max across rchunk iterations)
  * Compute the "live-arrays-visible" prediction (persistent + zeta_chunk
    transient, in GLOBAL bytes) and compare to observed.

The comparison metric is:
  pred_live_after_fit = gflat_acc + L_q + 4×centroid_layouts + N_sphere × sphere_buf
                        + zeta_chunk_transient + small (phase_table, etc.)
"""

import re
import sys
from pathlib import Path

# CrI3 production constants
NK = 36
NS_PHASE = 4  # bispinor layout shows ns=4 in (mu, nb, ns) ordering
NS = 2
MU_C = 1520  # charge channel μ (n_rmu_padded)
MU_T = 1504  # transverse channel μ
NB_L = 150
NB_R = 160
NGKMAX = 59990
NX, NY, NZ = 75, 75, 200
NQ = 36
NQ_DISK = 36
P_XY = 16


def gb_c128(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 16 / 1e9


def gb_i32(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 4 / 1e9


PRED_SPHERE_PER_BUF = gb_i32(NK, NX, NY, NZ)            # = 0.162 GB per buffer
PRED_SMALL = 0.04  # phase_table + g_index + small replicas


def gflat_acc_bytes(mu):
    return gb_c128(NQ_DISK, mu, NGKMAX)


def lq_bytes(mu):
    return gb_c128(NQ, mu, mu)


def centroids_bytes(mu):
    """At after_fit, 8 buffers (×2 layouts × 2 L/R)."""
    L = 2 * gb_c128(NK, mu, NB_L, NS_PHASE) + 2 * gb_c128(NK, NB_L, NS_PHASE, mu)
    R = 2 * gb_c128(NK, mu, NB_R, NS_PHASE) + 2 * gb_c128(NK, NB_R, NS_PHASE, mu)
    return L + R


def cross_channel_centroid_residual(mu_current):
    """If transverse channel: charge-channel R-centroids stay live as residual
    (just (NK, MU_C, NB_R, NS_PHASE) ×1 + transpose ×1 = ~1.12 GB)."""
    if mu_current == MU_C:
        return 0.0
    # residual from charge channel: R-centroid at MU_C, ×2 layouts (×1 each, not ×2)
    return gb_c128(NK, MU_C, NB_R, NS_PHASE) + gb_c128(NK, NB_R, NS_PHASE, MU_C)


def pred_live_after_fit(r_chunk, n_sphere, mu_current):
    zeta = gb_c128(NQ_DISK, mu_current, r_chunk)
    sphere = n_sphere * PRED_SPHERE_PER_BUF
    return (gflat_acc_bytes(mu_current) + lq_bytes(mu_current)
            + centroids_bytes(mu_current)
            + cross_channel_centroid_residual(mu_current)
            + sphere + zeta + PRED_SMALL)


def pred_live_pre_rchunk(n_sphere, mu_current):
    sphere = n_sphere * PRED_SPHERE_PER_BUF
    return (gflat_acc_bytes(mu_current) + lq_bytes(mu_current)
            + centroids_bytes(mu_current)
            + cross_channel_centroid_residual(mu_current)
            + sphere + PRED_SMALL)


def parse_run(path: Path):
    text = path.read_text()
    out = {"path": path.name}

    # plan
    m = re.search(r"band_chunk\s*=\s*(\d+)", text)
    out["band_chunk"] = int(m.group(1)) if m else None
    m = re.search(r"r_chunk\s*=\s*(\d+)\s*\((\d+) chunks\)", text)
    out["r_chunk"] = int(m.group(1)) if m else None
    out["n_chunks"] = int(m.group(2)) if m else None
    m = re.search(r"gflat_chunk_size\s*=\s*(\d+)", text)
    out["gflat_chunk_size"] = int(m.group(1)) if m else None
    m = re.search(r"HWM estimate\s*=\s*([\d.]+)\s*GB/dev[^\n]*\[bottleneck:\s*(\S+)", text)
    if m:
        out["hwm_pred_gb_per_dev"] = float(m.group(1))
        out["bottleneck"] = m.group(2).rstrip("]")
    # per-peak totals
    peaks = {}
    for pl, key in [("A_centroid", "A"), ("B_CCT_chol", "B"),
                    ("C_fit_one_rchunk", "C"), ("D_accumulate", "D"),
                    ("E_v_q", "E")]:
        m = re.search(rf"{pl}\.+\s+([\d.]+)\s*$", text, re.MULTILINE)
        if m:
            peaks[key] = float(m.group(1))
    out["peaks_gb_per_dev"] = peaks

    # parse probes
    probe_re = re.compile(
        r"\[mem_probe ([^\]]+)\] in_use=[\-\d.]+ GB\s+peak=[\-\d.]+ GB\s+"
        r"live_count=(\d+) live_total=([\d.]+) GB"
    )
    sig_re = re.compile(
        r"\[mem_probe ([^\]]+)\]\s+(\S+)\s+\(([^)]+)\) x (\d+) = ([\d.]+) GB"
    )

    after_fits = []  # (live_total, sphere_count, zeta_bytes)
    pre_rchunks = []
    after_accumulates = []
    post_v_qs = []
    pre_v_qs = []

    current = None
    bucket_after_fit = []
    bucket_pre_rchunk = []
    bucket_after_acc = []
    bucket_post_v_q = []
    bucket_pre_v_q = []

    for ln in text.splitlines():
        m = probe_re.match(ln)
        if m:
            if current:
                _flush(current, bucket_after_fit, bucket_pre_rchunk,
                       bucket_after_acc, bucket_post_v_q, bucket_pre_v_q)
            label = m.group(1)
            current = {"label": label, "live_total": float(m.group(3)),
                       "sigs": []}
            continue
        m = sig_re.match(ln)
        if m and current and m.group(1) == current["label"]:
            shape = tuple(int(x) for x in m.group(3).split(",") if x.strip())
            current["sigs"].append({
                "dtype": m.group(2), "shape": shape,
                "count": int(m.group(4)), "gb": float(m.group(5)),
            })
    if current:
        _flush(current, bucket_after_fit, bucket_pre_rchunk,
               bucket_after_acc, bucket_post_v_q, bucket_pre_v_q)

    out["after_fits"] = bucket_after_fit
    out["pre_rchunks"] = bucket_pre_rchunk
    out["after_accumulates"] = bucket_after_acc
    out["post_v_qs"] = bucket_post_v_q
    out["pre_v_qs"] = bucket_pre_v_q
    return out


def _flush(c, after_fit, pre_rchunk, after_acc, post_vq, pre_vq):
    lbl = c["label"]
    sphere_n = 0
    zeta_b = 0.0
    centroid_n = 0
    centroid_b = 0.0
    gflat_b = 0.0
    lq_b = 0.0
    mu_current = None  # 1520 charge, 1504 transverse
    for s in c["sigs"]:
        if s["dtype"] == "int32" and s["shape"] == (NK, NX, NY, NZ):
            sphere_n += s["count"]
        if (s["dtype"] == "complex128"
            and s["shape"] in [(NK, MU_C, NGKMAX), (NK, MU_T, NGKMAX)]):
            gflat_b += s["gb"]
            mu_current = s["shape"][1]
        if (s["dtype"] == "complex128"
            and s["shape"] in [(NK, MU_C, MU_C), (NK, MU_T, MU_T)]):
            lq_b += s["gb"]
            if mu_current is None:
                mu_current = s["shape"][1]
        # zeta_chunk shape (NK, mu, r) where mu in {1504, 1520} and r is r_chunk
        if (s["dtype"] == "complex128"
            and len(s["shape"]) == 3
            and s["shape"][0] == NK
            and s["shape"][1] in (MU_C, MU_T)
            and s["shape"][2] not in (MU_C, MU_T, NGKMAX)):
            zeta_b += s["gb"]
        # centroids: 4-dim, NK first, with mu in shape and nb in (150, 160)
        if (s["dtype"] == "complex128"
            and len(s["shape"]) == 4
            and s["shape"][0] == NK
            and (MU_C in s["shape"] or MU_T in s["shape"])
            and any(x in (NB_L, NB_R) for x in s["shape"])):
            centroid_n += s["count"]
            centroid_b += s["gb"]
    rec = {
        "label": lbl,
        "live_total": c["live_total"],
        "sphere_n": sphere_n,
        "zeta_b": zeta_b,
        "centroid_n": centroid_n,
        "centroid_b": centroid_b,
        "gflat_b": gflat_b,
        "lq_b": lq_b,
        "mu_current": mu_current,
    }
    if "after_fit_one_rchunk" in lbl:
        after_fit.append(rec)
    elif "pre_rchunk_loop" in lbl:
        pre_rchunk.append(rec)
    elif "after_accumulate" in lbl:
        after_acc.append(rec)
    elif "post_v_q" in lbl:
        post_vq.append(rec)
    elif "pre_v_q" in lbl:
        pre_vq.append(rec)


def report_config(label, parsed):
    print(f"## {label}")
    print(f"  Path: {parsed['path']}")
    print(f"  PLAN: band={parsed['band_chunk']} r={parsed['r_chunk']} cs={parsed['gflat_chunk_size']}")
    print(f"  HWM_pred = {parsed.get('hwm_pred_gb_per_dev','?')} GB/dev "
          f"[bottleneck: {parsed.get('bottleneck','?')}]")
    peaks = parsed.get("peaks_gb_per_dev", {})
    for k in "ABCDE":
        if k in peaks:
            print(f"    Peak {k}: {peaks[k]:.2f} GB/dev")

    print()
    print("  Live-arrays at after_fit_one_rchunk (per chunk):")
    if not parsed["after_fits"]:
        print("    (no after_fit_one_rchunk probe captured)")
    for rec in parsed["after_fits"]:
        mu = rec["mu_current"] or MU_C
        pred = pred_live_after_fit(parsed["r_chunk"], rec["sphere_n"], mu)
        obs = rec["live_total"]
        err = 100.0 * (pred - obs) / obs if obs > 0 else 0.0
        print(f"    {rec['label']:>40s}: live={obs:.2f} GB  "
              f"pred={pred:.2f} GB  err={err:+.1f}%  "
              f"sphere_n={rec['sphere_n']} mu={mu} zeta_b={rec['zeta_b']:.2f}")

    # Worst-case after_fit:
    if parsed["after_fits"]:
        worst = max(parsed["after_fits"], key=lambda r: r["live_total"])
        mu = worst["mu_current"] or MU_C
        pred = pred_live_after_fit(parsed["r_chunk"], worst["sphere_n"], mu)
        obs = worst["live_total"]
        err = 100.0 * (pred - obs) / obs if obs > 0 else 0.0
        print(f"  WORST-CASE after_fit ({worst['label']}): "
              f"obs={obs:.2f} GB pred={pred:.2f} GB err={err:+.1f}% mu={mu}")

    if parsed["pre_rchunks"]:
        print("  Live-arrays at pre_rchunk_loop:")
        for rec in parsed["pre_rchunks"]:
            mu = rec["mu_current"] or MU_C
            pred = pred_live_pre_rchunk(rec["sphere_n"], mu)
            obs = rec["live_total"]
            err = 100.0 * (pred - obs) / obs if obs > 0 else 0.0
            print(f"    {rec['label']:>40s}: live={obs:.2f} GB  "
                  f"pred={pred:.2f} GB  err={err:+.1f}%  sphere_n={rec['sphere_n']} mu={mu}")

    if parsed["pre_v_qs"]:
        print("  Live-arrays at pre_v_q:")
        for rec in parsed["pre_v_qs"]:
            obs = rec["live_total"]
            print(f"    {rec['label']:>40s}: live={obs:.2f} GB sphere_n={rec['sphere_n']}")
    if parsed["post_v_qs"]:
        print("  Live-arrays at post_v_q:")
        for rec in parsed["post_v_qs"]:
            obs = rec["live_total"]
            print(f"    {rec['label']:>40s}: live={obs:.2f} GB sphere_n={rec['sphere_n']}")

    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        parsed = parse_run(Path(p))
        report_config(Path(p).stem, parsed)
