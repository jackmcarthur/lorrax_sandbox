"""Compute predicted-vs-empirical scaling exponents for Agent B kgrid sweep."""
import json
import math
import numpy as np

with open("/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_b_planner_data.json") as f:
    data = json.load(f)

kg_meta = {
    "2x2x2": dict(nk=8, mu=48, ibz=3),
    "3x3x3": dict(nk=27, mu=192, ibz=4),
    "4x4x4": dict(nk=64, mu=432, ibz=8),
    "6x6x6": dict(nk=216, mu=1348, ibz=16),
}

# Use bfc_pre95 for mem_stats peak
print("\n=== Headline table: HWM_pred vs mem_stats peak ===")
print(f"{'kgrid':<6} {'nk':>4} {'μ':>5} {'r_chunk':>8} {'nc':>3} HWM_pred  mem_stats  %err     bottleneck")
results = []
for kg in ["2x2x2", "3x3x3", "4x4x4", "6x6x6"]:
    p = data[f"{kg}_bfc_pre95"]
    nk = kg_meta[kg]["nk"]
    mu = kg_meta[kg]["mu"]
    hwm = p["HWM_pred_gb"]
    peak = p["mem_stats_peak_gb"]
    perr = (hwm - peak) / peak * 100 if peak > 0 else float("nan")
    rch = p["r_chunk"]
    nc = p["n_chunks"]
    bn = p["bottleneck"]
    print(f"{kg:<6} {nk:>4} {mu:>5} {rch:>8} {nc:>3} {hwm:8.2f}  {peak:8.2f}  {perr:+6.1f}%  {bn}")
    results.append(dict(kg=kg, nk=nk, mu=mu, hwm_pred=hwm, mem_stats=peak, perr=perr, r_chunk=rch, n_chunks=nc, bottleneck=bn))

print("\n=== Predicted scaling exponents vs nk_full at fixed μ/nk_full ratio ===")
# Reference: with μ = 6 × nk, terms split into:
#   linear (nk^1): sphere_idx_replicated
#   nk^2: centroids_persist, A.centroid_out_filling, C/D.gflat_acc, E.zeta_L_all, psi_centroids
#   nk^3: P_l+P_r, C_q, L_q, P_pair_slots, zeta_out, zeta_chunk, V_acc, V_acc_full_BZ
#   indep (nk^0): fft_box, phase_table, accumulate_fft_box

# Collect per-term scaling
KEY_TERMS = [
    ("A.fft_box", 0),
    ("A.centroid_out_filling", 2),
    ("A.phase_table", 1),  # 16*nk*n_rtot - phase_table = nk (n_rtot fixed)
    ("A.sphere_idx_replicated", 1),
    ("B.centroids_persistent", 2),
    ("B.P_l_plus_P_r_open_spin", 3),
    ("B.C_q", 3),
    ("B.L_q", 3),
    ("C.P_pair_concurrent_slots", "nk*mu*r_chunk"),
    ("C.zeta_out", 3),
    ("C.centroids_persist", 2),
    ("C.gflat_acc", 2),
    ("C.L_q", 3),
    ("C.sphere_idx_replicated", 1),
    ("D.zeta_chunk", "nk*mu*r_chunk"),
    ("D.accumulate_fft_box", 0),
    ("D.centroids_persist", 2),
    ("D.gflat_acc", 2),
    ("D.L_q", 3),
    ("D.sphere_idx_replicated", 1),
    ("E.zeta_L_all", 2),
    ("E.V_acc", 3),
    ("E.V_acc_full_BZ", 3),
    ("E.zeta_L_on_x_axis", 1),  # μ × ngkmax / p_x : μ/(p_x) ~ μ ~ nk
    ("E.zeta_R_on_y_axis", 1),
    ("E.v_q_table_replicated", 1),  # n_q × ngkmax replicated
    ("E.psi_centroids_persistent", 2),
    ("E.V_q_block", 2),  # μ² (independent of nk)?
    ("E.g0_acc", 1),
    ("E.sphere_idx_replicated", 1),
]

# Empirical: log-log slope between adjacent kgrids
print(f"{'term':<35} {'pred_exp':>10}   {'2x2x2':>8}  {'3x3x3':>8}  {'4x4x4':>8}  {'6x6x6':>8}  {'empirical scaling vs nk':>30}")
nks = [kg_meta[kg]["nk"] for kg in ["2x2x2", "3x3x3", "4x4x4", "6x6x6"]]
analysis_rows = []
for term, pred_exp in KEY_TERMS:
    vals = []
    for kg in ["2x2x2", "3x3x3", "4x4x4", "6x6x6"]:
        v = data[f"{kg}_bfc_pre95"]["per_peak_gb"].get(term, 0)
        vals.append(v)
    # Compute log-log slopes (using nonzero pairs)
    valid = [(n, v) for n, v in zip(nks, vals) if v > 1e-5]
    if len(valid) >= 2:
        ns = np.log([v[0] for v in valid])
        vs = np.log([v[1] for v in valid])
        try:
            slope = np.polyfit(ns, vs, 1)[0]
        except Exception:
            slope = float("nan")
    else:
        slope = float("nan")
    print(f"{term:<35} {str(pred_exp):>10}   {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}  {vals[3]:8.4f}    empirical slope = {slope:+.2f}")
    analysis_rows.append(dict(term=term, pred_exp=pred_exp, vals=vals, slope=slope))

# Per-peak totals scaling
print("\n=== Peak totals (GB/dev) scaling ===")
print(f"{'peak':<25} {'2x2x2':>8}  {'3x3x3':>8}  {'4x4x4':>8}  {'6x6x6':>8}   empirical slope")
for peak in ["A_centroid", "B_CCT_chol", "C_fit_one_rchunk", "D_accumulate", "E_v_q"]:
    vals = []
    for kg in ["2x2x2", "3x3x3", "4x4x4", "6x6x6"]:
        v = data[f"{kg}_bfc_pre95"]["peak_totals_gb"].get(peak, 0)
        vals.append(v)
    valid = [(n, v) for n, v in zip(nks, vals) if v > 1e-5]
    if len(valid) >= 2:
        ns = np.log([v[0] for v in valid])
        vs = np.log([v[1] for v in valid])
        slope = np.polyfit(ns, vs, 1)[0]
    else:
        slope = float("nan")
    print(f"{peak:<25} {vals[0]:8.4f}  {vals[1]:8.4f}  {vals[2]:8.4f}  {vals[3]:8.4f}   {slope:+.2f}")

# Save
with open("/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_nonbispinor_kgrid_2026-05-18/agent_b_scaling_analysis.json", "w") as f:
    json.dump(dict(results=results, terms=analysis_rows), f, indent=2)
print("\nSaved analysis JSON.")
