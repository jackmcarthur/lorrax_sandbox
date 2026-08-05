"""Analyze the v2 16-GPU Σ^B end-to-end gate: per-(k, n) diff between run A & B.

Identical metric to the 2-GPU gate (analyze_gate.py at the parent run dir).
"""
import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14"
A = np.load(f"{BASE}/sigma_b_gate_16gpu_v2_A.npz")
B = np.load(f"{BASE}/sigma_b_gate_16gpu_v2_B.npz")

RYD_TO_EV = float(A["ryd_to_ev"])

for label, key in [("Sig^B (sig_x_b)", "sig_x_b"),
                   ("Sig_X total (scalar + Sig^B)", "sig_x_total"),
                   ("Sig_X scalar (charge)", "sig_x_scalar")]:
    sa = A[key] * RYD_TO_EV
    sb = B[key] * RYD_TO_EV
    print(f"\n=== {label} ===")
    print(f"  shape: {sa.shape}")
    nk, nb, _ = sa.shape

    da = np.real(np.diagonal(sa, axis1=1, axis2=2))
    db = np.real(np.diagonal(sb, axis1=1, axis2=2))
    diff_diag = db - da
    abs_diag = np.abs(diff_diag)
    abs_mat = np.abs(sb - sa)

    print(f"  diagonal |Delta|: max={abs_diag.max()*1000:.4f} meV  "
          f"mean={abs_diag.mean()*1000:.4f} meV")
    print(f"  matrix |Delta|:   max={abs_mat.max()*1000:.4f} meV  "
          f"mean={abs_mat.mean()*1000:.4f} meV")

    flat_idx = np.argsort(abs_diag, axis=None)[::-1][:5]
    print("  worst 5 diagonal (k, n):")
    for fi in flat_idx:
        k, n = np.unravel_index(fi, abs_diag.shape)
        print(f"    k={k:2d} n={n:2d}  A={da[k,n]:+.6f} eV  "
              f"B={db[k,n]:+.6f} eV  Delta={diff_diag[k,n]*1000:+.4f} meV")

sa_b = (A["sig_x_b"] * RYD_TO_EV)
sb_b = (B["sig_x_b"] * RYD_TO_EV)
da_b = np.real(np.diagonal(sa_b, axis1=1, axis2=2)) * 1000
db_b = np.real(np.diagonal(sb_b, axis1=1, axis2=2)) * 1000
diff_b = db_b - da_b
print(f"\n=== Sig^B[k, n, n] full per-(k, n) diagonal diff (real part, meV) ===")
print(f"  shape (nk, nb_sigma): {da_b.shape}")
print(f"  max |Delta|     = {np.abs(diff_b).max():.4f} meV")
print(f"  mean |Delta|    = {np.abs(diff_b).mean():.4f} meV")
print(f"  RMS  Delta      = {np.sqrt((diff_b**2).mean()):.4f} meV")

gate_value = np.abs(diff_b).max()
verdict = "PASS" if gate_value < 1.0 else "FAIL"
print(f"\n=== GATE VERDICT (max |Delta Sig^B diag| < 1 meV): {verdict}  ({gate_value:.4f} meV) ===")

sa_t = A["sig_x_total"] * RYD_TO_EV
sb_t = B["sig_x_total"] * RYD_TO_EV
da_t = np.real(np.diagonal(sa_t, axis1=1, axis2=2)) * 1000
db_t = np.real(np.diagonal(sb_t, axis1=1, axis2=2)) * 1000
diff_t = db_t - da_t
print(f"\n=== Sig_X total[k, n, n] full per-(k, n) diagonal diff (meV) ===")
print(f"  max |Delta|     = {np.abs(diff_t).max():.4f} meV")
print(f"  mean |Delta|    = {np.abs(diff_t).mean():.4f} meV")
print(f"  RMS  Delta      = {np.sqrt((diff_t**2).mean()):.4f} meV")

# Scalar charge channel — should be bit-identical
sa_s = A["sig_x_scalar"]; sb_s = B["sig_x_scalar"]
diff_s = sb_s - sa_s
print(f"\n=== Sig_X scalar (charge) — should be bit-identical ===")
print(f"  max |Delta|     = {np.abs(diff_s).max():.6e} (Ry, complex magnitude)")
