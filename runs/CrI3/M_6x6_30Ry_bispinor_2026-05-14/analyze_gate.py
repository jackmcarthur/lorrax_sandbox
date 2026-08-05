"""Analyze the Σ^B end-to-end gate: per-(k, n) diff between run A & B."""
import numpy as np

A = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/sigma_b_gate_A.npz")
B = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/sigma_b_gate_B.npz")

# Ryd-eV factor
RYD_TO_EV = float(A["ryd_to_ev"])

# Choose the comparison object.  Σ^B is "the bispinor part" (sig_x_b);
# total Σ_X = sig_x_scalar + sig_x_b is the full bispinor-aware bare Σ_X.
# Both gates are physically meaningful — print both.
for label, key in [("Σ^B (sig_x_b)", "sig_x_b"),
                   ("Σ_X total (scalar + Σ^B)", "sig_x_total"),
                   ("Σ_X scalar (charge)", "sig_x_scalar")]:
    sa = A[key] * RYD_TO_EV  # complex128 (nk, nb, nb), units eV
    sb = B[key] * RYD_TO_EV
    print(f"\n=== {label} ===")
    print(f"  shape: {sa.shape}")
    nk, nb, _ = sa.shape

    # Diagonal Σ[k, n, n]
    da = np.real(np.diagonal(sa, axis1=1, axis2=2))   # (nk, nb)
    db = np.real(np.diagonal(sb, axis1=1, axis2=2))
    diff_diag = db - da
    abs_diag = np.abs(diff_diag)

    # Per-element matrix diff (all m, n entries — real part)
    diff_mat_re = np.real(sb - sa)
    diff_mat_im = np.imag(sb - sa)
    abs_mat = np.abs(sb - sa)  # complex magnitude

    print(f"  diagonal |Δ|: max={abs_diag.max()*1000:.3f} meV"
          f"  ({abs_diag.max():.3e} eV)  "
          f"mean={abs_diag.mean()*1000:.4f} meV")
    print(f"  matrix |Δ|:   max={abs_mat.max()*1000:.3f} meV"
          f"  ({abs_mat.max():.3e} eV)  "
          f"mean={abs_mat.mean()*1000:.4f} meV")

    # Top-5 worst diagonal (k, n) entries
    flat_idx = np.argsort(abs_diag, axis=None)[::-1][:5]
    print(f"  worst 5 diagonal (k, n):")
    for fi in flat_idx:
        k, n = np.unravel_index(fi, abs_diag.shape)
        print(f"    k={k:2d} n={n:2d}  A={da[k,n]:+.6f} eV  "
              f"B={db[k,n]:+.6f} eV  Δ={diff_diag[k,n]*1000:+.4f} meV")

# Σ^B per (k, n) diagonal — full table for the gate (only Σ^B, the new
# transverse part).
print("\n\n=== Σ^B[k, n, n] full per-(k, n) diagonal diff (real part, meV) ===")
sa_b = (A["sig_x_b"] * RYD_TO_EV)
sb_b = (B["sig_x_b"] * RYD_TO_EV)
da_b = np.real(np.diagonal(sa_b, axis1=1, axis2=2)) * 1000  # meV
db_b = np.real(np.diagonal(sb_b, axis1=1, axis2=2)) * 1000
diff_b = db_b - da_b
print(f"  shape (nk, nb_sigma): {da_b.shape}")
print(f"  max |Δ Σ^B diag|     = {np.abs(diff_b).max():.4f} meV")
print(f"  mean |Δ Σ^B diag|    = {np.abs(diff_b).mean():.4f} meV")
print(f"  RMS  Δ Σ^B diag      = {np.sqrt((diff_b**2).mean()):.4f} meV")

# 1 meV gate verdict
gate_value = np.abs(diff_b).max()
verdict = "PASS" if gate_value < 1.0 else "FAIL"
print(f"\n=== GATE VERDICT (max |Δ Σ^B diag| < 1 meV): {verdict}  ({gate_value:.4f} meV) ===")

# Same for total Σ_X
sa_t = A["sig_x_total"] * RYD_TO_EV
sb_t = B["sig_x_total"] * RYD_TO_EV
da_t = np.real(np.diagonal(sa_t, axis1=1, axis2=2)) * 1000
db_t = np.real(np.diagonal(sb_t, axis1=1, axis2=2)) * 1000
diff_t = db_t - da_t
print(f"\n=== Σ_X total[k, n, n] full per-(k, n) diagonal diff (meV) ===")
print(f"  max |Δ Σ_X diag|     = {np.abs(diff_t).max():.4f} meV")
print(f"  mean |Δ Σ_X diag|    = {np.abs(diff_t).mean():.4f} meV")
print(f"  RMS  Δ Σ_X diag      = {np.sqrt((diff_t**2).mean()):.4f} meV")
