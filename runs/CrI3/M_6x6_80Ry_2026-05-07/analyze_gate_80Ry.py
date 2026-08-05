"""Analyze the Σ^B end-to-end gate at 80 Ry: per-(k, n) diff between run X & Y."""
import numpy as np

X = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/sigma_b_gate_80Ry_X.npz")
Y = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/sigma_b_gate_80Ry_Y.npz")

RYD_TO_EV = float(X["ryd_to_ev"])

for label, key in [("Σ^B (sig_x_b)", "sig_x_b"),
                   ("Σ_X total (scalar + Σ^B)", "sig_x_total"),
                   ("Σ_X scalar (charge)", "sig_x_scalar")]:
    sa = X[key] * RYD_TO_EV  # complex128 (nk, nb, nb), units eV
    sb = Y[key] * RYD_TO_EV
    print(f"\n=== {label} ===")
    print(f"  shape: {sa.shape}")
    nk, nb, _ = sa.shape

    da = np.real(np.diagonal(sa, axis1=1, axis2=2))   # (nk, nb)
    db = np.real(np.diagonal(sb, axis1=1, axis2=2))
    diff_diag = db - da
    abs_diag = np.abs(diff_diag)

    diff_mat_re = np.real(sb - sa)
    diff_mat_im = np.imag(sb - sa)
    abs_mat = np.abs(sb - sa)

    print(f"  diagonal |Δ|: max={abs_diag.max()*1000:.3f} meV"
          f"  ({abs_diag.max():.3e} eV)  "
          f"mean={abs_diag.mean()*1000:.4f} meV")
    print(f"  matrix |Δ|:   max={abs_mat.max()*1000:.3f} meV"
          f"  ({abs_mat.max():.3e} eV)  "
          f"mean={abs_mat.mean()*1000:.4f} meV")

    flat_idx = np.argsort(abs_diag, axis=None)[::-1][:5]
    print(f"  worst 5 diagonal (k, n):")
    for fi in flat_idx:
        k, n = np.unravel_index(fi, abs_diag.shape)
        print(f"    k={k:2d} n={n:2d}  X={da[k,n]:+.6f} eV  "
              f"Y={db[k,n]:+.6f} eV  Δ={diff_diag[k,n]*1000:+.4f} meV")

print("\n\n=== Σ^B[k, n, n] full per-(k, n) diagonal diff (real part, meV) ===")
sa_b = (X["sig_x_b"] * RYD_TO_EV)
sb_b = (Y["sig_x_b"] * RYD_TO_EV)
da_b = np.real(np.diagonal(sa_b, axis1=1, axis2=2)) * 1000  # meV
db_b = np.real(np.diagonal(sb_b, axis1=1, axis2=2)) * 1000
diff_b = db_b - da_b
print(f"  shape (nk, nb_sigma): {da_b.shape}")
print(f"  max |Δ Σ^B diag|     = {np.abs(diff_b).max():.4f} meV")
print(f"  mean |Δ Σ^B diag|    = {np.abs(diff_b).mean():.4f} meV")
print(f"  RMS  Δ Σ^B diag      = {np.sqrt((diff_b**2).mean()):.4f} meV")

gate_value = np.abs(diff_b).max()
verdict = "PASS" if gate_value < 1.0 else "FAIL"
print(f"\n=== GATE VERDICT (max |Δ Σ^B diag| < 1 meV): {verdict}  ({gate_value:.4f} meV) ===")

sa_t = X["sig_x_total"] * RYD_TO_EV
sb_t = Y["sig_x_total"] * RYD_TO_EV
da_t = np.real(np.diagonal(sa_t, axis1=1, axis2=2)) * 1000
db_t = np.real(np.diagonal(sb_t, axis1=1, axis2=2)) * 1000
diff_t = db_t - da_t
print(f"\n=== Σ_X total[k, n, n] full per-(k, n) diagonal diff (meV) ===")
print(f"  max |Δ Σ_X diag|     = {np.abs(diff_t).max():.4f} meV")
print(f"  mean |Δ Σ_X diag|    = {np.abs(diff_t).mean():.4f} meV")
print(f"  RMS  Δ Σ_X diag      = {np.sqrt((diff_t**2).mean()):.4f} meV")
