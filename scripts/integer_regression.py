"""Refit fit_one_rchunk's 21 DoE samples against a *minimal* set of
ACTUAL-VARIABLE products drawn from the small-kernel fits:

    A = n_k · n_rmu · cr / (p_x · p_y)          [pair-sized, from zct_lr]
    B = n_k · n_b · n_s · n_r / (p_x · p_y)      [full psiG cache, from load_psi_fft × nbc]
    C = n_k · n_rmu · n_b_sum · n_s / p_x        [centroid L+R, from pair_density]
    D = n_k · n_rmu² / (p_x · p_y)               [L_q sharded]
    E = n_k · n_rmu²                             [L_q replicated, collinear with D at fixed P]
    F = n_k · bc · n_s · cr / p_y                [rchunk_y, from load_psi_reshard]
    G = n_k · bc · n_s · n_r / (p_x · p_y)       [psi_G band-chunk, from load_psi_fft]
    H = n_k · bc · n_s · cr / (p_x · p_y)        [rchunk_xy, from load_psi_reshard]  *MISSING*
    I = n_k · bc · n_s · n_r / p_y               [psi_G slice on y-mesh only — hypothetical]

Goal: NNLS-fit `total` = Σ C_i · V_i with C_i NON-NEGATIVE, and report
which combinations come out as clean integers.  Accept non-integer
coefficients ONLY if a smaller variable-combo has the residual
absorbed cleanly.

No JAX, no GPU — runs offline on the existing samples.json.
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import nnls

ART = Path("/global/homes/j/jackm/software/lorrax_C/src/gw/aot_memory_model/artifacts")
samples = json.load(open(ART / "fit_one_rchunk__current__samples.json"))

_B = 16.0


def get(s, key, default=None):
    for d in (s["sys"], s["knobs"], s["mesh"]):
        if key in d:
            return d[key]
    return default


def primitives_at(s):
    n_k = int(np.prod(s["sys"]["kgrid"]))
    mu = s["sys"]["n_rmu"]
    n_s = s["sys"]["n_s"]
    n_b = s["sys"]["n_b"]
    nb_sum = s["sys"].get("n_b_sum") or (2 * n_b)
    fft = s["sys"].get("fft_grid")
    n_r = (fft[0] * fft[1] * fft[2]) if fft else s["sys"].get("n_r", 0)
    if n_r == 0:
        n_r = s["sys"].get("n_r", 0)
    cr = s["knobs"]["chunk_r"]
    bc = s["knobs"]["band_chunk"]
    px = s["mesh"]["p_x"]
    py = s["mesh"]["p_y"]
    P = px * py
    return {
        "A_pair":      _B * n_k * mu * cr / P,
        "B_psiG_full": _B * n_k * n_b * n_s * n_r / P,
        "C_centroid":  _B * n_k * mu * nb_sum * n_s / px,
        "D_Lq_shard":  _B * n_k * mu * mu / P,
        "E_Lq_rep":    _B * n_k * mu * mu,
        "F_rchunkY":   _B * n_k * bc * n_s * cr / py,
        "G_psiG_bc":   _B * n_k * bc * n_s * n_r / P,
        "H_rchunkXY":  _B * n_k * bc * n_s * cr / P,
    }


# Build design matrix
feat_names = list(primitives_at(samples[0]).keys())
X = np.array([[primitives_at(s)[f] for f in feat_names] for s in samples],
             dtype=np.float64)
y = np.array([s["meas"]["total"] for s in samples], dtype=np.float64)
print(f"Design matrix: {X.shape[0]} samples × {X.shape[1]} features")

# Collinearity diagnostics — Pearson correlation across samples
print("\nCross-feature Pearson correlation matrix (|ρ| > 0.95 = collinear):")
print(f"{'':12}" + "".join(f"{f.split('_')[0]:>8}" for f in feat_names))
X_norm = (X - X.mean(0)) / (X.std(0) + 1e-30)
corr = X_norm.T @ X_norm / X.shape[0]
for i, f in enumerate(feat_names):
    row = "".join(f"{corr[i, j]:>+8.2f}" for j in range(len(feat_names)))
    print(f"  {f:10s}" + row)

# Full NNLS fit
coefs, rnorm = nnls(X, y)
print(f"\nFull NNLS fit (7 primitives):")
for f, c in zip(feat_names, coefs):
    print(f"  C[{f:12s}] = {c:>7.3f}")
rms = rnorm / np.sqrt(len(y))
print(f"  RMS = {rms/1e6:.2f} MB")

# Drop collinear features and refit
# D and E always collinear (same μ² shape, differ by 1/P — P=4 fixed)
# so E is redundant if D is kept.  Drop E.
keep_idx = [i for i, f in enumerate(feat_names) if f != "E_Lq_rep"]
kept = [feat_names[i] for i in keep_idx]
coefs2, rnorm2 = nnls(X[:, keep_idx], y)
print(f"\nNNLS after dropping E_Lq_rep (collinear with D):")
for f, c in zip(kept, coefs2):
    print(f"  C[{f:12s}] = {c:>7.3f}")
rms2 = rnorm2 / np.sqrt(len(y))
print(f"  RMS = {rms2/1e6:.2f} MB")

# Check which predicted points have biggest residuals
y_pred = X[:, keep_idx] @ coefs2
resid = y - y_pred
print(f"\nResiduals (biggest 5):")
order = np.argsort(-np.abs(resid))[:5]
for idx in order:
    s = samples[idx]
    print(f"  [{idx}] μ={s['sys']['n_rmu']:>4} b={s['sys']['n_b']:>3} "
          f"s={s['sys']['n_s']} cr={s['knobs']['chunk_r']:>5} "
          f"bc={s['knobs']['band_chunk']:>3} mesh={s['mesh']['p_x']}x{s['mesh']['p_y']} | "
          f"actual {y[idx]/1e9:.3f} predicted {y_pred[idx]/1e9:.3f} "
          f"resid {resid[idx]/1e6:+.1f} MB")

# Round coefs to nearest integer and see how much RMS grows
coefs_int = np.round(coefs2).astype(int)
y_pred_int = X[:, keep_idx] @ coefs_int
rms_int = np.sqrt(np.mean((y - y_pred_int)**2))
print(f"\nRounded-to-integer coefficients:")
for f, c, ci in zip(kept, coefs2, coefs_int):
    mark = " ✓" if abs(c - ci) < 0.2 else ""
    print(f"  C[{f:12s}] = {c:>7.3f} → {ci}{mark}")
print(f"  RMS (integer) = {rms_int/1e6:.2f} MB  (vs real-valued {rms2/1e6:.2f} MB)")
