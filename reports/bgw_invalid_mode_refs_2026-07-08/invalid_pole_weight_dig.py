#!/usr/bin/env python3
"""Dig: WHY is LORRAX's zero<->2ry delta ~30x smaller than BGW's mode0<->mode2?

Compares, at the q->0 point of the same Si 4x4x4 system, the invalid-GN-pole
population AND the correlation-W weight it carries, in each code's own basis:

  BGW   : (G,G') plane-wave pairs of eps0mat.h5 (01_bgw_gn_ppm), exactly the
          reference counter (count_invalid_gn_poles.py) plus a |W_c| weighting
          W_c = v^1/2(q+G) (eps^-1 - delta)_GG' v^1/2(q+G') with the stored vcoul.
  LORRAX: (mu,nu) ISDF centroid pairs from w_copies_debug.h5 (April run of
          01_lorrax_gn_ppm, same 480 centroids, same WFN), exactly the
          fit_gn_ppm_from_wc_pair condition (minimax_screening.py:391-404):
          omega^2 = -(z^2) * Wp / (W0 - Wp), z = 2j Ry; invalid = not
          (|W0-Wp| > 1e-14 and finite and Re omega^2 > 0).

The delta Sigma_c(2ry - zero) is exactly the Sigma-contribution of the invalid
poles treated as 2 Ry poles with residue B = -1/2 W_c0 * Omega, so the natural
"how much is at stake" metric is the share of sum|W_c0| (and of sum|B|) carried
by invalid pairs.

Run: LORRAX_NGPU=1 lxrun python3 -u invalid_pole_weight_dig.py
"""
import h5py
import numpy as np

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band"
TOL_SMALL = 1.0e-6
OMEGA_P = 2.0  # Ry, both codes
FALLBACK = 2.0  # Ry, LORRAX ppm_fallback_omega


def read_complex(ds):
    a = np.asarray(ds)
    if a.dtype.kind == "c":
        return a
    if a.dtype.names:  # compound (r, i)
        return a[a.dtype.names[0]] + 1j * a[a.dtype.names[1]]
    if a.ndim >= 1 and a.shape[-1] == 2:
        return a[..., 0] + 1j * a[..., 1]
    raise ValueError(f"unrecognized complex layout {a.dtype} {a.shape}")


def stats(name, W_weight, invalid, considered):
    """Population + weight share of invalid pairs. W_weight = |W_c| per pair."""
    n_cons = int(considered.sum())
    n_inv = int(invalid.sum())
    w_tot = float(W_weight[considered].sum())
    w_inv = float(W_weight[invalid].sum())
    print(f"  {name}")
    print(f"    pairs considered : {n_cons}")
    print(f"    invalid          : {n_inv}  ({100.0 * n_inv / n_cons:.2f}%)")
    print(f"    sum|Wc| share    : {100.0 * w_inv / w_tot:.3f}%  "
          f"(invalid {w_inv:.4e} of {w_tot:.4e})")
    print(f"    mean|Wc| invalid : {w_inv / max(n_inv, 1):.4e}"
          f"   mean|Wc| valid: {(w_tot - w_inv) / max(n_cons - n_inv, 1):.4e}"
          f"   ratio: {(w_inv / max(n_inv, 1)) / ((w_tot - w_inv) / max(n_cons - n_inv, 1)):.3f}")
    return n_inv, n_cons, w_inv, w_tot


def bgw_q0():
    with h5py.File(f"{BASE}/01_bgw_gn_ppm/eps0mat.h5", "r") as f:
        nmtx = int(f["eps_header/gspace/nmtx"][0])
        vcoul = np.asarray(f["eps_header/gspace/vcoul"][0, :nmtx])
        m = f["mats/matrix"][0, 0]  # (nfreq, ncol, nrow, 2)
        e1 = m[0, :nmtx, :nmtx, 0] + 1j * m[0, :nmtx, :nmtx, 1]
        e2 = m[1, :nmtx, :nmtx, 0] + 1j * m[1, :nmtx, :nmtx, 1]
    eye = np.eye(nmtx)
    I1 = eye - e1
    I2 = eye - e2
    skip = (np.abs(I1) < TOL_SMALL) & (np.abs(I2) < TOL_SMALL)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = I2 / (I1 - I2)
    considered = ~skip
    invalid = considered & (np.real(ratio) < 0)
    # W_c in BGW hdf5 convention: matrix is eps^-1 with column-major (col, row);
    # for |W| weights the transpose question is irrelevant (|W^T| perm-invariant
    # under the same sqrt-v weighting).
    sqv = np.sqrt(vcoul)
    Wc = sqv[:, None] * (-I1) * sqv[None, :]  # eps^-1 - delta = -I1, static
    print("== BGW q->0 (eps0mat.h5, 537 G) ==")
    n_inv, n_cons, w_inv, w_tot = stats(
        "plane-wave (G,G') pairs", np.abs(Wc), invalid, considered)
    # where do the invalid pairs live? |G| index distribution (G sorted by |q+G|)
    idx = np.where(invalid)
    lo = nmtx // 4
    frac_core = float(np.mean((idx[0] < lo) & (idx[1] < lo)))
    print(f"    invalid pairs with both G-indices in the lowest quartile "
          f"(|G| small): {100.0 * frac_core:.2f}%")
    diag = int(np.sum(invalid[np.arange(nmtx), np.arange(nmtx)]))
    print(f"    invalid on diagonal: {diag}/{nmtx}")
    wing = np.zeros_like(invalid)
    wing[0, :] = True
    wing[:, 0] = True
    print(f"    invalid involving the wing row/col (G=0): "
          f"{int(np.sum(invalid & wing))} of {int(np.sum(considered & wing))} "
          "wing pairs")
    return invalid, np.abs(Wc)


def lorrax_q0():
    with h5py.File(f"{BASE}/01_lorrax_gn_ppm/w_copies_debug.h5", "r") as f:
        W0 = read_complex(f["W0_ppm_q000_munu"])
        Wp = read_complex(f["Wiwp_ppm_q000_munu"])
    z = 1j * OMEGA_P
    denom = W0 - Wp
    safe = np.abs(denom) > 1.0e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(safe, Wp / denom, 0.0)
    omega_sq = -(z * z) * ratio
    re = np.real(omega_sq)
    good = safe & np.isfinite(re) & (re > 0.0)
    considered = np.ones_like(good)  # LORRAX considers every (mu, nu) pair
    invalid = ~good
    print("\n== LORRAX q->0 (w_copies_debug.h5, 480 ISDF centroids) ==")
    stats("ISDF (mu,nu) centroid pairs", np.abs(W0), invalid, considered)
    diag = int(np.sum(invalid[np.arange(480), np.arange(480)]))
    print(f"    invalid on diagonal: {diag}/480")
    # Residue-weight at stake: B = -1/2 W0 * Omega; invalid get Omega = 2 Ry.
    omega_vals = np.where(good, np.sqrt(np.where(good, re, 1.0)), FALLBACK)
    B = 0.5 * np.abs(W0) * omega_vals
    print(f"    sum|B| share of invalid poles: "
          f"{100.0 * B[invalid].sum() / B.sum():.3f}%")
    # How close to the boundary are the invalid omega^2? (distribution of re)
    q = np.percentile(re[invalid & np.isfinite(re)], [5, 50, 95])
    print(f"    Re omega^2 of invalid poles (Ry^2): "
          f"p5={q[0]:.3e} p50={q[1]:.3e} p95={q[2]:.3e}")
    qv = np.percentile(np.abs(W0[invalid]), [50, 95, 99.9])
    print(f"    |Wc0| of invalid pairs: p50={qv[0]:.3e} p95={qv[1]:.3e} "
          f"p99.9={qv[2]:.3e};  global max|Wc0|={np.abs(W0).max():.3e}")
    # Diagonal dynamics: a physical plasmon-pole W_c(i*omega) DECAYS with
    # imaginary frequency (|Wp| < |W0|, same sign). Inverted dynamics
    # (|Wp| > |W0|) makes the GN denominator flip sign -> invalid.
    d0 = np.real(np.diag(W0))
    dp = np.real(np.diag(Wp))
    grow = np.abs(dp) > np.abs(d0)
    sign_flip = np.sign(dp) != np.sign(d0)
    print(f"    diagonal Re W: |W(iwp)| > |W(0)| for {int(grow.sum())}/480; "
          f"sign(W(iwp)) != sign(W(0)) for {int(sign_flip.sum())}/480")
    print(f"    diagonal medians: Re W0 = {np.median(d0):.4e}, "
          f"Re Wp = {np.median(dp):.4e}")


def main():
    bgw_q0()
    lorrax_q0()


if __name__ == "__main__":
    main()
