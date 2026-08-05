"""Wall-time extrapolation for the 12x12 / 80 Ry GW from the MEASURED 6x6 run.

Anchor: runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu/gw.out
        (16 GPU / 4 nodes, 30 Ry, nq=nk=36, n_rtot=46080, n_mu=1496->pad 1504,
         nband=200 / ncond=74 -> nb_total=308, ngkmax=1964, total 87.414 s)

Each timed section is scaled by its leading cost term.  Every exponent below is
the algorithmic scaling of that section, not a fit:

  cholesky / CCT factorization   nq * mu^3
  zeta solve (n_rtot RHS)        nq * mu^2 * n_rtot
  z_q_build (pair densities)     nk * mu * n_rtot * nb_total
  V_q                            nq * mu^2 * ngkmax
  chi0 + W (inversion per q)     nq * mu^3
  sigma (M W M^dag over k,q)     nk * nq * mu^2 * nb_total
  centroid load / misc           nk * mu * nb_total

Device count is held at 16 on both sides, so P cancels.
"""
import os
import sys

# ---- anchor (measured, from gw.out "--- Timing ---") ----------------------
A = dict(nq=36, nk=36, n_rtot=46080, mu=1504, nb=308, ngkmax=1964)
SECT = {                       # section -> (measured seconds, scaling key)
    "load_centroid_wfns": (3.054, "load"),
    "zeta_fit: cholesky": (5.981, "chol"),
    "zeta_fit: z_q_build": (5.961, "zq"),
    "zeta_fit: solve": (32.610, "solve"),
    "zeta_fit: other": (47.469 - 5.981 - 5.961 - 32.610, "load"),
    "V_q_compute": (4.868, "vq"),
    "chi0_W": (2.020, "chol"),
    "sigma": (29.972, "sig"),
}
ANCHOR_TOTAL = 87.414

# ---- target ---------------------------------------------------------------
T_BASE = dict(nq=144, nk=144, n_rtot=174960, ngkmax=8603)
MUS = [1600, 2400, 3008, 4000]        # padded to multiples of 16
NBS = [("wide  Sigma (ncond=300)", 662), ("narrow Sigma (ncond=74)", 436)]


def factor(key, T, mu, nb):
    rq, rk = T["nq"] / A["nq"], T["nk"] / A["nk"]
    rr = T["n_rtot"] / A["n_rtot"]
    rm = mu / A["mu"]
    rn = nb / A["nb"]
    rg = T["ngkmax"] / A["ngkmax"]
    return {
        "chol":  rq * rm ** 3,
        "solve": rq * rm ** 2 * rr,
        "zq":    rk * rm * rr * rn,
        "vq":    rq * rm ** 2 * rg,
        "sig":   rk * rq * rm ** 2 * rn,
        "load":  rk * rm * rn,
    }[key]


print("=" * 96)
print("WALL-TIME EXTRAPOLATION 6x6/30Ry -> 12x12/80Ry, 16 GPU (4 nodes), "
      "anchor = 87.414 s")
print(f"  anchor : nq={A['nq']} n_rtot={A['n_rtot']} mu={A['mu']} "
      f"nb_total={A['nb']} ngkmax={A['ngkmax']}")
print(f"  target : nq={T_BASE['nq']} n_rtot={T_BASE['n_rtot']} "
      f"ngkmax={T_BASE['ngkmax']}")
print("=" * 96)

for nb_label, nb in NBS:
    print()
    print(f"### {nb_label}  (nb_total = {nb})")
    hdr = f"{'section':<22}" + "".join(f"{('mu=' + str(m)):>13}" for m in MUS)
    print(hdr)
    totals = {m: 0.0 for m in MUS}
    for name, (t0, key) in SECT.items():
        row = f"{name:<22}"
        for m in MUS:
            t = t0 * factor(key, T_BASE, m, nb)
            totals[m] += t
            row += f"{t:>13.0f}"
        print(row + "   s")
    print("-" * len(hdr))
    row = f"{'TOTAL (s)':<22}" + "".join(f"{totals[m]:>13.0f}" for m in MUS)
    print(row)
    row = f"{'TOTAL (h:mm)':<22}" + "".join(
        f"{int(totals[m]//3600)}:{int(totals[m]%3600//60):02d}".rjust(13)
        for m in MUS)
    print(row)
    row = f"{'node-hours (4 nodes)':<22}" + "".join(
        f"{4 * totals[m] / 3600:>13.1f}" for m in MUS)
    print(row)
    row = f"{'speedup needed for':<22}" + "".join(
        f"{totals[m] / 14400:>13.2f}" for m in MUS)
    print(row + "   x  (>1 = exceeds the 4 h interactive QOS wall)")
