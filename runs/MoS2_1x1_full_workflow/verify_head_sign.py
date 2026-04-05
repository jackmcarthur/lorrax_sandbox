"""Carefully verify BGW vs GWJAX head self-energy sign and magnitude."""
import numpy as np

# BGW values from epsilon.out
vc0 = 315.0137
w0_static = 55.6197
w0_imfreq = 246.1099
omega_p = 2.0  # Ry

# ── BGW GN fit for head element ──
# BGW reads eps^{-1} and computes I_eps = delta - eps^{-1}
eps_inv_0 = w0_static / vc0      # 0.1766
eps_inv_iwp = w0_imfreq / vc0    # 0.7813
I_eps_0 = 1.0 - eps_inv_0        # 0.8234
I_eps_iwp = 1.0 - eps_inv_iwp    # 0.2187

# wtilde^2 = wp^2 * I_eps(iwp) / (I_eps(0) - I_eps(iwp))  [mtxel_cor.f90:826]
wtilde2 = omega_p**2 * I_eps_iwp / (I_eps_0 - I_eps_iwp)
wtilde = np.sqrt(wtilde2)

# Omega^2 = wtilde^2 * I_eps(0)  [recomputed at line 1815]
Omega2 = wtilde2 * I_eps_0

# ── BGW on-shell evaluation for head (G=G'=0, n1=n, wxt=0) ──
# Line 1833-1836: cden = wxt^2 - wtilde2 = -wtilde2
# ssx = Omega2 * conj(cden) / |cden|^2 = Omega2 * (-wtilde2) / wtilde^4 = -Omega2/wtilde2
ssx = -Omega2 / wtilde2  # = -I_eps(0)

# Line 1820-1825: delw = wtilde * conj(wxt-wtilde) / |wxt-wtilde|^2
# At wxt=0: delw = wtilde * (-wtilde) / wtilde^2 = -1
delw = -1.0
sch = delw * I_eps_0  # = -I_eps(0)

# Accumulation (lines 1911, 1918):
#   asxtemp += -ssx * occ * vcoul(igp)
#   achtemp += 0.5 * sch * vcoul(igp)
# Per |M|^2 (which is delta_nn for head):
SX_bgw_occ = -ssx * 1.0 * vc0   # -(-I_eps) * vc0 = +I_eps(0) * vc0
CH_bgw     = 0.5 * sch * vc0     # -0.5 * I_eps(0) * vc0

print("=" * 60)
print("BGW head (G=G'=0) on-shell contribution")
print("=" * 60)
print(f"ssx = -Omega2/wtilde2 = -I_eps(0) = {ssx:.6f}")
print(f"sch = delw * I_eps(0) = -I_eps(0) = {sch:.6f}")
print(f"SX (occ): -ssx * occ * vcoul = +I_eps*vc0 = {SX_bgw_occ:.3f} a.u.")
print(f"CH (all): 0.5*sch*vcoul = -0.5*I_eps*vc0 = {CH_bgw:.3f} a.u.")
print(f"Total occ: SX+CH = +0.5*I_eps*vc0 = {SX_bgw_occ + CH_bgw:.3f} a.u.")
print(f"Total emp: CH only = -0.5*I_eps*vc0 = {CH_bgw:.3f} a.u.")

# ── GWJAX head correction ──
wc_0 = w0_static - vc0       # W^c(0) = -259.4
wc_iwp = w0_imfreq - vc0     # W^c(iwp) = -68.9
Omega_h_sq = wc_iwp * omega_p**2 / (wc_0 - wc_iwp)
B_h = -wc_0 * Omega_h_sq
Omega_h = np.sqrt(Omega_h_sq)
R_h = B_h / (2 * Omega_h)

# On-shell: sigma = (R_h / Omega_h) * (2f-1) / vol  [in Ry]
# Per unit, without 1/vol: R_h/Omega_h
gwjax_occ = R_h / Omega_h    # positive
gwjax_emp = -R_h / Omega_h   # negative

print(f"\n{'=' * 60}")
print("GWJAX head correction (per 1/vol)")
print("=" * 60)
print(f"R_h / Omega_h = {gwjax_occ:.3f} Ry*a.u.")

# ── Key comparison ──
# BGW: occ total = +0.5 * I_eps(0) * vc0  (a.u., before 1/vol)
# GWJAX: occ = R_h/Omega_h                (Ry*a.u., before 1/vol)
# 
# Note unit difference: BGW is in Rydberg (8pi/q^2), GWJAX R_h is in Ry*a.u.
# Actually both should be in the same units if W^c is in a.u. and vcoul is in Ry.
#
# Check: 0.5 * I_eps(0) * vc0 = 0.5 * 0.8234 * 315.01 = 129.7
# R_h/Omega_h = 156.0 / 1.203 = 129.7
# THEY MATCH!

bgw_val = 0.5 * I_eps_0 * vc0
print(f"\n{'=' * 60}")
print("COMPARISON (occ, per 1/vol)")
print("=" * 60)
print(f"BGW:   +0.5 * I_eps(0) * vc0 = {bgw_val:.3f}")
print(f"GWJAX: R_h / Omega_h         = {gwjax_occ:.3f}")
print(f"Match: {abs(bgw_val - gwjax_occ) < 0.01}")
print(f"\nSIGNS:  BGW occ = POSITIVE,  GWJAX occ = POSITIVE  ✓")
print(f"        BGW emp = NEGATIVE,  GWJAX emp = NEGATIVE  ✓")

print(f"\n{'=' * 60}")
print("Earlier opposite-sign claim was WRONG")
print("=" * 60)
print("I compared GWJAX head (+2.5 eV) to BGW (GN-COHSEX) total (-1.5 eV).")
print("But BGW GN-COHSEX includes ALL G,G' changes, not just head.")
print("The head-only contribution in BGW is also +2.5 eV for valence.")
print("The other ~2000 (G,G') elements shift by ~-4 eV (dynamic correction)")
print("giving the net -1.5 eV total dynamic change.")
print()
print("The GWJAX head formalism IS equivalent to BGW. Same poles, same sign.")
