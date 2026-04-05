#!/usr/bin/env python3
"""Compare BGW vs GWJAX GN-GPP head formalism numerically."""
import numpy as np

# BGW values from epsilon.out
vc0 = 315.0137        # Vcoul head (MiniBZ)
w0_static = 55.6197   # Wcoul head (MiniBZ iw=1) = W(q=0,w=0)
w0_imfreq = 246.1099  # Wcoul head (MiniBZ iw=2) = W(q=0,w=iwp)
omega_p = 2.0         # Ry

# Cell volume from WFN
cell_vol = 612.23     # a.u.^3 (approximate)

print("=" * 70)
print("BGW formalism (freq_dep=3)")
print("=" * 70)

# BGW reads eps^{-1} from eps0mat.h5, then computes I_eps = delta - eps^{-1}
# But fixwings.f90 rescales the head of eps^{-1} so that:
#   eps^{-1}_head = Wcoul(MiniBZ) / Vcoul(MiniBZ)
eps_inv_0 = w0_static / vc0
eps_inv_iwp = w0_imfreq / vc0
print(f"eps^{{-1}}_head(0)   = {eps_inv_0:.6f}")
print(f"eps^{{-1}}_head(iwp) = {eps_inv_iwp:.6f}")

# I_eps = 1 - eps^{-1}  (line 812: for diagonal, I_epsRggp = ONE - epsRggp)
I_eps_0 = 1.0 - eps_inv_0
I_eps_iwp = 1.0 - eps_inv_iwp
print(f"I_eps(0)   = {I_eps_0:.6f}")
print(f"I_eps(iwp) = {I_eps_iwp:.6f}")

# GN fit (line 826): wtilde^2 = wp^2 * I_eps(2) / (I_eps(1) - I_eps(2))
wtilde2 = omega_p**2 * I_eps_iwp / (I_eps_0 - I_eps_iwp)
wtilde = np.sqrt(wtilde2) if wtilde2 > 0 else 0.0
print(f"wtilde^2   = {wtilde2:.6f} Ry^2")
print(f"wtilde     = {wtilde:.6f} Ry = {wtilde*13.6057:.3f} eV")

# Omega^2 = wtilde^2 * I_eps(0)  (line 1815)
Omega2 = wtilde2 * I_eps_0
print(f"Omega^2    = {Omega2:.6f} Ry^2")

# BGW on-shell (E=E_n, n1=n, wxt=0):
#   ssx = Omega2 / (0 - wtilde2) = Omega2/(-wtilde2) = I_eps(0)
#   Accumulated: asxtemp += -ssx * occ * vcoul
#   => SX_head(occ) = -I_eps(0) * 1 * vc0 * |M|^2
#   For PW: M_nn(q=0,G=0) involves the pair density rho_nn(G=0)
#   which = 1 (normalized). aqsntemp*conj(aqsmtemp) = |M|^2.
#
#   sch = delw * I_eps = -1 * I_eps(0) = -I_eps(0)
#   Accumulated: achtemp += 0.5 * sch * vcoul = -0.5 * I_eps(0) * vc0 * |M|^2
#
#   For occupied n1=n: Cor_head = SX + CH = -I_eps*vc0 + (-0.5*I_eps*vc0)
#     Wait that's not right. SX has the -occ factor:
#     SX_head = -(-I_eps(0)) * 1 * vc0 = I_eps(0) * vc0
#     Hmm, let me be more careful.
#
#  ssx = Omega2*conj(cden)/|cden|^2 where cden = wxt^2 - wtilde2 = -wtilde2
#  ssx = Omega2 * conj(-wtilde2) / |-wtilde2|^2 = Omega2/(-wtilde2) = I_eps(0)  [real]
#  Then: asxtemp += -ssx * occ * vcoul = -I_eps(0) * occ * vc0
#  Note I_eps(0) is NEGATIVE (eps^{-1} < 1 for screening), so -I_eps(0) > 0
#  SX contribution (occupied): -I_eps(0) * vc0 = -(1-eps^{-1}(0))*vc0 = (eps^{-1}(0)-1)*vc0
#
#  sch = delw * I_eps(0) = -1 * I_eps(0) = -I_eps(0)   [because delw = wtilde/(0-wtilde) = -1]
#  achtemp += 0.5 * sch * vcoul = -0.5 * I_eps(0) * vc0

# But this is only the n1=n term. The FULL sum over n1 is more complex.
# Let me just compute the n1=n on-shell term to compare.

# BGW: n1=n on-shell contribution
sx_bgw_n1n = -I_eps_0 * vc0  # per |M|^2, times occ
ch_bgw_n1n = -0.5 * I_eps_0 * vc0  # per |M|^2, all n1

print(f"\nBGW n1=n on-shell (per |M|^2, in a.u.):")
print(f"  SX (occ only): {sx_bgw_n1n:.3f}")
print(f"  CH (all n1):   {ch_bgw_n1n:.3f}")
print(f"  Total (occ):   {sx_bgw_n1n + ch_bgw_n1n:.3f}")

print("\n" + "=" * 70)
print("GWJAX formalism (head_correction.py)")
print("=" * 70)

# GWJAX fits W^c = W - V directly
wc_0 = w0_static - vc0
wc_iwp = w0_imfreq - vc0
print(f"W^c(0)   = {wc_0:.3f} a.u.")
print(f"W^c(iwp) = {wc_iwp:.3f} a.u.")

# GN model: W^c(w) = B_h / (w^2 - Omega_h^2)
Omega_h_sq = wc_iwp * omega_p**2 / (wc_0 - wc_iwp)
B_h = -wc_0 * Omega_h_sq
Omega_h = np.sqrt(abs(Omega_h_sq))
R_h = B_h / (2.0 * Omega_h)
print(f"Omega_h^2 = {Omega_h_sq:.6f} Ry^2")
print(f"Omega_h   = {Omega_h:.6f} Ry = {Omega_h*13.6057:.3f} eV")
print(f"B_h       = {B_h:.3f} Ry^2 * a.u.")
print(f"R_h       = {R_h:.3f} Ry * a.u.")

# GWJAX on-shell sigma_head = (R_h/Omega_h) * (2f-1) / cell_volume
# For occupied: +R_h/(Omega_h * vol)
# For empty:    -R_h/(Omega_h * vol)
sig_occ_ry = R_h / (Omega_h * cell_vol)
sig_emp_ry = -R_h / (Omega_h * cell_vol)
print(f"\nGWJAX on-shell head sigma (Ry): occ={sig_occ_ry:.6f}, emp={sig_emp_ry:.6f}")
print(f"GWJAX on-shell head sigma (eV): occ={sig_occ_ry*13.6057:.3f}, emp={sig_emp_ry*13.6057:.3f}")

# Verify: GWJAX sums over ALL bands n1 with the PPM self-energy formula.
# The on-shell n1=n contribution is:
#   sigma^c_nn(E_n) = (1/vol) * B_h / (2*Omega_h) * [f_n/Omega_h + (1-f_n)/(-Omega_h)]
#                   = (R_h/vol) * [f_n/Omega_h - (1-f_n)/Omega_h]
#                   = (R_h/vol/Omega_h) * (2f-1)
# This is what compute_head_sigma_diagonal does.

# But wait: in BGW, the self-energy sums over ALL n1 (occupied for SX, all for CH).
# The n1=n term is just one contribution. GWJAX's formula gives the total sum over
# all n1 assuming |M_nn1(q=0,G=0)|^2 = delta_{n,n1} (PW identity).

print("\n" + "=" * 70)
print("CRITICAL: Is the GWJAX formula the sum or just n1=n?")
print("=" * 70)
print("In PW basis: rho_{nn1}(q=0, G=0) = delta_{n,n1}")
print("=> Only n1=n survives in the G=0 head sum")
print("=> GWJAX's 'sum over all n1' reduces to just the n1=n term")
print("=> This is correct for the head contribution!")

print("\n" + "=" * 70)
print("Pole position comparison")
print("=" * 70)
print(f"BGW  wtilde   = {wtilde:.6f} Ry")
print(f"GWJAX Omega_h = {Omega_h:.6f} Ry")
print(f"Ratio         = {wtilde/Omega_h:.6f}")

# Are these supposed to be equal? Let's check algebraically.
# BGW: wtilde^2 = wp^2 * I_eps(iwp) / (I_eps(0) - I_eps(iwp))
# GWJAX: Omega_h^2 = wc_iwp * wp^2 / (wc_0 - wc_iwp)
#
# Relationship: I_eps(iw) = 1 - eps^{-1}(iw) and wc(iw) = (eps^{-1}(iw)-1)*vc0 = -I_eps(iw)*vc0
# So: wc_0 = -I_eps(0)*vc0, wc_iwp = -I_eps(iwp)*vc0
#
# GWJAX: Omega_h^2 = [-I_eps(iwp)*vc0] * wp^2 / [-I_eps(0)*vc0 - (-I_eps(iwp)*vc0)]
#                   = [-I_eps(iwp)*vc0] * wp^2 / [vc0*(-I_eps(0) + I_eps(iwp))]
#                   = -I_eps(iwp) * wp^2 / (-I_eps(0) + I_eps(iwp))
#                   = I_eps(iwp) * wp^2 / (I_eps(0) - I_eps(iwp))
#                   = wtilde^2_BGW  ✓

print("\nAlgebraic verification:")
print(f"  GWJAX Omega_h^2 = wc_iwp * wp^2 / (wc_0 - wc_iwp)")
print(f"  = [-I_eps(iwp)*vc0] * wp^2 / [vc0*(I_eps(iwp) - I_eps(0))]")
print(f"  = I_eps(iwp) * wp^2 / (I_eps(0) - I_eps(iwp))")
print(f"  = wtilde^2_BGW  ✓")
print(f"  => Pole positions are IDENTICAL")

print("\n" + "=" * 70)
print("On-shell self-energy comparison")
print("=" * 70)
# BGW for n1=n (the only term for G=0 head):
# For occupied n:
#   Sigma^c_SX = -ssx * occ * vcoul where ssx = Omega2/(0-wtilde2) = I_eps(0)
#   => -I_eps(0) * 1 * vc0   [note: vcoul already includes 1/vol from mini-BZ average? No!]
#   Actually vcoul(igp) in BGW is in Rydberg, and the matrix elements M are normalized
#   with 1/sqrt(vol). So |M|^2 * vcoul = (1/vol) * vcoul.
#
#   Sigma^c_CH = 0.5 * sch * vcoul where sch = -I_eps(0) (delw=-1)
#   => 0.5 * (-I_eps(0)) * vc0 / vol
#
#   Total for occ n: (-I_eps + 0.5*(-I_eps)) * vc0 / vol = -1.5 * I_eps(0) * vc0 / vol
#   Hmm, that doesn't look right. In BGW SX is for occupied n1 only, CH is for all.
#   Since n1=n is the only term for G=0:
#     If n is occupied: SX + CH = [-I_eps(0) + 0.5*(-I_eps(0))] * vc0 / vol
#     If n is empty: only CH = 0.5*(-I_eps(0)) * vc0 / vol (no SX contribution)
#
# Wait, I need to be more careful. For empty states, SX is zero (occ=0).
# For the CH term, delw = wtilde/(E-E_n1-wtilde).
# When E=E_n and n1=n (occupied): delw = wtilde/(-wtilde) = -1
# When E=E_n and n1=n (empty): same thing, delw = -1
# CH sums over ALL n1, and for n1=n: sch = -I_eps(0)
# But for empty n, the SX term vanishes.
#
# So:
#   occ n: SX + CH = -I_eps * vc0/vol + 0.5*(-I_eps)*vc0/vol = -1.5*I_eps*vc0/vol
#   emp n: CH only = 0.5*(-I_eps)*vc0/vol

# Hmm wait, BGW's SX formula is different from CH. Let me re-examine.
# From the code:
#   ssx = Omega2/(wxt^2 - wtilde2)  [normal case]
#   At wxt=0: ssx = Omega2/(-wtilde2) = I_eps(0)
#   asxtemp += -ssx * occ * vcoul
#   So SX contribution = -I_eps(0) * occ * vcoul * |M|^2
#
#   sch = delw * I_eps = -1 * I_eps(0) = -I_eps(0)
#   achtemp += 0.5 * sch * vcoul
#   So CH contribution = -0.5 * I_eps(0) * vcoul * |M|^2
#
# For the head: |M|^2 should be unity (rho_nn(G=0)=1 in PW with 1/vol norm)
# BGW's aqsntemp*conj(aqsmtemp) gives |M|^2 without vol factor.
# The vol factor comes from the normalization of the PW basis.
#
# Actually, I realize I need to check BGW's vcoul normalization carefully.
# vcoul(ig) in BGW is the Coulomb kernel 8pi/|q+G|^2 * (truncation) in Rydberg.
# For the head (q+G=0), vcoul(1) = vc0 from mini-BZ average.
# The self-energy in BGW is: Sigma = (1/Nk) * sum_q ...
# and the 1/vol factor comes from the 1/Nk and the BZ integration.
# For 1x1 (Nk=1): 1/Nk = 1.

# Let me just compute the numerical values and compare.
# BGW n1=n head contribution (in Rydberg/a.u.):
# SX: -I_eps(0) * vc0  (for occ)
# CH: -0.5 * I_eps(0) * vc0  (for all)
# Note: these are in "per matrix element" units. The full sigma has (1/Nk) * |M|^2.
# For 1x1 at Gamma, Nk=1 and |M(q=0,G=0)|^2 per vol is 1/vol.

sx_n1n_au = -I_eps_0 * vc0  # in a.u. per |M|^2
ch_n1n_au = -0.5 * I_eps_0 * vc0

print(f"BGW raw (per |M|^2, a.u.):")
print(f"  SX(occ): {sx_n1n_au:.3f}")
print(f"  CH:      {ch_n1n_au:.3f}")
print(f"  occ total: {sx_n1n_au + ch_n1n_au:.3f}")
print(f"  emp total: {ch_n1n_au:.3f}")

# Now BGW sigma in eV with 1/(Nk*vol) for the head:
# Actually in BGW |M|^2 already has 1/vol built in from the wfn normalization.
# So Sigma = (1/Nk) * SX_raw * |M_PW|^2 where |M_PW|^2 ~ 1 for head
# For the head of a semiconductor at Gamma: rho_nn(G=0) is exactly delta_nn
# but there's a normalization issue. Let me just compare shifts.

# GWJAX gives: sigma_head = R_h/(Omega_h*vol) * (2f-1) in Ry
# = B_h/(2*Omega_h^2*vol) * (2f-1)   since R_h = B_h/(2*Omega_h)
# = (-wc_0 * Omega_h^2)/(2*Omega_h^2*vol) * (2f-1)
# = -wc_0/(2*vol) * (2f-1)
# = (I_eps(0)*vc0)/(2*vol) * (2f-1)
# For occ: = I_eps(0)*vc0/(2*vol) * 1 = I_eps(0)*vc0/(2*vol)
# For emp: = I_eps(0)*vc0/(2*vol) * (-1) = -I_eps(0)*vc0/(2*vol)

gwjax_occ_ry = I_eps_0 * vc0 / (2 * cell_vol)
gwjax_emp_ry = -I_eps_0 * vc0 / (2 * cell_vol)

print(f"\nGWJAX head sigma (Ry):")
print(f"  occ: {gwjax_occ_ry:.6f} = {gwjax_occ_ry*13.6057:.3f} eV")
print(f"  emp: {gwjax_emp_ry:.6f} = {gwjax_emp_ry*13.6057:.3f} eV")

# BGW should give for the n1=n HEAD contribution (in same units):
# occ: (1/Nk) * [SX + CH] * |M|^2 = (SX_raw + CH_raw) / vol   [Nk=1, |M|^2=1/vol]
# Hmm, this is confusing. Let me just compute what BGW's head sigma
# evaluates to by comparing Cor' with and without head.

print("\n" + "=" * 70)
print("Direct comparison: GWJAX head shift vs BGW Cor' difference")
print("=" * 70)
print(f"GWJAX head correction (from gw.out): val=+2.513 eV, cond=-2.513 eV")
print(f"BGW GN Cor' - BGW COHSEX Cor' should give the dynamic correction.")

# From our data:
# BGW GN Cor' for band 19: 6.957 eV
# BGW COHSEX Cor' for band 19: 8.495 eV
# Difference: 6.957 - 8.495 = -1.538 eV
# This is the BGW dynamic correction (GN changes relative to COHSEX)
# It includes ALL (G,G') elements, not just the head.
print(f"\nBGW GN - BGW COHSEX (band 19): {6.957 - 8.495:.3f} eV")
print(f"BGW GN - BGW COHSEX (band 27): {-4.225 - (-6.183):.3f} eV")
print(f"GWJAX head shift (band 19, occ): +2.513 eV")
print(f"GWJAX head shift (band 27, emp): -2.513 eV")
print(f"\nThe GWJAX head shift is ~+2.5 eV for occ but BGW dynamic correction is ~-1.5 eV")
print(f"=> GWJAX head has OPPOSITE SIGN from what the dynamic correction should be!")
