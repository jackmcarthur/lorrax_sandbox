#!/usr/bin/env python3
"""
Mini-BZ averaging of the 2D Coulomb head: impact on GN parameters.
===================================================================

BerkeleyGW's minibzaverage_2d_oneoverq2 (Common/minibzaverage.f90:97-186)
fits a model for EPSILON (not ε⁻¹) and averages W = v/ε over the mini-BZ:

  Model:   ε(q, ω) = 1 + v(q) · q² · γ(ω)        [line 119, 165]
  where:   v(q) = (1 - e^{-zc·q}) / q²             [slab-truncated, unnormalized]
  and:     γ(ω) = (1/ε⁻¹(q₀,ω) - 1) / (q₀²·v(q₀))   [line 124]

  Average: Wcoul(ω) = (8π/N) Σ_i v(q_i) / ε(q_i, ω)    [lines 164-166, 180]
  Vcoul    = (8π/N) Σ_i v_full(q_i)                      [lines 152-154, 178]

  where v_full includes the qz-dependent truncation factor, while the
  model uses the qz=0 simplified v(q) = (1-e^{-zc·q})/q².

Note: α=0 (line 128: "Set to zero for now"), so exp(-α·q) = 1.

Critical: W = v/ε, NOT ε⁻¹·v evaluated from a model of ε⁻¹. The model is
for ε, and the Coulomb singularity at q→0 is tamed by the 1/ε denominator.

This script computes the averaging integral analytically (radial, circular
mini-BZ) and compares the GN parameters obtained from:
  A) Averaged: Wcoul(ω) = ⟨v/ε⟩_BZ  with ε from the model
  B) Raw:      Wcoul(ω) = v(q₀)/ε(q₀,ω)  at the single shifted q-point
"""

import numpy as np
from scipy import integrate

# ── MoS2 lattice ──
bohr = 0.529177210903
a = 3.164292 / bohr   # bohr
c = 12.0 / bohr       # bohr
zc = c / 2.0          # slab truncation parameter = c/2
A_cell = a**2 * np.sqrt(3) / 2
A_BZ = (2 * np.pi)**2 / A_cell
Q_bz = np.sqrt(A_BZ / np.pi)  # circular BZ radius
b_mag = 4 * np.pi / (a * np.sqrt(3))
q0 = 0.001 * b_mag     # shifted q-point

# ── BGW output values ──
eps_inv_head_0 = 1.152298946517484      # ε⁻¹(q₀, ω=0)
eps_inv_head_iwp = 0.9945864734352702   # ε⁻¹(q₀, ω=iωp)  [from "GN epsinv head (iw=2)"]
vc0_bgw = 315.0137       # BGW Vcoul head (MiniBZ)
wc0_bgw_static = 55.6197 # BGW Wcoul head (MiniBZ iw=1)
wc0_bgw_imfreq = 246.1099  # BGW Wcoul head (MiniBZ iw=2)
omega_p = 2.0  # Ry

# ── Model functions (matching BGW minibzaverage.f90:119-166) ──

def v_unnorm(q):
    """Unnormalized slab Coulomb: (1 - e^{-zc·q})/q². [BGW line 164]"""
    if q < 1e-15:
        return zc  # Taylor limit
    return (1.0 - np.exp(-zc * q)) / q**2

def v_full(q):
    """Full slab Coulomb in Ry: 8π·v_unnorm(q). [BGW line 178]"""
    return 8.0 * np.pi * v_unnorm(q)

def gamma_from_epsinv(epsinv_q0):
    """BGW's γ calibration from ε⁻¹ at shifted q-point. [line 124]
    Model: 1/ε⁻¹(q) = ε(q) = 1 + v(q)·q²·γ
    → γ = (1/ε⁻¹ - 1) / (q₀²·v(q₀))
    """
    return (1.0 / epsinv_q0 - 1.0) / (q0**2 * v_unnorm(q0))

def eps_model(q, gamma):
    """Model dielectric function. [line 165]"""
    return 1.0 + v_unnorm(q) * q**2 * gamma

def W_model(q, gamma):
    """Screened Coulomb from model: W = 8π·v_unnorm/ε. [line 166, 180]"""
    return 8.0 * np.pi * v_unnorm(q) / eps_model(q, gamma)

# ── Compute γ at both frequencies ──
gamma_0 = gamma_from_epsinv(eps_inv_head_0)
gamma_iwp = gamma_from_epsinv(eps_inv_head_iwp)

print("=" * 72)
print("MoS₂ 1×1: Mini-BZ Head Averaging (corrected model: ε, not ε⁻¹)")
print("=" * 72)
print(f"  q₀ = {q0:.6f} bohr⁻¹,  Q_BZ = {Q_bz:.4f} bohr⁻¹")
print(f"  zc = {zc:.4f} bohr,  cell_vol = {A_cell*c:.2f} bohr³")
print(f"\n  ε⁻¹(q₀, 0)   = {eps_inv_head_0:.6f}  →  ε(q₀, 0)   = {1/eps_inv_head_0:.6f}")
print(f"  ε⁻¹(q₀, iωp) = {eps_inv_head_iwp:.6f}  →  ε(q₀, iωp) = {1/eps_inv_head_iwp:.6f}")
print(f"  γ(0)   = {gamma_0:.6e}")
print(f"  γ(iωp) = {gamma_iwp:.6e}")

# Verify model reproduces input at q₀
eps_check_0 = eps_model(q0, gamma_0)
eps_check_iwp = eps_model(q0, gamma_iwp)
print(f"\n  Verify: ε_model(q₀, 0)   = {eps_check_0:.6f}  (should be {1/eps_inv_head_0:.6f})")
print(f"  Verify: ε_model(q₀, iωp) = {eps_check_iwp:.6f}  (should be {1/eps_inv_head_iwp:.6f})")

# ── Method A: Average W over mini-BZ using model ──
# Wcoul(ω) = (2π/A_BZ) ∫₀^Q W_model(q, γ(ω)) · q · dq
# Vcoul    = (2π/A_BZ) ∫₀^Q v_full(q) · q · dq

def avg_Vcoul():
    """⟨v(q)⟩ over circular mini-BZ."""
    def integrand(q):
        return v_full(q) * q
    result, _ = integrate.quad(integrand, 1e-12, Q_bz)
    return 2.0 * np.pi / A_BZ * result

def avg_Wcoul(gamma):
    """⟨W(q,ω)⟩ = ⟨v/ε⟩ over circular mini-BZ."""
    def integrand(q):
        return W_model(q, gamma) * q
    result, _ = integrate.quad(integrand, 1e-12, Q_bz)
    return 2.0 * np.pi / A_BZ * result

Vcoul_A = avg_Vcoul()
Wcoul_A_0 = avg_Wcoul(gamma_0)
Wcoul_A_iwp = avg_Wcoul(gamma_iwp)

print(f"\n── Method A: average model over mini-BZ ──")
print(f"  Vcoul   = {Vcoul_A:.4f} a.u.  (BGW: {vc0_bgw:.4f})")
print(f"  Wcoul(0)   = {Wcoul_A_0:.4f} a.u.  (BGW: {wc0_bgw_static:.4f})")
print(f"  Wcoul(iωp) = {Wcoul_A_iwp:.4f} a.u.  (BGW: {wc0_bgw_imfreq:.4f})")

# ── Method B: Raw single q-point, no averaging ──
Vcoul_B = v_full(q0)
Wcoul_B_0 = W_model(q0, gamma_0)   # = v(q₀)/ε(q₀,0) = v(q₀)·ε⁻¹(q₀,0)
Wcoul_B_iwp = W_model(q0, gamma_iwp)

print(f"\n── Method B: raw v(q₀) and W(q₀) = v/ε at single point ──")
print(f"  v(q₀)      = {Vcoul_B:.4f} a.u.")
print(f"  W(q₀, 0)   = {Wcoul_B_0:.4f} a.u.")
print(f"  W(q₀, iωp) = {Wcoul_B_iwp:.4f} a.u.")

# ── GN fit from both methods ──
def fit_gn(wc_0, wc_iwp, vc, wp):
    """Fit GN model to W^c = W - V at two frequencies.
    Returns dict with Omega, B, R, on_shell shift per 1/vol.
    """
    wc0 = wc_0 - vc   # correlation part
    wci = wc_iwp - vc
    denom = wc0 - wci
    if abs(denom) < 1e-30:
        return {'Omega_sq': 0, 'Omega': 0, 'B': 0, 'R': 0, 'shift_per_vol': 0,
                'Wc0': wc0, 'Wci': wci}
    Omega_sq = wci * wp**2 / denom
    B = -wc0 * Omega_sq
    Omega = np.sqrt(abs(Omega_sq)) if Omega_sq > 0 else np.sqrt(abs(Omega_sq))
    R = B / (2 * Omega) if Omega > 1e-30 else 0.0
    shift = R / Omega if Omega > 1e-30 else 0.0  # = -Wc0/2
    return {'Omega_sq': Omega_sq, 'Omega': Omega, 'B': B, 'R': R,
            'shift_per_vol': shift, 'Wc0': wc0, 'Wci': wci, 'valid': Omega_sq > 0}

gn_A = fit_gn(Wcoul_A_0, Wcoul_A_iwp, Vcoul_A, omega_p)
gn_B = fit_gn(Wcoul_B_0, Wcoul_B_iwp, Vcoul_B, omega_p)
gn_bgw = fit_gn(wc0_bgw_static, wc0_bgw_imfreq, vc0_bgw, omega_p)

vol = A_cell * c
ryd2ev = 13.6056980659

print(f"\n{'=' * 72}")
print("GN fit comparison")
print("=" * 72)
hdr = f"{'':>25} {'A: ⟨v/ε⟩':>14} {'B: v₀/ε₀':>14} {'BGW actual':>14}"
print(hdr)
print(f"  {'-' * 62}")
for key, label in [('Wc0', 'W^c(0) (a.u.)'), ('Wci', 'W^c(iωp) (a.u.)'),
                    ('Omega_sq', 'Ω² (Ry²)'), ('Omega', 'Ω (Ry)'),
                    ('B', 'B (Ry²·a.u.)'), ('R', 'R (Ry·a.u.)'),
                    ('shift_per_vol', 'R/Ω (a.u.)')]:
    a_val = gn_A[key]
    b_val = gn_B[key]
    bgw_val = gn_bgw[key]
    print(f"  {label:>25} {a_val:14.4f} {b_val:14.4f} {bgw_val:14.4f}")

# On-shell shifts in eV
shift_A = gn_A['shift_per_vol'] / vol * ryd2ev
shift_B = gn_B['shift_per_vol'] / vol * ryd2ev
shift_bgw = gn_bgw['shift_per_vol'] / vol * ryd2ev

print(f"\n  {'occ shift (eV)':>25} {shift_A:+14.4f} {shift_B:+14.4f} {shift_bgw:+14.4f}")
print(f"  {'emp shift (eV)':>25} {-shift_A:+14.4f} {-shift_B:+14.4f} {-shift_bgw:+14.4f}")

print(f"\n── Key question: does γ cancel in Ω²? ──")
if gn_A['Omega_sq'] > 0 and gn_B['Omega_sq'] > 0:
    print(f"  Ω²_A / Ω²_B = {gn_A['Omega_sq'] / gn_B['Omega_sq']:.6f}")
    print(f"  (Would be 1.0 if γ cancelled — it does NOT because W = v/ε, not ε⁻¹·v)")
else:
    print(f"  Ω²_A = {gn_A['Omega_sq']:.6f}, Ω²_B = {gn_B['Omega_sq']:.6f}")
    if gn_B['Omega_sq'] < 0:
        print(f"  Method B gives Ω² < 0: unphysical pole (model breaks down at single point)")

print(f"\n── How well does the circular-BZ model reproduce BGW? ──")
print(f"  Vcoul:   model={Vcoul_A:.2f}, BGW={vc0_bgw:.2f},"
      f" diff={Vcoul_A - vc0_bgw:+.2f} ({(Vcoul_A/vc0_bgw - 1)*100:+.1f}%)")
print(f"  Wcoul(0):   model={Wcoul_A_0:.2f}, BGW={wc0_bgw_static:.2f},"
      f" diff={Wcoul_A_0 - wc0_bgw_static:+.2f} ({(Wcoul_A_0/wc0_bgw_static - 1)*100:+.1f}%)")
print(f"  Wcoul(iωp): model={Wcoul_A_iwp:.2f}, BGW={wc0_bgw_imfreq:.2f},"
      f" diff={Wcoul_A_iwp - wc0_bgw_imfreq:+.2f} ({(Wcoul_A_iwp/wc0_bgw_imfreq - 1)*100:+.1f}%)")

# ── Show how ε⁻¹ and ε vary with q ──
print(f"\n── Model ε⁻¹(q) profile at ω=0 ──")
print(f"  {'q (bohr⁻¹)':>12} {'v(q)':>12} {'ε':>10} {'ε⁻¹':>10} {'W=v/ε':>12}")
for q_frac in [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    q_test = q_frac * Q_bz
    v_test = v_full(q_test)
    eps_test = eps_model(q_test, gamma_0)
    print(f"  {q_test:12.4f} {v_test:12.2f} {eps_test:10.4f} {1/eps_test:10.4f} {v_test/eps_test:12.2f}")
