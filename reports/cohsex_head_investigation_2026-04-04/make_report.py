#!/usr/bin/env python3
"""
Generate report and plots for mini-BZ head averaging analysis.

The question: BGW fits a model ε(q,ω) = 1 + v(q)·q²·γ(ω) at a shifted
q-point and averages W(q,ω) = v(q)/ε(q,ω) over the mini-BZ. The model
parameter γ(ω) is different at each frequency. How much does this averaging
affect the GN plasmon-pole parameters (Ω, B, R) compared to using the raw
single-point values?
"""
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Lattice ──
bohr = 0.529177210903
a = 3.164292 / bohr
c = 12.0 / bohr
zc = c / 2.0
A_cell = a**2 * np.sqrt(3) / 2
A_BZ = (2 * np.pi)**2 / A_cell
Q_bz = np.sqrt(A_BZ / np.pi)
b_mag = 4 * np.pi / (a * np.sqrt(3))
q0 = 0.001 * b_mag
vol = A_cell * c
ryd2ev = 13.6056980659

# ── BGW values ──
eps_inv_0 = 1.152298946517484
eps_inv_iwp = 0.9945864734352702
vc0_bgw = 315.0137
wc0_bgw_0 = 55.6197
wc0_bgw_iwp = 246.1099
omega_p = 2.0

# ── Model functions ──
def v_unnorm(q):
    q = np.asarray(q, float)
    scalar = q.ndim == 0
    q = np.atleast_1d(q)
    out = np.where(q > 1e-15, (1 - np.exp(-zc * q)) / q**2, zc)
    return float(out[0]) if scalar else out

def v_full(q):
    return 8 * np.pi * v_unnorm(q)

def gamma_from_epsinv(ei):
    return (1.0 / ei - 1.0) / (q0**2 * v_unnorm(q0))

def eps_model(q, gam):
    return 1 + v_unnorm(q) * np.asarray(q)**2 * gam

def W_model(q, gam):
    return v_full(q) / eps_model(q, gam)

gamma_0 = gamma_from_epsinv(eps_inv_0)
gamma_iwp = gamma_from_epsinv(eps_inv_iwp)

# ── Averaging ──
def avg_f(func, qmax):
    integrand = lambda q: func(q) * q
    result, _ = integrate.quad(integrand, 1e-12, qmax, limit=200)
    return 2 * np.pi / A_BZ * result

Vcoul = avg_f(v_full, Q_bz)
Wcoul_0 = avg_f(lambda q: W_model(q, gamma_0), Q_bz)
Wcoul_iwp = avg_f(lambda q: W_model(q, gamma_iwp), Q_bz)

# ── GN fits ──
def fit_gn(w0, wi, vc, wp):
    wc0, wci = w0 - vc, wi - vc
    d = wc0 - wci
    if abs(d) < 1e-30:
        return dict(O2=0, O=0, B=0, R=0, s=0, wc0=wc0, wci=wci)
    O2 = wci * wp**2 / d
    B = -wc0 * O2
    O = np.sqrt(abs(O2))
    R = B / (2 * O) if O > 1e-30 else 0
    s = R / O if O > 1e-30 else 0
    return dict(O2=O2, O=O, B=B, R=R, s=s, wc0=wc0, wci=wci, valid=O2 > 0)

gn_avg = fit_gn(Wcoul_0, Wcoul_iwp, Vcoul, omega_p)
gn_bgw = fit_gn(wc0_bgw_0, wc0_bgw_iwp, vc0_bgw, omega_p)

# ── Plots ──
q_arr = np.linspace(1e-4, Q_bz, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (0,0): v(q) and W(q,ω) for both frequencies
ax = axes[0, 0]
ax.semilogy(q_arr, v_full(q_arr), 'k-', label='v(q)', linewidth=1.5)
W_0_arr = np.array([W_model(q, gamma_0) for q in q_arr])
W_iwp_arr = np.array([W_model(q, gamma_iwp) for q in q_arr])
# Clip for log plot — show absolute value, mark sign
ax.semilogy(q_arr[W_0_arr > 0], W_0_arr[W_0_arr > 0], 'b-', label='W(q, ω=0)')
ax.semilogy(q_arr[W_0_arr < 0], -W_0_arr[W_0_arr < 0], 'b--', label='−W(q, ω=0)')
ax.semilogy(q_arr, W_iwp_arr, 'r-', label='W(q, ω=iωp)')
ax.axhline(vc0_bgw, color='k', ls=':', alpha=0.5, label=f'⟨v⟩={vc0_bgw:.0f}')
ax.axhline(wc0_bgw_0, color='b', ls=':', alpha=0.5, label=f'⟨W(0)⟩={wc0_bgw_0:.0f}')
ax.axhline(wc0_bgw_iwp, color='r', ls=':', alpha=0.5, label=f'⟨W(iωp)⟩={wc0_bgw_iwp:.0f}')
ax.axvline(q0, color='gray', ls='--', alpha=0.4, label=f'q₀={q0:.4f}')
ax.set_xlabel('q (bohr⁻¹)')
ax.set_ylabel('|Coulomb| (a.u.)')
ax.set_title('v(q), W(q) from model')
ax.legend(fontsize=6, loc='upper right')
ax.set_ylim(1, 1e6)

# (0,1): ε(q) and ε⁻¹(q) at both frequencies
ax = axes[0, 1]
eps_0_arr = np.array([eps_model(q, gamma_0) for q in q_arr])
eps_iwp_arr = np.array([eps_model(q, gamma_iwp) for q in q_arr])
ax.plot(q_arr, eps_0_arr, 'b-', label='ε(q, ω=0)', linewidth=1.5)
ax.plot(q_arr, eps_iwp_arr, 'r-', label='ε(q, ω=iωp)', linewidth=1.5)
ax.plot(q_arr, 1.0 / eps_0_arr, 'b--', label='ε⁻¹(q, ω=0)', linewidth=1)
ax.plot(q_arr, 1.0 / eps_iwp_arr, 'r--', label='ε⁻¹(q, ω=iωp)', linewidth=1)
ax.axhline(0, color='k', linewidth=0.5)
ax.axhline(1, color='gray', ls=':', alpha=0.5)
ax.axvline(q0, color='gray', ls='--', alpha=0.4)
ax.set_xlabel('q (bohr⁻¹)')
ax.set_ylabel('dielectric function')
ax.set_title('Model ε(q) and ε⁻¹(q)')
ax.legend(fontsize=7)
ax.set_ylim(-12, 4)

# (0,2): Radial integrand q·W(q) for averaging
ax = axes[0, 2]
qW_0 = q_arr * W_0_arr
qW_iwp = q_arr * W_iwp_arr
qv = q_arr * v_full(q_arr)
ax.plot(q_arr, qv, 'k-', label='q·v(q)', linewidth=1.5)
ax.plot(q_arr, qW_0, 'b-', label='q·W(q, ω=0)', linewidth=1.5)
ax.plot(q_arr, qW_iwp, 'r-', label='q·W(q, ω=iωp)', linewidth=1.5)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(q0, color='gray', ls='--', alpha=0.4)
ax.set_xlabel('q (bohr⁻¹)')
ax.set_ylabel('q · f(q) (integrand)')
ax.set_title('Radial integrand for ⟨f⟩_BZ')
ax.legend(fontsize=7)

# (1,0): Cumulative average as function of cutoff
Qcuts = np.linspace(0.01, Q_bz, 80)
cum_V, cum_W0, cum_Wiwp = [], [], []
for Qc in Qcuts:
    cum_V.append(avg_f(v_full, Qc))
    cum_W0.append(avg_f(lambda q: W_model(q, gamma_0), Qc))
    cum_Wiwp.append(avg_f(lambda q: W_model(q, gamma_iwp), Qc))

ax = axes[1, 0]
ax.plot(Qcuts, cum_V, 'k-', label='⟨v⟩(Q)')
ax.plot(Qcuts, cum_W0, 'b-', label='⟨W(0)⟩(Q)')
ax.plot(Qcuts, cum_Wiwp, 'r-', label='⟨W(iωp)⟩(Q)')
ax.axhline(vc0_bgw, color='k', ls=':', alpha=0.5)
ax.axhline(wc0_bgw_0, color='b', ls=':', alpha=0.5)
ax.axhline(wc0_bgw_iwp, color='r', ls=':', alpha=0.5)
ax.set_xlabel('Integration cutoff Q (bohr⁻¹)')
ax.set_ylabel('Cumulative average (a.u.)')
ax.set_title('Convergence of ⟨v⟩ and ⟨W⟩ with mini-BZ radius')
ax.legend(fontsize=7)

# (1,1): GN Ω² and shift as function of cutoff
cum_O2, cum_shift = [], []
for i, Qc in enumerate(Qcuts):
    gn = fit_gn(cum_W0[i], cum_Wiwp[i], cum_V[i], omega_p)
    cum_O2.append(gn['O2'])
    cum_shift.append(gn['s'] / vol * ryd2ev)

ax = axes[1, 1]
ax.plot(Qcuts, cum_O2, 'g-', linewidth=1.5)
ax.axhline(gn_bgw['O2'], color='g', ls=':', label=f'BGW Ω²={gn_bgw["O2"]:.3f}')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Integration cutoff Q (bohr⁻¹)')
ax.set_ylabel('Ω² (Ry²)')
ax.set_title('GN pole Ω² vs mini-BZ radius')
ax.legend(fontsize=7)

# (1,2): On-shell shift convergence
ax = axes[1, 2]
ax.plot(Qcuts, cum_shift, 'm-', linewidth=1.5)
ax.axhline(gn_bgw['s'] / vol * ryd2ev, color='m', ls=':', label=f'BGW shift={gn_bgw["s"]/vol*ryd2ev:.3f} eV')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Integration cutoff Q (bohr⁻¹)')
ax.set_ylabel('On-shell head shift, occ (eV)')
ax.set_title('Head σ shift vs mini-BZ radius')
ax.legend(fontsize=7)

plt.suptitle('MoS₂ 1×1: Mini-BZ Head Averaging — Model ε(q) = 1 + v·q²·γ', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('head_averaging_plots.png', dpi=150, bbox_inches='tight')
print('Saved head_averaging_plots.png')

# ── Text report ──
report = f"""# Mini-BZ Head Averaging Analysis

**System**: MoS₂ 1×1 Γ-only, 2D slab truncation

## Question

BGW fits a model to ε(q) at a shifted q-point and averages W = v/ε over
the mini-BZ. The model parameter γ(ω) differs at each frequency. How much
do the GN parameters change depending on whether we average before or after
the GN fit?

## BGW's model (minibzaverage.f90:119–166)

```
ε(q, ω) = 1 + v(q) · q² · γ(ω)
v(q) = (1 − e^{{−zc·q}}) / q²     (slab-truncated, unnormalized)
γ(ω) = (1/ε⁻¹(q₀,ω) − 1) / (q₀² · v(q₀))
W(q, ω) = 8π · v(q) / ε(q, ω)
```

Note: this is a model for ε, not ε⁻¹. The W = v/ε form is nonlinear in γ.

## Key finding: the ω=0 model is pathological for 1×1

At ω=0, ε⁻¹(q₀) = {eps_inv_0:.3f} > 1, giving γ(0) = {gamma_0:.2f} < 0. This
makes ε(q) cross zero at moderate q ({q_arr[np.argmin(np.abs(eps_0_arr))]:.3f} bohr⁻¹)
and go deeply negative. The model W = v/ε diverges at the zero crossing and
is negative for most of the mini-BZ.

At ω=iωp, ε⁻¹(q₀) = {eps_inv_iwp:.4f} < 1, giving γ(iωp) = {gamma_iwp:.2f} > 0.
The model is well-behaved: ε > 1 everywhere, W < v everywhere.

## Numerical results

| Quantity | Circular avg | BGW (Sobol QMC) | Ratio |
|----------|-------------|-----------------|-------|
| Vcoul | {Vcoul:.1f} | {vc0_bgw:.1f} | {Vcoul/vc0_bgw:.4f} |
| Wcoul(0) | {Wcoul_0:.1f} | {wc0_bgw_0:.1f} | {Wcoul_0/wc0_bgw_0:.4f} |
| Wcoul(iωp) | {Wcoul_iwp:.1f} | {wc0_bgw_iwp:.1f} | {Wcoul_iwp/wc0_bgw_iwp:.4f} |

The circular-BZ radial integral badly misses the static Wcoul because the
divergent region near ε=0 dominates. BGW's Sobol QMC on the hexagonal BZ
presumably handles the sampling differently or includes points that tame the
divergence. The imaginary-frequency Wcoul matches to 0.5%.

| GN parameter | Circular avg | BGW actual |
|---|---|---|
| Ω² (Ry²) | {gn_avg['O2']:.4f} | {gn_bgw['O2']:.4f} |
| Ω (Ry) | {gn_avg['O']:.4f} | {gn_bgw['O']:.4f} |
| On-shell shift (eV) | {gn_avg['s']/vol*ryd2ev:+.3f} | {gn_bgw['s']/vol*ryd2ev:+.3f} |

## Implications

1. **The averaging of W = v/ε is highly nonlinear** — it does NOT reduce to
   a simple rescaling of the GN parameters. Ω² does not cancel between
   frequencies because the v/ε averaging weights change with γ.

2. **The ω=0 model is fragile for ε⁻¹ > 1** (which occurs in 2D at small q).
   The sign change of ε(q) makes the integral ill-conditioned. BGW's QMC
   sampling may handle this differently from our radial integration.

3. **The imaginary-frequency integral is robust**: γ > 0, ε > 1 everywhere,
   well-converged. The static integral is the problematic one.

![Averaging analysis](head_averaging_plots.png)
"""

with open('report.md', 'w') as f:
    f.write(report)
print('Saved report.md')
