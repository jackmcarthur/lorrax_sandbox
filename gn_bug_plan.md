# GN-PPM Head Correction and HL-GPP Implementation Plan

## Status

The static COHSEX head correction is implemented and working: `apply_head_correction()`
in `gw/gw_jax.py` adds the q=0 head as a rank-1 correction to V and W, and the
COH = ½(W-V) subtraction cancels the divergence. MoS2 COHSEX matches BGW to 67 meV.

For GN-PPM, `apply_head_diagonal` should remain **false**. BGW treats the head
contribution to Σ^c as negligible (~0.5 meV for 2D MoS2 — see CHANGELOG entry
"Root cause of 2D GN-GPP head discrepancy"). The large q→0 Coulomb divergence
enters through the mini-BZ averaged vcoul in the **bare exchange** Σ_x, not
through the correlation Σ_c.

The remaining ~1 eV GN-PPM body error for 2D MoS2 (1324 meV at 3×3, 1019 meV at
4×4) is NOT from the head. It is from the ISDF PPM pole extraction mixing
G-vector channels. See CHANGELOG "Current status" section.

## Remaining head-related work: GN head correction for dynamic sigma

The head GN fit module (`gw/head_correction.py`) exists but is not currently used
in the GN-PPM path. If needed in future, the equations are:

Given scalar head values $w_1 = W^c_{00}(q{=}0, \omega_1)$ and $w_2 = W^c_{00}(q{=}0, \omega_2)$:

$$\Omega_h^2 = \frac{w_1 \omega_1^2 - w_2 \omega_2^2}{w_1 - w_2}, \qquad B_h = w_1(\omega_1^2 - \Omega_h^2), \qquad R_h = \frac{B_h}{2\Omega_h}$$

Head self-energy (diagonal only, from $\rho^{mn}_{q=0}(G{=}0) = \delta_{mn}$):

$$\Sigma^{c,\text{head}}_{nn}(\mathbf{k}, E) = R_h \left[ \frac{f_{n\mathbf{k}}}{E - \epsilon_{n\mathbf{k}} + \Omega_h} + \frac{1 - f_{n\mathbf{k}}}{E - \epsilon_{n\mathbf{k}} - \Omega_h} \right]$$

## HL-GPP equations (for future implementation)

HL (Hybertsen-Louie) PPM matches $W^c(0)$ and the $1/\omega^2$ asymptotic tail
rather than GN's two imaginary-frequency fit. For the scalar head channel:

**Direct HL from $W^c$:** Given $w_0 = W^c_{00}(0)$ and the asymptotic coefficient
$M_w = \lim_{\omega\to\infty} \omega^2 W^c_{00}(\omega)$:

$$\Omega_h^2 = -\frac{M_w}{w_0}, \qquad R_h = \frac{M_w}{2\Omega_h} = -\frac{\Omega_h w_0}{2}$$

**Finite real-frequency approximation** (using $W^c_{00}(\omega_s)$ at large real $\omega_s$, e.g. 40–60 Ry):

$$M_w \approx \omega_s^2 \operatorname{Re} W^c_{00}(\omega_s)$$

Then use the same $\Omega_h, R_h$ formulas above. This is valid only if $\omega_s$ is
in the asymptotic $1/\omega^2$ regime for that element.

**Exact two-point fit** (if $\omega_s$ is not fully asymptotic):

$$\Omega_h^2 = \omega_s^2 \frac{w_h(\omega_s)}{w_h(\omega_s) - w_h(0)}, \qquad R_h = -\frac{\Omega_h w_h(0)}{2}$$

The self-energy formula is the same as the GN case above — only the pole
parameters differ.

## Implementation entry points

- Head added to V/W: `gw/gw_jax.py:apply_head_correction()` (~line 237)
- PPM extraction: `gw/minimax_screening.py:extract_gn_ppm_parameters_from_Wc()`
- PPM sigma: `gw/ppm_sigma.py:compute_sigma_c_ppm_omega_grid()`
- Head override inputs: `vhead`, `whead_0freq`, `whead_imfreq` in cohsex.in
- Head scalar fit: `gw/head_correction.py` (existing module, currently unused for PPM)
