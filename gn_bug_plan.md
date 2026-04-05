Here is a denser handoff note with the requested implementation change.

---

## GWJAX head-correction bug

**Problem:** GWJAX currently handles the (q=0,;G=G'=0) head in a way that is not basis-equivalent to BerkeleyGW. The core GWJAX self-energy path is the ISDF one: build (G_k) and (W_q), FFT to real-space lattice coordinates, take the elementwise product in ISDF pair indices ((\mu,\nu)), then project back to bands. That is the correct path for the **body** of (W), but not for the head.

In BerkeleyGW / plane waves, the head is its own exact scalar channel (W^c_{00}(q=0,\omega)). For (q=0,;G=0), the pair density is exactly the spatial average of (\psi^**{m\mathbf{k}}\psi*{n\mathbf{k}}), so
[
\rho^{mn}*{q=0}(G=0)=\langle m\mathbf{k}|n\mathbf{k}\rangle=\delta*{mn}.
]
Therefore the head contributes **only to diagonal self-energy matrix elements**. This is also the logic behind the Gaussian periodic GW head correction: the (G=0) head modifies only diagonal (\Sigma_{nn}).

The bug is that GWJAX currently tries to represent this special plane-wave scalar head inside the ISDF body basis, via the (g_\mu=\langle G=0|\zeta_{q=0,\mu}\rangle) channel, instead of treating it separately. That introduces a basis-set inequivalence: in plane waves the head is an exact separate basis direction, while in ISDF it gets spread over body degrees of freedom. Because the 2D head is large, this likely distorts the effective dynamical treatment and is the plausible origin of the large MoS(_2) discrepancy even when benzene works.

## Required fix

Do **not** add the head into the ISDF (W_{\mu\nu}) body at all. Do not “subtract it back out” later. Instead, never put it into the body pipeline in the first place. Implement it in a separate module and add it afterward as an independent diagonal correction.

The body GN fitting / body (\Sigma) path should see only the non-head ISDF (W). The head should be handled entirely outside that path.

## Equations to implement

Let
[
w_j \equiv W^c_{00}(q=0,\omega_j)
]
be the scalar head values at the two GN fit frequencies (\omega_1,\omega_2).

Fit a separate scalar GN model
[
W^c_{00}(q=0,\omega)=\frac{B_h}{\omega^2-\Omega_h^2}.
]

From the two sampled values,
[
\Omega_h^2=\frac{w_1\omega_1^2-w_2\omega_2^2}{w_1-w_2},
\qquad
B_h=w_1(\omega_1^2-\Omega_h^2)=w_2(\omega_2^2-\Omega_h^2),
\qquad
R_h=\frac{B_h}{2\Omega_h}.
]

Then add the head contribution directly in band space:
[
\boxed{
\Sigma^{c,\mathrm{head}}_{mn}(\mathbf{k},E)
===========================================

\delta_{mn},
R_h\left[
\frac{f_{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}+\Omega_h-i0^+}
+
\frac{1-f_{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}-\Omega_h+i0^+}
\right]
}
]
where (f_{n\mathbf{k}}=1) for occupied states and (0) for empty states.

Equivalently:

* valence (v):
  [
  \Sigma^{c,\mathrm{head}}*{v}(\mathbf{k},E)
  =
  \frac{R_h}{E-\epsilon*{v\mathbf{k}}+\Omega_h-i0^+},
  ]

* conduction (c):
  [
  \Sigma^{c,\mathrm{head}}*{c}(\mathbf{k},E)
  =
  \frac{R_h}{E-\epsilon*{c\mathbf{k}}-\Omega_h+i0^+}.
  ]

This is the only head formula that should be used in the final self-energy. There should be no attempt to rebuild this through the ISDF ((\mu,\nu)) body representation.

## How to structure it

Create a separate module, e.g.
`gw/head_correction.py`
or similar.

It should do only three things:

1. Read / receive the scalar head samples (W^c_{00}(q=0,\omega_1)), (W^c_{00}(q=0,\omega_2)).
2. Fit (\Omega_h,B_h,R_h).
3. Evaluate (\Sigma^{c,\mathrm{head}}_{nn\mathbf{k}}(E)) for the requested bands / frequencies and return a diagonal correction array.

## Output / debugging requirements

* Do **not** mix the head contribution into the normal body column(s) of `sigma_freq_debug`.
* Add it as its own separate column / field in `sigma_freq_debug`, so it is easy to inspect independently.
* But when printing the final QP energies, **do add it directly** into the total (\Sigma) used for the printed QP result.

So debugging output should expose

* body correlation contribution,
* head correction contribution,
* total used for QP energies.

## One-sentence summary

The bug is that GWJAX currently treats the plane-wave (q=0,G=0) head as if it belonged in the ISDF body path; the fix is to never add it to the ISDF body at all, fit it as its own scalar GN pole, and add its exact diagonal-only self-energy contribution afterward in a separate module and a separate `sigma_freq_debug` column, while including it in the final printed QP energies.

## Implementation entry points (from Claude)

The head is currently added to W via `apply_head_correction()` in `gw/gw_jax.py` (~line 233), which adds `(wcoul0 / cell_volume) * |ζ(G=0)⟩⟨ζ(G=0)|` to W_q at q=0 (line 287-289). The PPM extraction in `gw/ppm_sigma.py` (lines 186-210) then builds Wc = W_headed - V_headed, so the head leaks into the ISDF body PPM parameters B and Ω. To implement the fix: stop adding the head to W before PPM extraction (pass `W_nohead` instead), and add the new scalar head GN correction as a separate diagonal term after the body Σ^c is computed. The `vhead`/`whead_0freq`/`whead_imfreq` override parameters in `cohsex.in` (see `docs_gwjax/COHSEX_INPUT.md`) already provide the scalar head values needed for the separate GN fit.
