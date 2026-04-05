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



## Testing HL-GPP rather than GN-GPP to check improvement
equations from ChatGPT:
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
`gw_isdf/head_correction.py`
or similar.

It should do only three things:

1. Read / receive the scalar head samples (W^c_{00}(q=0,\omega_1)), (W^c_{00}(q=0,\omega_2)).
2. Fit (\Omega_h,B_h,R_h).
3. Evaluate (\Sigma^{c,\mathrm{head}}_{nn\mathbf{k}}(E)) for the requested bands / frequencies and return a diagonal correction array.

## Output / debugging requirements

* Do **not** mix the head contribution into the normal body column(s) of `sigma_freq_debug`.
* Add it as its own separate column / field in `sigma_freq_debug`, so it is easy to inspect independently.
* But when printing the final QP energies, **do add it directly** into the total (\Sigma) used for the printed QP result.

Yes. The multipole-W paper gives exactly the right lens for this: the HL plasmon-pole model is equivalent to matching the response at (\omega=0) and matching its **leading (1/\omega^2) coefficient at (\omega\to\infty)**, while GN is the two-point fit at (0) and (i\omega_p). Leon et al. make this explicit: for a one-pole model (X^{\mathrm{PP}}(z)=2\Omega R/(z^2-\Omega^2)), GN matches (X(0)) and (X(i\omega_p)), whereas HL matches (X(0)) and the asymptotic coefficient, i.e. the (1/z^2) tail or “(z_2=\infty)” condition. They also show the resulting self-energy from a pole expansion has exactly the same occupied/unoccupied denominator structure we wrote before. 

For your head module, the cleanest thing is to fit the **scalar head channel directly**. Let
[
w_h(\omega)\equiv W^c_{00}(q=0,\omega)
]
denote the scalar correlation head after whatever mini-BZ / truncated-Coulomb averaging you use for the (q=0) head. Since for (q=0,;G=0),
[
\rho^{mn}(G=0)=\langle m\mathbf{k}|n\mathbf{k}\rangle=\delta_{mn},
]
this head contributes only to diagonal (\Sigma_{nn}), exactly as in the Gaussian periodic GW head correction. 

Use a one-pole HL form
[
w_h^{\mathrm{HL}}(\omega)=\frac{2\Omega_h R_h}{\omega^2-\Omega_h^2}.
]
Then
[
w_h^{\mathrm{HL}}(0)=-\frac{2R_h}{\Omega_h},
\qquad
\lim_{\omega\to\infty}\omega^2 w_h^{\mathrm{HL}}(\omega)=2\Omega_h R_h.
]
So if you define
[
w_0 \equiv w_h(0),
\qquad
M_w \equiv \lim_{\omega\to\infty}\omega^2 w_h(\omega),
]
the **true HL** parameters are
[
\boxed{\Omega_h^2=-\frac{M_w}{w_0}},
\qquad
\boxed{R_h=\frac{M_w}{2\Omega_h}=-\frac{\Omega_h w_0}{2}}.
]

That is the direct HL analogue of your current GN head fit. Once you have ((\Omega_h,R_h)), the diagonal head self-energy is just the same pole formula as before, only with HL-fitted parameters:
[
\boxed{
\Sigma^{c,\mathrm{head}}_{nn}(\mathbf{k},E)
===========================================

R_h\left[
\frac{f_{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}+\Omega_h-i0^+}
+
\frac{1-f_{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}-\Omega_h+i0^+}
\right]
}
]
with (f_{n\mathbf{k}}=1) for occupied, (0) for empty. This denominator structure is the same one that appears in the multipole self-energy expression of Leon et al. 

If you do **not** have a genuine (\omega=\infty) quantity and instead want to approximate it from a large real frequency (\omega_s) such as (50) Ry, there are two distinct things you can do:

1. **HL-style asymptotic approximation**
   Assume (\omega_s) is already in the asymptotic regime and set
   [
   M_w \approx \omega_s^2,\mathrm{Re},w_h(\omega_s).
   ]
   Then use
   [
   \boxed{
   \Omega_h^2 \approx -\frac{\omega_s^2,\mathrm{Re},w_h(\omega_s)}{w_h(0)},
   \qquad
   R_h \approx \frac{\omega_s^2,\mathrm{Re},w_h(\omega_s)}{2\Omega_h}.
   }
   ]
   This is the closest thing to “HL with (\infty) replaced by (50) Ry”. It is only sensible if (50) Ry is well beyond all relevant plasmon structure for that head element, so that the response is already in its (1/\omega^2) tail. Because the exact tail coefficient is real, taking the real part is the natural thing to do at large real (\omega). The multipole paper explicitly says the HL condition is the (\omega\to\infty) leading-order coefficient, not a generic finite-(\omega) sample. 

2. **Finite-real-frequency two-point fit**
   If you do not trust (50) Ry to be fully asymptotic, then you are no longer really doing HL; you are doing a one-pole fit constrained by (w_h(0)) and (w_h(\omega_s)) on the real axis. For
   [
   w_h^{\mathrm{PP}}(\omega)=\frac{2\Omega_h R_h}{\omega^2-\Omega_h^2},
   ]
   the exact two-point solve is
   [
   \boxed{
   \Omega_h^2=\omega_s^2,\frac{w_h(\omega_s)}{w_h(\omega_s)-w_h(0)},
   \qquad
   R_h=-\frac{\Omega_h,w_h(0)}{2}.
   }
   ]
   If (w_h(\omega_s)\approx M_w/\omega_s^2), this reduces to the HL formula above. So this is the better formula if you literally want to use the computed value at (50) Ry without assuming it is fully asymptotic.

If instead you want to fit the **head polarizability** or reducible response first, say
[
\Pi_h(\omega)\equiv \Pi_{00}(q=0,\omega),
]
then use the same HL logic on (\Pi_h):
[
\Pi_h^{\mathrm{HL}}(\omega)=\frac{2\Omega_\Pi R_\Pi}{\omega^2-\Omega_\Pi^2},
\qquad
\Pi_0\equiv \Pi_h(0),
\qquad
M_\Pi\equiv \lim_{\omega\to\infty}\omega^2\Pi_h(\omega),
]
so that
[
\boxed{
\Omega_\Pi^2=-\frac{M_\Pi}{\Pi_0},
\qquad
R_\Pi=\frac{M_\Pi}{2\Omega_\Pi}.
}
]
Then convert to the scalar screened head through the Dyson relation. For a scalar head channel with bare head factor (v_h),
[
W_h(\omega)=\frac{v_h}{1-v_h\Pi_h(\omega)},
\qquad
W_h^c(\omega)=W_h(\omega)-v_h.
]
Plugging in the one-pole (\Pi_h) gives another one-pole form for (W_h^c):
[
W_h^c(\omega)=\frac{2\Omega_\Pi v_h^2 R_\Pi}{\omega^2-\Omega_\Pi^2-2\Omega_\Pi v_h R_\Pi}
\equiv \frac{2\Omega_h R_h}{\omega^2-\Omega_h^2},
]
with
[
\boxed{
\Omega_h^2=\Omega_\Pi^2+2\Omega_\Pi v_h R_\Pi
=\Omega_\Pi^2\bigl(1-v_h\Pi_0\bigr),
}
]
and
[
\boxed{
2\Omega_h R_h = 2\Omega_\Pi v_h^2 R_\Pi = v_h^2 M_\Pi.
}
]
So if what you actually have is (\Pi_h(0)) and either (\Pi_h(\infty)) or (\Pi_h(50,\mathrm{Ry})), you can fit (\Pi_h) first and then map to (W_h^c) with these formulas.

For your bug fix, though, I would strongly prefer the **direct (W_h^c) fit** if you already have the scalar head (W^c_{00}(q=0,\omega)) in the same normalization that your sigma code uses. That avoids extra ambiguity about which (\Pi) object you mean and how the mini-BZ averaged (q=0) head normalization enters.

So the practical formulas I would hand over are:

* **Direct HL-style head from (W^c):**
  [
  w_0=W^c_{00}(0),\qquad
  M_w \approx \omega_s^2,\mathrm{Re},W^c_{00}(\omega_s)\quad (\omega_s=50\ \mathrm{Ry}),
  ]
  [
  \Omega_h^2=-M_w/w_0,\qquad R_h=M_w/(2\Omega_h),
  ]
  [
  \Sigma^{c,\mathrm{head}}*{nn}(\mathbf{k},E)
  =
  R_h\left[
  \frac{f*{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}+\Omega_h-i0^+}

-

\frac{1-f_{n\mathbf{k}}}{E-\epsilon_{n\mathbf{k}}-\Omega_h+i0^+}
\right].
]

* **If using the finite value at (50) Ry literally instead of asymptotically:**
  [
  \Omega_h^2=\omega_s^2\frac{w_h(\omega_s)}{w_h(\omega_s)-w_h(0)},
  \qquad
  R_h=-\frac{\Omega_h w_h(0)}{2}.
  ]

The main conceptual warning is: **HL is not “sample at a large real frequency”; HL is “match the (1/\omega^2) tail.”** Replacing (\infty) by (50) Ry is only justified if that specific head element is already in its asymptotic regime there. The multipole paper is very explicit that the HL condition is the asymptotic leading-order coefficient, i.e. the (\omega\to\infty) limit. 

If you want, I can next rewrite this in exactly the variable names you would likely have in GWJAX, e.g. `w_head_0`, `w_head_50ry`, `omega_head_hl`, `R_head_hl`, and `sigma_head_freq[nk,nb,nfreq]`.
