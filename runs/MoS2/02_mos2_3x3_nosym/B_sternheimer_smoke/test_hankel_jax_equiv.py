"""Equivalence test: scipy hankel_l vs JAX spherical_hankel_table_batch_jax.

Compares per-projector Hankel transforms produced by the current scipy
path against the JAX path on the actual MoS2 pseudopotential data.
Measures wall-time too so we know the GPU win.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env; set_default_env()
import numpy as np
import jax
import jax.numpy as jnp

from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.species import extract_species
from psp.radial_tables import (
    hankel_l, projector_table, projector_deriv_table,
)
from psp.radial.radial_jax import spherical_hankel_table_batch_jax

wfn = WFNReader('WFN.h5')
pseudos = load_pseudopotentials('.')
nspinor = int(wfn.nspinor)
ecut = float(wfn.ecutwfc)
q_max = float(np.sqrt(ecut)) * 1.01
n_q = 4000
q_grid = np.linspace(0.0, q_max, n_q, dtype=np.float64)

species_list = extract_species(pseudos, nspinor=nspinor)

def simpson_w(n_r):
    sw = np.ones(n_r)
    sw[1:-1:2] = 4.0/3.0
    sw[2:-1:2] = 2.0/3.0
    sw[0] = sw[-1] = 1.0/3.0
    return sw

print("\n══ Forward projector Hankel: scipy hankel_l vs JAX batch ══")
max_abs_F = 0.0
max_rel_F = 0.0
t_scipy_F = 0.0
t_jax_F   = 0.0
for sp in species_list:
    weights = simpson_w(len(sp.r)) * sp.rab

    # Reference: scipy per-projector
    t0 = time.perf_counter()
    F_scipy = np.stack([projector_table(sp, ip, q_grid)
                         for ip in range(sp.n_proj)])
    t_scipy_F += time.perf_counter() - t0

    # JAX batch per unique l
    F_jax = np.zeros_like(F_scipy)
    ls = np.asarray(sp.proj_l, dtype=int)
    t0 = time.perf_counter()
    for l_val in np.unique(ls):
        idx = np.where(ls == l_val)[0]
        F_block = spherical_hankel_table_batch_jax(
            int(l_val),
            jnp.asarray(sp.r, dtype=jnp.float64),
            jnp.asarray(sp.beta_r[idx], dtype=jnp.float64),
            jnp.asarray(q_grid, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64),
        )
        F_block.block_until_ready()
        F_jax[idx] = np.asarray(F_block)
    t_jax_F += time.perf_counter() - t0

    diff = F_scipy - F_jax
    a = float(np.max(np.abs(diff)))
    rel = a / max(float(np.max(np.abs(F_scipy))), 1e-30)
    max_abs_F = max(max_abs_F, a)
    max_rel_F = max(max_rel_F, rel)
    print(f"  {sp.element}: |Δ|_∞={a:.3e}  rel={rel:.3e}  "
          f"t_scipy={t_scipy_F:.2f}s  t_jax={t_jax_F:.2f}s")

print(f"\n  Forward summary: max abs={max_abs_F:.3e}, max rel={max_rel_F:.3e}")
print(f"  scipy total: {t_scipy_F:.2f}s  |  jax total: {t_jax_F:.2f}s")

print("\n══ Deriv (H_{l+1}) Hankel: scipy projector_deriv_table vs JAX batch ══")
max_abs_D = 0.0
max_rel_D = 0.0
t_scipy_D = 0.0
t_jax_D   = 0.0
for sp in species_list:
    weights = simpson_w(len(sp.r)) * sp.rab

    # Reference: scipy per-projector deriv table.
    # Note: projector_deriv_table returns the FULL dG/dq (with /q^l division).
    # For equivalence, compare directly to the JAX H_{l+1} path with the same
    # post-processing — easier: compare just the raw H_{l+1} Hankel itself.

    # Recompute scipy raw H_{l+1}: integrand is β = (β/r) · r → use sp.beta_r * sp.r.
    H_scipy = np.zeros((sp.n_proj, n_q), dtype=np.float64)
    t0 = time.perf_counter()
    for ip in range(sp.n_proj):
        l = int(sp.proj_l[ip])
        beta_full = sp.beta_r[ip] * sp.r        # restore β(r)
        H_scipy[ip] = hankel_l(l + 1, sp.r, beta_full, q_grid, sp.rab)
    t_scipy_D += time.perf_counter() - t0

    # JAX batch: same integrand, but call spherical_hankel_table_batch_jax
    # at l+1.  Group by *unique l+1* to use one batch per group.
    H_jax = np.zeros_like(H_scipy)
    lp1s = np.asarray(sp.proj_l, dtype=int) + 1
    beta_full_br = sp.beta_r * sp.r[None, :]    # (n_proj, n_r) — β(r)
    t0 = time.perf_counter()
    for l_val in np.unique(lp1s):
        idx = np.where(lp1s == l_val)[0]
        H_block = spherical_hankel_table_batch_jax(
            int(l_val),
            jnp.asarray(sp.r, dtype=jnp.float64),
            jnp.asarray(beta_full_br[idx], dtype=jnp.float64),
            jnp.asarray(q_grid, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64),
        )
        H_block.block_until_ready()
        H_jax[idx] = np.asarray(H_block)
    t_jax_D += time.perf_counter() - t0

    diff = H_scipy - H_jax
    a = float(np.max(np.abs(diff)))
    rel = a / max(float(np.max(np.abs(H_scipy))), 1e-30)
    max_abs_D = max(max_abs_D, a)
    max_rel_D = max(max_rel_D, rel)
    print(f"  {sp.element}: |Δ|_∞={a:.3e}  rel={rel:.3e}")

print(f"\n  Deriv summary: max abs={max_abs_D:.3e}, max rel={max_rel_D:.3e}")
print(f"  scipy total: {t_scipy_D:.2f}s  |  jax total: {t_jax_D:.2f}s")

# Tolerance check.  scipy uses double-precision Bessel; JAX uses Miller's
# backward recurrence which has comparable accuracy near machine eps.
# We expect rel < 1e-10 for both.  If anything larger, surface for review.
TOL = 1e-9
fail = (max_rel_F > TOL) or (max_rel_D > TOL)
print(f"\n  Tolerance: relative error < {TOL:.0e}")
if fail:
    print(f"  [WARN] FORWARD rel={max_rel_F:.2e}  DERIV rel={max_rel_D:.2e}")
    sys.exit(1)
print("  [PASS] JAX Hankel matches scipy within tolerance.")
