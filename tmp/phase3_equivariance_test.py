"""Phase 3: orbit accumulator + finalize equivariance + brute-force parity.

Equivariance test (the cleanest single check):

    For any sym op s₀, applying s₀ to all input points before the orbit
    Lloyd update should produce the SAME canonical reps as not applying it.

Concretely:
    rep'_a = orbit_lloyd_step(positions, reps, ρ)
    rep'_b = orbit_lloyd_step(s₀ ⋅ positions, reps, ρ)         # ρ unchanged
    => rep'_a == rep'_b   (after canonicalisation)

This holds because the orbit assignment + fold-back accumulator project
out the symmetry-image redundancy: a point and its s₀-image both fold
back into the same rep-frame displacement.

Additionally, for a sanity check, we run a brute-force (numpy) reference
implementation of the orbit Lloyd update and verify the JIT'd kernels
match it on a small synthetic system.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_A/src")

from runtime import set_default_env
set_default_env()
import jax
import jax.numpy as jnp
import numpy as np

from file_io import WFNReader
from common import symmetry_maps
from centroid.charge_density import get_charge_density
from centroid.kmeans_isdf import (
    kmeans_pp_init,
    assign_labels_orbit_chunked,
    _orbit_local_update_accumulators,
    _orbit_finalize_update,
    build_min_image_offsets,
)
from centroid.orbit_syms import (
    build_real_space_syms,
    canonicalize_orbit,
)


def orbit_lloyd_step(positions, reps, rho, metric, offsets, R, Rinv, tau):
    """One sym-aware Lloyd iteration: assign + accumulate + finalize."""
    n_rep = reps.shape[0]
    labels, _, tie_mask = assign_labels_orbit_chunked(
        positions, reps, metric, n_rep,
        offsets=offsets, Rinv=Rinv, tau=tau,
    )
    sum_wd, sum_w = _orbit_local_update_accumulators(
        positions, reps, rho, labels, tie_mask,
        n_rep, metric, offsets, R, Rinv, tau,
    )
    new_reps, movement_sq = _orbit_finalize_update(
        reps, sum_wd, sum_w, metric, offsets, Rinv, tau,
    )
    return new_reps, movement_sq


def setup(label, wfn_path, save_dir, n_rep, seed=0):
    print(f"\n══════ {label} ══════")
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    R, Rinv, tau = build_real_space_syms(wfn, sym)
    n_sym = int(R.shape[0])
    rho_jax = jnp.asarray(get_charge_density(
        wfn=wfn, sym=sym, source="qe_save", save_dir=save_dir,
    ), dtype=jnp.float64)
    Nx, Ny, Nz = rho_jax.shape
    avec = jnp.asarray(np.asarray(wfn.avec), dtype=jnp.float64)
    metric = avec @ avec.T
    offsets = jnp.asarray(build_min_image_offsets(metric))
    fx, fy, fz = (jnp.linspace(0, 1, n, endpoint=False, dtype=jnp.float64)
                  for n in (Nx, Ny, Nz))
    positions = jnp.stack(jnp.meshgrid(fx, fy, fz, indexing="ij"),
                          axis=-1).reshape(-1, 3)
    rho_flat = rho_jax.reshape(-1)
    reps = kmeans_pp_init(
        positions, rho_flat, metric, n_rep,
        jax.random.PRNGKey(seed),
        offsets=offsets, Rinv=Rinv, tau=tau,
    )
    print(f"ntran={n_sym}, fft_grid=({Nx}, {Ny}, {Nz}), n_rep={n_rep}")
    return positions, reps, rho_flat, metric, offsets, R, Rinv, tau, (Nx, Ny, Nz)


def equivariance_test(positions, reps, rho, metric, offsets, R, Rinv, tau,
                      fft_grid):
    """For each sym op s₀, apply it to positions and check that the
    resulting Lloyd step reproduces the same canonical reps.

    The transformed positions ``s₀ ⋅ x`` are snapped back to the FFT grid
    to suppress fp64 drift from the matrix-multiply / mod-1 wrap; the
    snap is exact when (R, τ) is grid-commensurate (verified by
    ``build_real_space_syms`` via the atomic-symmetry validator).

    This also requires ρ to be sym-invariant (true for ``rho_from_qe_save``).
    The pos_b vs pos_a permutation re-orders the per-index ρ array, which
    is what ``rho_b`` should be set to — we re-gather ρ at each new
    physical location below.
    """
    new_a, _ = orbit_lloyd_step(positions, reps, rho, metric, offsets,
                                R, Rinv, tau)
    new_a = np.asarray(new_a)
    n_sym = int(R.shape[0])
    fft_arr = jnp.asarray(np.array(fft_grid))
    pos_np = np.asarray(positions)
    pos_idx_orig = np.round(pos_np * np.array(fft_grid)).astype(int) % fft_grid

    # Build a lookup: integer-index triple → flat point index
    Nx, Ny, Nz = fft_grid
    lin_lookup = np.full(Nx * Ny * Nz, -1, dtype=np.int64)
    lin_orig = (pos_idx_orig[:, 0] * Ny * Nz
                + pos_idx_orig[:, 1] * Nz
                + pos_idx_orig[:, 2])
    lin_lookup[lin_orig] = np.arange(pos_np.shape[0])

    max_drift = 0.0
    for s0 in range(n_sym):
        # Transform points then snap to grid — exact when grid-commensurate.
        pos_b_raw = positions @ Rinv[s0].T + tau[s0]
        idx_b = jnp.round(pos_b_raw * fft_arr).astype(jnp.int32) % fft_arr
        pos_b = idx_b.astype(jnp.float64) / fft_arr             # snapped, on-grid

        # Re-gather ρ at each new physical location: pos_b[i] is the
        # transformed location of the original point i. The ρ value at
        # that physical location is ρ at whichever ORIGINAL point lives there.
        idx_b_np = np.asarray(idx_b)
        lin_b = (idx_b_np[:, 0] * Ny * Nz
                 + idx_b_np[:, 1] * Nz
                 + idx_b_np[:, 2])
        src_idx = lin_lookup[lin_b]                              # (P,) original indices
        rho_b = rho[jnp.asarray(src_idx)]

        new_b, _ = orbit_lloyd_step(pos_b, reps, rho_b, metric, offsets,
                                    R, Rinv, tau)
        new_b = np.asarray(new_b)
        drift = (new_a - new_b)
        drift = drift - np.round(drift)
        max_drift = max(max_drift, float(np.max(np.abs(drift))))
    print(f"equivariance: max drift over all {n_sym} sym ops = {max_drift:.2e}  "
          f"({'PASS' if max_drift < 1e-10 else 'FAIL'})")


def brute_force_orbit_lloyd_step(positions_np, reps_np, rho_np, metric_np,
                                 offsets_np, R_np, Rinv_np, tau_np,
                                 tie_tol=1e-10):
    """Brute-force NumPy reference for one orbit Lloyd step.
    Returns canonicalised new reps."""
    P = positions_np.shape[0]
    n_rep = reps_np.shape[0]
    n_sym = R_np.shape[0]

    # 1. Assign: orbit distance d²(P, n_rep, n_sym)
    d2 = np.full((P, n_rep, n_sym), np.inf)
    for s in range(n_sym):
        image_s = reps_np @ Rinv_np[s].T + tau_np[s]              # (n_rep, 3)
        delta_raw = positions_np[:, None, :] - image_s[None, :, :]
        delta_w = delta_raw - np.round(delta_raw)
        for o in offsets_np:
            df = delta_w + o
            d2_o = np.einsum("pmi,ij,pmj->pm", df, metric_np, df)
            d2[:, :, s] = np.minimum(d2[:, :, s], d2_o)
    orbit_d = d2.min(axis=2)
    labels = orbit_d.argmin(axis=1)
    best_d = orbit_d[np.arange(P), labels]
    d2_winner = d2[np.arange(P), labels, :]
    tie_mask = d2_winner <= best_d[:, None] + tie_tol             # (P, n_sym)

    # 2. Accumulate
    n_tied = tie_mask.sum(axis=1).astype(float)
    inv_n = np.where(n_tied > 0, 1.0 / n_tied, 0.0)
    sum_wd = np.zeros((n_rep, 3))
    sum_w = np.zeros((n_rep,))
    for s in range(n_sym):
        # δ_image = min-image(positions - image of rep_per_point under s)
        image_p = reps_np[labels] @ Rinv_np[s].T + tau_np[s]      # (P, 3)
        delta_raw = positions_np - image_p
        delta_w = delta_raw - np.round(delta_raw)
        # Find min-image offset n* per point (we only need the vector)
        d2_per_o = np.full((P, len(offsets_np)), np.inf)
        for oi, o in enumerate(offsets_np):
            df = delta_w + o
            d2_per_o[:, oi] = np.einsum("pi,ij,pj->p", df, metric_np, df)
        best_o = d2_per_o.argmin(axis=1)
        delta_image = delta_w + offsets_np[best_o]                # (P, 3)
        delta_rep = delta_image @ R_np[s].T                       # fold-back
        w_s = rho_np * inv_n * tie_mask[:, s].astype(float)
        for p in range(P):
            sum_wd[labels[p]] += w_s[p] * delta_rep[p]
            sum_w[labels[p]] += w_s[p]

    # 3. Finalize
    avg = np.where(sum_w[:, None] > 0,
                   sum_wd / np.maximum(sum_w[:, None], 1e-10), 0.0)
    new_reps = (reps_np + avg) % 1.0
    # Canonicalise
    new_reps_canon = []
    for r in new_reps:
        images = (r @ Rinv_np.transpose(0, 2, 1) + tau_np) % 1.0
        keys = np.round(images * 1e12).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        new_reps_canon.append(images[order[0]])
    return np.array(new_reps_canon)


def brute_force_parity_test(positions, reps, rho, metric, offsets,
                            R, Rinv, tau, P_subsample=2000):
    """JIT'd one-step output should match brute-force NumPy reference."""
    rng = np.random.default_rng(0)
    idx = rng.choice(positions.shape[0], P_subsample, replace=False)
    pos_sub = positions[jnp.asarray(idx)]
    rho_sub = rho[jnp.asarray(idx)]

    new_jit, _ = orbit_lloyd_step(pos_sub, reps, rho_sub, metric, offsets,
                                  R, Rinv, tau)
    new_jit = np.asarray(new_jit)
    new_ref = brute_force_orbit_lloyd_step(
        np.asarray(pos_sub), np.asarray(reps), np.asarray(rho_sub),
        np.asarray(metric), np.asarray(offsets),
        np.asarray(R), np.asarray(Rinv), np.asarray(tau),
    )
    drift = new_jit - new_ref
    drift = drift - np.round(drift)
    max_drift = float(np.max(np.abs(drift)))
    print(f"brute-force parity: max drift = {max_drift:.2e}  "
          f"({'PASS' if max_drift < 1e-10 else 'WARN' if max_drift < 1e-5 else 'FAIL'})")


def main():
    for label, wfn, save, n_rep in [
        ("MoS2 (ntran=2)",
         "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5",
         "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/scf/MoS2.save",
         16),
        ("Si (ntran=48)",
         "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_3200c/WFN.h5",
         "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/scf/silicon.save",
         8),
    ]:
        positions, reps, rho, metric, offsets, R, Rinv, tau, fft_grid = setup(
            label, wfn, save, n_rep,
        )
        equivariance_test(positions, reps, rho, metric, offsets,
                          R, Rinv, tau, fft_grid)
        brute_force_parity_test(positions, reps, rho, metric, offsets,
                                R, Rinv, tau, P_subsample=1500)


if __name__ == "__main__":
    main()
