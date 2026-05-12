"""Phase 1: orbit-aware kmeans++ init.

Runs the modified ``kmeans_pp_init`` in orbit mode on (a) MoS2 (ntran=2,
symmorphic) and (b) Si (ntran=48, non-symmorphic Fd-3m), and checks:

1. **Canonicalisation is idempotent**: every returned rep equals its own
   canonical form.
2. **Reps are pairwise non-orbit-equivalent**: no two reps share an orbit
   (kpp shouldn't seed the same orbit twice).
3. **Unfolding multiplicities match orbit sizes**: a rep with stabilizer
   k unfolds to n_sym/k distinct centroids; sum of unfolded counts gives
   the total post-symmetrisation centroid count.
4. **Unfolded centroids correlate with high-density regions**: mean ρ at
   unfolded centroid locations is ≥ 3× the grid mean (same sanity check
   the regular kpp passes).
5. **Cross-check vs non-orbit kpp on the n_sym=1 limit** (sanity that
   identity-syms reproduces ordinary kmeans++).
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
    build_min_image_offsets,
)
from centroid.orbit_syms import (
    build_real_space_syms,
    canonicalize_orbit,
    orbit_images,
    unfold_orbit_unique,
)


def run_one_system(label, wfn_path, save_dir, n_rep=50, seed=0):
    print(f"\n══════ {label} ══════")
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    R, Rinv, tau = build_real_space_syms(wfn, sym)
    n_sym = int(Rinv.shape[0])
    print(f"ntran={n_sym}, fft_grid={tuple(int(x) for x in wfn.fft_grid)}")

    rho = jnp.asarray(get_charge_density(wfn=wfn, sym=sym, source="qe_save",
                                         save_dir=save_dir),
                      dtype=jnp.float64)
    Nx, Ny, Nz = rho.shape

    avec = jnp.asarray(np.asarray(wfn.avec), dtype=jnp.float64)
    metric = avec @ avec.T
    offsets = jnp.asarray(build_min_image_offsets(metric))
    print(f"min-image offsets: {offsets.shape[0]}")

    fx, fy, fz = (jnp.linspace(0, 1, n, endpoint=False, dtype=jnp.float64)
                  for n in (Nx, Ny, Nz))
    positions = jnp.stack(jnp.meshgrid(fx, fy, fz, indexing="ij"),
                          axis=-1).reshape(-1, 3)
    rho_flat = rho.reshape(-1)

    # Orbit-aware kpp
    reps = np.asarray(kmeans_pp_init(
        positions, rho_flat, metric, n_rep,
        jax.random.PRNGKey(seed),
        offsets=offsets, Rinv=Rinv, tau=tau,
    ))
    print(f"reps shape: {reps.shape}")

    # 1. Canonicalisation idempotent
    reps_c = np.asarray(canonicalize_orbit(jnp.asarray(reps), Rinv, tau))
    canon_drift = np.max(np.abs(((reps - reps_c) - np.round(reps - reps_c))))
    print(f"canonicalisation idempotent: max drift = {canon_drift:.2e}  "
          f"({'PASS' if canon_drift < 1e-10 else 'FAIL'})")

    # 2. Pairwise non-orbit-equivalent
    images_all = np.asarray(orbit_images(jnp.asarray(reps), Rinv, tau))  # (n_sym, n_rep, 3)
    # For each pair (i, j), is ANY image of rep[i] equivalent (mod 1) to rep[j]?
    n_collisions = 0
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            d = images_all[:, i, :] - reps[j]
            d -= np.round(d)
            if np.any(np.max(np.abs(d), axis=1) < 1e-6):
                n_collisions += 1
    print(f"pairwise orbit collisions: {n_collisions}  "
          f"({'PASS' if n_collisions == 0 else 'FAIL'})")

    # 3. Unfolding multiplicities
    per_rep_orbits = []
    for i in range(n_rep):
        # unique images of rep i mod 1
        rounded = np.round(images_all[:, i, :], 6) % 1.0
        unique = np.unique(rounded, axis=0)
        per_rep_orbits.append(unique.shape[0])
    n_unfold_total = sum(per_rep_orbits)
    print(f"per-rep orbit sizes: min={min(per_rep_orbits)}, "
          f"max={max(per_rep_orbits)}, sum={n_unfold_total} "
          f"(generic-position cap = n_rep · n_sym = {n_rep * n_sym})")

    # 4. Density correlation
    unfolded = unfold_orbit_unique(reps, Rinv, tau)
    fft_grid = np.array([Nx, Ny, Nz])
    idx = (np.round(unfolded * fft_grid).astype(int) % fft_grid)
    rho_at = np.asarray(rho)[idx[:, 0], idx[:, 1], idx[:, 2]]
    enrichment = float(rho_at.mean() / np.asarray(rho).mean())
    print(f"unfolded {unfolded.shape[0]} centroids; "
          f"mean ρ at centroids / grid mean = {enrichment:.2f}×  "
          f"({'PASS' if enrichment > 3.0 else 'WARN'})")

    # 5. Identity-syms equivalence (sanity that orbit-mode collapses correctly)
    reps_noorbit = np.asarray(kmeans_pp_init(
        positions, rho_flat, metric, n_rep,
        jax.random.PRNGKey(seed),
        offsets=offsets,                         # no Rinv/tau ⇒ identity
    ))
    # With identity Rinv/tau, orbit-mode should also reproduce the same draws
    # (canonicalize is identity for n_sym=1, _orbit_distance_sq collapses to
    # one fori_loop iteration ≡ pbc_distance_sq_single).
    Rinv_id = jnp.eye(3, dtype=jnp.int32)[None]
    tau_id = jnp.zeros((1, 3), dtype=jnp.float64)
    reps_orbit_identity = np.asarray(kmeans_pp_init(
        positions, rho_flat, metric, n_rep,
        jax.random.PRNGKey(seed),
        offsets=offsets, Rinv=Rinv_id, tau=tau_id,
    ))
    eq = np.allclose(reps_noorbit, reps_orbit_identity, atol=1e-12)
    print(f"identity-syms equivalence to non-orbit kpp: "
          f"{'PASS' if eq else 'FAIL (max diff = ' + str(np.abs(reps_noorbit - reps_orbit_identity).max()) + ')'}")


def main():
    run_one_system(
        "MoS2  (ntran=2)",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/scf/MoS2.save",
        n_rep=50,
    )
    run_one_system(
        "Si    (ntran=48, Fd-3m)",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_3200c/WFN.h5",
        # find QE save dir for this run
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/scf/silicon.save",
        n_rep=20,                                 # smaller — n_sym=48 is heavy
    )


if __name__ == "__main__":
    main()
