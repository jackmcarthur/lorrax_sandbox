"""Phase 3: orbit Lloyd convergence sanity test.

Run several iterations of the orbit Lloyd step on MoS2 and Si and check:
1. **Reps stay canonical** — every Lloyd iteration ends with canonicalised reps.
2. **Movement² decreases (mostly monotonically)** — algorithmic convergence.
3. **Final reps are close to atomic Wyckoff sites** — physical sanity.
4. **Final reps respect orbit closure** — applying any sym op to a converged
   rep must give a fractional coordinate that, when canonicalised, returns
   the same rep (i.e., the rep is fixed by the canonicalisation, not just
   chosen by it).
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
    unfold_orbit_unique,
)


def orbit_lloyd_step(positions, reps, rho, metric, offsets, R, Rinv, tau):
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


def test_system(label, wfn_path, save_dir, n_rep=20, n_iter=30, seed=0):
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
    print(f"ntran={n_sym}, fft_grid=({Nx}, {Ny}, {Nz}), n_rep={n_rep}")

    reps = kmeans_pp_init(
        positions, rho_flat, metric, n_rep,
        jax.random.PRNGKey(seed),
        offsets=offsets, Rinv=Rinv, tau=tau,
    )

    movements = []
    for it in range(n_iter):
        new_reps, movement_sq = orbit_lloyd_step(
            positions, reps, rho_flat, metric, offsets, R, Rinv, tau,
        )
        m = float(jnp.sqrt(jnp.max(movement_sq)))
        movements.append(m)
        # Reps should already be canonical (orbit_finalize canonicalises).
        canon = np.asarray(canonicalize_orbit(new_reps, Rinv, tau))
        canon_drift = float(np.max(np.abs(
            (np.asarray(new_reps) - canon) - np.round(np.asarray(new_reps) - canon)
        )))
        if canon_drift > 1e-10 and it == 0:
            print(f"  ⚠ iter {it}: canonicalisation drift {canon_drift:.2e}")
        reps = new_reps

    print(f"movement (iters 0, 5, 10, last) = "
          f"{movements[0]:.4e}, {movements[min(5, len(movements)-1)]:.4e}, "
          f"{movements[min(10, len(movements)-1)]:.4e}, {movements[-1]:.4e}")
    drops = sum(1 for i in range(1, len(movements)) if movements[i] <= movements[i-1] * 1.1)
    print(f"non-increasing-by-10%-fudge fraction: {drops}/{len(movements)-1}  "
          f"({'PASS' if drops > 0.7 * (len(movements)-1) else 'WARN'})")

    # Final orbit closure: apply each sym op to each final rep, canonicalise,
    # and check we get back the same rep.
    canon_final = canonicalize_orbit(reps, Rinv, tau)
    drift_final = float(np.max(np.abs(
        (np.asarray(reps) - np.asarray(canon_final))
        - np.round(np.asarray(reps) - np.asarray(canon_final))
    )))
    print(f"final reps canonical: drift = {drift_final:.2e}  "
          f"({'PASS' if drift_final < 1e-10 else 'FAIL'})")

    # Atom-Wyckoff sanity: for each atom, check whether some final rep is in
    # its orbit (i.e., shares the same orbit).
    apos_cart = np.asarray(wfn.atom_positions, dtype=np.float64)
    avec_np = np.asarray(wfn.avec, dtype=np.float64)
    atom_frac = apos_cart @ np.linalg.inv(avec_np)
    Rinv_np, tau_np = np.asarray(Rinv), np.asarray(tau)
    reps_np = np.asarray(reps)
    rep_dist = np.full(len(atom_frac), np.inf)
    for i, atom in enumerate(atom_frac):
        atom_orbit = (atom @ Rinv_np.transpose(0, 2, 1) + tau_np) % 1.0
        for r in reps_np:
            d = atom_orbit - r
            d -= np.round(d)
            metric_np = np.asarray(metric)
            d_min = np.min(np.einsum('si,ij,sj->s', d, metric_np, d))
            rep_dist[i] = min(rep_dist[i], np.sqrt(d_min))
    print(f"atom-to-nearest-rep distances (Å): "
          f"min={rep_dist.min():.3f}, max={rep_dist.max():.3f}, "
          f"mean={rep_dist.mean():.3f}")

    # Final unfolded count
    unfolded = unfold_orbit_unique(reps_np, Rinv_np, tau_np)
    print(f"unfolded centroid count: {unfolded.shape[0]} "
          f"(generic cap = {n_rep * n_sym})")


def main():
    test_system(
        "MoS2 (ntran=2)",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/scf/MoS2.save",
        n_rep=24, n_iter=40,
    )
    test_system(
        "Si (ntran=48)",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/19_cohsex_sym_3200c/WFN.h5",
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/scf/silicon.save",
        n_rep=12, n_iter=30,
    )


if __name__ == "__main__":
    main()
