"""Verify the proper Gram K_q=0 is PSD for all four channels.

K_q=0(μ, λ) = Σ_{k, n_l, n_r} ρ*(μ) ρ(λ)
with ρ_{n_l, n_r, k}(r) = Σ_{ab} ψ_l_n_l, k, a*(r) γ̃_{ab} ψ_r_n_r, k, b(r).

This is a Gram of {ρ_{n_l, n_r, k}(.)} at points (μ, λ) — PSD by
construction.  If empirically PSD with finite Cholesky residual for
all four μ_L, the previous "indefinite-CCT" finding was an artifact
of the Schur-product CCT form, not a structural property of the
ISDF problem.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from common import Meta, symmetry_maps
from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
from common.load_wfns import load_centroids_band_chunked
from file_io import WFNReader


LORRAX_RUN = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex"
)
WFN_PATH = LORRAX_RUN / "WFN.h5"
CENTROIDS_PATH = LORRAX_RUN / "centroids_frac_640.txt"
BAND_RANGE = (0, 24)
LEFT_RANGE = (0, 8)
RIGHT_RANGE = (8, 24)


def _make_mesh():
    devs = jax.devices()
    n = len(devs)
    if n == 1:
        return Mesh(np.asarray(devs).reshape(1, 1), axis_names=("x", "y"))
    raise RuntimeError("smoke is single-GPU only")


def _load_centroid_indices(meta):
    coords = np.loadtxt(CENTROIDS_PATH, dtype=np.float64)
    grid = np.array(meta.fft_grid)
    idx = np.rint(coords * grid).astype(np.int64) % grid
    seen, unique = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return jnp.asarray(np.asarray(unique, dtype=np.int64))


def main():
    print(f"jax devices: {jax.devices()}", flush=True)
    mesh = _make_mesh()

    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    nb = BAND_RANGE[1] - BAND_RANGE[0]
    meta = Meta.from_system(
        wfn, sym, nval=BAND_RANGE[1], ncond=0, nband=nb,
        n_rmu=640, bispinor=True,
    )
    centroid_idx = _load_centroid_indices(meta)
    print(f"loaded {centroid_idx.shape[0]} centroids; ns={meta.nspinor}", flush=True)

    psi_rmu, _psi_rmuT = load_centroids_band_chunked(
        wfn, sym, meta, centroid_idx,
        bispinor=True, mesh_xy=mesh,
        band_range=BAND_RANGE,
        band_chunk_size=min(64, nb),
    )
    # psi_rmu: (nk, nb, 4, n_rmu)
    psi_l = psi_rmu[:, LEFT_RANGE[0]:LEFT_RANGE[1], :, :]    # (k, nl, 4, μ)
    psi_r = psi_rmu[:, RIGHT_RANGE[0]:RIGHT_RANGE[1], :, :]  # (k, nr, 4, μ)
    print(f"  ψ_l shape={tuple(psi_l.shape)}, ψ_r shape={tuple(psi_r.shape)}",
          flush=True)

    gammas = [gamma0, gamma1, gamma2, gamma3]

    print("\n=== Direct-enumeration proper Gram K_q=0 per channel ===")
    print("    K_q=0(μ,λ) = Σ_{k,n_l,n_r} ρ*(μ) · ρ(λ),  "
          "ρ = Σ_{ab} ψ_l*_a γ̃_{ab} ψ_r,b\n", flush=True)

    for mu_L, gtilde in enumerate(gammas):
        # ρ(k, n_l, n_r, μ) = Σ_{ab} ψ_l*[k, n_l, a, μ] · γ̃[a, b] · ψ_r[k, n_r, b, μ]
        rho = jnp.einsum('klaμ,ab,krbμ->klrμ',
                         jnp.conj(psi_l), gtilde.astype(jnp.complex128), psi_r,
                         optimize=True)
        # K_q=0(μ, λ) = Σ_{k,nl,nr} ρ*(μ) ρ(λ)
        K = jnp.einsum('klrμ,klrλ->μλ', jnp.conj(rho), rho, optimize=True)
        herm_err = float(jnp.max(jnp.abs(K - jnp.conj(K.T))))
        norm = float(jnp.max(jnp.abs(K)))
        # eigvalsh for PSD check
        evs = jnp.linalg.eigvalsh(0.5 * (K + jnp.conj(K.T)))
        ev_min = float(jnp.min(evs))
        ev_max = float(jnp.max(evs))
        # Cholesky with the same trace-ridge as scalar path
        ridge = 1e-14 * jnp.abs(jnp.trace(K)) * jnp.eye(K.shape[0])
        L = jnp.linalg.cholesky(K + ridge)
        chol_finite = bool(jnp.all(jnp.isfinite(L)))
        if chol_finite:
            chol_resid = float(jnp.max(jnp.abs(L @ jnp.conj(L.T) - K)))
        else:
            chol_resid = float('nan')
        marker = " (INDEFINITE!)" if ev_min < -1e-10 * abs(ev_max) else ""
        print(
            f"  μ_L={mu_L}: |K|≤{norm:.3e}  "
            f"eig∈[{ev_min:.3e}, {ev_max:.3e}]{marker}  "
            f"herm_rel={herm_err / max(norm, 1e-300):.3e}  "
            f"chol_finite={chol_finite}  chol_resid={chol_resid:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
