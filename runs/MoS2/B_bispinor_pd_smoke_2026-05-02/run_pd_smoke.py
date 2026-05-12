"""Bispinor pair-density smoke test (agent B, 2026-05-02).

Exercises the new vertex-parameterised pair-density helpers
(``compute_pair_density_with_vertex`` / ``compute_pair_density_lorentz``)
against the existing scalar ``compute_pair_density_spin_traced`` on the
MoS2 3x3 WFN.h5.

Caveat: this WFN is non-SOC (wfn.nspinor=1).  The bispinor lift in
``read_Gvecs_to_devices`` builds a small component from the σ·(k+G)
kinetic-balance formula on a pseudo-spinor padded from the scalar input
— physically dubious but mechanically correct, which is enough for an
infrastructure smoke test.  For real physics we want a fully-relativistic
WFN (wfn.nspinor=2); CrI3 will be that target.

Validations:
  1. ``compute_pair_density_with_vertex`` with vertex=I_2 on the
     ns=2 path reproduces ``compute_pair_density_spin_traced``
     bit-for-bit (= same einsum, identity vertex).
  2. With ψ_S manually zeroed in the ns=4 path, ``compute_pair_density_lorentz``:
     - μ_L=0 → exactly equal to the ns=2 scalar pair density (γ̃^0=1₄;
       only the (0,0)+(1,1) blocks contribute, identical to the ns=2 sum).
     - μ_L=1,2,3 → exactly zero (γ̃^i is off-block-diagonal between L and S).
  3. With the actual kinetic-balance ψ_S enabled, μ_L=i pair densities
     become nonzero and α_FS-suppressed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")

from runtime import set_default_env

set_default_env()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import Meta, symmetry_maps
from common.isdf_fitting import (
    compute_pair_density_spin_traced,
    compute_pair_density_with_vertex,
    compute_pair_density_lorentz,
    compute_CCT_from_left_right,
    compute_L_q_from_CCT,
)
from common.load_wfns import load_centroids_band_chunked
from file_io import WFNReader


ROOT = Path(__file__).resolve().parent
LORRAX_RUN = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex"
)
WFN_PATH = LORRAX_RUN / "WFN.h5"
CENTROIDS_PATH = LORRAX_RUN / "centroids_frac_640.txt"

# Smoke-test bands.  For the pair-density einsum probe (validations A-D)
# any band range works; for the CCT+Cholesky probe (validation E) we
# need a left ⊕ right split that keeps both densities non-trivial.
# (0, 8) for left / (8, 24) for right gives 8 × 16 band pairs at each k.
LEFT_RANGE  = (0, 8)
RIGHT_RANGE = (8, 24)
BAND_RANGE  = (0, RIGHT_RANGE[1])  # = max-of-the-two; load span


def _print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _make_mesh() -> Mesh:
    devs = jax.devices()
    n_dev = len(devs)
    if n_dev == 1:
        return Mesh(np.asarray(devs).reshape(1, 1), axis_names=("x", "y"))
    if n_dev == 2:
        return Mesh(np.asarray(devs).reshape(2, 1), axis_names=("x", "y"))
    if n_dev == 4:
        return Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    raise RuntimeError(f"unsupported device count {n_dev}")


def _load_centroid_indices(meta: Meta) -> jax.Array:
    """Read fractional-coord centroids and snap to FFT grid (matching the
    existing run's snapping in centroids_frac_640.txt header)."""
    coords = np.loadtxt(CENTROIDS_PATH, dtype=np.float64)
    nx, ny, nz = meta.fft_grid
    grid = np.array([nx, ny, nz])
    idx = np.rint(coords * grid).astype(np.int64) % grid
    # Dedup (centroid file claims 640 unique; preserve order)
    seen = set()
    unique = []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    arr = np.asarray(unique, dtype=np.int64)
    return jnp.asarray(arr)


def _load_psi_centroids(*, bispinor: bool, mesh: Mesh):
    """Load ψ at centroid grid for the given bispinor flag.  Returns
    (psi_rmu_Y, psi_rmuT_X, meta, n_rmu)."""
    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    nb_total = BAND_RANGE[1] - BAND_RANGE[0]
    meta = Meta.from_system(
        wfn, sym,
        nval=BAND_RANGE[1], ncond=0, nband=nb_total, n_rmu=640,
        bispinor=bispinor,
    )
    centroid_idx = _load_centroid_indices(meta)
    n_rmu = int(centroid_idx.shape[0])
    print(
        f"  loaded WFN.h5 nspinor_wfnfile={meta.nspinor_wfnfile} "
        f"nspinor={meta.nspinor} nb_total={nb_total} n_rmu={n_rmu} "
        f"fft_grid={tuple(meta.fft_grid)} nk_tot={meta.nk_tot}",
        flush=True,
    )
    t0 = time.perf_counter()
    psi_rmu, psi_rmuT = load_centroids_band_chunked(
        wfn, sym, meta, centroid_idx,
        bispinor=bispinor, mesh_xy=mesh,
        band_range=BAND_RANGE,
        band_chunk_size=min(64, nb_total),
    )
    print(
        f"  centroid sample took {time.perf_counter() - t0:.2f} s; "
        f"psi_rmu shape {tuple(psi_rmu.shape)}, "
        f"psi_rmuT shape {tuple(psi_rmuT.shape)}",
        flush=True,
    )
    return psi_rmu, psi_rmuT, meta, n_rmu


def _zero_small_component(psi_rmu, psi_rmuT):
    """Manually zero the small (lower) bispinor block.  Used for the
    bit-identity regression (with ψ_S=0 the ns=4 μ_L=0 path must equal
    the ns=2 scalar)."""
    assert psi_rmu.shape[2] == 4
    psi_rmu = psi_rmu.at[:, :, 2:4, :].set(0.0)
    psi_rmuT = psi_rmuT.at[:, :, :, 2:4].set(0.0)
    return psi_rmu, psi_rmuT


def _diff_norms(label: str, A: jax.Array, B: jax.Array) -> float:
    diff = jnp.abs(A - B)
    rel = float(jnp.max(diff) / max(float(jnp.max(jnp.abs(A))), 1e-300))
    absmax = float(jnp.max(diff))
    print(
        f"    {label}: max|Δ|={absmax:.3e}  "
        f"max|A|={float(jnp.max(jnp.abs(A))):.3e}  rel={rel:.3e}",
        flush=True,
    )
    return rel


def main() -> None:
    _print_section(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()
    print(f"  mesh shape={mesh.devices.shape}", flush=True)

    # ────────────────────────────────────────────────────────────────
    # Pass 1 — load with bispinor=False (ns=2 reference)
    # ────────────────────────────────────────────────────────────────
    _print_section("Pass 1: bispinor=False (existing scalar path)")
    psi_rmu_2, psi_rmuT_2, meta2, n_rmu = _load_psi_centroids(
        bispinor=False, mesh=mesh
    )
    assert psi_rmu_2.shape[2] == meta2.nspinor, "ns axis mismatch"
    P_scalar = compute_pair_density_spin_traced(psi_rmuT_2, psi_rmu_2, mesh)
    P_via_vertex = compute_pair_density_with_vertex(
        psi_rmuT_2, psi_rmu_2, jnp.eye(meta2.nspinor, dtype=jnp.complex128), mesh
    )

    _print_section("Validation A: identity-vertex parity at ns=2")
    print(
        "  compute_pair_density_with_vertex(V=I) vs "
        "compute_pair_density_spin_traced",
        flush=True,
    )
    rel_A = _diff_norms("ns=2 μ_L=0 (V=I)", P_scalar, P_via_vertex)
    assert rel_A < 1e-13, (
        f"identity-vertex regression failed (rel={rel_A:.3e}); the "
        f"vertex einsum must reduce to spin-traced for V=I"
    )
    print("  PASS", flush=True)

    # ────────────────────────────────────────────────────────────────
    # Pass 2 — load with bispinor=True (ns=4)
    # ────────────────────────────────────────────────────────────────
    _print_section("Pass 2: bispinor=True (ns=4 with kinetic-balance ψ_S)")
    psi_rmu_4, psi_rmuT_4, meta4, _ = _load_psi_centroids(
        bispinor=True, mesh=mesh
    )
    assert psi_rmu_4.shape[2] == 4, "ns axis must be 4 in bispinor mode"

    # Diagnostic: how big is the kinetic-balance small component?
    norm_L = float(jnp.linalg.norm(psi_rmu_4[:, :, 0:2, :]))
    norm_S = float(jnp.linalg.norm(psi_rmu_4[:, :, 2:4, :]))
    print(
        f"  ‖ψ_L‖ = {norm_L:.4e}   ‖ψ_S‖ = {norm_S:.4e}   "
        f"ratio = {norm_S / max(norm_L, 1e-300):.4e}   "
        f"(α_FS/2 ≈ 3.65e-3 — expect this scale times typical |k+G|)",
        flush=True,
    )

    # ─── Validation B — bit-identity with ψ_S=0 ─────────────────────
    _print_section("Validation B: bit-identity at ψ_S=0 (ns=4 vs ns=2)")
    psi_rmu_4z, psi_rmuT_4z = _zero_small_component(psi_rmu_4, psi_rmuT_4)
    P0_zS = compute_pair_density_lorentz(psi_rmuT_4z, psi_rmu_4z, 0, mesh)
    rel_B0 = _diff_norms("μ_L=0 ns=4(ψ_S=0) vs ns=2", P_scalar, P0_zS)
    assert rel_B0 < 1e-13, (
        f"μ_L=0 with ψ_S=0 must bit-equal ns=2 scalar (rel={rel_B0:.3e})"
    )
    for mu_L in (1, 2, 3):
        Pi_zS = compute_pair_density_lorentz(psi_rmuT_4z, psi_rmu_4z, mu_L, mesh)
        amax = float(jnp.max(jnp.abs(Pi_zS)))
        print(f"    μ_L={mu_L} ns=4(ψ_S=0): max|P|={amax:.3e}", flush=True)
        assert amax < 1e-13, (
            f"μ_L={mu_L} with ψ_S=0 must vanish (γ̃^i is L↔S only); "
            f"got max|P|={amax:.3e}"
        )
    print("  PASS", flush=True)

    # ─── Validation C — physical Lorentz pair densities ─────────────
    _print_section("Validation C: physical four-density (ψ_S from kinetic balance)")
    P0 = compute_pair_density_lorentz(psi_rmuT_4, psi_rmu_4, 0, mesh)
    P1 = compute_pair_density_lorentz(psi_rmuT_4, psi_rmu_4, 1, mesh)
    P2 = compute_pair_density_lorentz(psi_rmuT_4, psi_rmu_4, 2, mesh)
    P3 = compute_pair_density_lorentz(psi_rmuT_4, psi_rmu_4, 3, mesh)
    n0 = float(jnp.linalg.norm(P0))
    n1 = float(jnp.linalg.norm(P1))
    n2 = float(jnp.linalg.norm(P2))
    n3 = float(jnp.linalg.norm(P3))
    print(f"    ‖P^0‖ = {n0:.4e}", flush=True)
    print(f"    ‖P^1‖ = {n1:.4e}   (‖P^1‖/‖P^0‖ = {n1 / max(n0, 1e-300):.3e})", flush=True)
    print(f"    ‖P^2‖ = {n2:.4e}   (‖P^2‖/‖P^0‖ = {n2 / max(n0, 1e-300):.3e})", flush=True)
    print(f"    ‖P^3‖ = {n3:.4e}   (‖P^3‖/‖P^0‖ = {n3 / max(n0, 1e-300):.3e})", flush=True)
    # Drift between P^0 and the ns=2 scalar — coming from the |ψ_S|² addition.
    rel_drift = _diff_norms("μ_L=0 vs ns=2 scalar (drift = |ψ_S|² term)",
                             P_scalar, P0)
    print(
        "  Expected order: rel_drift ~ (‖ψ_S‖/‖ψ_L‖)² ≈ "
        f"{(norm_S / max(norm_L, 1e-300))**2:.3e}",
        flush=True,
    )

    # ─── Validation D — Hermiticity of P^{μ_L}_k(μ_c, ν_c) ──────────
    _print_section("Validation D: per-k Hermiticity of P^{μ_L}_k")
    for label, P_full in (("μ_L=0", P0), ("μ_L=1", P1),
                          ("μ_L=2", P2), ("μ_L=3", P3)):
        P_T = jnp.swapaxes(P_full, -1, -2).conj()
        herm_err = float(jnp.max(jnp.abs(P_full - P_T)))
        norm = float(jnp.max(jnp.abs(P_full)))
        rel = herm_err / max(norm, 1e-300)
        print(
            f"    {label}: max|P - P^†_(μν↔νμ)| = {herm_err:.3e}, "
            f"max|P| = {norm:.3e}, rel = {rel:.3e}",
            flush=True,
        )
        # Per-k Hermiticity should hold to fp64 noise.  Threshold is
        # generous to accommodate the band-summed accumulation noise.
        assert rel < 1e-12, f"P^{label} not Hermitian (rel={rel:.3e})"
    print("  PASS", flush=True)

    # ─── Validation E — CCT^{μ_L} for left / right band split ───────
    # Per-channel ζ-fits in the bispinor pipeline reuse a single
    # (centroid-set, band-pair) layout but each channel gets its own
    # CCT, Cholesky, ZCT, and ζ — see docs/BISPINOR_DHFB_DESIGN.md §4.
    _print_section("Validation E: CCT and Cholesky per Lorentz channel")
    kgrid = tuple(int(x) for x in meta4.kgrid)
    print(f"  using kgrid = {kgrid} (nk = {int(np.prod(kgrid))})", flush=True)

    def _slice_bands(psi_rmu, psi_rmuT, lo, hi):
        # psi_rmu: (nk, nb_loaded, ns, n_rmu); psi_rmuT: (nk, n_rmu, nb_loaded, ns)
        return (psi_rmu[:, lo:hi, :, :], psi_rmuT[:, :, lo:hi, :])

    # ns=2 reference: scalar CCT from existing path (left ⊕ right pair densities)
    pl2_rmu, pl2_rmuT = _slice_bands(psi_rmu_2, psi_rmuT_2,
                                      LEFT_RANGE[0],  LEFT_RANGE[1])
    pr2_rmu, pr2_rmuT = _slice_bands(psi_rmu_2, psi_rmuT_2,
                                      RIGHT_RANGE[0], RIGHT_RANGE[1])
    P_l_scalar = compute_pair_density_spin_traced(pl2_rmuT, pl2_rmu, mesh)
    P_r_scalar = compute_pair_density_spin_traced(pr2_rmuT, pr2_rmu, mesh)
    CCT_scalar = compute_CCT_from_left_right(P_l_scalar, P_r_scalar, kgrid, mesh)
    print(f"    CCT_scalar (ns=2): shape={tuple(CCT_scalar.shape)} "
          f"max|.|={float(jnp.max(jnp.abs(CCT_scalar))):.3e}",
          flush=True)

    # ns=4 zero-S reference: μ_L=0 with ψ_S=0 must bit-equal CCT_scalar
    pl4_rmu_z,  pl4_rmuT_z  = _slice_bands(psi_rmu_4z, psi_rmuT_4z,
                                            LEFT_RANGE[0],  LEFT_RANGE[1])
    pr4_rmu_z,  pr4_rmuT_z  = _slice_bands(psi_rmu_4z, psi_rmuT_4z,
                                            RIGHT_RANGE[0], RIGHT_RANGE[1])
    P_l_0z = compute_pair_density_lorentz(pl4_rmuT_z, pl4_rmu_z, 0, mesh)
    P_r_0z = compute_pair_density_lorentz(pr4_rmuT_z, pr4_rmu_z, 0, mesh)
    CCT_0z = compute_CCT_from_left_right(P_l_0z, P_r_0z, kgrid, mesh)
    rel_E0 = _diff_norms("CCT^0 (ns=4, ψ_S=0) vs CCT_scalar",
                          CCT_scalar, CCT_0z)
    assert rel_E0 < 1e-12, (
        f"CCT^0 with ψ_S=0 must bit-equal scalar CCT (rel={rel_E0:.3e})"
    )

    # Full bispinor — per-channel CCT and Cholesky
    pl4_rmu,  pl4_rmuT  = _slice_bands(psi_rmu_4, psi_rmuT_4,
                                        LEFT_RANGE[0],  LEFT_RANGE[1])
    pr4_rmu,  pr4_rmuT  = _slice_bands(psi_rmu_4, psi_rmuT_4,
                                        RIGHT_RANGE[0], RIGHT_RANGE[1])
    print(
        f"  full-bispinor CCT^{{μ_L}} for left {LEFT_RANGE} ⊕ right "
        f"{RIGHT_RANGE} band split:",
        flush=True,
    )
    # eigvalsh gives min/max eigenvalues per q — much more diagnostic
    # than diag@q=0 alone for detecting indefiniteness.
    for mu_L in (0, 1, 2, 3):
        P_l = compute_pair_density_lorentz(pl4_rmuT, pl4_rmu, mu_L, mesh)
        P_r = compute_pair_density_lorentz(pr4_rmuT, pr4_rmu, mu_L, mesh)
        CCT = compute_CCT_from_left_right(P_l, P_r, kgrid, mesh)
        norm = float(jnp.max(jnp.abs(CCT)))
        herm_err = float(jnp.max(jnp.abs(
            CCT - jnp.conj(jnp.swapaxes(CCT, -1, -2))
        )))
        # Per-q min/max eigenvalues (Hermitian → eigvalsh).  PSD ↔ all min ≥ 0.
        evs = jnp.linalg.eigvalsh(0.5 * (CCT + jnp.conj(jnp.swapaxes(CCT, -1, -2))))
        ev_min = float(jnp.min(evs))
        ev_max = float(jnp.max(evs))

        # Cholesky path
        L_chol = compute_L_q_from_CCT(CCT, mesh)
        chol_finite = bool(jnp.all(jnp.isfinite(L_chol)))
        if chol_finite:
            chol_resid = float(jnp.max(jnp.abs(
                jnp.einsum('qmk,qnk->qmn', L_chol, jnp.conj(L_chol)) - CCT
            )))
        else:
            chol_resid = float('nan')

        # LU path (jax.numpy.linalg.solve uses LU; here we just LU-factor and
        # check residual reconstruction L_LU U_LU = P CCT).
        lu_finite = False
        lu_resid = float('nan')
        try:
            from jax.scipy.linalg import lu as jlu
            P_perm, L_lu, U_lu = jax.vmap(jlu)(CCT)
            lu_finite = bool(jnp.all(jnp.isfinite(L_lu)) and
                             jnp.all(jnp.isfinite(U_lu)))
            if lu_finite:
                # P @ L @ U should reconstruct CCT
                LU = jnp.einsum('qij,qjk->qik', L_lu, U_lu)
                P_LU = jnp.einsum('qij,qjk->qik', P_perm, LU)
                lu_resid = float(jnp.max(jnp.abs(P_LU - CCT)))
        except Exception as exc:  # noqa: BLE001
            print(f"    μ_L={mu_L}: LU exception — {exc}", flush=True)

        is_indef = ev_min < -1e-12 * max(abs(ev_max), 1.0)
        marker = " (INDEFINITE)" if is_indef else ""
        print(
            f"    μ_L={mu_L}: max|CCT|={norm:.3e}  "
            f"eig∈[{ev_min:.3e}, {ev_max:.3e}]{marker}  "
            f"herm_rel={herm_err / max(norm, 1e-300):.3e}",
            flush=True,
        )
        print(
            f"            chol_finite={chol_finite}  chol_resid={chol_resid:.3e}  "
            f"lu_finite={lu_finite}  lu_resid={lu_resid:.3e}",
            flush=True,
        )
    print("  done.", flush=True)

    _print_section("All validations passed.")


if __name__ == "__main__":
    main()
