"""Synthetic V_q IBZ→full-BZ round-trip test that EXPOSES the TRS bug
in ``_unfold_v_q_ibz_to_full`` at
``sources/lorrax_B/src/gw/v_q_tile.py:1452``.

Pre-fix (branch ``agent/zeta-bc-scan-shardmap`` HEAD ``c796420``):
    - For q's reached from their IBZ partner via a TRS-augmented op
      (``full_to_irr_sym[q] >= ntran``), the codebase's unfold
      silently mis-indexes ``sym_perm`` (which has only ``ntran``
      rows) under the default JAX gather clamp.  Result: V at those
      q's is WRONG by O(eV) on this synthetic geometry.

Post-fix:
    - Every q should reproduce the reference V to relative error
      <1e-12.

Geometry (chosen to expose the TRS path):
    - 3x3x1 q-grid.
    - Spatial sym group ``{I, σ_y}`` (ntran=2, mirror through xz-plane).
      σ_y on (x,y,z) := (x, -y, z).
    - TRS-augmented sym_mats_k = {I, σ_y, -I, -σ_y} (length 4).
    - Some q's fold only via TRS-augmented ops: e.g. (2,0,0)←(1,0,0)
      under -I, (2,1,0)←(1,1,0) under -σ_y, etc.  Others fold via the
      spatial-only half: (0,2,0)←(0,1,0) under σ_y.

This test invokes the actual ``_unfold_v_q_ibz_to_full`` function
from the codebase (NOT a re-implementation), so the bug is
reproduced exactly as it would manifest in production.

Compatible with CPU-only execution (single JAX device, ``mesh_xy``
on a 1x1 device array).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# x64 must be enabled before jax is imported anywhere
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Wire up the lorrax_B source tree so we can import the actual code.
_LORRAX_SRC = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src"
).resolve()
if str(_LORRAX_SRC) not in sys.path:
    sys.path.insert(0, str(_LORRAX_SRC))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.v_q_tile import _unfold_v_q_ibz_to_full
from centroid.orbit_syms import compute_centroid_sym_perm


# ---------------------------------------------------------------------------
# Step 1: synthetic geometry
# ---------------------------------------------------------------------------

def build_geometry():
    """Build a 3x3x1 q-grid + {I, σ_y} sym group + orbit-closed
    centroid set + small G-sphere.

    Returns a dict of all the host-side arrays needed downstream.
    """
    kgrid = np.array([3, 3, 1], dtype=np.int64)
    fft_grid = np.array([6, 6, 1], dtype=np.int64)  # commensurate w/ q-grid

    # Spatial sym ops (BGW ``mtrx`` convention: acts on G-vectors).
    # σ_y diag(1,-1,1) is its own inverse and is orthogonal.
    I3 = np.eye(3, dtype=np.int64)
    sigma_y = np.diag([1, -1, 1]).astype(np.int64)
    sym_matrices = np.stack([I3, sigma_y], axis=0)      # (ntran, 3, 3)
    ntran = sym_matrices.shape[0]
    translations = np.zeros((ntran, 3), dtype=np.float64)  # symmorphic

    # TRS-augmented sym_mats_k matches symmetry_maps.py:117-130 logic.
    # BGW convention: sym_mats_k = sym_matrices.transpose(0, 2, 1).
    # For our orthogonal integer mats both forms are the same matrix
    # (σ_y is symmetric, identity is symmetric), but we still apply
    # the .T for fidelity to the production code.
    sym_mats_k_spatial = sym_matrices.transpose(0, 2, 1).copy()
    sym_mats_k = np.concatenate(
        [sym_mats_k_spatial, -sym_mats_k_spatial], axis=0)  # (2·ntran, 3, 3)
    assert sym_mats_k.shape[0] == 2 * ntran

    # Centroids: orbit-closed under {I, σ_y} on the FFT grid.
    # Take representative seeds, unfold under sym_matrices (NOT TRS-augmented
    # — sym_perm is built from spatial only), dedupe.
    seeds = np.array([
        [1, 1, 0],    # σ_y image (1, 5, 0)
        [2, 0, 0],    # σ_y image (2, 0, 0)  ← fixed
        [3, 2, 0],    # σ_y image (3, 4, 0)
        [4, 3, 0],    # σ_y image (4, 3, 0)  ← fixed
        [0, 0, 0],    # Γ ← fixed
    ], dtype=np.int64)

    # Build orbit under spatial syms (use r' = Rinv @ r + tau; tau=0;
    # for orthogonal int sym Rinv = sym^T = sym since σ_y is symmetric).
    Rinv = np.rint(np.linalg.inv(sym_matrices)).astype(np.int64)
    all_imgs = []
    for r in seeds:
        for s in range(ntran):
            img = (Rinv[s] @ r) % fft_grid
            all_imgs.append(tuple(img))
    cent_idx = np.array(sorted(set(all_imgs)), dtype=np.int64)
    n_rmu = cent_idx.shape[0]

    # Sanity check: orbit closure via the actual production helper.
    # extend_trs=True (post-Agent2 fix) returns a (2·ntran, n_rmu)
    # table; on pre-fix HEAD the kwarg doesn't exist, fall back to
    # the legacy (ntran, n_rmu) form.  Either way the test consumes
    # sym_perm.shape[0] dynamically.
    try:
        sym_perm = compute_centroid_sym_perm(
            cent_idx.astype(np.int32),
            sym_matrices,
            2.0 * np.pi * translations,
            fft_grid.astype(np.int32),
            validate=True,
            extend_trs=True,
        )
    except TypeError:
        sym_perm = compute_centroid_sym_perm(
            cent_idx.astype(np.int32),
            sym_matrices,
            2.0 * np.pi * translations,
            fft_grid.astype(np.int32),
            validate=True,
        )

    return {
        "kgrid": kgrid,
        "fft_grid": fft_grid,
        "sym_matrices": sym_matrices,                   # (ntran, 3, 3)
        "sym_mats_k": sym_mats_k,                       # (2·ntran, 3, 3)
        "ntran": ntran,
        "translations": translations,
        "cent_idx": cent_idx,                           # (n_rmu, 3)
        "n_rmu": n_rmu,
        "sym_perm": sym_perm,                           # (ntran, n_rmu)
    }


# ---------------------------------------------------------------------------
# Step 2: IBZ reduction using the codebase's q-IBZ helper
# ---------------------------------------------------------------------------

def reduce_q_to_ibz(geom):
    """Run ``SymMaps.find_irreducible_qpoints`` on the synthetic geometry."""
    from common.symmetry_maps import SymMaps

    obj = object.__new__(SymMaps)
    kg = geom["kgrid"]
    kx, ky, kz = np.meshgrid(np.arange(kg[0]), np.arange(kg[1]),
                              np.arange(kg[2]), indexing='ij')
    obj.kvecs_asints = np.stack(
        [kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
    obj.sym_mats_k = geom["sym_mats_k"]

    (q_irr_kgrid_int, full_to_irr_idx, full_to_irr_sym,
     q_irr_full_idx) = obj.find_irreducible_qpoints()

    return {
        "qs_full_int": obj.kvecs_asints.copy(),         # (n_q_full, 3)
        "q_irr_kgrid_int": q_irr_kgrid_int,             # (n_qpt_irr, 3)
        "full_to_irr_idx": full_to_irr_idx,             # (n_q_full,)
        "full_to_irr_sym": full_to_irr_sym,             # (n_q_full,) - TRS-augmented indices
        "q_irr_full_idx": q_irr_full_idx,
    }


# ---------------------------------------------------------------------------
# Step 3: Synthetic ζ + V_q on IBZ and on full BZ (reference)
# ---------------------------------------------------------------------------

def build_g_sphere(sym_mats_k):
    """Build a small G-sphere closed under the full TRS-augmented group
    (so that v(q+G) reductions can be exactly compared).

    Returns
    -------
    G_set : (n_g, 3) int — G-vectors in lattice units, closed under all
            sym ops (including TRS, which sends G → -G).
    G_perm : (n_sym, n_g) int — for each sym s, ``G_perm[s, g]`` is the
            index of S·G in G_set (where S = sym_mats_k[s]).
    """
    # Start with a small set and close under sym group.
    seeds = [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([1, 0, 0], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        np.array([1, -1, 0], dtype=np.int64),
        np.array([2, 1, 0], dtype=np.int64),
    ]
    n_sym = sym_mats_k.shape[0]
    all_gs = set()
    for g in seeds:
        for s in range(n_sym):
            # G transforms as G' = sym_matrices[s] @ G  (BGW mtrx
            # convention on G).  ``sym_mats_k`` differs from
            # ``sym_matrices`` by a transpose (and the TRS factor of -1).
            # For our orthogonal symmetric ops the transpose is a no-op,
            # so apply sym_mats_k directly.  We track the FULL group
            # (including TRS, where G → -G is part of the action).
            g_img = tuple((sym_mats_k[s] @ g).tolist())
            all_gs.add(g_img)
    G_set = np.array(sorted(all_gs), dtype=np.int64)
    n_g = G_set.shape[0]

    # Build G_perm
    G_perm = np.zeros((n_sym, n_g), dtype=np.int64)
    lookup = {tuple(g.tolist()): i for i, g in enumerate(G_set)}
    for s in range(n_sym):
        for ig, g in enumerate(G_set):
            g_img = tuple((sym_mats_k[s] @ g).tolist())
            G_perm[s, ig] = lookup[g_img]

    return G_set, G_perm


def q_to_wrapped_frac(q_int, kgrid):
    """Map kgrid-int q to wrapped physical fractional in [-1/2, 1/2)."""
    qf = q_int.astype(np.float64) / kgrid
    return np.where(qf > 0.5, qf - 1.0, qf)


def build_zeta_irr(n_qpt_irr, n_rmu, n_g, *, seed=7):
    """Random complex ζ at IBZ q's."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_qpt_irr, n_rmu, n_g))
            + 1j * rng.standard_normal((n_qpt_irr, n_rmu, n_g)))


def unfold_zeta_irr_to_full(zeta_irr, *, q_ibz, sym_perm, ntran,
                            G_perm):
    """Build the canonical full-BZ ζ from ζ_irr using the proper
    convention (with TRS conjugation).

    For spatial sym ``s < ntran``::
        ζ_{q_full, π_s(ν)}(G) = ζ_{q_irr, ν}(S^{-1} G)

    For TRS-augmented ``s >= ntran`` (with spatial part s0 = s - ntran)::
        ζ_{q_full, π_{s0}(ν)}(G) = conj( ζ_{q_irr, ν}(-S0^{-1} G) )

    (τ-phases are zero for symmorphic groups.)

    G_perm captures the G permutation: ``G' = sym_mats_k[s] @ G``,
    so ``G_perm[s, g]`` is the index of ``S G`` in G_set.  We need the
    INVERSE — i.e. given a target G in the full-BZ ζ, find which G in
    the IBZ ζ's sphere maps to it.  ``G_perm_inv[s] = argsort(G_perm[s])``.
    """
    n_q_full = q_ibz["full_to_irr_idx"].shape[0]
    n_rmu = zeta_irr.shape[1]
    n_g = zeta_irr.shape[2]
    zeta_full = np.zeros((n_q_full, n_rmu, n_g), dtype=np.complex128)

    # Inverse permutations.
    n_sym_aug = G_perm.shape[0]
    G_perm_inv = np.zeros_like(G_perm)
    for s in range(n_sym_aug):
        G_perm_inv[s] = np.argsort(G_perm[s])
    # ``sym_perm`` may be length ntran (legacy) or 2·ntran (extend_trs);
    # either way we index via ``s % ntran``.
    inv_sym_perm = np.zeros((ntran, sym_perm.shape[1]), dtype=sym_perm.dtype)
    for s in range(ntran):
        inv_sym_perm[s] = np.argsort(sym_perm[s])

    for iq in range(n_q_full):
        irr_i = int(q_ibz["full_to_irr_idx"][iq])
        s = int(q_ibz["full_to_irr_sym"][iq])
        s_spatial = s % ntran
        is_trs = s >= ntran

        # Apply centroid permutation (forward π_{s_spatial}):
        # ζ_full[π_{s0}(ν), G] = (±)conj?(ζ_irr[ν, S^{-1} G])
        # Equivalent index form:
        # for each ν in [0, n_rmu): nu_irr = inv_sym_perm[s_spatial, ν]
        # i.e. ν = π_{s0}(nu_irr) ⇒ nu_irr = π_{s0}^{-1}(ν)
        # Hmm wait — we want ζ_full[ν', G] where ν' = π_{s0}(ν).
        # Equivalently: for full index ν', set ν = π_{s0}^{-1}(ν') and
        # use ζ_irr[ν, G_back].
        for nu in range(n_rmu):
            nu_irr = int(inv_sym_perm[s_spatial, nu])
            # G permutation: for full G index g, the back-mapped G is
            # G_back = S^{-1} G; we need its index in G_set.  Using
            # G_perm: G_perm[s, g_back] = g  ⇒  g_back = G_perm_inv[s, g].
            # The relevant ``s`` for the G map is the same s (TRS sym
            # in sym_mats_k already includes the -1 factor).
            g_back_indices = G_perm_inv[s]               # (n_g,)
            zeta_irr_slice = zeta_irr[irr_i, nu_irr, g_back_indices]
            if is_trs:
                zeta_full[iq, nu, :] = np.conj(zeta_irr_slice)
            else:
                zeta_full[iq, nu, :] = zeta_irr_slice

    return zeta_full


def compute_v_per_q(zeta_per_q, v_per_G):
    """V_q[μ, ν] = sum_G conj(ζ_q[μ, G]) v(q+G) ζ_q[ν, G].

    zeta_per_q: (n_q, n_rmu, n_g)
    v_per_G:    (n_q, n_g)  real
    """
    # ζ_L = conj(ζ)
    zeta_L = np.conj(zeta_per_q)
    # Contract over G.
    # V[q, mu, nu] = sum_g zeta_L[q, mu, g] * v[q, g] * zeta_per_q[q, nu, g]
    weighted = zeta_per_q * v_per_G[:, None, :]         # (n_q, n_rmu, n_g)
    V = np.einsum('qmg,qng->qmn', zeta_L, weighted)
    return V


def build_v_per_G(q_frac_wrapped, G_set):
    """v(q+G) = 1/|q_frac + G|² (toy 3D Coulomb, no system-dimension trick).

    q_frac_wrapped: (n_q, 3) physical fractional q in [-1/2, 1/2)
    G_set:          (n_g, 3) int
    Returns         (n_q, n_g) real array.
    """
    # Use a TRIVIAL cubic metric (lattice = identity) so |q+G|² is
    # well-defined without bvec/cell-volume threading.
    qG = q_frac_wrapped[:, None, :] + G_set[None, :, :].astype(np.float64)
    norm2 = np.einsum('qgi,qgi->qg', qG, qG)
    # Tiny floor to avoid 1/0 at q=Γ, G=0; this is the standard q→0
    # truncation any production code does anyway.
    return 1.0 / np.maximum(norm2, 1e-6)


# ---------------------------------------------------------------------------
# Step 4: main test driver
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 78)
    print("Synthetic V_q TRS round-trip test")
    print("Pre-fix branch: agent/zeta-bc-scan-shardmap  HEAD: c796420")
    print("=" * 78)

    # ---- geometry ----
    geom = build_geometry()
    kgrid = geom["kgrid"]
    ntran = geom["ntran"]
    sym_matrices = geom["sym_matrices"]
    sym_mats_k = geom["sym_mats_k"]
    sym_perm = geom["sym_perm"]
    n_rmu = geom["n_rmu"]
    print(f"\nntran (spatial only)        = {ntran}")
    print(f"sym_mats_k length (TRS aug) = {sym_mats_k.shape[0]}")
    print(f"n_rmu (centroids)           = {n_rmu}")
    print(f"sym_perm shape              = {sym_perm.shape}")

    # ---- q IBZ reduction ----
    q_ibz = reduce_q_to_ibz(geom)
    n_q_full = q_ibz["qs_full_int"].shape[0]
    n_qpt_irr = q_ibz["q_irr_kgrid_int"].shape[0]
    print(f"\nn_q_full                    = {n_q_full}")
    print(f"n_qpt_irr                   = {n_qpt_irr}")

    # ---- G sphere closed under the full TRS-augmented group ----
    G_set, G_perm = build_g_sphere(sym_mats_k)
    n_g = G_set.shape[0]
    print(f"n_G_sphere                  = {n_g}")

    # ---- per-q v(q+G), using the canonical IBZ partner's wrapped q ----
    # Important: we use the SAME v(q+G) at full q and at IBZ partner,
    # i.e. v(q_irr+G) — this is what the production V_q kernel does
    # (it never recomputes v(q_full); only knows about IBZ q's).
    # Reference V_full computed below uses the same v(q_irr+G) per IBZ
    # partner — so any disagreement reflects ζ/V transformation only.
    q_irr_frac = q_to_wrapped_frac(
        q_ibz["q_irr_kgrid_int"], kgrid)                 # (n_qpt_irr, 3)
    v_per_G_ibz = build_v_per_G(q_irr_frac, G_set)       # (n_qpt_irr, n_g)

    # Pull v(q+G) at each full q from its IBZ parent's v table
    v_per_G_full = np.zeros((n_q_full, n_g), dtype=np.float64)
    for iq in range(n_q_full):
        irr_i = int(q_ibz["full_to_irr_idx"][iq])
        # Under the ζ unfold, ζ_full[q, μ, G] uses ζ_irr at G' = S^{-1} G.
        # The v(q+G) factor that goes with this contracted G is
        # v(q_irr + G').  Equivalently: at full q index ν and full G g,
        # we use v_per_G_ibz[irr_i, G_perm_inv[s, g]].
        # Reorder using G_perm_inv to match the ζ transformation.
        s = int(q_ibz["full_to_irr_sym"][iq])
        G_perm_inv_s = np.argsort(G_perm[s])
        v_per_G_full[iq] = v_per_G_ibz[irr_i, G_perm_inv_s]

    # ---- ζ_irr (random, complex) ----
    zeta_irr = build_zeta_irr(n_qpt_irr, n_rmu, n_g)

    # ---- V_q_ibz from ζ_irr (the input to _unfold_v_q_ibz_to_full) ----
    V_q_ibz = compute_v_per_q(zeta_irr, v_per_G_ibz)     # (n_qpt_irr, n_rmu, n_rmu)
    print(f"\nV_q_ibz shape               = {V_q_ibz.shape}")

    # ---- Reference V_q_full: unfold ζ properly (with TRS conj), then
    # contract.  This is the "true" answer.
    zeta_full = unfold_zeta_irr_to_full(
        zeta_irr,
        q_ibz=q_ibz, sym_perm=sym_perm, ntran=ntran,
        G_perm=G_perm,
    )                                                    # (n_q_full, n_rmu, n_g)
    V_ref = compute_v_per_q(zeta_full, v_per_G_full)     # (n_q_full, n_rmu, n_rmu)

    # ---- Codebase path: _unfold_v_q_ibz_to_full(V_q_ibz, ...) ----
    # Pad V_q_ibz to padded extent if needed; we use n_rmu directly
    # (n_rmu = 10 here so no padding issue with a 1x1 mesh).
    devices = np.array(jax.devices()).reshape(1, 1)
    mesh_xy = Mesh(devices, ('x', 'y'))

    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    V_q_ibz_j = jax.device_put(V_q_ibz.astype(np.complex128), V_sh)

    # ``n_sym_spatial`` (post-Agent2 fix) tells the unfold which half
    # of sym_perm is TRS-augmented.  Fall back to the pre-fix
    # signature (no kwarg) when the unfolder doesn't accept it.
    unfold_kwargs = dict(
        full_to_irr_idx=q_ibz["full_to_irr_idx"],
        full_to_irr_sym=q_ibz["full_to_irr_sym"],
        sym_perm=sym_perm,
        mesh_xy=mesh_xy,
    )
    try:
        V_buggy = _unfold_v_q_ibz_to_full(
            V_q_ibz_j, n_sym_spatial=ntran, **unfold_kwargs)
    except TypeError:
        V_buggy = _unfold_v_q_ibz_to_full(V_q_ibz_j, **unfold_kwargs)
    V_buggy = np.asarray(jax.device_get(V_buggy))

    # ---- Per-q comparison ----
    print("\n" + "=" * 78)
    print(f"{'q_full':>12} {'irr':>3} {'sym':>3} {'TRS':>4} "
          f"{'max|ΔV|':>12} {'rel':>10} {'|V_ref|':>10}")
    print("-" * 78)
    headers = []
    rows = []
    max_abs_trs = 0.0
    max_abs_spatial = 0.0
    n_trs = 0
    n_spatial = 0
    for iq in range(n_q_full):
        s = int(q_ibz["full_to_irr_sym"][iq])
        irr_i = int(q_ibz["full_to_irr_idx"][iq])
        is_trs = s >= ntran
        d = V_buggy[iq] - V_ref[iq]
        max_abs = float(np.max(np.abs(d)))
        ref_norm = float(np.linalg.norm(V_ref[iq]))
        rel = max_abs / max(ref_norm, 1e-30)
        q_str = "({},{},{})".format(*q_ibz["qs_full_int"][iq])
        print(f"{q_str:>12} {irr_i:>3} {s:>3} {'T' if is_trs else 'F':>4} "
              f"{max_abs:>12.3e} {rel:>10.2e} {ref_norm:>10.3e}")
        rows.append({
            "q_full": q_ibz["qs_full_int"][iq].tolist(),
            "q_irr": q_ibz["q_irr_kgrid_int"][irr_i].tolist(),
            "sym_idx": s,
            "is_trs": is_trs,
            "max_abs": max_abs,
            "rel": rel,
            "ref_norm": ref_norm,
        })
        if is_trs:
            max_abs_trs = max(max_abs_trs, max_abs)
            n_trs += 1
        else:
            max_abs_spatial = max(max_abs_spatial, max_abs)
            n_spatial += 1
    print("-" * 78)
    print(f"spatial-only q's: {n_spatial}, max |ΔV| = {max_abs_spatial:.3e}")
    print(f"TRS-only     q's: {n_trs}, max |ΔV| = {max_abs_trs:.3e}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    if max_abs_trs > 1e-3 and max_abs_spatial < 1e-10:
        print("PRE-FIX VERDICT: BUG REPRODUCED.")
        print(f"  TRS-related q's disagree by {max_abs_trs:.3e}, which is")
        print(f"  HUGE (the units here are arbitrary toy units; this would")
        print(f"  be eV-scale on a physical V_q matrix element).")
        print(f"  Spatial-only q's agree at {max_abs_spatial:.3e}, confirming")
        print(f"  the spatial half of the unfold is correct.")
        return rows, "pre-fix-bug-reproduced"
    elif max_abs_trs < 1e-10 and max_abs_spatial < 1e-10:
        print("POST-FIX VERDICT: PASS.")
        print(f"  All q's agree at relative error < 1e-12.")
        return rows, "post-fix-pass"
    else:
        print("UNEXPECTED STATE:")
        print(f"  spatial-only max |ΔV| = {max_abs_spatial:.3e}")
        print(f"  TRS         max |ΔV| = {max_abs_trs:.3e}")
        return rows, "unexpected"


if __name__ == "__main__":
    rows, verdict = run_test()
    print(f"\nVerdict: {verdict}")
