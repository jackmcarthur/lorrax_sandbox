#!/usr/bin/env python3
"""Diagnose SymMaps wavefunction rotation by comparing rotated IBZ wavefunctions
against directly-computed nosym wavefunctions.

Strategy:
  1. Load sym WFN.h5 (8 IBZ k-points, ntran=12 symmorphic symmetries)
  2. Load nosym WFN.h5 (64 direct k-points, ntran=1)
  3. For each full BZ k-point that needs a nontrivial symmetry rotation:
     a. Rotate the IBZ wavefunction using SymMaps (G-vector rotation + spinor)
     b. Compare against the nosym wavefunction at that k-point
     c. For degenerate spinor pairs: check that the 2D subspace matches

The inner product between two spinor wavefunctions is:
    <ψ_a | ψ_b> = Σ_G Σ_s  conj(c_a(s,G)) * c_b(s,G)
where s ∈ {0,1} (spinor) and G runs over the G-sphere.

For a degenerate pair (n, n+1), the overlap matrix is:
    O_ij = <nosym_i | rotated_j>,   i,j ∈ {n, n+1}
The pair spans the same 2D subspace iff Σ_ij |O_ij|² ≈ 2.0
(equivalently: singular values of O are both ≈ 1).
"""

import sys
import numpy as np
import h5py
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal WFN.h5 reader (standalone, no LORRAX imports)
# ---------------------------------------------------------------------------

class WFNData:
    """Lightweight reader for BerkeleyGW WFN.h5 files."""

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            # K-points — shape is (nk, 3) already in this HDF5 layout
            self.kpoints = f["mf_header/kpoints/rk"][:]  # (nk, 3)
            self.nkpts = self.kpoints.shape[0]

            # Number of G-vectors per k-point
            self.ngk = f["mf_header/kpoints/ngk"][:]  # (nk,)
            self.ngktot = int(np.sum(self.ngk))

            # Band energies — shape (nspin, nk, nband)
            el_raw = f["mf_header/kpoints/el"][:]
            if el_raw.ndim == 3:
                self.energies = el_raw[0]  # → (nk, nband)
            else:
                self.energies = el_raw

            # Symmetry — mtrx is (ntran_stored, 3, 3)
            self.ntran = int(f["mf_header/symmetry/ntran"][()])
            self.sym_matrices = f["mf_header/symmetry/mtrx"][:]  # (ntran_stored, 3, 3)
            self.frac_trans = f["mf_header/symmetry/tnp"][:]     # (ntran_stored, 3)

            # Crystal
            self.bvec = f["mf_header/crystal/bvec"][:]  # (3, 3)
            self.kgrid = f["mf_header/kpoints/kgrid"][:]  # (3,)
            self.shift = f["mf_header/kpoints/shift"][:]  # (3,)
            self.nband = int(f["mf_header/kpoints/mnband"][()])

            # G-vectors — (ngktot, 3) already
            self.gvecs = f["wfns/gvecs"][:]  # (ngktot, 3)

            # Wavefunction coefficients — (nband, nspinor, ngktot, 2=re/im)
            self.coeffs = f["wfns/coeffs"][:]  # (nband, 2, ngktot, 2)

        # Cumulative G-vector starts per k-point
        self.kpt_starts = np.zeros(self.nkpts, dtype=np.int64)
        self.kpt_starts[1:] = np.cumsum(self.ngk[:-1])

    def get_gvec_nk(self, ik):
        """G-vectors for k-point ik. Returns (ngk, 3) int array."""
        s = self.kpt_starts[ik]
        return self.gvecs[s : s + self.ngk[ik]]

    def get_cnk(self, ik, ib):
        """Spinor wavefunction coefficients for band ib at k-point ik.
        Returns complex array of shape (2, ngk)."""
        s = self.kpt_starts[ik]
        e = s + self.ngk[ik]
        raw = self.coeffs[ib, :, s:e, :]  # (2, ngk, 2)
        return raw[:, :, 0] + 1j * raw[:, :, 1]  # (2, ngk)


# ---------------------------------------------------------------------------
# Symmetry rotation helpers (standalone reimplementation of SymMaps logic)
# ---------------------------------------------------------------------------

def build_sym_mats_k(sym_matrices, ntran):
    """Build k-space symmetry matrices (spatial + time-reversal).
    sym_matrices: (ntran, 3, 3) real-space matrices.
    Returns (2*ntran, 3, 3) integer matrices for k-space."""
    # k-space: transpose of real-space
    spatial = sym_matrices[:ntran].transpose(0, 2, 1).copy()
    # Time-reversal: negate
    time_rev = -spatial
    return np.concatenate([spatial, time_rev], axis=0)


def build_full_bz_kpoints(kgrid, shift):
    """Generate the full BZ k-point grid. Returns (nk, 3) array."""
    kx = np.arange(kgrid[0]) / kgrid[0] + shift[0] / (2 * kgrid[0])
    ky = np.arange(kgrid[1]) / kgrid[1] + shift[1] / (2 * kgrid[1])
    kz = np.arange(kgrid[2]) / kgrid[2] + shift[2] / (2 * kgrid[2])
    gx, gy, gz = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def find_ibz_mapping(full_kpts, ibz_kpts, sym_mats_k, tol=1e-6):
    """For each full BZ k-point, find the IBZ k-point and sym op that maps to it.

    Returns:
        irk_map: (nk_full,) — index into ibz_kpts
        sym_map: (nk_full,) — index into sym_mats_k
    """
    nk_full = len(full_kpts)
    irk_map = np.full(nk_full, -1, dtype=np.int32)
    sym_map = np.full(nk_full, -1, dtype=np.int32)

    # Pre-compute S @ k_irk for all (irk, sym) pairs
    # Skbar[irk, isym, :] = sym_mats_k[isym] @ ibz_kpts[irk]
    Skbar = np.einsum("sij,kj->ksi", sym_mats_k, ibz_kpts)  # (nirk, nsym, 3)
    Skbar_mod = Skbar % 1.0
    Skbar_mod[Skbar_mod > 1 - tol] = 0.0

    for ikf, kf in enumerate(full_kpts):
        kf_mod = kf % 1.0
        kf_mod[kf_mod > 1 - tol] = 0.0
        diff = np.abs(Skbar_mod - kf_mod[None, None, :])
        diff = np.minimum(diff, 1 - diff)  # periodic
        match = np.all(diff < tol, axis=2)  # (nirk, nsym)
        hits = np.argwhere(match)
        if len(hits) > 0:
            irk_map[ikf] = hits[0, 0]
            sym_map[ikf] = hits[0, 1]

    return irk_map, sym_map


def compute_spinor_rotation(R_cart):
    """Quaternion-based spinor rotation from a 3x3 Cartesian rotation matrix.
    Returns a (2, 2) complex unitary matrix."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    Q = np.zeros((4, 4))
    Q[0, 0] = R_cart[0, 0] + R_cart[1, 1] + R_cart[2, 2]
    Q[0, 1] = Q[1, 0] = R_cart[1, 2] - R_cart[2, 1]
    Q[0, 2] = Q[2, 0] = R_cart[2, 0] - R_cart[0, 2]
    Q[0, 3] = Q[3, 0] = R_cart[0, 1] - R_cart[1, 0]
    Q[1, 1] = R_cart[0, 0] - R_cart[1, 1] - R_cart[2, 2]
    Q[1, 2] = Q[2, 1] = R_cart[0, 1] + R_cart[1, 0]
    Q[1, 3] = Q[3, 1] = R_cart[0, 2] + R_cart[2, 0]
    Q[2, 2] = -R_cart[0, 0] + R_cart[1, 1] - R_cart[2, 2]
    Q[2, 3] = Q[3, 2] = R_cart[1, 2] + R_cart[2, 1]
    Q[3, 3] = -R_cart[0, 0] - R_cart[1, 1] + R_cart[2, 2]

    eigvals, eigvecs = np.linalg.eigh(Q)
    q = eigvecs[:, np.argmax(eigvals)]
    q /= np.linalg.norm(q)
    q0, q1, q2, q3 = q

    theta = 2 * np.arccos(np.clip(q0, -1, 1))
    sin_half = np.sqrt(1 - q0**2)

    if sin_half < 1e-8:
        return np.eye(2, dtype=complex)

    n = np.array([q1, q2, q3]) / sin_half
    n /= np.linalg.norm(n)

    U = np.cos(theta / 2) * np.eye(2, dtype=complex) - 1j * np.sin(theta / 2) * (
        n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z
    )
    return U


def build_all_spinor_rotations(sym_mats_k, bvec):
    """Build spinor rotation matrices for all sym ops (spatial + time-reversal).
    bvec: (3,3) reciprocal lattice vectors (rows = b1, b2, b3)."""
    nsym = len(sym_mats_k)
    U_all = np.zeros((nsym, 2, 2), dtype=complex)
    # bvec rows = b1, b2, b3, so B_T = bvec.T = columns are b-vectors
    B_T = bvec.T
    B_T_inv = np.linalg.inv(B_T)
    for i in range(nsym):
        # Convert crystal → Cartesian rotation
        R_cart = B_T_inv @ sym_mats_k[i] @ B_T
        R_cart = np.around(R_cart, decimals=10)
        U_all[i] = compute_spinor_rotation(R_cart)
    return U_all


def rotate_gvecs(gvecs_irk, sym_mat_k, k_irk):
    """Rotate G-vectors from IBZ k-point to full BZ k-point.

    gvecs_irk: (ngk, 3) integer G-vectors at the IBZ k-point
    sym_mat_k: (3, 3) integer symmetry matrix in k-space
    k_irk: (3,) IBZ k-point in crystal coords

    Returns: (ngk, 3) rotated G-vectors
    """
    # Compute G-shift from BZ wrapping
    q_full = sym_mat_k @ k_irk
    q_inzone = q_full % 1.0
    q_inzone[q_inzone > 0.9999] = 0.0
    G_shift = (q_inzone - q_full).astype(int)

    # Rotate: G' = S^T @ G (using einsum for batch)
    G_rot = np.einsum("ij,kj->ki", sym_mat_k.astype(np.int32), gvecs_irk)
    G_rot -= G_shift
    return G_rot


def rotate_wfn(cnk_irk, sym_idx, ntran, U_spinor):
    """Apply symmetry rotation to wavefunction coefficients.

    cnk_irk: (2, ngk) complex spinor coefficients at IBZ k-point
    sym_idx: index into sym_mats_k (>= ntran means time-reversal)
    ntran: number of spatial symmetries
    U_spinor: (2, 2) spinor rotation matrix

    Returns: (2, ngk) rotated coefficients
    """
    c = cnk_irk.copy()
    if sym_idx >= ntran:
        c = np.conj(c)  # time-reversal: u(-k)(G) = u*(k)(-G)
    return U_spinor @ c  # (2, 2) @ (2, ngk) → (2, ngk)


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def build_gvec_index_map(gvecs_a, gvecs_b):
    """Build mapping from G-vectors of set A to indices in set B.

    Returns: idx_map (len(gvecs_a),) where idx_map[i] = j means
             gvecs_a[i] == gvecs_b[j], or -1 if not found.
    """
    # Hash-based lookup for speed
    b_dict = {}
    for j, g in enumerate(gvecs_b):
        b_dict[tuple(g)] = j

    idx = np.full(len(gvecs_a), -1, dtype=np.int64)
    for i, g in enumerate(gvecs_a):
        key = tuple(g)
        if key in b_dict:
            idx[i] = b_dict[key]
    return idx


def compute_overlap(cnk_a, cnk_b, gmap):
    """Compute <ψ_a | ψ_b> where a and b live on different G-vector lists.

    cnk_a: (2, ngk_a) — coefficients on G-list A (the reference, nosym)
    cnk_b: (2, ngk_b) — coefficients on G-list B (the rotated)
    gmap: (ngk_b,) — gmap[j] = index into A's G-list, or -1

    Returns: complex overlap scalar.
    """
    valid = gmap >= 0
    if not np.any(valid):
        return 0.0 + 0.0j

    # Gather A coefficients at the matched positions
    a_at_match = cnk_a[:, gmap[valid]]  # (2, n_matched)
    b_at_match = cnk_b[:, valid]        # (2, n_matched)

    return np.sum(np.conj(a_at_match) * b_at_match)


def check_subspace_overlap(overlaps_2x2):
    """Check if two degenerate pairs span the same 2D subspace.

    overlaps_2x2: (2, 2) complex matrix where O[i,j] = <nosym_i | rotated_j>

    Returns: Frobenius norm squared = Σ |O_ij|² (should be ≈ 2.0 for perfect match).
    Also returns singular values for diagnostics.
    """
    fro2 = np.sum(np.abs(overlaps_2x2) ** 2)
    svs = np.linalg.svd(overlaps_2x2, compute_uv=False)
    return fro2, svs


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    sym_wfn_path = Path("../01_si_4x4x4_nosymmorphic/qe/nscf/WFN.h5")
    nosym_wfn_path = Path("../02_si_4x4x4_nosym/qe/nscf/WFN.h5")

    # Resolve relative to script location
    base = Path(__file__).resolve().parent
    sym_wfn_path = base / ".." / ".." / "01_si_4x4x4_nosymmorphic" / "qe" / "nscf" / "WFN.h5"
    nosym_wfn_path = base / ".." / "qe" / "nscf" / "WFN.h5"

    print("Loading sym WFN (IBZ)...")
    sym_wfn = WFNData(sym_wfn_path)
    print(f"  {sym_wfn.nkpts} IBZ k-points, ntran={sym_wfn.ntran}, "
          f"nband={sym_wfn.nband}, ngk range=[{sym_wfn.ngk.min()}, {sym_wfn.ngk.max()}]")

    print("Loading nosym WFN (full BZ)...")
    nosym_wfn = WFNData(nosym_wfn_path)
    print(f"  {nosym_wfn.nkpts} k-points, ntran={nosym_wfn.ntran}, "
          f"nband={nosym_wfn.nband}")

    # Build full BZ from sym WFN's grid
    print(f"\nBuilding full BZ from kgrid={sym_wfn.kgrid}, shift={sym_wfn.shift}")
    full_kpts = build_full_bz_kpoints(sym_wfn.kgrid, sym_wfn.shift)
    nk_full = len(full_kpts)
    print(f"  {nk_full} full BZ k-points")

    # Build symmetry mappings
    sym_mats_k = build_sym_mats_k(sym_wfn.sym_matrices, sym_wfn.ntran)
    irk_map, sym_map = find_ibz_mapping(full_kpts, sym_wfn.kpoints, sym_mats_k)
    print(f"  Mapped {np.sum(irk_map >= 0)}/{nk_full} k-points to IBZ")

    unmapped = np.sum(irk_map < 0)
    if unmapped > 0:
        print(f"  WARNING: {unmapped} k-points could not be mapped!")
        bad = np.where(irk_map < 0)[0]
        for ib in bad[:5]:
            print(f"    k={full_kpts[ib]}")

    # Build spinor rotations
    U_spinor = build_all_spinor_rotations(sym_mats_k, sym_wfn.bvec)

    # Match full BZ k-points to nosym k-points
    nosym_kmap = {}
    for ink, k in enumerate(nosym_wfn.kpoints):
        k_mod = k % 1.0
        k_mod[k_mod > 0.9999] = 0.0
        nosym_kmap[tuple(np.round(k_mod, 6))] = ink

    # Degeneracy threshold (eV)
    DEGEN_TOL = 0.001

    # Which bands to test (0-indexed, first 16 = 8 val + 8 cond for Si)
    test_bands = list(range(16))

    print(f"\n{'='*90}")
    print(f"Testing wavefunction rotation for bands {test_bands[0]}–{test_bands[-1]}")
    print(f"{'='*90}\n")

    results = []

    for ikf in range(nk_full):
        irk = irk_map[ikf]
        isym = sym_map[ikf]
        if irk < 0:
            continue

        # Skip identity (sym 0 at the IBZ k-point itself)
        k_full = full_kpts[ikf]
        k_irk = sym_wfn.kpoints[irk]

        # Find matching nosym k-point
        kf_mod = k_full % 1.0
        kf_mod[kf_mod > 0.9999] = 0.0
        key = tuple(np.round(kf_mod, 6))
        ink_nosym = nosym_kmap.get(key)
        if ink_nosym is None:
            continue

        # Is this a nontrivial rotation?
        is_identity = np.allclose(sym_mats_k[isym], np.eye(3))
        sym_label = f"sym={isym:2d}" + (" (identity)" if is_identity else "")

        # ---- Rotate G-vectors ----
        gvecs_irk = sym_wfn.get_gvec_nk(irk)
        gvecs_rot = rotate_gvecs(gvecs_irk, sym_mats_k[isym], k_irk)

        # G-vectors at the nosym k-point
        gvecs_nosym = nosym_wfn.get_gvec_nk(ink_nosym)

        # Build map: rotated G → nosym G index
        gmap = build_gvec_index_map(gvecs_nosym, gvecs_rot)
        n_matched = np.sum(gmap >= 0)
        n_total = len(gvecs_nosym)
        match_frac = n_matched / n_total if n_total > 0 else 0

        # ---- Band-by-band overlap ----
        nosym_energies = nosym_wfn.energies[ink_nosym]  # (nband,)

        # ---- Detect degeneracy groups (may be >2 for cubic symmetry) ----
        degen_groups = []
        ib = 0
        while ib < len(test_bands):
            nb = test_bands[ib]
            group = [nb]
            while ib + 1 < len(test_bands):
                nb_next = test_bands[ib + 1]
                if abs(nosym_energies[nb_next] - nosym_energies[nb]) < DEGEN_TOL:
                    group.append(nb_next)
                    ib += 1
                else:
                    break
            degen_groups.append(group)
            ib += 1

        # ---- Build overlap matrix for each group ----
        band_results = []
        for group in degen_groups:
            ndeg = len(group)

            # Build ndeg × ndeg overlap matrix:
            # O[i,j] = <nosym_group[i] | rotated_group[j]>
            O = np.zeros((ndeg, ndeg), dtype=complex)
            for ii, ni in enumerate(group):
                c_nosym = nosym_wfn.get_cnk(ink_nosym, ni)
                for jj, nj in enumerate(group):
                    c_irk = sym_wfn.get_cnk(irk, nj)
                    c_rot = rotate_wfn(c_irk, isym, sym_wfn.ntran, U_spinor[isym])
                    O[ii, jj] = compute_overlap(c_nosym, c_rot, gmap)

            fro2 = np.sum(np.abs(O) ** 2)
            svs = np.linalg.svd(O, compute_uv=False)
            band_results.append({
                "bands": tuple(group),
                "type": f"group_{ndeg}",
                "fro2": fro2,
                "expected_fro2": float(ndeg),  # perfect match: Σ|O|² = ndeg
                "svs": svs,
            })

        # Summarize for this k-point
        max_err = 0.0
        for br in band_results:
            err = abs(br["expected_fro2"] - br["fro2"])
            max_err = max(max_err, err)

        flag = " *** BAD" if max_err > 0.05 else ""
        results.append({
            "ikf": ikf, "irk": irk, "isym": isym,
            "k_full": k_full, "k_irk": k_irk,
            "is_identity": is_identity,
            "match_frac": match_frac,
            "max_err": max_err,
            "band_results": band_results,
        })

        # Print summary line
        print(f"ikf={ikf:3d}  k=({k_full[0]:6.3f},{k_full[1]:6.3f},{k_full[2]:6.3f})  "
              f"irk={irk}  {sym_label:20s}  "
              f"G-match={match_frac:.3f}  max_err={max_err:.6f}{flag}")

        # Print per-band detail if there's a problem
        if max_err > 0.05:
            for br in band_results:
                err = abs(br["expected_fro2"] - br["fro2"])
                ndeg = len(br["bands"])
                svs_str = ", ".join(f"{s:.4f}" for s in br["svs"])
                if err > 0.01:
                    print(f"    bands {br['bands']} ({ndeg}-fold): "
                          f"||O||²={br['fro2']:.4f}/{br['expected_fro2']:.0f}  "
                          f"svs=[{svs_str}]")

    # ---- Summary table ----
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    good = [r for r in results if r["max_err"] < 0.01]
    bad = [r for r in results if r["max_err"] >= 0.01]
    identity = [r for r in results if r["is_identity"]]
    nontrivial = [r for r in results if not r["is_identity"]]

    print(f"Total k-points tested:     {len(results)}")
    print(f"  Identity sym (trivial):  {len(identity)}")
    print(f"  Nontrivial rotation:     {len(nontrivial)}")
    print(f"  GOOD (err < 0.01):       {len(good)}")
    print(f"  BAD  (err >= 0.01):      {len(bad)}")

    if bad:
        print(f"\nBAD k-points:")
        for r in bad:
            print(f"  ikf={r['ikf']:3d}  k=({r['k_full'][0]:6.3f},{r['k_full'][1]:6.3f},{r['k_full'][2]:6.3f})  "
                  f"irk={r['irk']}  sym={r['isym']:2d}  G-match={r['match_frac']:.3f}  "
                  f"max_err={r['max_err']:.6f}")

    # Check if G-vector matching is the issue
    low_gmatch = [r for r in results if r["match_frac"] < 0.99]
    if low_gmatch:
        print(f"\nK-points with low G-vector match fraction (<0.99):")
        for r in low_gmatch:
            print(f"  ikf={r['ikf']:3d}  G-match={r['match_frac']:.3f}  sym={r['isym']}")


if __name__ == "__main__":
    main()
