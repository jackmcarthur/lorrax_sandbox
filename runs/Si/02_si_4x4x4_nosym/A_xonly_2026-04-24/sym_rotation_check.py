"""Check LORRAX's sym-unfolded bands ↔ nosym bands overlap at the same full-BZ k.

For each off-axis full-BZ k-point the symmetric WFN produces via rotation of
some IBZ k_bar, we compute <ψ_rot_n | ψ_nosym_m> for all bands in a window
and check whether it is block-diagonal with blocks matching degenerate
groups (each block should be unitary).
"""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")

import numpy as np
from file_io import WFNReader
from common import symmetry_maps

SYM_WFN  = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5"
NOS_WFN  = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5"
NBAND    = 20             # bands to check (0-indexed up to nband-1)
DEGEN_TOL_EV = 5e-3       # group bands closer than this in energy into one block

wfn_sym  = WFNReader(SYM_WFN)
wfn_nos  = WFNReader(NOS_WFN)
sym = symmetry_maps.SymMaps(wfn_sym)
print(f"sym WFN ntran={wfn_sym.ntran}; LORRAX's sym_mats_k length={len(sym.sym_mats_k)}")
print(f"nosym WFN ntran={wfn_nos.ntran}")
print(f"sym IBZ: {wfn_sym.nkpts} k-points; unfolded: {sym.unfolded_kpts.shape[0]}")
print(f"nosym full: {wfn_nos.nkpts} k-points")
print(f"U_spinor shape: {sym.U_spinor.shape}")


def kindex_nosym(kfrac, kpoints_nos, tol=2e-3):
    for i, k in enumerate(kpoints_nos):
        d = kfrac - k
        d = d - np.round(d)
        if np.max(np.abs(d)) < tol:
            return i
    return None


def compute_overlap_on_fft_grid(cnk_A, gvecs_A, cnk_B, gvecs_B, fft_grid):
    """Put wavefunctions onto a shared FFT grid and compute <A|B> matrix."""
    nA = cnk_A.shape[0]
    nB = cnk_B.shape[0]
    nx, ny, nz = [int(x) for x in fft_grid]
    # Build dense grids per band
    def to_grid(cnks, gvecs):
        n = cnks.shape[0]
        grid = np.zeros((n, 2, nx, ny, nz), dtype=np.complex128)
        gx = np.mod(gvecs[:, 0], nx)
        gy = np.mod(gvecs[:, 1], ny)
        gz = np.mod(gvecs[:, 2], nz)
        for i in range(n):
            grid[i, 0, gx, gy, gz] = cnks[i, 0]
            grid[i, 1, gx, gy, gz] = cnks[i, 1]
        return grid.reshape(n, -1)
    GA = to_grid(cnk_A, gvecs_A)
    GB = to_grid(cnk_B, gvecs_B)
    # Overlap <A|B> = sum_g A*(g) B(g)
    return GA.conj() @ GB.T


def degenerate_blocks(energies, tol=DEGEN_TOL_EV):
    """Return list of (start, end) index pairs grouping near-degenerate bands."""
    blocks = []
    i = 0
    while i < len(energies):
        j = i + 1
        while j < len(energies) and abs(energies[j] - energies[i]) < tol:
            j += 1
        blocks.append((i, j))
        i = j
    return blocks


results = []
RY2EV = 13.6056980659
for nk_full in range(sym.unfolded_kpts.shape[0]):
    k_full = np.asarray(sym.unfolded_kpts[nk_full])
    # get sym op + irreducible k
    sym_idx, kbar_idx, sym_krep = sym._get_symmetry_context(nk_full)
    # Skip identity (trivial — sym==nosym at those points)
    trivial = (sym_idx == 0)

    nk_nos = kindex_nosym(k_full, wfn_nos.kpoints)
    if nk_nos is None:
        continue

    # Sym-rotated bands (using LORRAX machinery)
    band_indices = np.arange(NBAND)
    cnk_rot = sym.get_cnk_fullzone_batch(wfn_sym, band_indices, nk_full)
    gvecs_rot = sym.get_gvecs_kfull(wfn_sym, nk_full)
    # Nosym true bands at same k_full
    cnk_true = wfn_nos.get_cnk_batch(nk_nos, band_indices)
    gvecs_true = wfn_nos.get_gvec_nk(nk_nos)

    # Normalize (each band's <ψ|ψ> should be 1)
    fft_grid = wfn_sym.fft_grid
    S = compute_overlap_on_fft_grid(cnk_rot, gvecs_rot, cnk_true, gvecs_true, fft_grid)

    # Degenerate groupings (use nosym energies — same DFT result so sym/nosym match)
    energies_ev = wfn_nos.energies[0, nk_nos, :NBAND] * RY2EV
    blocks = degenerate_blocks(energies_ev)

    # Check block-diagonal structure
    diag_norm = 0.0
    offdiag_norm = 0.0
    block_unitary_err = 0.0
    for (i0, i1) in blocks:
        for (j0, j1) in blocks:
            blk = S[i0:i1, j0:j1]
            if i0 == j0:
                diag_norm += np.sum(np.abs(blk) ** 2)
                # block should be unitary
                U = blk
                Ud = U @ U.conj().T
                eye = np.eye(U.shape[0])
                block_unitary_err += np.linalg.norm(Ud - eye)
            else:
                offdiag_norm += np.sum(np.abs(blk) ** 2)

    results.append((nk_full, sym_idx, kbar_idx, trivial, tuple(k_full),
                    diag_norm, offdiag_norm, block_unitary_err,
                    [(i0, i1, energies_ev[i0]) for (i0, i1) in blocks]))

print(f"\n{'nk_full':>7} {'sym':>4} {'kbar':>4} {'trivial':>8} "
      f"{'diag²':>10} {'offdiag²':>10} {'Σ|UU†-I|':>10}   degen blocks (E_ev)")
for r in results[:32]:
    nk, si, kb, triv, kf, dn, on, be, blks = r
    blocks_str = ",".join(f"{i0}-{i1-1}" for (i0, i1, _) in blks[:8])
    print(f"{nk:>7} {si:>4} {kb:>4} {triv!s:>8} {dn:>10.4f} {on:>10.4f} {be:>10.3f}   [{blocks_str}]")

# Summary: for non-trivial sym ops, how many show non-trivial off-diag?
nontrivial = [r for r in results if not r[3]]
print(f"\nSummary over {len(nontrivial)} non-trivial-sym full-BZ k-points:")
bad_offdiag = [r for r in nontrivial if r[6] > 1e-3]
bad_unitary = [r for r in nontrivial if r[7] > 1e-3]
print(f"  off-block leakage > 1e-3: {len(bad_offdiag)} / {len(nontrivial)}")
print(f"  block unitarity err > 1e-3: {len(bad_unitary)} / {len(nontrivial)}")
if bad_offdiag:
    print(f"\n  worst off-diag leakage:")
    worst = sorted(bad_offdiag, key=lambda r: -r[6])[:5]
    for r in worst:
        print(f"    nk_full={r[0]} sym_idx={r[1]} k_full={r[4]}  offdiag²={r[6]:.4f}")
