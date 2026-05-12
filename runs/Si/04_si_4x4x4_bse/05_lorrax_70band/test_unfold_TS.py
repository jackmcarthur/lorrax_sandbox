"""Test 1: Gram-matrix orthonormality of LORRAX-unfolded ψ at full-BZ k-points
that are reached via T·S unfolding (sym_idx >= ntran).

If LORRAX's get_cnk_fullzone[_batch] is correct for T·S, then the unfolded
band coefficients at every full-BZ k should still be orthonormal: ⟨n|m⟩ = δ_nm.

This test ALSO compares the gram matrix at:
 (a) a kfull reached via spatial-only sym (sym_idx < ntran) — should pass either way
 (b) a kfull reached via T·S sym (sym_idx >= ntran) — passes only if T·S handling is correct

If (a) passes and (b) fails, the bug is in the T·S branch.
"""
import sys
import numpy as np

sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')

from file_io import WFNReader
from common import symmetry_maps

WFN_PATH = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/05_lorrax_70band/WFN.h5'

w = WFNReader(WFN_PATH)
sym = symmetry_maps.SymMaps(w)
ntran = w.ntran
nk_full = sym.nk_tot
nb_test = 16  # check first 16 bands

print(f"WFN: ntran={ntran}, nk_full={nk_full}, n bands tested={nb_test}")
print(f"Full-BZ k-points reached via spatial sym vs T·S:")
spatial_kfull = []
ts_kfull = []
for kfull in range(nk_full):
    sym_idx = int(sym.irk_sym_map[kfull])
    if sym_idx < ntran:
        spatial_kfull.append((kfull, sym_idx))
    else:
        ts_kfull.append((kfull, sym_idx, sym_idx - ntran))

print(f"  spatial-only: {len(spatial_kfull)} (sym_idx < {ntran})")
print(f"  T·S:          {len(ts_kfull)} (sym_idx >= {ntran})")

# Pick representative k-points
def pick_diverse(klist, n=4):
    """Pick a few k-points with distinct irreducible parents and non-zero τ involvement."""
    seen = set()
    out = []
    for k_info in klist:
        kfull = k_info[0]
        irrk = int(sym.irk_to_k_map[kfull])
        sym_idx = k_info[1]
        spatial_idx = sym_idx if sym_idx < ntran else sym_idx - ntran
        tau = w.translations[spatial_idx]
        nonzero_tau = bool(np.any(np.abs(tau) > 1e-6))
        key = (irrk, nonzero_tau)
        if key not in seen:
            seen.add(key)
            out.append(k_info + (irrk, nonzero_tau, tau))
            if len(out) >= n:
                break
    return out

print("\n=== Checking spatial-only kfull (sanity baseline) ===")
for k_info in pick_diverse(spatial_kfull, 3):
    kfull = k_info[0]
    sym_idx = k_info[1]
    irrk = k_info[2]
    nz_tau = k_info[3]
    tau = k_info[4]
    cnk_full = sym.get_cnk_fullzone_batch(w, np.arange(nb_test), kfull)
    # cnk_full shape: (nb, 2, ngk)
    # Gram: ⟨n|m⟩ = Σ_{σ,G} c*_n,σ(G) · c_m,σ(G)
    G = np.einsum('nsG,msG->nm', np.conj(cnk_full), cnk_full)
    # Diagonal should be ~1; off-diagonal should be ~0
    diag = np.real(np.diag(G))
    off = G - np.diag(np.diag(G))
    max_off = float(np.max(np.abs(off)))
    diag_dev = float(np.max(np.abs(diag - 1.0)))
    print(f"  kfull={kfull:2d}  irrk={irrk}  sym={sym_idx:2d}(spatial)  τ-nonzero={nz_tau}  τ={tau}")
    print(f"        diag dev from 1: {diag_dev:.3e}    max off-diag: {max_off:.3e}")

print("\n=== Checking T·S kfull (this is where the bug should manifest) ===")
for k_info in pick_diverse(ts_kfull, 3):
    kfull = k_info[0]
    sym_idx = k_info[1]
    spatial_idx = k_info[2]
    irrk = k_info[3]
    nz_tau = k_info[4]
    tau = k_info[5]
    cnk_full = sym.get_cnk_fullzone_batch(w, np.arange(nb_test), kfull)
    G = np.einsum('nsG,msG->nm', np.conj(cnk_full), cnk_full)
    diag = np.real(np.diag(G))
    off = G - np.diag(np.diag(G))
    max_off = float(np.max(np.abs(off)))
    diag_dev = float(np.max(np.abs(diag - 1.0)))
    print(f"  kfull={kfull:2d}  irrk={irrk}  sym={sym_idx:2d}(T·S, spatial_idx={spatial_idx})  τ-nonzero={nz_tau}  τ={tau}")
    print(f"        diag dev from 1: {diag_dev:.3e}    max off-diag: {max_off:.3e}")

print("\n=== Summary across all T·S full-BZ k-points ===")
all_diag_devs = []
all_off_max = []
for k_info in ts_kfull:
    kfull = k_info[0]
    cnk_full = sym.get_cnk_fullzone_batch(w, np.arange(nb_test), kfull)
    G = np.einsum('nsG,msG->nm', np.conj(cnk_full), cnk_full)
    diag = np.real(np.diag(G))
    off = G - np.diag(np.diag(G))
    all_diag_devs.append(float(np.max(np.abs(diag - 1.0))))
    all_off_max.append(float(np.max(np.abs(off))))
print(f"  T·S k-points ({len(ts_kfull)} total):")
print(f"    diag dev from 1: max = {max(all_diag_devs):.3e}, mean = {np.mean(all_diag_devs):.3e}")
print(f"    off-diag       : max = {max(all_off_max):.3e},   mean = {np.mean(all_off_max):.3e}")

print("\n=== Same summary across spatial-only full-BZ k-points ===")
all_diag_devs = []
all_off_max = []
for k_info in spatial_kfull:
    kfull = k_info[0]
    cnk_full = sym.get_cnk_fullzone_batch(w, np.arange(nb_test), kfull)
    G = np.einsum('nsG,msG->nm', np.conj(cnk_full), cnk_full)
    diag = np.real(np.diag(G))
    off = G - np.diag(np.diag(G))
    all_diag_devs.append(float(np.max(np.abs(diag - 1.0))))
    all_off_max.append(float(np.max(np.abs(off))))
print(f"  spatial-only k-points ({len(spatial_kfull)} total):")
print(f"    diag dev from 1: max = {max(all_diag_devs):.3e}, mean = {np.mean(all_diag_devs):.3e}")
print(f"    off-diag       : max = {max(all_off_max):.3e},   mean = {np.mean(all_off_max):.3e}")
