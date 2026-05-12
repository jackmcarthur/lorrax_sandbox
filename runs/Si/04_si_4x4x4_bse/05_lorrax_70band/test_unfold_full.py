"""Gram-matrix orthonormality at EVERY full-BZ k, broken down by sym op used.

If ⟨n|m⟩ ≠ δ_nm at some kfull, then the unfolded coefficients are wrong.
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
nb_test = 16

print(f"WFN: ntran={ntran}, nk_full={nk_full}")

# For each unique (sym_idx, has_nonzero_tau), pick one example kfull
buckets = {}
for kfull in range(nk_full):
    sym_idx = int(sym.irk_sym_map[kfull])
    spatial_idx = sym_idx if sym_idx < ntran else sym_idx - ntran
    tau = w.translations[spatial_idx]
    nonzero_tau = bool(np.any(np.abs(tau) > 1e-6))
    is_identity = bool(np.allclose(sym.sym_mats_k[spatial_idx], np.eye(3, dtype=np.int32)))
    bucket_key = (sym_idx, nonzero_tau, is_identity)
    buckets.setdefault(bucket_key, []).append(kfull)

print(f"\n{len(buckets)} unique (sym_idx, nonzero_tau, is_identity) buckets:")
for key, kfulls in sorted(buckets.items()):
    sym_idx, nz_tau, is_id = key
    print(f"  sym_idx={sym_idx:2d}  nonzero_tau={nz_tau!s:5s}  is_identity={is_id!s:5s}  → {len(kfulls)} kfulls (e.g. {kfulls[0]})")

# Test gram at one representative kfull from each bucket
print(f"\nGram orthonormality test (first {nb_test} bands):")
print(f"{'kfull':>5} {'irrk':>4} {'sym_idx':>7} {'nz_tau':>6} {'is_id':>5} {'diag_dev':>10} {'max_off':>10}")
worst_off_kfull = None
worst_off = 0.0
for key, kfulls in sorted(buckets.items()):
    sym_idx, nz_tau, is_id = key
    kfull = kfulls[0]
    irrk = int(sym.irk_to_k_map[kfull])
    cnk_full = sym.get_cnk_fullzone_batch(w, np.arange(nb_test), kfull)  # (nb, 2, ngk)
    G = np.einsum('nsG,msG->nm', np.conj(cnk_full), cnk_full)
    diag = np.real(np.diag(G))
    off = G - np.diag(np.diag(G))
    max_off = float(np.max(np.abs(off)))
    diag_dev = float(np.max(np.abs(diag - 1.0)))
    flag = " ***" if max_off > 1e-6 or diag_dev > 1e-6 else ""
    print(f"{kfull:>5} {irrk:>4} {sym_idx:>7} {nz_tau!s:>6} {is_id!s:>5} {diag_dev:>10.2e} {max_off:>10.2e}{flag}")
    if max_off > worst_off:
        worst_off = max_off
        worst_off_kfull = (kfull, sym_idx, nz_tau, is_id)

print(f"\nWorst off-diagonal: {worst_off:.3e} at kfull={worst_off_kfull[0]} (sym_idx={worst_off_kfull[1]}, τ={'nonzero' if worst_off_kfull[2] else 'zero'}, id={'yes' if worst_off_kfull[3] else 'no'})")

if worst_off > 1e-6:
    print(f"\n⚠️  Gram-matrix test FAILS at some kfull. The unfolded ψ at those k is NOT orthonormal.")
    print(f"   This is the SymMaps bug for the broken WFN.")
else:
    print(f"\n✓  Gram-matrix test PASSES everywhere. Unfolded ψ are orthonormal.")
    print(f"   The PPM Σ_c bug must be elsewhere (NOT in SymMaps wfn unfolding).")
