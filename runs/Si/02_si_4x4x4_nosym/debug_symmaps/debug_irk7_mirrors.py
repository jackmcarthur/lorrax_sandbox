"""Debug irk=7 mirror failures.

ikf=30: irk=7, sym=7 (mirror: swaps k2↔k3), k_irk=(0.25,0.5,0.75) → k_full=(0.25,0.75,0.5)
ikf=39: irk=7, sym=8 (mirror: swaps k1↔k2), k_irk=(0.25,0.5,0.75) → k_full=(0.5,0.25,0.75)

Both have U_spinor = I and ||O||² = 1.0 (half of expected 2.0).
The same mirrors PASS from irk=4 (sym=7 at ikf=9, sym=8 at ikf=18).

Strategy:
1. Compare G-vector sets in detail between the rotated and nosym spheres
2. Check if the G-shift is correct
3. Manually trace a few G-vectors through the rotation and verify they land correctly
4. Compare irk=7+sym=7 (FAIL) against irk=4+sym=7 (PASS) to find the difference
"""
import numpy as np, h5py
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps

wfn = WFNReader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/qe/nscf/WFN.h5')
sym = SymMaps(wfn)

with h5py.File('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5', 'r') as f:
    nosym_kpts = f['mf_header/kpoints/rk'][:]
    nosym_ngk = f['mf_header/kpoints/ngk'][:]
    nosym_gvecs = f['wfns/gvecs'][:]
    nosym_coeffs = f['wfns/coeffs'][:]
nosym_starts = np.zeros(len(nosym_kpts), dtype=np.int64)
nosym_starts[1:] = np.cumsum(nosym_ngk[:-1])
def nosym_get_gvec(ik):
    s = nosym_starts[ik]; return nosym_gvecs[s:s+nosym_ngk[ik]]
def nosym_get_cnk(ik, ib):
    s = nosym_starts[ik]; e = s + nosym_ngk[ik]
    raw = nosym_coeffs[ib, :, s:e, :]; return raw[:,:,0] + 1j*raw[:,:,1]
nosym_kmap = {}
for ink, k in enumerate(nosym_kpts):
    km = k % 1.0; km[km > 0.9999] = 0.0
    nosym_kmap[tuple(np.round(km, 6))] = ink

# ---- FAILING case: ikf=30, irk=7, sym=7 ----
ikf_fail = 30
irk_fail = sym.irk_to_k_map[ikf_fail]
isym_fail = sym.irk_sym_map[ikf_fail]
k_irk = wfn.kpoints[irk_fail]
kf = sym.unfolded_kpts[ikf_fail]
kfm = kf % 1.0; kfm[kfm > 0.9999] = 0.0
ink_fail = nosym_kmap[tuple(np.round(kfm, 6))]

print(f"FAILING: ikf={ikf_fail}, irk={irk_fail}, sym={isym_fail}")
print(f"  k_irk = {k_irk}")
print(f"  k_full = {kfm}")
print(f"  S_k[{isym_fail}] = {sym.sym_mats_k[isym_fail].tolist()}")

# G-shift computation (replicate SymMaps logic)
S_k = sym.sym_mats_k[isym_fail]
q_full = S_k @ k_irk
q_inzone = q_full % 1.0
q_inzone[q_inzone > 0.9999] = 0.0
G_shift = (q_inzone - q_full).astype(int)
print(f"  q_full = S_k @ k_irk = {q_full}")
print(f"  q_inzone = {q_inzone}")
print(f"  G_shift = {G_shift}")

# G-vectors: rotated vs nosym
gvecs_rot_fail = sym.get_gvecs_kfull(wfn, ikf_fail)
gvecs_nosym_fail = nosym_get_gvec(ink_fail)
gvecs_irk = wfn.get_gvec_nk(irk_fail)

# Check first few G-vectors in detail
print(f"\n  First 5 G-vectors at irk={irk_fail}:")
for i in range(5):
    g = gvecs_irk[i]
    g_rot = gvecs_rot_fail[i]
    print(f"    G_irk={g}  →  G_rot = S_k@G - G_shift = {S_k.astype(int) @ g} - {G_shift} = {g_rot}")

# ---- PASSING case: ikf=9, irk=4, sym=7 (same mirror!) ----
ikf_pass = 9
irk_pass = sym.irk_to_k_map[ikf_pass]
isym_pass = sym.irk_sym_map[ikf_pass]
k_irk_pass = wfn.kpoints[irk_pass]
kf_pass = sym.unfolded_kpts[ikf_pass]
kfm_pass = kf_pass % 1.0; kfm_pass[kfm_pass > 0.9999] = 0.0
ink_pass = nosym_kmap[tuple(np.round(kfm_pass, 6))]

print(f"\nPASSING: ikf={ikf_pass}, irk={irk_pass}, sym={isym_pass}")
print(f"  k_irk = {k_irk_pass}")
print(f"  k_full = {kfm_pass}")

q_full_pass = S_k @ k_irk_pass
q_inzone_pass = q_full_pass % 1.0
q_inzone_pass[q_inzone_pass > 0.9999] = 0.0
G_shift_pass = (q_inzone_pass - q_full_pass).astype(int)
print(f"  q_full = {q_full_pass}")
print(f"  q_inzone = {q_inzone_pass}")
print(f"  G_shift = {G_shift_pass}")

# Key comparison: does the FAILING case have a nonzero G_shift 
# while the PASSING case doesn't?
print(f"\n  G_shift comparison: FAIL={G_shift}, PASS={G_shift_pass}")

# Now check: per-G overlap contribution
# For the failing case, compute overlap contribution from each G-vector
gvecs_rot_fail = sym.get_gvecs_kfull(wfn, ikf_fail)
gvecs_nosym_fail = nosym_get_gvec(ink_fail)

# Band 0
c_nosym = nosym_get_cnk(ink_fail, 0)
c_rot = sym.get_cnk_fullzone(wfn, 0, ikf_fail)

# Build map
bd = {tuple(g): j for j, g in enumerate(gvecs_rot_fail)}
ov_per_G = []
for i, g in enumerate(gvecs_nosym_fail):
    j = bd.get(tuple(g))
    if j is not None:
        contrib = np.conj(c_nosym[0,i])*c_rot[0,j] + np.conj(c_nosym[1,i])*c_rot[1,j]
        ov_per_G.append((i, j, g, abs(contrib), contrib))

# Sort by magnitude
ov_per_G.sort(key=lambda x: -x[3])
total_ov = sum(x[4] for x in ov_per_G)
print(f"\n  Band 0 overlap: {total_ov:.6f} (|ov|={abs(total_ov):.6f})")
print(f"  Top 10 G-vector contributions:")
for i, j, g, mag, ov in ov_per_G[:10]:
    # Check: what's the irk G-vector that maps to this?
    # gvecs_rot[j] = S_k @ gvecs_irk[j] - G_shift
    g_irk = gvecs_irk[j]
    print(f"    nosym_G[{i}]={g} = rot_G[{j}] (from irk_G={g_irk})  "
          f"|contrib|={mag:.6f}  contrib={ov:.4f}")

# Now the critical test: what if we DON'T apply the G_shift?
print(f"\n  What if G_shift were 0 instead of {G_shift}?")
gvecs_rot_noshift = (S_k.astype(int) @ gvecs_irk.T).T  # no G_shift subtraction
bd2 = {tuple(g): j for j, g in enumerate(gvecs_rot_noshift)}
n_match_noshift = sum(1 for g in gvecs_nosym_fail if tuple(g) in bd2)
print(f"  G-match without shift: {n_match_noshift}/{len(gvecs_nosym_fail)}")

# And what about the OPPOSITE sign of G_shift?
gvecs_rot_negshift = (S_k.astype(int) @ gvecs_irk.T).T + G_shift  # opposite sign
bd3 = {tuple(g): j for j, g in enumerate(gvecs_rot_negshift)}
n_match_negshift = sum(1 for g in gvecs_nosym_fail if tuple(g) in bd3)
print(f"  G-match with +G_shift: {n_match_negshift}/{len(gvecs_nosym_fail)}")

# Test overlap with no G_shift
ov_noshift = 0.0j
for i, g in enumerate(gvecs_nosym_fail):
    j = bd2.get(tuple(g))
    if j is not None:
        ov_noshift += np.conj(c_nosym[0,i])*c_rot[0,j] + np.conj(c_nosym[1,i])*c_rot[1,j]
print(f"  Band 0 overlap without G_shift: |ov|={abs(ov_noshift):.6f}")

# Test overlap with +G_shift
ov_negshift = 0.0j
for i, g in enumerate(gvecs_nosym_fail):
    j = bd3.get(tuple(g))
    if j is not None:
        ov_negshift += np.conj(c_nosym[0,i])*c_rot[0,j] + np.conj(c_nosym[1,i])*c_rot[1,j]
print(f"  Band 0 overlap with +G_shift: |ov|={abs(ov_negshift):.6f}")
