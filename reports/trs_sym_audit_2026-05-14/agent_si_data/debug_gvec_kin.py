"""Debug: kinetic energies at hand-rotated G-list vs nosym G-list at same k."""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader

sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
nos = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5")
s = sym._ensure_sym()
kirr = 4
sidx = 5
kf = 11
sym_krep = np.asarray(s.sym_mats_k[sidx], dtype=np.int64)
g_irr = sym._gvecs_raw[sym._kpt_starts[kirr]:sym._kpt_starts[kirr] + sym.ngk[kirr]]
g_rot = np.einsum("ij,kj->ki", sym_krep, g_irr)
diff = nos.kpoints - s.unfolded_kpts[kf]
diff_w = diff - np.round(diff)
j = int(np.argmin(np.max(np.abs(diff_w), axis=1)))
g_nos = nos._gvecs_raw[nos._kpt_starts[j]:nos._kpt_starts[j] + nos.ngk[j]]
k_full = s.unfolded_kpts[kf]
k_irr = np.array(sym.kpoints[kirr])

bvec = sym.bvec
print("k_full:", k_full, "  k_irr:", k_irr, "   S*k_irr:", sym_krep @ k_irr)


def kin(k, gvecs):
    kg = (k[None, :] + gvecs.astype(float))
    cart = kg @ bvec
    return np.sum(cart ** 2, axis=1)

ke_rot = kin(k_full, g_rot)
ke_irr = kin(k_irr, g_irr)
ke_nos = kin(k_full, g_nos)
print("max diff sorted (rot,irr):", np.max(np.abs(np.sort(ke_rot) - np.sort(ke_irr))))
print("max ke_rot:", np.max(ke_rot), "   max ke_nos:", np.max(ke_nos))
print("  2*ecutwfc:", 2 * sym.ecutwfc)

g_rot_set = set(map(tuple, g_rot.tolist()))
g_nos_set = set(map(tuple, g_nos.tolist()))
only_rot = list(g_rot_set - g_nos_set)
only_nos = list(g_nos_set - g_rot_set)
print("|only in rot|=%d  |only in nosym|=%d" % (len(only_rot), len(only_nos)))
ke_only_rot = kin(k_full, np.array(only_rot))
ke_only_nos = kin(k_full, np.array(only_nos))
print("only-in-rot kin range:", np.min(ke_only_rot), np.max(ke_only_rot))
print("only-in-nos kin range:", np.min(ke_only_nos), np.max(ke_only_nos))
print("only-in-rot G values (first 20):", only_rot[:20])
print("only-in-nos G values (first 20):", only_nos[:20])

# Now check inverse: does sym_mats_k[s].T @ g_nos cover g_irr?
sym_inv = np.linalg.inv(sym_krep).astype(np.int64)
g_inv_rot = np.einsum("ij,kj->ki", sym_inv, g_nos)
g_inv_rot_set = set(map(tuple, g_inv_rot.tolist()))
g_irr_set = set(map(tuple, g_irr.tolist()))
print("|g_inv_rot ∩ g_irr|=%d  |g_inv_rot \\ g_irr|=%d  |g_irr \\ g_inv_rot|=%d" %
      (len(g_inv_rot_set & g_irr_set), len(g_inv_rot_set - g_irr_set), len(g_irr_set - g_inv_rot_set)))

# Check using mtrx[s] (column action) instead of sym_mats_k (row)
mtrx_s = np.asarray(sym.sym_matrices[sidx], dtype=np.int64)
g_rot_col = np.einsum("ij,kj->ki", mtrx_s, g_irr)
g_rot_col_set = set(map(tuple, g_rot_col.tolist()))
print("|g_rot_col ∩ g_nos|=%d  |only_col|=%d" % (len(g_rot_col_set & g_nos_set), len(g_rot_col_set - g_nos_set)))

# Also check 'g_full_lorrax' from gvecs(k='full_bz')
g_full_lorrax = np.asarray(sym.gvecs(k='full_bz'))[kf]
g_full_lorrax_valid = g_full_lorrax[:sym.ngk_valid(k='full_bz')[kf]]
g_lorrax_set = set(map(tuple, g_full_lorrax_valid.tolist()))
print("|g_lorrax ∩ g_nos|=%d   |g_lorrax ∩ g_rot|=%d" %
      (len(g_lorrax_set & g_nos_set), len(g_lorrax_set & g_rot_set)))
