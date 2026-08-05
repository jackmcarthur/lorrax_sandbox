"""Resolve the kg0 confusion."""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader

sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
s = sym._ensure_sym()
kf = 11
kirr = int(s.irr_idx_k[kf])
sidx = int(s.sym_idx_k[kf])
sym_krep = np.asarray(s.sym_mats_k[sidx], dtype=np.int64)

# WfnLoader.gvecs subtracts kg0 via sym._get_umklapp_vector
# Let's also compute it the simple way to compare
k_full = np.asarray(s.unfolded_kpts[kf])
skbar = sym_krep @ np.asarray(sym.kpoints[kirr])
kg0_simple = np.rint(k_full - skbar).astype(np.int64)
print("Simple kg0:", kg0_simple)
print("k_full:", k_full)
print("S*k_irr:", skbar)

# What does _get_umklapp_vector return?
kg0_lorrax = s._get_umklapp_vector(sym, kf, sidx, kirr, sym_krep)
print("LORRAX kg0:", kg0_lorrax)

# Now LORRAX g_rot:
g_irr = sym._gvecs_raw[sym._kpt_starts[kirr]:sym._kpt_starts[kirr] + sym.ngk[kirr]]
g_rot = np.einsum("ij,kj->ki", sym_krep, g_irr)
g_lorrax = g_rot - kg0_lorrax[None, :]
print("g_rot[:3]:", g_rot[:3])
print("g_lorrax[:3]:", g_lorrax[:3])
# Compare to LORRAX gvecs output
g_full_loader = np.asarray(sym.gvecs(k='full_bz'))[kf][:sym.ngk_valid(k='full_bz')[kf]]
print("g_full_loader[:3]:", g_full_loader[:3])
print("max |g_lorrax - g_full_loader|:", np.max(np.abs(g_lorrax - g_full_loader)))
