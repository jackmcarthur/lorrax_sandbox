"""Debug: with kg0 subtracted does the hand-rotated G-list match nosym?"""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader

sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
nos = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5")
s = sym._ensure_sym()

# Try k_f=11
for kf in [11, 12, 15, 27, 31, 49]:
    kirr = int(s.irr_idx_k[kf])
    sidx = int(s.sym_idx_k[kf])
    sym_krep = np.asarray(s.sym_mats_k[sidx], dtype=np.int64)
    g_irr = sym._gvecs_raw[sym._kpt_starts[kirr]:sym._kpt_starts[kirr] + sym.ngk[kirr]]
    g_rot = np.einsum("ij,kj->ki", sym_krep, g_irr)
    # Compute kg0 (BGW: k_full = S k_irr + kg0)
    k_full = s.unfolded_kpts[kf]
    skbar = sym_krep @ sym.kpoints[kirr]
    kg0 = np.rint(k_full - skbar).astype(np.int64)
    g_rot_shift = g_rot - kg0[None, :]
    # nosym side
    diff = nos.kpoints - k_full
    diff_w = diff - np.round(diff)
    j = int(np.argmin(np.max(np.abs(diff_w), axis=1)))
    g_nos = nos._gvecs_raw[nos._kpt_starts[j]:nos._kpt_starts[j] + nos.ngk[j]]
    s1 = set(map(tuple, g_rot.tolist()))
    s2 = set(map(tuple, g_rot_shift.tolist()))
    s3 = set(map(tuple, g_nos.tolist()))
    print(f"k_f={kf:2d}  k_irr={kirr} s={sidx:2d} kg0={kg0}: rot ∩ nos={len(s1 & s3):3d}  rot-kg0 ∩ nos={len(s2 & s3):3d}  |nos|={len(s3):3d}")
