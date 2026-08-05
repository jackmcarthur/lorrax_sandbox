"""Companion to _tt_covar_control.py: report the raw magnitude of the in-plane
Lorentz tile V_qmunu_TT_22 (the (y,y) transverse current block) across the full
v_q file. Prints n_rmu_T (centroid count = tile dim), median|TT_22|, max|TT_22|.
Used to check whether the magnitude inflation seen in the muddied prior sweep
(102->small, 410->2.49) persists under clean identical methodology.
"""
import sys, numpy as np, h5py
DIR = sys.argv[1] if len(sys.argv) > 1 else 'runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5'
with h5py.File(DIR, 'r') as f:
    tt22 = f['V_qmunu_TT_22'][()]          # (nq, n_rmu_T, n_rmu_T) complex
    cc   = f['V_qmunu_CC'][()]
mag = np.abs(tt22)
print(f"DIR={DIR}")
print(f"  V_qmunu_TT_22 shape = {tt22.shape}  (n_rmu_T = {tt22.shape[1]})")
print(f"  V_qmunu_CC   shape = {cc.shape}    (n_rmu_C = {cc.shape[1]})")
print(f"  median|TT_22| = {np.median(mag):.6e}")
print(f"  max|TT_22|    = {mag.max():.6e}")
print(f"  mean|TT_22|   = {mag.mean():.6e}")
