"""Compare LORRAX's spinor rotation matrices against a from-scratch BGW
convention (R_cart = bvec @ mtrx @ bvecinv).

The questions:
1. Does LORRAX's syms_crystal_to_cartesian convention (B_T_inv @ sym_mats_k @ B_T)
   produce the same R_cart as BGW (bvec @ mtrx @ bvecinv)?
2. Does the U_spinor matrix LORRAX builds agree with one built from BGW's R_cart?
"""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader

sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
s = sym._ensure_sym()

bvec = sym.bvec   # what is this convention?
mtrx = sym.sym_matrices    # (48,3,3) int — BGW "mtrx" indexed [ntran,3,3]
sym_mats_k = s.sym_mats_k  # (96,3,3) int = sym_matrices.transpose(0,2,1) +TRS

print("bvec:")
print(bvec)
print("bvec.T:")
print(bvec.T)

# LORRAX cartesian rotation (from syms_crystal_to_cartesian)
B_T = bvec
B_T_inv = np.linalg.inv(B_T)
R_lorrax = np.einsum('ij,njk,kl->nil', B_T_inv, sym_mats_k[:48], B_T)
print("R_lorrax[0]:")
print(R_lorrax[0])

# BGW convention: R_cart = bvec @ mtrx @ bvecinv (with BGW's bvec already transposed)
# If LORRAX bvec == BGW's bvec (== B^T), then BGW's R_cart = B^T @ mtrx @ (B^T)^-1
R_bgw = np.einsum('ij,njk,kl->nil', B_T, mtrx[:48], B_T_inv)
print("R_bgw[0]:")
print(R_bgw[0])

# Try a few non-identity ops
for s_idx in [1, 5, 9, 13, 24]:
    print(f"\n# sym op s={s_idx}")
    print(f"  mtrx[s]      = {mtrx[s_idx].tolist()}")
    print(f"  sym_mats_k[s] = mtrx[s].T = {sym_mats_k[s_idx].tolist()}")
    print(f"  R_lorrax[s]:")
    print("   ", R_lorrax[s_idx].round(5).tolist())
    print(f"  R_bgw[s]:")
    print("   ", R_bgw[s_idx].round(5).tolist())
    # check if R_lorrax = R_bgw or R_lorrax = R_bgw.T
    diff_eq = np.linalg.norm(R_lorrax[s_idx] - R_bgw[s_idx])
    diff_T = np.linalg.norm(R_lorrax[s_idx] - R_bgw[s_idx].T)
    print(f"  |R_lorrax - R_bgw|     = {diff_eq:.3e}")
    print(f"  |R_lorrax - R_bgw.T|   = {diff_T:.3e}")
    # also check det
    print(f"  det(R_lorrax) = {np.linalg.det(R_lorrax[s_idx]):.4f}  det(R_bgw) = {np.linalg.det(R_bgw[s_idx]):.4f}")
