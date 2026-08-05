"""Localize the bispinor TT Lorentz-unfold bug.

Replicates SymMaps.syms_crystal_to_cartesian + R_proper from the raw WFN.h5
and tests three things in isolation:
  (1) is R_proper ORTHOGONAL per op? (esp. the C3 ops) — pure numpy.
  (2) does the exact unfold_v_q_bispinor_lorentz einsum preserve the
      Lorentz trace on a synthetic symmetric tile, per op?
  (3) eigenvalue/trace of R^T V R vs R V R^T (transpose convention effect).
No GPU/JAX needed — the einsum is reproduced with numpy.
"""
import sys, numpy as np, h5py

WFN = sys.argv[1]
with h5py.File(WFN, 'r') as f:
    avec = f['/mf_header/crystal/avec'][()]          # (3,3) real-space lattice
    mtrx = f['/mf_header/symmetry/mtrx'][()]         # (48,3,3) int, BGW col form
    ntran = int(f['/mf_header/symmetry/ntran'][()])
mtrx = np.asarray(mtrx)[:ntran].astype(np.float64)
avec = np.asarray(avec, dtype=np.float64)
print(f"ntran={ntran}  avec=\n{np.round(avec,4)}")

# --- replicate syms_crystal_to_cartesian (symmetry_maps.py:1301-1310) ---
A_T = avec.T
A_T_inv = np.linalg.inv(A_T)
R_cart = np.einsum('ij,njk,kl->nil', A_T, mtrx, A_T_inv)
R_cart = np.around(R_cart, 10)
# R_proper: det-flip (symmetry_maps.py:1025-1027)
det = np.linalg.det(R_cart)
R_proper = np.where(det[:, None, None] < 0, -R_cart, R_cart)

print("\n op | det(R_cart) | order | |R_cart R^T - I|inf | |R_proper R^T - I|inf")
def order(R):
    M = R.copy();
    for k in range(1, 13):
        if np.allclose(M, np.eye(3), atol=1e-6): return k
        M = M @ R
    return -1
c3_ops = []
for s in range(ntran):
    Rc = R_cart[s]; Rp = R_proper[s]
    oc = abs(Rc.T @ Rc - np.eye(3)).max()
    op = abs(Rp.T @ Rp - np.eye(3)).max()
    o = order(Rc)
    tag = ""
    if abs(det[s]-1) < 1e-6 and o == 3:
        c3_ops.append(s); tag = "  <-- C3 (proper, order 3)"
    print(f" {s:2d} | {det[s]:+.3f} | {o:2d} | {oc:.2e} | {op:.2e}{tag}")

# --- (2)/(3) trace preservation of the exact einsum, per op -----------
# Synthetic symmetric in-plane-anisotropic Lorentz tile M[a,b] (3x3),
# standing in for the centroid-traced Sigma^B 3x3 block.
rng = np.random.default_rng(0)
M = rng.standard_normal((3, 3)); M = M + M.T          # symmetric, like Sigma^B block
print(f"\nsynthetic M (sym): tr={np.trace(M):.4f}")
print(" op | tr(R^T M R)[code einsum] | tr(R M R^T)[derivation] | both==tr(M)?")
for s in (c3_ops + [s for s in range(ntran) if s not in c3_ops]):
    R = R_proper[s]
    # CODE einsum: out_ij = sum_ab R[a,i] R[b,j] M[a,b]  ==  R^T M R
    out_code = np.einsum('ai,bj,ab->ij', R, R, M)
    # DERIVATION text: out_ij = sum_ab R[i,a] R[j,b] M[a,b]  ==  R M R^T
    out_deriv = np.einsum('ia,jb,ab->ij', R, R, M)
    tc, td = np.trace(out_code), np.trace(out_deriv)
    print(f" {s:2d} | {tc:+.5f} | {td:+.5f} | {abs(tc-np.trace(M))<1e-9 and abs(td-np.trace(M))<1e-9}")

print("\nIf |R_proper R^T - I|inf is large for the C3 ops => R NOT orthogonal "
      "=> trace broken => bug is R_proper (cartesian convention). "
      "If orthogonal AND einsum preserves trace => bug is downstream "
      "(scalar-unfold feed / sym_idx pairing), not this helper.")
