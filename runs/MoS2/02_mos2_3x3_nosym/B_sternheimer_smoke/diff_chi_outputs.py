"""Compare run_sternheimer outputs A vs B element-wise.

Surfaces |Δ|_∞ and rel-Δ_∞ for chi, dchi/dq, and S-tensor at every q.
Also prints a worst-offender summary so we know if the JAX Hankel
plumbing leaked any error into the JVP path.
"""
from __future__ import annotations
import sys
import h5py
import numpy as np

A = h5py.File(sys.argv[1] if len(sys.argv) > 1 else 'stern_new3.h5', 'r')
B = h5py.File(sys.argv[2] if len(sys.argv) > 2 else 'stern_jaxhankel.h5', 'r')

def mx(x): return float(np.max(np.abs(x)))

print(f"\nA = {A.filename}")
print(f"B = {B.filename}")
print()

# χ + ∂χ/∂q per q
worst_chi_abs = worst_chi_rel = 0.0
worst_dchi_abs = worst_dchi_rel = 0.0
print("  q     │   |Δ chi|        rel       │   |Δ ∂chi/∂q|     rel")
print("  ──────┼─────────────────────────────┼─────────────────────────────")
nq = sum(1 for k in A.keys() if k.startswith('q_') and k[2:].isdigit())
for iq in range(nq):
    g = f'q_{iq}'
    cA = np.asarray(A[g]['chi_col'])
    cB = np.asarray(B[g]['chi_col'])
    dA = np.asarray(A[g]['dchi_col_dq'])
    dB = np.asarray(B[g]['dchi_col_dq'])

    da = mx(cA - cB);  ra = da / max(mx(cA), 1e-300)
    da2 = mx(dA - dB); ra2 = da2 / max(mx(dA), 1e-300)
    worst_chi_abs = max(worst_chi_abs, da); worst_chi_rel = max(worst_chi_rel, ra)
    worst_dchi_abs = max(worst_dchi_abs, da2); worst_dchi_rel = max(worst_dchi_rel, ra2)
    print(f"  {iq}    │   {da:8.2e}     {ra:8.2e}   │   {da2:8.2e}      {ra2:8.2e}")

# S-tensor
S_A = np.asarray(A['s_tensor_q0'])
S_B = np.asarray(B['s_tensor_q0'])
ds = mx(S_A - S_B)
rs = ds / max(mx(S_A), 1e-300)

print()
print(f"  S-tensor: |Δ|_∞ = {ds:.2e}   rel = {rs:.2e}")
print()
print(f"  Worst chi:    abs = {worst_chi_abs:.2e}   rel = {worst_chi_rel:.2e}")
print(f"  Worst ∂chi:   abs = {worst_dchi_abs:.2e}   rel = {worst_dchi_rel:.2e}")
print(f"  Worst S:      abs = {ds:.2e}   rel = {rs:.2e}")

# Detailed dchi at one q for visualisation
print(f"\n  Sample ∂χ/∂q at q[4] (should be (1, 3) shape):")
print(f"    A: {np.asarray(A['q_4']['dchi_col_dq']).flatten()}")
print(f"    B: {np.asarray(B['q_4']['dchi_col_dq']).flatten()}")
