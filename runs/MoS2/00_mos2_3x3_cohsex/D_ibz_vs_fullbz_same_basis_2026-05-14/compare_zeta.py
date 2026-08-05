"""Compare ζ_q tensors between Run A (IBZ) and Run B (full BZ).

Run A has 5 q-slots on disk (IBZ q's only).
Run B has 9 q-slots on disk (full BZ).

We need to identify which Run-B slots correspond to the same q_irr as
Run A's 5 slots, then compare ζ at those slots.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
jax.config.update("jax_enable_x64", True)
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "run_A_ibz", "tmp", "zeta_q.h5")
B = os.path.join(HERE, "run_B_fullbz", "tmp", "zeta_q.h5")

print(f"Run A zeta: {A}")
print(f"Run B zeta: {B}")

with h5py.File(A, "r") as fA, h5py.File(B, "r") as fB:
    def keys(f):
        out = []
        f.visit(lambda n: out.append(n))
        return out
    print("\nA datasets:")
    for k in keys(fA):
        try:
            shp = fA[k].shape
            dt = fA[k].dtype
            print(f"  {k}: shape={shp}, dtype={dt}")
        except Exception:
            print(f"  {k}: group")
    print("\nB datasets:")
    for k in keys(fB):
        try:
            shp = fB[k].shape
            dt = fB[k].dtype
            print(f"  {k}: shape={shp}, dtype={dt}")
        except Exception:
            print(f"  {k}: group")
