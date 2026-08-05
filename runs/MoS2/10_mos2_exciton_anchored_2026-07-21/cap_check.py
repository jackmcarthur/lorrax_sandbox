import os, numpy as np, jax
from jax.sharding import Mesh
from isdf import core
m = Mesh(np.array(jax.devices()[:1]).reshape(1,1), ('x','y'))
need = 74*2412**2*16/1024**3
print(f"IBZ stack (74, 2412) = {need:.2f} GiB; default cap = {core._REPLICATED_CHOL_MAX_STACK_BYTES/1024**3:.2f} GiB")
try:
    k = core._resolve_solver_kind_charge(m,'auto',n_rmu=2412,nq=74,charge_zeta_solve='rank_truncate')
    print("SILENT DOWNGRADE (bad):", k)
except ValueError as e:
    print("RAISES as intended ->", str(e)[:120].replace("\n"," "))
print("cholesky still allowed ->",
      core._resolve_solver_kind_charge(m,'auto',n_rmu=2412,nq=74,charge_zeta_solve='cholesky'))
