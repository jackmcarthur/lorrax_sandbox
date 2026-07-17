import sys
import numpy as np
sys.path.insert(0, ".")
from proto1_prep import Fixture, relF
fx = Fixture("MoS2_3x3")
for q in range(fx.nq):
    Vd = fx.make_Vq(fx.ZG[q], q)
    _, K2, n = fx.Kvecs(q)
    print(f"q={q} qfr={np.round(fx.qfr[q],3)} relF={relF(Vd, fx.Vqmunu[q]):.3e} "
          f"ngk={n} nK2>30={int(np.sum(K2 > 30.0))} maxK2={K2.max():.3f}")
# with hard vcoul cutoff at 30 Ry
print("with vcoul cutoff 30 Ry:")
import types
for q in range(fx.nq):
    v, n = fx.vq(q)
    _, K2, _ = fx.Kvecs(q)
    v = np.where(K2 > 30.0, 0.0, v)
    A = fx.ZG[q][:, :n] * np.sqrt(v[:n])[None, :]
    Vd = np.conj(A) @ A.T
    print(f"q={q} relF={relF(Vd, fx.Vqmunu[q]):.3e}")
