import sys
import numpy as np
sys.path.insert(0, ".")
from proto1_prep import Fixture, relF
fx = Fixture("MoS2_3x3")
for k in range(fx.nk):
    ug = fx.u_grid(k, nbmax=fx.nb)[:, :, fx.rmu_flat]
    # best per-band scalar (phase+scale) alignment
    num = np.einsum("nsm,nsm->n", np.conj(ug), fx.psi[k])
    den = np.einsum("nsm,nsm->n", np.conj(ug), ug)
    sc = num / den
    resid = np.linalg.norm(sc[:, None, None] * ug - fx.psi[k]) / np.linalg.norm(fx.psi[k])
    # one global scalar
    scg = np.vdot(ug, fx.psi[k]) / np.vdot(ug, ug)
    residg = np.linalg.norm(scg * ug - fx.psi[k]) / np.linalg.norm(fx.psi[k])
    print(f"k={k} global-scalar resid={residg:.3e}  per-band-scalar resid={resid:.3e} "
          f"|sc| spread=[{np.abs(sc).min():.3e},{np.abs(sc).max():.3e}]")
