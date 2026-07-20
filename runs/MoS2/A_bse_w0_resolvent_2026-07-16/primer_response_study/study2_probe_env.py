"""study2_probe_env — environment + C_q-spectrum probe for the two owner
investigations (STUDY 1 principled LR basis; STUDY 2 cutoff audit).
READ-ONLY on fixtures and on sources/lorrax_A.  Reuses the validated
REFERENCE_arbitrary_q_vq loaders verbatim."""
import numpy as np

try:
    import scipy
    from scipy.special import jv, jn_zeros, jnp_zeros
    print("scipy", scipy.__version__, "jv/jn_zeros/jnp_zeros OK")
    print("  j0 zeros:", np.round(jn_zeros(0, 3), 4))
    print("  j0' zeros:", np.round(jnp_zeros(0, 3), 4))
except Exception as e:
    print("scipy FAIL:", repr(e))

import REFERENCE_arbitrary_q_vq as R

for name in ("MoS2_3x3", "MoS2_6x6"):
    fx = R.load_fixture(name)
    C_q = R.build_cq(fx)
    lams = []
    for q in range(fx["nq"]):
        lam = np.linalg.eigvalsh(0.5 * (C_q[q] + C_q[q].conj().T))
        lams.append(lam)
    lams = np.array(lams)                     # (nq, n_mu), ascending
    lmax = lams[:, -1]
    lmin = np.clip(lams[:, 0], 1e-300, None)
    cond = lmax / lmin
    print(f"{name}: n_mu={fx['n_mu']} nq={fx['nq']}  lam_max med "
          f"{np.median(lmax):.3e}  lam_min med {np.median(lmin):.2e}  "
          f"cond med {np.median(cond):.2e} max {np.max(cond):.2e}")
    print(f"    cond^-1/2 (crossover eps*) med {1/np.sqrt(np.median(cond)):.2e}")
    for eps in (1e-3, 1e-4, 1e-5, 1e-6):
        below = np.mean(np.sum(lams < (eps * lmax[:, None]), axis=1))
        print(f"    eps_rel={eps:.0e}: mean #eig below eps*lam_max = "
              f"{below:.1f} / {fx['n_mu']}")
