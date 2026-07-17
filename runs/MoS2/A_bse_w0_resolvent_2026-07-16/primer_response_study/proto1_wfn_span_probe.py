import sys
import numpy as np
import h5py
sys.path.insert(0, ".")
from proto1_prep import Fixture

fx = Fixture("MoS2_3x3")
base = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex"
cands = {
    "dir_WFN": f"{base}/05_lorrax_cohsex_native/WFN.h5",
    "qe_nscf_WFN": f"{base}/qe/nscf/WFN.h5",
    "WFN_qp": f"{base}/05_lorrax_cohsex_native/WFN_qp.h5",
}
for lbl, path in cands.items():
    try:
        fx.wfn_path = path
        fx._wfn_cache = None
        resids = []
        for k in (1, 4):
            ug = fx.u_grid(k)                      # ALL stored bands
            ugc = ug[:, :, fx.rmu_flat].reshape(ug.shape[0], -1)
            psc = fx.psi[k].reshape(fx.nb, -1)
            # project psi rows onto span(ug rows): least squares
            coef, *_ = np.linalg.lstsq(ugc.T, psc.T, rcond=None)
            resid = np.linalg.norm(ugc.T @ coef - psc.T) / np.linalg.norm(psc)
            resids.append(resid)
        print(f"{lbl:>12s}: span-projection resid k=1,4: "
              + " ".join(f"{r:.3e}" for r in resids))
    except Exception as e:
        print(f"{lbl:>12s}: {type(e).__name__}: {e}")
