"""probe2: per-row consistency of exact pair FFT F_p vs fitted rho~ = x_p.ZG.

Tests q0=0 (kmq = k, no wraps) vs q0=1 (wraps active) and the umklapp-phase
variant. Also raw-vs-fit per-row for a few (k,c,v)."""
import numpy as np
import prep
from prep import (load_fixture, load_wfn, grid_geometry, relF, wfn_u_on_grid,
                  sphere_K, vdim_at)

fx = load_fixture("mos2_3x3")
grid_geometry(fx)
wfn = load_wfn(fx, prep.FIX["mos2_3x3"]["wfn"])
occ = fx.nocc
vb = list(range(occ - 3, occ)); cb = list(range(occ, occ + 3))
u_cache = {k: wfn_u_on_grid(fx, wfn, k, vb + cb) for k in range(fx.nk)}
kg = np.array(fx.kgrid)

for q0 in [0, 1, 2]:
    n = int(fx.ngk[q0])
    fi = prep.flat_idx(fx, fx.gvec[q0])[:n]
    ph = np.exp(-2j * np.pi * (fx.rfrac @ fx.qwrap[q0]))
    rows_dev, rows_dev_ph = [], []
    for k in range(fx.nk):
        kmq = int(fx.kmq_idx[q0, k])
        # integer G0(k) = (k - q)_unwrapped - wrapped, in reciprocal units:
        # k3 - q3 (raw ints) vs wrapped kmq3
        d3raw = fx.iq3[k] - fx.iq3[q0]
        G0k = (d3raw - ((d3raw) % kg)) // kg  # integer division: unwrap - wrap
        uc = u_cache[kmq][3:]
        uv = u_cache[k][:3]
        rho = np.einsum("csm,vsm->cvm", np.conj(uc), uv).reshape(9, fx.n_rtot)
        # variant A: no umklapp phase (code cyclic convention)
        boxA = np.fft.fftn((rho * ph[None, :]).reshape(9, *fx.fg),
                           axes=(1, 2, 3), norm="backward").reshape(9, -1)[:, fi]
        # variant B: with e^{+2pi i G0.r} on the conduction conj leg
        phG0 = np.exp(2j * np.pi * (fx.rfrac @ G0k.astype(float)))
        boxB = np.fft.fftn((rho * (ph * phG0)[None, :]).reshape(9, *fx.fg),
                           axes=(1, 2, 3), norm="backward").reshape(9, -1)[:, fi]
        # fitted prediction: x rows at centroids . ZG
        A = fx.psi[kmq][cb]
        Bv = fx.psi[k][vb]
        x = np.einsum("csm,vsm->cvm", np.conj(A), Bv).reshape(9, fx.nmu)
        fit = x @ fx.ZG[q0][:, :n]
        for r in range(9):
            rows_dev.append(relF(boxA[r], fit[r]))
            rows_dev_ph.append(relF(boxB[r], fit[r]))
    print(f"q0={q0} iq3={fx.iq3[q0]}: F-vs-fit per-row relF  "
          f"noG0phase med={np.median(rows_dev):.3e} max={np.max(rows_dev):.3e} | "
          f"withG0 med={np.median(rows_dev_ph):.3e} max={np.max(rows_dev_ph):.3e}")
print("PROBE2 DONE")
