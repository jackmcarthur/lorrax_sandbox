"""probe: (1) disk V_qmunu G=0 convention; (2) WFN <-> psi_full_y identity."""
import numpy as np
import prep
from prep import (load_fixture, load_wfn, grid_geometry, sphere_K, vdim_at,
                  make_Vq_generic, relF, wfn_u_on_grid)

fx = load_fixture("mos2_3x3")
grid_geometry(fx)

# (1) disk V convention: full v vs G=0-dropped vs 3D-v variants
for tag in ["slab_full", "slab_noG0", "bare3d_full", "bare3d_noG0"]:
    errs = []
    for q in range(fx.nq):
        Kf, n = sphere_K(fx, q)
        v, denom = vdim_at(fx, Kf)
        if tag.startswith("bare3d"):
            zero = denom < 1e-12
            v = np.where(zero, 0.0, 8.0 * np.pi / np.where(zero, 1, denom) / fx.celvol)
        if tag.endswith("noG0"):
            g0 = np.all(fx.gvec[q][:, :n] == 0, axis=0)
            v = v.copy(); v[:n][g0] = 0.0
        errs.append(relF(make_Vq_generic(fx.ZG[q], v, n), fx.Vdisk[q]))
    print(f"[V] {tag:>14}: med={np.median(errs):.3e} max={np.max(errs):.3e} "
          f"per-q={['%.1e' % e for e in errs]}")

# scalar-align check on the best variant (is it just a scale?)
q = 1
Kf, n = sphere_K(fx, q)
v, _ = vdim_at(fx, Kf)
g0 = np.all(fx.gvec[q][:, :n] == 0, axis=0)
v2 = v.copy(); v2[:n][g0] = 0.0
Vt = make_Vq_generic(fx.ZG[q], v2, n)
sc = (np.vdot(fx.Vdisk[q].ravel(), Vt.ravel()) / np.vdot(Vt.ravel(), Vt.ravel())).real
print(f"[V] slab_noG0 q=1: scalar-align factor={sc:.6f} relF after={relF(sc*Vt, fx.Vdisk[q]):.3e}")

# (2) WFN identity: qe/nscf/WFN.h5
wfn_path = f"{prep.BASE}/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5"
wfn = load_wfn(fx, wfn_path)
print(f"[psi] rk match: {np.max(np.abs(wfn['rk'] - fx.qfr)):.2e}")
u = wfn_u_on_grid(fx, wfn, 0, list(range(4)))[:, :, fx.rmu_flat]  # (4, ns, nmu)
p = fx.psi[0][:4]
for b in range(4):
    num = np.vdot(p[b], u[b])
    den = np.vdot(u[b], u[b])
    r = num / den
    res = relF(r * u[b], p[b])
    print(f"[psi] k=0 band {b}: best scalar r={np.abs(r):.6e} phase={np.angle(r):+.4f} "
          f"resid={res:.3e}  (norm_u={np.linalg.norm(u[b]):.4e} norm_psi={np.linalg.norm(p[b]):.4e})")
# global scalar across bands?
rs = []
for b in range(4):
    rs.append(np.vdot(u[b], p[b]) / np.vdot(u[b], u[b]))
print(f"[psi] per-band scalars: {['%.4e %+.3f' % (abs(x), np.angle(x)) for x in rs]}")
print("PROBE DONE")
