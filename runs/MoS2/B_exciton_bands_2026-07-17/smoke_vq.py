"""smoke_vq — validate the production vq_interp port on the MoS2 3x3
640-centroid fixture (the reference impl's pinned fixture).

Stages: load -> gates -> prepare_coarse (device) -> b26p fit -> nulls ->
(a) LOO ladder with eval_vq_host, thresholds = the reference e2e smoke
    (B med<=2.2e-2, max<=5.5e-2; exc med<=1.0/max<=4.0 meV) — measured
    reference baseline B 1.409e-2/3.553e-2, exc 0.642/2.542 meV;
(b) jitted eval parity vs host at on-grid + off-grid Q (<=1e-9);
(c) compile census: one compile serves all Q; sharding P('x','y').
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from bse import vq_interp as vqi

FX = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp"
THRESH = {"B_med": 2.2e-2, "B_max": 5.5e-2, "exc_med": 1.00, "exc_max": 4.00}

t0 = time.time()
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
C_q = vqi.build_cq(zx)
vqi.run_gates(zx, C_q)
with mesh:
    prep = vqi.prepare_coarse(zx, C_q, mesh)
des = vqi.lr_design_blocks(zx, prep)
coeffs = vqi.fit_lr_model(des)
vqi.run_nulls(zx, prep, des, coeffs)

# ---- (a) LOO ladder (honest refits) with the host evaluator ----
Bs, excs = [], []
for q0 in range(zx["nq"]):
    train = [q for q in range(zx["nq"]) if q != q0]
    C_loo = vqi.fit_lr_model(des, exclude=q0)
    Vp = vqi.eval_vq_host(zx, prep, des, C_loo, zx["qfr"][q0], train=train)
    x = vqi.gap_window_pairs(zx, q0)
    B_true = vqi.b_block(x, vqi.make_vq(zx, zx["ZG"][q0], q0))
    Bp = vqi.b_block(x, Vp)
    Bs.append(vqi.relF(Bp, B_true))
    D, Hdir = vqi.build_hdir(zx, q0)
    dev = np.abs(vqi.exciton_evs(zx, D, Hdir, Bp)
                 - vqi.exciton_evs(zx, D, Hdir, B_true))
    excs.append(float(np.max(dev) * vqi.RY2MEV))
    print(f"  q0={q0}: B={Bs[-1]:.3e}  exc={excs[-1]:.3f} meV", flush=True)

got = {"B_med": float(np.median(Bs)), "B_max": float(np.max(Bs)),
       "exc_med": float(np.median(excs)), "exc_max": float(np.max(excs))}
ok = True
print("\n  ===== LOO thresholds (reference e2e smoke) =====")
for k, thr in THRESH.items():
    good = got[k] <= thr
    ok &= good
    print(f"    {k:<8s} got {got[k]:.3e}  <=  {thr:.3e}  "
          f"{'OK' if good else '** FAIL **'}")

# ---- (b) jit-vs-host parity + (c) census/sharding ----
with mesh:
    eval_jit = vqi.make_eval_vq(zx, prep, des, mesh, None)
    pinvF = jnp.asarray(vqi.stencil_pinv(zx["qfr"], vqi.stencil_r7(zx)))
    cpk = vqi.pack_coeffs(des, coeffs)
    tests = [zx["qfr"][1], np.array([0.11, 0.07, 0.0]),
             np.array([-0.23, 0.4, 0.0]), np.array([1.0 / 6.0, 0.0, 0.0])]
    for i, qf in enumerate(tests):
        Vj = eval_jit(jnp.asarray(qf), prep["V_SRc"], pinvF, cpk)
        if i == 0:
            assert Vj.sharding.spec == P("x", "y"), \
                f"tile sharding {Vj.sharding.spec} != P('x','y')"
        Vh = vqi.eval_vq_host(zx, prep, des, coeffs, qf)
        r = vqi.relF(np.asarray(jax.device_get(Vj)), Vh)
        print(f"    [jit] Q={np.array2string(np.asarray(qf), precision=4)} "
              f"jit-vs-host relF = {r:.3e}")
        ok &= r <= 1e-9
    ncomp = eval_jit._cache_size()
    print(f"    [census] eval_vq jit cache size after {len(tests)} Q: {ncomp}")
    ok &= ncomp == 1

print(f"\n[smoke_vq] {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
sys.exit(0 if ok else 1)
