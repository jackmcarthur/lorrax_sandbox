"""Mini-BZ head averaging validation — checks (a) bit-identity, (b) point-vs-
cell-average vs §16.4b, (c) seed-independence.  Host + 1 GPU.  Fixture: MoS2 6×6
(the arbitrary_q_bse.md §16 audit grid).  Every number is grep-tagged [RESULT]."""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from bse import vq_interp as vqi
from gw.coulomb import base as cb

FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/"
      "interp_study/mos2_6x6/lorrax/tmp")
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/minibz_head_averaging_2026-07-20"

zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
bvec = np.asarray(zx["bvec"], dtype=np.float64)     # cartesian recip (rows)
celvol = float(zx["celvol"])
kgrid = tuple(int(s) for s in zx["kgrid"])
zc = float(np.pi / bvec[2, 2])
print(f"[INFO] MoS2 fixture kgrid={kgrid} celvol={celvol:.3f} zc={zc:.4f} "
      f"|b1|={np.linalg.norm(bvec[0]):.4f}")

results = {}

# ---------------------------------------------------------------------------
# (b) POINT value vs mini-BZ CELL AVERAGE — reproduce arbitrary_q_bse.md §16.4b.
#   At Q=(1/N,0,0), G*=0, average the bare slab v over the N×N mini-BZ cell.
#   Reference ratios point/avg: 0.873@3, 0.908@6, 0.936@12, 0.951@24, 0.958@48.
# ---------------------------------------------------------------------------
print("\n=== (b) point-vs-cell-average (slab head), §16.4b reproduction ===")
ref_ratio = {3: 0.873, 6: 0.908, 12: 0.936, 24: 0.951, 48: 0.958}
b_rows = []
for N in (3, 6, 12, 24, 48):
    Qf = np.array([1.0 / N, 0.0, 0.0])
    shift = Qf @ bvec                                   # cartesian Q+G* (G*=0)
    qn = float(np.linalg.norm(shift))
    v_point, _ = cb._minibz_kernel_bare(shift, np.zeros((1, 3)), kind="slab", zc=zc)
    v_point = float(v_point[0])
    dq = cb.minibz_voronoi_batches(bvec, (N, N, 1), nsamples=2**18,
                                   qmc_reps=8, nmax=3, is_2d=True)
    q0sph2 = cb.minibz_inscribed_sphere_r2(bvec, (N, N, 1), is_2d=True)
    v_avg = cb.minibz_average(shift, dq, kind="slab", celvol=celvol,
                              n_kpts=N * N, q0sph2=q0sph2, zc=zc,
                              analytic_sphere=False, adaptive=True)
    ratio = v_point / v_avg
    err = abs(ratio - ref_ratio[N])
    tag = "OK" if err < 0.02 else "**CHECK**"
    print(f"[RESULT] (b) N={N:>2d} |Q+G*|={qn:.3e} v_point={v_point:.4g} "
          f"v_cellavg={v_avg:.4g} point/avg={ratio:.3f} (ref {ref_ratio[N]:.3f}) "
          f"Δ={err:.3f} {tag}")
    b_rows.append((N, qn, v_point, v_avg, ratio, ref_ratio[N]))
results["b_point_vs_avg"] = np.array(b_rows)

# ---------------------------------------------------------------------------
# (c) SEED-INDEPENDENCE.
#   3D bulk: analytic-sphere head stable across seeds; pure-Sobol drifts.
#   2D slab_lr head (the eval_vq head): stable across seeds.
# ---------------------------------------------------------------------------
print("\n=== (c) seed-independence ===")
q0sph2_3d = cb.minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=False)
nk = int(np.prod(kgrid))
def head3d(seed, analytic):
    dq = cb.minibz_voronoi_batches(bvec, kgrid, nsamples=2**16, qmc_reps=1,
                                   nmax=3 if analytic else 1, is_2d=False,
                                   seed_offset=seed)
    return cb.minibz_average(np.zeros(3), dq, kind="bulk_3d", celvol=celvol,
                             n_kpts=nk, q0sph2=q0sph2_3d,
                             analytic_sphere=analytic,
                             adaptive=not analytic)
a0, a1 = head3d(0, True), head3d(777, True)
p0, p1 = head3d(0, False), head3d(777, False)
an_spread = abs(a0 - a1) / abs(0.5 * (a0 + a1))
pu_spread = abs(p0 - p1) / abs(0.5 * (p0 + p1))
print(f"[RESULT] (c) 3D analytic-sphere head  seed0={a0:.5g} seed777={a1:.5g} "
      f"rel-spread={an_spread:.2e}  {'OK' if an_spread < 5e-3 else '**CHECK**'}")
print(f"[RESULT] (c) 3D pure-Sobol head       seed0={p0:.5g} seed777={p1:.5g} "
      f"rel-spread={pu_spread:.2e}  (baseline: drifts)")
print(f"[RESULT] (c) 3D analytic/pure spread ratio = {an_spread/max(pu_spread,1e-30):.3e} "
      f"({'analytic MORE stable' if an_spread < pu_spread else '**CHECK**'})")

# 2D slab_lr head (eval_vq head) at a near-Γ Q — two seeds
alpha = vqi.ALPHA
def head2d_lr(seed):
    Qf = np.array([1.0 / kgrid[0], 0.0, 0.0])
    shift = Qf @ bvec
    dq = cb.minibz_voronoi_batches(bvec, kgrid, nsamples=2**16, qmc_reps=1,
                                   nmax=3, is_2d=True, seed_offset=seed)
    q0s = cb.minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=True)
    return cb.minibz_average(shift, dq, kind="slab_lr", celvol=celvol, n_kpts=nk,
                             q0sph2=q0s, alpha=alpha, zc=zc,
                             analytic_sphere=False, adaptive=True)
s0, s1 = head2d_lr(0), head2d_lr(777)
sp = abs(s0 - s1) / abs(0.5 * (s0 + s1))
print(f"[RESULT] (c) 2D slab_lr head          seed0={s0:.5g} seed777={s1:.5g} "
      f"rel-spread={sp:.2e}  {'OK' if sp < 1e-2 else '**CHECK**'}")
results["c_seed"] = np.array([a0, a1, p0, p1, s0, s1])

# ---------------------------------------------------------------------------
# (a) BIT-IDENTITY: eval_vq(flag=OFF) == eval_vq(flag=ON, sentinel gstar=-1).
#     And ON with a real per-Q head differs (injection active + rank-1 on G*).
# ---------------------------------------------------------------------------
print("\n=== (a) eval_vq bit-identity (flag OFF vs ON-sentinel) + injection ===")
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
C_q = vqi.build_cq(zx)
with mesh:
    prep = vqi.prepare_coarse(zx, C_q, mesh)
des = vqi.lr_design_blocks(zx, prep)
coeffs = vqi.fit_lr_model(des)
cpk = vqi.pack_coeffs(des, coeffs)
pinvF = jnp.asarray(vqi.stencil_pinv(zx["qfr"], vqi.stencil_r7(zx)))
Qtest = np.array([1.0 / kgrid[0], 0.0, 0.0])            # near-Γ test Q

with mesh:
    ev_off = vqi.make_eval_vq(zx, prep, des, mesh, None, head_minibz_average=False)
    ev_on = vqi.make_eval_vq(zx, prep, des, mesh, None, head_minibz_average=True)
    V_off = np.asarray(jax.device_get(ev_off(jnp.asarray(Qtest), prep["V_SRc"], pinvF, cpk)))
    # ON with sentinel (gstar=-1, head=0) must EXACTLY reproduce OFF
    V_on_noop = np.asarray(jax.device_get(ev_on(
        jnp.asarray(Qtest), prep["V_SRc"], pinvF, cpk,
        jnp.asarray(0.0, jnp.float64), jnp.asarray(-1, jnp.int32))))
    # ON with the real mini-BZ head
    gstar, head_val = vqi.minibz_head_vlr(zx, prep, Qtest, alpha=alpha)
    V_on = np.asarray(jax.device_get(ev_on(
        jnp.asarray(Qtest), prep["V_SRc"], pinvF, cpk,
        jnp.asarray(head_val, jnp.float64), jnp.asarray(gstar, jnp.int32))))

max_abs_noop = float(np.max(np.abs(V_on_noop - V_off)))
n_calls = ev_on._cache_size()
print(f"[RESULT] (a) OFF vs ON-sentinel max|ΔV| = {max_abs_noop:.3e}  "
      f"{'BIT-IDENTICAL' if max_abs_noop == 0.0 else ('OK<1e-15' if max_abs_noop < 1e-15 else '**FAIL**')}")
# point value that was replaced (bare slab_lr / celvol at Q+G*)
Kstar = bvec.T @ (Qtest + np.asarray(prep["GS"], float)[:, gstar])
K2star = float(Kstar @ Kstar)
f2ds = 1.0 - np.exp(-zc*np.hypot(Kstar[0], Kstar[1]))*np.cos(Kstar[2]*zc)
v_point_star = 8*np.pi/K2star*f2ds/celvol*np.exp(-K2star/(4*alpha**2))
print(f"[RESULT] (a) gstar={gstar} |Q+G*|²={K2star:.4e} "
      f"v_LR point={v_point_star:.5g} <v_LR>_mBZ={head_val:.5g} "
      f"avg/point={head_val/v_point_star:.3f}")
diff_on = float(np.max(np.abs(V_on - V_off)))
print(f"[RESULT] (a) ON(real head) vs OFF max|ΔV| = {diff_on:.3e} "
      f"({'injection ACTIVE' if diff_on > 0 else '**no change**'})")
print(f"[RESULT] (a) eval_vq compile count (ON) = {n_calls} (expect 1)")
results["a_bit"] = np.array([max_abs_noop, diff_on, gstar, head_val,
                             v_point_star, float(n_calls)])

np.savez(f"{OUT}/validate_minibz.npz", **results)
print(f"\n[INFO] saved {OUT}/validate_minibz.npz")
print("[INFO] DONE")
