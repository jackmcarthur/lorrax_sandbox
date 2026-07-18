"""Per-block timing of vq_interp.run_gates internals (finding the ~13.5 s)."""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from bse import vq_interp as vqi

FX12 = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_bands_perf_2026-07-18/12x12/tmp"

t0 = time.time()
zx = vqi.load_zeta_coarse(f"{FX12}/isdf_tensors_640.h5", f"{FX12}/zeta_q.h5")
print(f"load {time.time()-t0:.2f}s", flush=True)
t0 = time.time(); C_q = vqi.build_cq(zx); print(f"build_cq {time.time()-t0:.2f}s", flush=True)

t0 = time.time()
n0 = int(zx["ngk"][0])
zt = vqi.to_sphere(zx, vqi.recon(zx, 0), 0)
r = vqi.relF(zt[:, :n0], zx["ZG"][0][:, :n0])
print(f"recon_roundtrip {time.time()-t0:.2f}s ({r:.3e})", flush=True)

t0 = time.time()
k2max = max(np.max(np.sum((zx["bvec"].T @ (zx["qfr"][q][:, None]
                           + zx["gvec"][q][:, :int(zx["ngk"][q])]
                           .astype(np.float64))) ** 2, axis=0))
            for q in range(zx["nq"]))
print(f"k2max {time.time()-t0:.2f}s", flush=True)

t0 = time.time(); vp = vqi.v_sphere_padded(zx); print(f"v_sphere_padded {time.time()-t0:.2f}s", flush=True)
t0 = time.time(); vd = vqi._batched_vq_relF(zx["ZG"], vp, zx["Vqmunu"])
print(f"batched_vq_relF {time.time()-t0:.2f}s (max {np.max(vd):.3e})", flush=True)

t0 = time.time()
q = 0
X = np.empty((zx["nk"], zx["nb"], zx["nb"], zx["n_mu"]), dtype=np.complex128)
for k in range(zx["nk"]):
    kq = vqi.kq_index(zx, k, q)
    X[k] = np.einsum("nsm,Msm->nMm", np.conj(zx["psi"][kq]), zx["psi"][k])
t1 = time.time()
print(f"XHX einsum loop {t1-t0:.2f}s", flush=True)
X = X.reshape(-1, zx["n_mu"])
G = np.conj(X.T) @ X
print(f"XHX gemm {time.time()-t1:.2f}s (relF {vqi.relF(G, C_q[0]):.3e})", flush=True)

t0 = time.time()
for q in range(2):
    v, n = vqi.v_sphere(zx, q)
    vs, _ = vqi.v_sphere(zx, q, kind="slab_sr", alpha=0.63)
    vl, _ = vqi.v_sphere(zx, q, kind="slab_lr", alpha=0.63)
print(f"split gates {time.time()-t0:.2f}s", flush=True)
