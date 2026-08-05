import numpy as np, h5py
W = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect/work"
def load(d):
    with h5py.File(f"{W}/{d}/tmp/isdf_tensors_1600.h5","r") as f:
        return {k: np.asarray(f[k]) for k in ("V_qmunu","W0_qmunu","G0_mu_nu")}
def frob(a,b): return float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(a.ravel()),1e-300))
runs={"ridge6_1x1":"ridge6_1x1","ridge6_2x2":"ridge6_2x2","cusolv_4x4":"cur_1600_full","noridge_2x2":"fix_1600_2x2","noridge_1x1":"fix_1600_1x1"}
D={k:load(v) for k,v in runs.items()}
pairs=[("ridge6_1x1","ridge6_2x2"),   # MESH-INVARIANCE with ridge (KEY)
       ("ridge6_2x2","cusolv_4x4"),   # ridge-2x2 vs physical-4x4(noridge)
       ("ridge6_1x1","cusolv_4x4"),
       ("noridge_1x1","ridge6_1x1")]  # how much ridge perturbs on 1x1
for a,b in pairs:
    print(f"== {a} vs {b} ==  " + "  ".join(f"{ds}={frob(D[a][ds],D[b][ds]):.3e}" for ds in ("V_qmunu","W0_qmunu","G0_mu_nu")))
