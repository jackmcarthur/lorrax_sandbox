import numpy as np, h5py
W = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect/work"
def load(d):
    with h5py.File(f"{W}/{d}/tmp/isdf_tensors_1600.h5", "r") as f:
        return {k: np.asarray(f[k]) for k in ("V_qmunu", "W0_qmunu", "G0_mu_nu", "whead", "vhead")}
def frob(a, b):
    return float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(a.ravel()), 1e-300))
runs = ["fix_1600_1x1", "fix_1600_2x2", "base_1600_full_2x2", "cur_1600_full"]
lab = {"fix_1600_1x1":"fix_1x1","fix_1600_2x2":"fix_2x2","base_1600_full_2x2":"cusolv_2x2","cur_1600_full":"cusolv_4x4"}
data = {r: load(r) for r in runs}
pairs = [("fix_1600_1x1","fix_1600_2x2"),        # dense mesh-invariance (1x1 vs 2x2)
         ("fix_1600_2x2","base_1600_full_2x2"),  # dense vs cusolvermp @ 2x2
         ("fix_1600_1x1","cur_1600_full"),       # dense-1x1 vs cusolvermp-4x4 (physical ref)
         ("base_1600_full_2x2","cur_1600_full"), # ORIGINAL drift 2x2 vs 4x4
         ("fix_1600_2x2","cur_1600_full")]       # dense-2x2 vs cusolvermp-4x4
for dA, dB in pairs:
    print(f"\n== {lab[dA]} vs {lab[dB]} ==")
    for ds in ("V_qmunu","W0_qmunu","G0_mu_nu"):
        print(f"   {ds:10s} frob-rel={frob(data[dA][ds], data[dB][ds]):.4e}")
    print(f"   vhead {lab[dA]}={complex(data[dA]['vhead']):.6g}  {lab[dB]}={complex(data[dB]['vhead']):.6g}")
