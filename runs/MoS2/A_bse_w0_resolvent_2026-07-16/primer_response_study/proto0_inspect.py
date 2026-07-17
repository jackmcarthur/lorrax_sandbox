"""proto0 fixture inspection: dataset shapes + key scalars for every fixture."""
import h5py, numpy as np, sys

FIX = {
 "mos2_3x3_isdf": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/isdf_tensors_640.h5",
 "mos2_3x3_zeta": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5",
 "mos2_6x6_isdf": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/isdf_tensors_640.h5",
 "mos2_6x6_zeta": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/zeta_q.h5",
 "mos2_6x6_wfn":  "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/qe/nscf/WFN.h5",
 "mos2_3x3_wfn":  "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/WFN.h5",
 "mos2_4x4_isdf": "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/01_mos2_4x4_cohsex_gnppm/C_lorrax_gnppm_replicated_postproc/tmp/isdf_tensors_640.h5",
 "si_isdf":       "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/work_sym/tmp/isdf_tensors_792.h5",
}

for tag, path in FIX.items():
    print(f"===== {tag}: {path}")
    try:
        with h5py.File(path, "r") as f:
            def v(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if obj.size <= 12:
                        print(f"  {name:55s} {str(obj.shape):22s} {obj.dtype}  val={np.asarray(obj[()]).ravel()}")
                    else:
                        print(f"  {name:55s} {str(obj.shape):22s} {obj.dtype}")
            f.visititems(v)
    except Exception as e:
        print(f"  ERROR: {e}")
print("INSPECT DONE")
