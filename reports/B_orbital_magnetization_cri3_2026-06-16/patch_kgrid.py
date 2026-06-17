import h5py, numpy as np
WFN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/qe/nscf_bandpath/WFN.h5"
with h5py.File(WFN,"r+") as f:
    kg = f["mf_header/kpoints/kgrid"]
    print("before:", kg[()])
    kg[...] = np.array([240,240,1], dtype=kg.dtype)
    print("after :", kg[()])
print("patched kgrid -> (240,240,1) so SymMaps builds the integer lookup; "
      "k-coords and wavefunctions untouched (band path has no true uniform grid).")
