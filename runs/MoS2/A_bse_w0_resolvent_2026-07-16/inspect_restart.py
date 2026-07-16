import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
keys = ["V_qmunu", "V_qmunu_nohead", "W0_qmunu", "W0_qmunu_nohead",
        "psi_full_y", "enk_full", "kgrid", "G0_mu_nu", "vhead", "whead"]
for k in keys:
    if k in f:
        d = f[k]
        extra = ""
        if hasattr(d, "attrs") and "W0_ready" in d.attrs:
            extra = f"  W0_ready={bool(d.attrs['W0_ready'])}"
        try:
            shp = d.shape
        except Exception:
            shp = "scalar"
        print(f"{k:22s} shape={shp} dtype={d.dtype}{extra}")
    else:
        print(f"{k:22s} ABSENT")
# n_occ from enk_full
enk = np.asarray(f["enk_full"][:])
print("enk_full nk,nb =", enk.shape)
f.close()
