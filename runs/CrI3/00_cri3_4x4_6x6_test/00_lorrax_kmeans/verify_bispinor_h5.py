"""Sanity-check the bispinor_isdf.h5 file: structure, shapes, magnitudes."""

from __future__ import annotations
import numpy as np
import h5py
from pathlib import Path

H5_PATH = Path(__file__).resolve().parent / "bispinor_isdf.h5"


def main():
    print(f"Opening {H5_PATH} (size {H5_PATH.stat().st_size/1e9:.2f} GB)\n")
    with h5py.File(H5_PATH, "r") as f:
        print("=== top-level keys ===")
        for k in f.keys():
            print(f"  /{k}  ({type(f[k]).__name__})")

        print("\n=== meta ===")
        for k, v in f["meta"].attrs.items():
            print(f"  {k}: {v}")

        print("\n=== centroid index sets ===")
        for name in ("scalar_centroid_indices", "current_centroid_indices",
                     "union_r_indices"):
            ds = f[name]
            print(f"  /{name}: shape={ds.shape}  dtype={ds.dtype}  "
                  f"min={ds[:].min()}  max={ds[:].max()}")

        print("\n=== zeta_lorentz/* ===")
        for mu_L in (0, 1, 2, 3):
            ds = f[f"zeta_lorentz/zeta_{mu_L}"]
            data = ds[:]
            print(f"  zeta_{mu_L}: shape={data.shape}  dtype={data.dtype}  "
                  f"|z|max={np.abs(data).max():.3e}  "
                  f"|z|mean={np.abs(data).mean():.3e}  "
                  f"centroid_set='{ds.attrs.get('centroid_set','?')}'")

        print("\n=== V_lorentz/* ===")
        v_grp = f["V_lorentz"]
        keys_sorted = sorted(v_grp.keys(),
                             key=lambda s: (int(s.split('_')[1]), int(s.split('_')[2])))
        for key in keys_sorted:
            data = v_grp[key][:]
            herm_dev = float(np.max(np.abs(
                data - np.conj(data.transpose(0, 2, 1)))))
            mu_L = int(key.split('_')[1])
            nu_L = int(key.split('_')[2])
            herm_str = "(N/A — off-diagonal)"
            if mu_L == nu_L:
                herm_str = f"|V−V†|max(per-q)={herm_dev:.3e}"
            print(f"  {key}: shape={data.shape}  dtype={data.dtype}  "
                  f"|V|max={np.abs(data).max():.3e}  "
                  f"|V|mean={np.abs(data).mean():.3e}  "
                  f"{herm_str}")

        # Cross-check V^{ij}(q) = (V^{ji}(q))^T  (P^T is symmetric in ij, so
        # V^{i,j}_{μν} = Σ_G ζ^{i,*}_μ P^T_{ij} v ζ^j_ν = (Σ_G ζ^{j,*}_ν P^T_{ji} v ζ^i_μ)^*
        #               = (V^{j,i}_{νμ})^*  →  V^{ij}(q) = V^{ji}(q)^H
        v12 = f["V_lorentz/V_1_2"][:]
        v21 = f["V_lorentz/V_2_1"][:]
        herm12 = float(np.max(np.abs(v12 - np.conj(v21.transpose(0, 2, 1)))))
        print(f"\n  |V^12 − (V^21)^H|max = {herm12:.3e}  "
              f"(should be ~0)")


if __name__ == "__main__":
    main()
