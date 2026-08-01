"""Print band-count metadata for one or more BGW WFN.h5 files.

Read-only. Copy of /scratch2/08271/jackmc/mos2_4x4_test/probe_wfn_bands.py
(the b300 band-count evidence tool) so the fastloop deck build is
self-contained. The login node's h5tools are HDF5 1.8 and cannot open
these 1.14 files — run this in-container.
"""
import sys

import h5py


def main():
    for path in sys.argv[1:]:
        try:
            with h5py.File(path, "r") as f:
                kp = f["mf_header/kpoints"]
                out = {k: kp[k][()] for k in
                       ("mnband", "nspin", "nspinor", "nrk", "nelec")
                       if k in kp}
                print(f"{path}")
                for k, v in out.items():
                    print(f"    {k:10s} = {v}")
                if "el" in kp:
                    print(f"    el.shape   = {kp['el'].shape}")
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            print(f"{path}: FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
