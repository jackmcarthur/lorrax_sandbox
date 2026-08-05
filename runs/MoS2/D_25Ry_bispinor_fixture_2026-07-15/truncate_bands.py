"""Truncate a BGW WFN.h5 to the lowest NB_KEEP bands (fixture-generation tool).

Same procedure as the 2026-07-02 bispinor fixture (82->34): band axis is the
leading dim of wfns/coeffs and the trailing dim of kpoints/el and kpoints/occ.
Everything else copies verbatim.  Usage:
    python3 truncate_bands.py WFN.h5 WFN_34b.h5 34
"""
import sys

import h5py

src_path, dst_path, nb_keep = sys.argv[1], sys.argv[2], int(sys.argv[3])

with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if name == "wfns/coeffs":
            data = obj[:nb_keep]
        elif name in ("mf_header/kpoints/el", "mf_header/kpoints/occ"):
            data = obj[..., :nb_keep]
        elif name == "mf_header/kpoints/mnband":
            data = nb_keep
        else:
            data = obj[()]
        dst.create_dataset(name, data=data)

    src.visititems(visit)

with h5py.File(dst_path, "r") as f:
    nb, ns, ngk = f["wfns/coeffs"].shape[:3]
    assert nb == nb_keep and int(f["mf_header/kpoints/mnband"][()]) == nb_keep
    print(f"OK: {dst_path} coeffs ({nb},{ns},{ngk},...) el/occ trailing dim {nb_keep}")
