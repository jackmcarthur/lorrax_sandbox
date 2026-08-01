"""Parity gate for the 2026-07-31 template validation smoke.

Compares the fresh run against the pinned baseline run_800c_merged
(archived 2026-07-30).  Tolerance 1e-8 eV on every numeric column: the
FFT-FFI certification measured 2.5e-14 eV h5 parity for the backend swap,
so 1e-8 is ~5-6 orders above measured numerical noise and 5 orders below
physical significance (~1e-3 eV); bitwise equality is NOT expected because
the FFI FFT/GEMM backends reorder floating-point reductions.

Usage: python compare_valsmoke.py BASELINE_DIR NEW_DIR
Exit 0 = all files within tolerance; 1 = any mismatch (printed).
"""
import re
import sys

import numpy as np

TOL = 1e-8
DAT_FILES = ["eqp0.dat", "eqp1.dat", "eqp_g0w0.dat", "sigma_diag.dat"]
H5_FILE = "sigma_mnk.h5"


_NUM = re.compile(r"[-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?")

def load_numbers(path):
    # Rows may be whitespace-separated floats OR key=value tokens
    # (eqp_g0w0.dat, sigma_diag.dat: "n=0  E_DFT= -66.16 ...").
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            vals = [float(t.replace("D", "e").replace("d", "e"))
                    for t in _NUM.findall(line)]
            if vals:
                rows.append(vals)
    return rows


def main(base, new):
    failed = False
    for f in DAT_FILES:
        try:
            a = load_numbers(f"{base}/{f}")
            b = load_numbers(f"{new}/{f}")
        except OSError as e:
            print(f"[compare] {f}: MISSING ({e})")
            failed = True
            continue
        if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
            print(f"[compare] {f}: SHAPE MISMATCH rows {len(a)} vs {len(b)}")
            failed = True
            continue
        d = max((max(abs(x - y) for x, y in zip(ra, rb)) if ra else 0.0)
                for ra, rb in zip(a, b))
        verdict = "OK" if d <= TOL else "FAIL"
        if d > TOL:
            failed = True
        print(f"[compare] {f}: rows={len(a)} max|delta|={d:.3e} {verdict} (tol {TOL:g})")

    try:
        import h5py
        worst = (0.0, "<none>")
        shapes_bad = []

        with h5py.File(f"{base}/{H5_FILE}", "r") as fa, \
                h5py.File(f"{new}/{H5_FILE}", "r") as fb:
            names = []
            fa.visit(lambda n: names.append(n)
                     if isinstance(fa[n], h5py.Dataset) else None)
            for n in names:
                if n not in fb:
                    shapes_bad.append(n + " (missing)")
                    continue
                da, db = fa[n][...], fb[n][...]
                if da.shape != db.shape:
                    shapes_bad.append(f"{n} {da.shape} vs {db.shape}")
                    continue
                if da.size == 0 or not np.issubdtype(da.dtype, np.number):
                    continue
                d = float(np.max(np.abs(np.asarray(da, dtype=np.complex128)
                                        - np.asarray(db, dtype=np.complex128))))
                if d > worst[0]:
                    worst = (d, n)
        if shapes_bad:
            print(f"[compare] {H5_FILE}: SHAPE/MISSING: {shapes_bad}")
            failed = True
        verdict = "OK" if worst[0] <= TOL else "FAIL"
        if worst[0] > TOL:
            failed = True
        print(f"[compare] {H5_FILE}: max|delta|={worst[0]:.3e} "
              f"(worst dataset: {worst[1]}) {verdict} (tol {TOL:g})")
    except Exception as e:  # noqa: BLE001 - report, don't crash the gate
        print(f"[compare] {H5_FILE}: ERROR {e!r}")
        failed = True

    print("[compare] VERDICT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
