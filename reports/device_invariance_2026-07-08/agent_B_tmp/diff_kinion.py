import sys, h5py, numpy as np
def load(p):
    with h5py.File(p, 'r') as f:
        return {k: f[k][...] for k in f.keys()}
a, b = load(sys.argv[1]), load(sys.argv[2])
print("datasets:", list(a.keys()))
for k in a:
    if a[k].shape != b[k].shape:
        print(k, "SHAPE MISMATCH", a[k].shape, b[k].shape); continue
    if a[k].dtype.kind in 'fc':
        d = np.abs(a[k] - b[k])
        print(f"{k}: shape {a[k].shape} maxdiff {d.max():.6e}")
        if d.max() > 0 and a[k].ndim >= 2:
            i = np.unravel_index(d.argmax(), d.shape)
            print(f"   argmax {i}: {a[k][i]} vs {b[k][i]}")
        # diagonal diffs if square
        if a[k].ndim == 3 and a[k].shape[1] == a[k].shape[2]:
            dd = np.abs(np.diagonal(a[k]-b[k], axis1=1, axis2=2))
            print(f"   diag maxdiff {dd.max():.6e} (Ry?) at {np.unravel_index(dd.argmax(), dd.shape)}")
    else:
        print(f"{k}: identical={np.array_equal(a[k], b[k])}")
