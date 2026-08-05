import sys, h5py, numpy as np
p4, p16 = sys.argv[1], sys.argv[2]
def walk(f, pre=""):
    out = {}
    for k in f:
        if isinstance(f[k], h5py.Group): out.update(walk(f[k], pre+k+"/"))
        else: out[pre+k] = f[k]
    return out
with h5py.File(p4,'r') as f4, h5py.File(p16,'r') as f16:
    d4, d16 = walk(f4), walk(f16)
    print("keys 4g:", sorted(d4.keys()))
    for k in sorted(d4):
        if k not in d16: print(k, "MISSING in 16g"); continue
        a, b = d4[k][...], d16[k][...]
        if a.shape != b.shape: print(f"{k}: SHAPE {a.shape} vs {b.shape}"); continue
        if a.dtype.kind in 'fc':
            d = np.abs(a-b); rel = d.max()/max(np.abs(a).max(), 1e-300)
            flag = "  <<<<" if d.max() > 1e-6*max(np.abs(a).max(),1) else ""
            print(f"{k}: shape {a.shape} |a|max {np.abs(a).max():.4e} maxdiff {d.max():.4e} rel {rel:.2e}{flag}")
        else:
            print(f"{k}: nonfloat identical={np.array_equal(a,b)}")
