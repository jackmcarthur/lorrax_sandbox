import sys, h5py, numpy as np

path = sys.argv[1]
print(f"=== {path} ===")
with h5py.File(path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  DSET {name:60s} shape={obj.shape} dtype={obj.dtype}")
    f.visititems(show)
    print("--- attrs on groups ---")
    def show_attrs(name):
        obj = f[name]
        for k, v in obj.attrs.items():
            sv = v
            if isinstance(v, np.ndarray) and v.size > 8:
                sv = f"array{v.shape}"
            print(f"  ATTR {name}/{k} = {sv}")
    f.visit(lambda n: show_attrs(n))
    for k, v in f.attrs.items():
        print(f"  ROOT ATTR {k} = {v}")
    # try to print isdf_header specifics
    if "isdf_header" in f:
        g = f["isdf_header"]
        print("--- isdf_header datasets ---")
        for k in g.keys():
            d = g[k]
            if isinstance(d, h5py.Dataset):
                arr = d[()]
                if np.ndim(arr) == 0 or (hasattr(arr, "size") and arr.size <= 12):
                    print(f"    {k} = {arr}")
                else:
                    print(f"    {k}: shape={arr.shape} dtype={arr.dtype}")
