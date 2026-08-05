"""Localize the pad0-vs-pad12 P=1 gnppm residual: byte-compare every dumped
intermediate, then the sigma columns."""
import re
import sys
import numpy as np
import h5py

W = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/.padprobe"
A, B = f"{W}/pad0", f"{W}/pad12"


def cmp_h5(rel):
    print(f"--- {rel} ---")
    try:
        fa = h5py.File(f"{A}/{rel}", "r")
        fb = h5py.File(f"{B}/{rel}", "r")
    except OSError as e:
        print(f"  missing: {e}")
        return
    def walk(g, prefix=""):
        for k in g:
            obj = g[k]
            name = f"{prefix}/{k}"
            if isinstance(obj, h5py.Group):
                walk(obj, name)
            else:
                yield_name.append(name)
    yield_name = []
    walk(fa)
    for name in yield_name:
        if name not in fb:
            print(f"  {name}: only in pad0")
            continue
        a = np.asarray(fa[name])
        b = np.asarray(fb[name])
        if a.shape != b.shape:
            # padded in-memory dumps may differ in extent; compare logical block
            sl = tuple(slice(0, min(sa, sb)) for sa, sb in zip(a.shape, b.shape))
            a2, b2 = a[sl], b[sl]
            tag = f"shape {a.shape} vs {b.shape}, logical block"
        else:
            a2, b2 = a, b
            tag = f"shape {a.shape}"
        if a2.dtype.kind in "cf":
            d = np.abs(a2 - b2)
            na = np.abs(a2).max() or 1.0
            bit = "BIT-IDENTICAL" if not d.any() else f"max|d|={d.max():.3e} rel={d.max()/na:.3e}"
        else:
            bit = "BIT-IDENTICAL" if np.array_equal(a2, b2) else "DIFFERS (int)"
        print(f"  {name}: {tag}: {bit}")
    fa.close(); fb.close()


def cmp_census():
    for leg, path in (("pad0", A), ("pad12", B)):
        text = open(f"{path}/run.log").read()
        nodes = re.findall(r'window "(\w+)" \((?:crossing|Laplace)\): (\d+) nodes', text)
        m = re.search(r"GN invalid modes: (\d+)/(\d+)", text)
        u = re.search(r"unfulfilled=([\d.]+)%", text)
        print(f"  {leg}: windows={nodes} invalid={m.groups() if m else None} unfulfilled={u.group(1) if u else None}%")


def cmp_sigma():
    sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/reports/device_invariance_2026-07-08")
    from agent_A_diff_outputs import parse_sigma_diag, parse_eqp
    sa = parse_sigma_diag(f"{A}/sigma_diag_gnppm_test.dat")
    sb = parse_sigma_diag(f"{B}/sigma_diag_gnppm_test.dat")
    keys = sorted(set(sa) & set(sb))
    dX = np.array([sb[k]["sigX"] - sa[k]["sigX"] for k in keys])
    dC = np.array([sb[k]["sigC"] - sa[k]["sigC"] for k in keys])
    ims = np.array([abs(sa[k]["sigC"].imag) for k in keys])
    print(f"  sigX: max|d| = {np.abs(dX).max():.3e}")
    print(f"  sigC: max|Re d| = {np.abs(dC.real).max():.3e}  max|Im d| = {np.abs(dC.imag).max():.3e}")
    off = ims < 100.0
    print(f"  sigC off-pole (|Im|<100, {off.sum()} rows): max|Re d| = {np.abs(dC.real[off]).max():.3e}")
    print(f"  d/|Im| max: {(np.abs(dC.real)/np.maximum(ims,1)).max():.3e}")
    for f in ("eqp0.dat", "eqp1.dat"):
        ea, eb = parse_eqp(f"{A}/{f}"), parse_eqp(f"{B}/{f}")
        ks = sorted(set(ea) & set(eb))
        d = np.array([eb[k][1] - ea[k][1] for k in ks])
        print(f"  {f}: max|d| = {np.abs(d).max():.3e}")


print("== census =="); cmp_census()
print("== zeta =="); cmp_h5("tmp/zeta_q.h5")
print("== isdf_tensors (642 vs 654 in-memory dump) ==")
import os
for fn in sorted(os.listdir(f"{A}/tmp")):
    if fn.startswith("isdf_tensors") and fn.endswith(".h5"):
        # pad12's file may be named isdf_tensors_654.h5 — match by prefix
        cands = [g for g in os.listdir(f"{B}/tmp") if g.startswith("isdf_tensors") and g.endswith(".h5")]
        pair = fn if fn in cands else (cands[0] if cands else None)
        if pair:
            print(f"  ({fn} vs {pair})")
            fa = h5py.File(f"{A}/tmp/{fn}", "r"); fb = h5py.File(f"{B}/tmp/{pair}", "r")
            common = [k for k in fa if k in fb and hasattr(fa[k], "shape")]
            for k in common:
                a, b = np.asarray(fa[k]), np.asarray(fb[k])
                sl = tuple(slice(0, min(sa2, sb2)) for sa2, sb2 in zip(a.shape, b.shape))
                a2, b2 = a[sl], b[sl]
                if a2.dtype.kind in "cf":
                    d = np.abs(a2 - b2)
                    na = np.abs(a2).max() or 1.0
                    s = "BIT-IDENTICAL" if not d.any() else f"max|d|={d.max():.3e} rel={d.max()/na:.3e}"
                else:
                    s = "BIT-IDENTICAL" if np.array_equal(a2, b2) else "DIFFERS"
                print(f"    {k}: {a.shape} vs {b.shape}: {s}")
            fa.close(); fb.close()
        break
print("== sigma / eqp =="); cmp_sigma()
