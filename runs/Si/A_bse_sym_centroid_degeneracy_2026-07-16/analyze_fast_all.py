#!/usr/bin/env python3
"""Fast dense-BSE degeneracy analysis across band windows, both centroid arms.

The gate's ``_build_dense_H`` builds the screened direct term Kd with an
O(nk^2) Python k,k' loop — fine for the 2v2c gate (N=36/256) but hours at
8v8c (N=4096).  This script provides a VECTORISED Kd (inner k'-loop lifted
into batched einsums; identical formula) and PROVES it bit-equal to the
reused ``_build_dense_H`` at 2v2c before using it at the large windows.

Diagonal D and exchange Kx are already loop-free in ``_build_dense_H`` and
are reproduced here with the identical einsum formulas; the 2v2c bit-equal
check covers them too.
"""
import sys
import os
import json
import numpy as np

LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
sys.path.insert(0, os.path.join(LROOT, "tests"))

import jax
jax.config.update("jax_enable_x64", True)

from bse import bse_io
from test_bse_dense_reference import _build_dense_H  # reference (validation)

RY = 13.6056980659


def fast_H(data):
    """Vectorised dense H; same formulas as _build_dense_H, batched k'."""
    psi_c = np.asarray(data["psi_c"])   # (k,c,s,μ)
    psi_v = np.asarray(data["psi_v"])   # (k,v,s,ν)
    eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
    V_q0 = np.asarray(data["V_q0"]); W_q = np.asarray(data["W_q"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    grid = (nkx, nky, nkz); nk = nkx * nky * nkz
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
    N = nc * nv * nk

    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0, optimize=True)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M, optimize=True) / nk

    Wflat = W_q.reshape(nmu, nmu, nk)  # (μ,ν,qflat) C-order
    ck = np.array(np.unravel_index(np.arange(nk), grid)).T  # (nk,3)
    qidx = np.empty((nk, nk), dtype=int)
    for k in range(nk):
        q = (ck[k][None, :] - ck) % np.array(grid)
        qidx[k] = np.ravel_multi_index(q.T, grid)

    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    for k in range(nk):
        Wq_k = np.transpose(Wflat[:, :, qidx[k]], (2, 0, 1))          # (kp,μ,ν)
        Pc = np.einsum("csm,KCsm->KcCm", np.conj(psi_c[k]), psi_c, optimize=True)
        Pv = np.einsum("vsn,KVsn->KvVn", psi_v[k], np.conj(psi_v), optimize=True)
        tmp = np.einsum("KcCm,Kmn->KcCn", Pc, Wq_k, optimize=True)
        Kd[:, :, k, :, :, :] = np.einsum(
            "KcCn,KvVn->cvCVK", tmp, Pv, optimize=True) / nk

    H = (np.diag(D.reshape(-1).astype(np.complex128))
         + Kx.reshape(N, N) - Kd.reshape(N, N))
    return H


def cluster(ev_eV, gap_meV=1.0):
    gap = gap_meV * 1e-3
    groups, cur = [], [0]
    for i in range(1, len(ev_eV)):
        if ev_eV[i] - ev_eV[i - 1] > gap:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


def analyze(inp, arm, nv, nc):
    restart = bse_io._find_restart_file(inp)
    data = bse_io._load_ring_subset(restart, n_val=nv, n_cond=nc,
                                    px=1, py=1, input_file=inp)
    H = fast_H(data)
    Hh = 0.5 * (H + H.conj().T)
    ev = np.sort(np.linalg.eigvalsh(Hh)) * RY
    groups = cluster(ev, 1.0)
    manifolds = []
    for gi, g in enumerate(groups[:6]):
        idx = np.array(g)
        manifolds.append({
            "manifold": gi, "size": len(g),
            "mean_eV": float(ev[idx].mean()),
            "max_intra_split_ueV": float((ev[idx].max() - ev[idx].min()) * 1e6),
        })
    return ev, manifolds


def main():
    # ---- validation: fast_H == _build_dense_H at 2v2c (bit-level) ----
    inp_old = f"{RUN}/work_old/cohsex_si_test.in"
    r = bse_io._find_restart_file(inp_old)
    d = bse_io._load_ring_subset(r, n_val=2, n_cond=2, px=1, py=1,
                                 input_file=inp_old)
    Href, _, _, _ = _build_dense_H(d)
    Hf = fast_H(d)
    rel = np.linalg.norm(Hf - Href) / max(np.linalg.norm(Href), 1e-300)
    print(f"[validate] fast_H vs _build_dense_H (2v2c): rel-err {rel:.2e}",
          flush=True)
    assert rel < 1e-12, f"fast_H NOT equal to reference: {rel:.2e}"
    print("[validate] PASS — vectorised builder is bit-equal.\n", flush=True)

    windows = [(4, 4), (6, 6), (8, 8)]
    summary = {}
    for arm in ("old", "sym"):
        inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
        for (nv, nc) in windows:
            tag = f"{nv}v{nc}c"
            ev, manifolds = analyze(inp, arm, nv, nc)
            summary[f"{arm}_{tag}"] = {
                "lowest_24_eV": [float(x) for x in ev[:24]],
                "manifolds": manifolds,
            }
            json.dump({"arm": arm, "window": tag,
                       "lowest_24_eV": [float(x) for x in ev[:24]],
                       "manifolds": manifolds},
                      open(f"{RUN}/results_{arm}_{tag}.json", "w"), indent=2)
            print(f"[{arm} {tag}] lowest-6 manifolds (gap<1meV):", flush=True)
            for m in manifolds:
                print(f"    mfd{m['manifold']} size={m['size']} "
                      f"mean={m['mean_eV']:.6f}eV "
                      f"split={m['max_intra_split_ueV']:.3f}ueV", flush=True)
            print(flush=True)

    json.dump(summary, open(f"{RUN}/summary_windows.json", "w"), indent=2)
    print("wrote summary_windows.json", flush=True)


if __name__ == "__main__":
    main()
