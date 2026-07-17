#!/usr/bin/env python3
"""Calibrate the BSE Γ-on-site degeneracy gate (robust, full_multiplet-style).

Auto-detect a degeneracy-closed (nv,nc) window at Γ, build H_Γ = D+Kx−Kd with
(a) RAW production q=0 tiles and (b) little-group-SYMMETRIZED q=0 tiles.  The
COVARIANT (sym-tile) spectrum defines the true exciton multiplets; measure how
much the RAW tiles split those same states (state-matched by sorted index).
"""
import os, sys, json
import numpy as np, h5py

LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/fixture_run"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from bse import bse_io
from file_io.wfn_loader import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm
from gw.degen_average import round_band_window_to_closed_shell, TOL_DEGENERACY_RY
RY = 13.6056980659


def build_H(pc, pv, ec, ev, V0, W0):
    nc = pc.shape[0]; nv = pv.shape[0]
    M = np.einsum("csm,vsm->cvm", np.conj(pc), pv, optimize=True)
    Kx = np.einsum("cvm,mn,CVn->cvCV", np.conj(M), V0, M, optimize=True)
    Pc = np.einsum("csm,Csm->cCm", np.conj(pc), pc, optimize=True)
    Pv = np.einsum("vsn,Vsn->vVn", pv, np.conj(pv), optimize=True)
    Kd = np.einsum("cCn,vVn->cvCV",
                   np.einsum("cCm,mn->cCn", Pc, W0, optimize=True), Pv, optimize=True)
    D = np.zeros((nc, nv, nc, nv), complex)
    for c in range(nc):
        for v in range(nv):
            D[c, v, c, v] = ec[c] - ev[v]
    H = (D + Kx - Kd).reshape(nc * nv, nc * nv)
    return 0.5 * (H + H.conj().T)


def symmetrize_q0(T0, alpha, ntran):
    nmu = T0.shape[0]
    acc = np.zeros((nmu, nmu), complex)
    for s in range(ntran):
        a = alpha[s]
        acc += T0[np.ix_(a, a)]
    return acc / ntran


def main():
    inp = os.path.join(RUN, "gnppm_test.in")
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(os.path.join(RUN, "WFN.h5"), backend="eager")
    ntran = int(wfn.ntran)
    FFT = np.asarray(wfn.fft_grid, dtype=np.int64)
    cfrac = np.loadtxt(os.path.join(RUN, "centroids_frac_399.txt"))
    ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
    alpha, _ = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)

    with h5py.File(restart, "r") as f:
        enk = np.asarray(f["enk_full"][:])
    nk, nb = enk.shape
    n_occ = bse_io.resolve_n_occ(enk, input_file=inp)
    print(f"restart nk={nk} nb={nb} nelec={n_occ} ntran={ntran} FFT={tuple(FFT)}", flush=True)

    e_g = enk[0:1, :]
    out = {}
    for tgt in (4, 6, 8):
        b_v = round_band_window_to_closed_shell(e_g, n_occ - tgt, TOL_DEGENERACY_RY, "up")
        b_c = round_band_window_to_closed_shell(e_g, min(n_occ + tgt, nb), TOL_DEGENERACY_RY, "down")
        nv = n_occ - b_v; nc = b_c - n_occ
        if nv <= 0 or nc <= 0:
            continue
        data = bse_io._load_ring_subset(restart, n_val=nv, n_cond=nc, px=1, py=1, input_file=inp)
        pc = np.asarray(data["psi_c"])[0]; pv = np.asarray(data["psi_v"])[0]
        ec = np.asarray(data["eps_c"])[0]; ev = np.asarray(data["eps_v"])[0]
        V0 = np.asarray(data["V_q0"]); W0 = np.asarray(data["W_q"])[:, :, 0, 0, 0]
        V0s = symmetrize_q0(V0, alpha, ntran); W0s = symmetrize_q0(W0, alpha, ntran)
        ev_raw = np.sort(np.linalg.eigvalsh(build_H(pc, pv, ec, ev, V0, W0))) * RY
        ev_cov = np.sort(np.linalg.eigvalsh(build_H(pc, pv, ec, ev, V0s, W0s))) * RY
        # true multiplets = degenerate groups of the COVARIANT spectrum, tol 5μeV
        tol = 5e-6
        grp = [[0]]
        for i in range(1, len(ev_cov)):
            (grp.append([i]) if ev_cov[i] - ev_cov[i-1] > tol else grp[-1].append(i))
        mults = [g for g in grp if len(g) > 1]
        cov_splits = [float((ev_cov[g[-1]]-ev_cov[g[0]])*1e6) for g in mults]
        raw_splits = [float((ev_raw[np.array(g)].max()-ev_raw[np.array(g)].min())*1e6) for g in mults]
        max_eig_shift = float(np.max(np.abs(ev_raw - ev_cov))*1e6)
        rec = {"nv": nv, "nc": nc, "N": nc*nv, "b_v": b_v, "b_c": b_c,
               "n_mult": len(mults), "sizes": [len(g) for g in mults],
               "cov_split_max": max(cov_splits) if cov_splits else 0.0,
               "raw_split_max": max(raw_splits) if raw_splits else 0.0,
               "max_eig_shift_ueV": max_eig_shift}
        out[f"tgt{tgt}"] = rec
        dV = float(np.abs(V0s-V0).max()/max(np.abs(V0).max(),1e-30))
        dW = float(np.abs(W0s-W0).max()/max(np.abs(W0).max(),1e-30))
        print(f"tgt{tgt}: nv={nv} nc={nc} N={nc*nv} mults={len(mults)} sizes={rec['sizes']}", flush=True)
        print(f"   COV split max={rec['cov_split_max']:.4f}μeV  RAW split max={rec['raw_split_max']:.4f}μeV  "
              f"max|λraw-λcov|={max_eig_shift:.4f}μeV", flush=True)
        print(f"   cov_splits(μeV)={[round(s,3) for s in cov_splits]}", flush=True)
        print(f"   raw_splits(μeV)={[round(s,3) for s in raw_splits]}", flush=True)
        print(f"   tile sym-change: V0 rel={dV:.3e} W0 rel={dW:.3e}", flush=True)
    json.dump(out, open("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/fix_validate/calib.json", "w"), indent=2)


if __name__ == "__main__":
    main()
