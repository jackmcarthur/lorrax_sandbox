#!/usr/bin/env python3
"""Calibrate the BSE Γ-on-site degeneracy gate against the committed MoS2 gnppm
fixture.  Mirrors the gate: auto-detect a degeneracy-closed (nv,nc) window at Γ,
build H_Γ = D + Kx − Kd with the production (head-injected) q=0 tiles, eigvalsh,
and report the intra-multiplet splitting — for RAW and little-group-SYMMETRIZED
q=0 tiles.
"""
import os, sys, json
import numpy as np, h5py

LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/fixture_run"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from bse import bse_io
from gw.degen_average import round_band_window_to_closed_shell, TOL_DEGENERACY_RY
RY = 13.6056980659


def gamma_onsite_H(data, kg=0):
    pc = np.asarray(data["psi_c"])[kg]   # (c,s,μ)
    pv = np.asarray(data["psi_v"])[kg]   # (v,s,μ)
    ec = np.asarray(data["eps_c"])[kg]; ev = np.asarray(data["eps_v"])[kg]
    V0 = np.asarray(data["V_q0"])                 # (μ,μ)
    W0 = np.asarray(data["W_q"])[:, :, 0, 0, 0]   # q=0 (μ,μ)
    return _build(pc, pv, ec, ev, V0, W0)


def _build(pc, pv, ec, ev, V0, W0):
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


def clusters(ev, gap):
    g = [[0]]
    for i in range(1, len(ev)):
        (g.append([i]) if ev[i] - ev[i - 1] > gap else g[-1].append(i))
    return g


def max_split(H, gap_ev=1e-3):
    ev = np.sort(np.linalg.eigvalsh(H)) * RY
    cl = clusters(ev, gap_ev)
    mults = [c for c in cl if len(c) > 1]
    splits = [float((ev[c[-1]] - ev[c[0]]) * 1e6) for c in mults]  # μeV
    sizes = [len(c) for c in mults]
    return (max(splits) if splits else 0.0), sizes, splits, ev


def main():
    inp = os.path.join(RUN, "gnppm_test.in")
    restart = bse_io._find_restart_file(inp)
    with h5py.File(restart, "r") as f:
        enk = np.asarray(f["enk_full"][:])   # (nk, nb) Ry
    nk, nb = enk.shape
    n_occ = bse_io.resolve_n_occ(enk, input_file=inp)
    print(f"restart: nk={nk} nb={nb} n_occ(nelec)={n_occ}", flush=True)

    # Γ = k-index 0 (C-order (0,0,0)). Auto-detect a degeneracy-closed (nv,nc)
    # window at Γ: reuse the same helper with a single-k energies row so
    # min-over-k == the Γ gap.
    e_g = enk[0:1, :]   # (1, nb)
    for tgt in (4, 6, 8):
        b_v = round_band_window_to_closed_shell(e_g, n_occ - tgt, TOL_DEGENERACY_RY, "up")
        b_c = round_band_window_to_closed_shell(e_g, min(n_occ + tgt, nb), TOL_DEGENERACY_RY, "down")
        nv = n_occ - b_v; nc = b_c - n_occ
        if nv <= 0 or nc <= 0:
            print(f"  target {tgt}: degenerate window empty (nv={nv} nc={nc}); skip", flush=True)
            continue
        data = bse_io._load_ring_subset(restart, n_val=nv, n_cond=nc, px=1, py=1,
                                        input_file=inp)
        H = gamma_onsite_H(data)
        ms, sizes, splits, ev = max_split(H)
        print(f"  target {tgt} -> closed window nv={nv} nc={nc} (N={nc*nv}): "
              f"multiplet sizes={sizes} splits(μeV)={[round(s,3) for s in splits]} "
              f"MAX={ms:.4f} μeV", flush=True)
        print(f"       Γ valence bottom boundary b_v={b_v} (gap "
              f"{float(enk[0,b_v]-enk[0,b_v-1])*RY*1e3 if b_v>0 else float('inf'):.2f} meV), "
              f"cond top boundary b_c={b_c} (gap "
              f"{float(enk[0,b_c]-enk[0,b_c-1])*RY*1e3 if b_c<nb else float('inf'):.2f} meV)", flush=True)
        # DFT degeneracy of the kept bands at Γ (μeV)
        vv = enk[0, n_occ-nv:n_occ]*RY*1e6; cc = enk[0, n_occ:n_occ+nc]*RY*1e6
        print(f"       Γ valence ε(μeV rel): {np.round(vv-vv[0],2)}", flush=True)
        print(f"       Γ cond    ε(μeV rel): {np.round(cc-cc[0],2)}", flush=True)


if __name__ == "__main__":
    main()
