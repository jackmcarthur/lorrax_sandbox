#!/usr/bin/env python3
"""DIAG2 Task-1 (B) finalize: tile effect on GENUINE full-BSE multiplets.

closed_window_full.py showed the covariant-tile full-H spectrum barely moves
(2031.50->2031.41 μeV on the size-8 cluster) — so the 2031 μeV "manifold split"
is physically-distinct excitons (energy-clustering artifact), NOT a broken
multiplet.  To isolate the tile effect in the FULL coupled BSE, identify the
genuine degenerate multiplets from the COVARIANT-tile spectrum (states that ARE
degenerate when tiles obey symmetry), then measure how much RAW tiles split
those same states.  That split = the tile-induced symmetry breaking in the full
BSE (vs the 36 μeV of the ISOLATED Γ-on-site block).

Usage: python3 full_multiplet.py sym
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from common.symmetry_maps import unfold_v_q
from centroid.orbit_syms import compute_centroid_sym_perm
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24]); GRID = (4, 4, 4); NK = 64


def build_full_H(psi_c, psi_v, eps_c, eps_v, V0, Wflat):
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v, optimize=True)
    Dcvk = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V0, optimize=True)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M, optimize=True) / NK
    ck = np.array(np.unravel_index(np.arange(NK), GRID)).T
    qidx = np.empty((NK, NK), int)
    for k in range(NK):
        qidx[k] = np.ravel_multi_index(((ck[k][None] - ck) % np.array(GRID)).T, GRID)
    Kd = np.zeros((nc, nv, NK, nc, nv, NK), complex)
    for k in range(NK):
        Wq_k = np.transpose(Wflat[:, :, qidx[k]], (2, 0, 1))
        Pc = np.einsum("csm,KCsm->KcCm", np.conj(psi_c[k]), psi_c, optimize=True)
        Pv = np.einsum("vsn,KVsn->KvVn", psi_v[k], np.conj(psi_v), optimize=True)
        tmp = np.einsum("KcCm,Kmn->KcCn", Pc, Wq_k, optimize=True)
        Kd[:, :, k, :, :, :] = np.einsum("KcCn,KvVn->cvCVK", tmp, Pv, optimize=True) / NK
    N = nc * nv * NK
    H = np.diag(Dcvk.reshape(-1).astype(complex)) + Kx.reshape(N, N) - Kd.reshape(N, N)
    return 0.5 * (H + H.conj().T)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac * FFT[None]).astype(np.int64) % FFT[None]
    sym_perm, L_table = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT,
        validate=True, extend_trs=True)
    q_int = np.asarray(sym.q_irr_kgrid_int); irr_idx = np.asarray(sym.irr_idx_q)
    sym_idx = np.asarray(sym.sym_idx_q); kg = np.asarray(GRID, float)
    q_irr_frac = (np.where(q_int > kg / 2, q_int - kg, q_int).astype(float)) / kg
    n_ibz = q_int.shape[0]; sym_mats_k = np.asarray(sym.sym_mats_k)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:]); enk = np.asarray(f["enk_full"][:])
        Vq = np.asarray(f["V_qmunu"][:]); Wq = np.asarray(f["W0_qmunu"][:])
    nmu = psi.shape[3]
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))

    def do_unfold(V_ibz):
        return np.asarray(unfold_v_q(
            jnp.asarray(V_ibz), irr_idx=irr_idx, sym_idx=sym_idx, sym_perm=sym_perm,
            L_table=L_table, q_irr_frac=q_irr_frac, mesh_xy=mesh, n_sym_spatial=ntran))

    parent_flat = np.array([np.ravel_multi_index(tuple(q_int[i] % np.array(GRID)), GRID)
                            for i in range(n_ibz)])

    def stab_symmetrize(T_ibz):
        Ts = np.zeros_like(T_ibz)
        for i in range(n_ibz):
            qf = q_irr_frac[i]
            stab = [s for s in range(ntran)
                    if np.all(np.abs(sym_mats_k[s] @ qf - qf - np.rint(sym_mats_k[s] @ qf - qf)) < 1e-6)]
            acc = np.zeros((nmu, nmu), complex); Ti = T_ibz[i]
            for s in stab:
                qL = L_table[s].astype(float) @ qf
                acc += np.exp(2j * np.pi * (qL[:, None] - qL[None, :])) * Ti[np.ix_(sym_perm[s], sym_perm[s])]
            Ts[i] = acc / len(stab)
        return Ts

    V_cov = do_unfold(stab_symmetrize(Vq[parent_flat]))
    W_cov = do_unfold(stab_symmetrize(Wq[parent_flat]))

    # full-H, no head, [0,8) x [8,16)
    psi_v = psi[:, 0:8]; psi_c = psi[:, 8:16]
    eps_v = enk[:, 0:8]; eps_c = enk[:, 8:16]
    H_raw = build_full_H(psi_c, psi_v, eps_c, eps_v, Vq[0], np.transpose(Wq, (1, 2, 0)))
    H_cov = build_full_H(psi_c, psi_v, eps_c, eps_v, V_cov[0], np.transpose(W_cov, (1, 2, 0)))
    ev_raw = np.sort(np.linalg.eigvalsh(H_raw)) * RY   # eV
    ev_cov = np.sort(np.linalg.eigvalsh(H_cov)) * RY

    # tile-induced eigenvalue shift (state-matched by sorted index)
    max_shift = float(np.max(np.abs(ev_raw - ev_cov)) * 1e6)   # μeV
    print(f"[{arm}] max |λ_raw - λ_cov| over ALL states = {max_shift:.3f} μeV", flush=True)

    # genuine multiplets = degenerate groups in the COVARIANT spectrum (tol 5 μeV)
    tol = 5e-6  # eV
    groups = [[0]]
    for i in range(1, len(ev_cov)):
        (groups.append([i]) if ev_cov[i] - ev_cov[i - 1] > tol else groups[-1].append(i))
    mults = [g for g in groups if len(g) > 1]
    print(f"  genuine multiplets (covariant-tile degenerate groups, tol 5μeV): "
          f"{len(mults)} (sizes {sorted(set(len(g) for g in mults))})", flush=True)
    # for each, covariant split (~0) vs raw split (tile-induced)
    rows = []
    worst = (0.0, None)
    for g in mults[:30]:
        gi = np.array(g)
        cov_split = float((ev_cov[gi].max() - ev_cov[gi].min()) * 1e6)
        raw_split = float((ev_raw[gi].max() - ev_raw[gi].min()) * 1e6)
        rows.append({"size": len(g), "mean_eV": float(ev_cov[gi].mean()),
                     "cov_split_ueV": cov_split, "raw_split_ueV": raw_split})
        if raw_split > worst[0]:
            worst = (raw_split, len(g), float(ev_cov[gi].mean()))
    for r in rows[:12]:
        print(f"    mult sz={r['size']} @ {r['mean_eV']:.4f}eV: "
              f"cov_split={r['cov_split_ueV']:.3f}μeV  raw_split={r['raw_split_ueV']:.3f}μeV",
              flush=True)
    print(f"  WORST genuine-multiplet raw-tile split: {worst[0]:.3f} μeV "
          f"(size {worst[1] if len(worst)>1 else '?'})", flush=True)

    out = {"arm": arm, "max_eig_shift_ueV": max_shift, "n_multiplets": len(mults),
           "worst_raw_split_ueV": worst[0], "multiplets": rows}
    json.dump(out, open(f"{RUN}/diag2/full_multiplet_{arm}.json", "w"), indent=2)
    print(f"  wrote full_multiplet_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
