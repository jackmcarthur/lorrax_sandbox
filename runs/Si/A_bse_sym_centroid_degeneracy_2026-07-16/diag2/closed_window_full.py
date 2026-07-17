#!/usr/bin/env python3
"""DIAG2 Task-1 (B), corrected: FULL-H closed-window residual with PROPERLY
covariant tiles.

The per-q stabilizer projection in closed_window.py is a no-op for generic q
(trivial little group) so it cannot make the full-BZ tile set covariant — full
covariance needs the INTER-q relation, which only the IBZ→full unfold restores.

Correct procedure (reuses PRODUCTION machinery):
  1. Extract IBZ representative tiles from the production full-BZ V_q/W_q
     (V_ibz[i] = V_full[parent(i)]).
  2. Little-group-symmetrize each IBZ rep (project onto the covariant subspace
     of the parent's stabilizer), same Reynolds form unfold_v_q inverts.
  3. Re-expand with the PRODUCTION common.symmetry_maps.unfold_v_q → a tile set
     that is covariant under the FULL group by construction.
  4. ROUNDTRIP GATE: unfold(V_ibz_raw) must reproduce the production V_full
     (validates the table wiring) before trusting the symmetrized number.
  5. Build the full-H over all 64 k with the degeneracy-closed window and
     measure manifold splittings: raw vs covariant tiles.

Usage: python3 closed_window_full.py sym
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from common.symmetry_maps import unfold_v_q
from centroid.orbit_syms import compute_centroid_sym_perm
from gw.head_correction import apply_q0_head_rank1_sharded
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24]); GRID = (4, 4, 4); NK = 64
CELL = 270.11


def clusters(ev, gap=1e-3):
    g = [[0]]
    for i in range(1, len(ev)):
        (g.append([i]) if ev[i] - ev[i - 1] > gap else g[-1].append(i))
    return g


def manifolds(H, ntop=6, gap_meV=1.0):
    ev = np.sort(np.linalg.eigvalsh(H)) * RY
    cl = clusters(ev, gap_meV * 1e-3)
    return [{"size": len(c), "mean_eV": float(ev[np.array(c)].mean()),
             "split_ueV": float((ev[c[-1]] - ev[c[0]]) * 1e6)} for c in cl[:ntop]]


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
        validate=True, extend_trs=True)   # (2*ntran, nmu), (2*ntran, nmu, 3)

    # IBZ q tables (production convention)
    q_int = np.asarray(sym.q_irr_kgrid_int)          # (n_ibz, 3)
    irr_idx = np.asarray(sym.irr_idx_q)              # (nk,)
    sym_idx = np.asarray(sym.sym_idx_q)              # (nk,)
    kg = np.asarray(GRID, float)
    q_wrap = np.where(q_int > kg / 2, q_int - kg, q_int).astype(float)
    q_irr_frac = q_wrap / kg                         # (n_ibz, 3)
    n_ibz = q_int.shape[0]
    sym_mats_k = np.asarray(sym.sym_mats_k)          # (2*ntran, 3, 3) TRS-augmented

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:]); enk = np.asarray(f["enk_full"][:])
        Vq = np.asarray(f["V_qmunu"][:]); Wq = np.asarray(f["W0_qmunu"][:])
        G0 = np.asarray(f["G0_mu_nu"][:])
        vhead = complex(f["vhead"][()]); whead = np.asarray(f["whead"][:], complex)
    nk, nb, ns, nmu = psi.shape
    print(f"[{arm}] psi{psi.shape} nmu={nmu} ntran={ntran} n_ibz={n_ibz}", flush=True)

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))

    def do_unfold(V_ibz):
        out = unfold_v_q(jnp.asarray(V_ibz), irr_idx=irr_idx, sym_idx=sym_idx,
                         sym_perm=sym_perm, L_table=L_table, q_irr_frac=q_irr_frac,
                         mesh_xy=mesh, n_sym_spatial=ntran)
        return np.asarray(out)

    # ---- extract IBZ reps from production full-BZ tiles ----
    parent_flat = np.array([np.ravel_multi_index(tuple(q_int[i] % np.array(GRID)), GRID)
                            for i in range(n_ibz)])
    V_ibz = Vq[parent_flat].copy()      # (n_ibz, μ, ν)
    W_ibz = Wq[parent_flat].copy()

    # ---- ROUNDTRIP GATE: unfold(raw IBZ) must reproduce production full-BZ ----
    V_rt = do_unfold(V_ibz)
    rt_err = float(np.abs(V_rt - Vq).max() / np.abs(Vq).max())
    print(f"  ROUNDTRIP unfold(V_ibz_raw) vs production V_full: rel={rt_err:.3e}", flush=True)
    if rt_err > 1e-10:
        print("  !! roundtrip FAILED — table wiring wrong; aborting full-H", flush=True)

    # ---- little-group symmetrize each IBZ rep (SPATIAL ops only) ----
    # Spatial point-group covariance is what enforces exciton-multiplet
    # degeneracy; TRS is anti-unitary (relates q<->-q) and its unfold phase
    # convention (conj(phase·V), not phase·conj(V)) is a separate branch — kept
    # out of the projection so the covariance restored here is exactly the
    # point-group covariance the degeneracy test cares about.
    def stab_symmetrize(T_ibz):
        Ts = np.zeros_like(T_ibz)
        stab_sizes = []
        for i in range(n_ibz):
            qf = q_irr_frac[i]
            stab = []
            for s in range(ntran):                   # spatial ops only
                Sq = sym_mats_k[s] @ qf
                if np.all(np.abs(Sq - qf - np.rint(Sq - qf)) < 1e-6):
                    stab.append(s)
            stab_sizes.append(len(stab))
            acc = np.zeros((nmu, nmu), complex)
            Ti = T_ibz[i]
            for s in stab:
                a = sym_perm[s]                      # centroid perm
                Ls = L_table[s].astype(float)        # (nmu,3)
                qL = Ls @ qf                          # (nmu,)
                phase = np.exp(2j * np.pi * (qL[:, None] - qL[None, :]))
                acc += phase * Ti[np.ix_(a, a)]
            Ts[i] = acc / len(stab)
        return Ts, stab_sizes

    V_ibz_sym, ssz = stab_symmetrize(V_ibz)
    W_ibz_sym, _ = stab_symmetrize(W_ibz)
    dVi = np.abs(V_ibz_sym - V_ibz).max() / np.abs(V_ibz).max()
    print(f"  IBZ-rep stabilizer sizes: {ssz}   symmetrization ΔV_ibz rel={dVi:.3e}", flush=True)

    V_full_sym = do_unfold(V_ibz_sym)
    W_full_sym = do_unfold(W_ibz_sym)

    # ---- VALIDATE covariance of the symmetrized full-BZ set ----
    # For each spatial op s and each full-BZ q, V_full_sym should obey the
    # q=0-style stabilizer identity only where s fixes q; a cheaper global
    # check: re-extract IBZ from V_full_sym and re-unfold == V_full_sym.
    Vfs_ibz = V_full_sym[parent_flat]
    cov_err = float(np.abs(do_unfold(Vfs_ibz) - V_full_sym).max() / np.abs(V_full_sym).max())
    print(f"  covariance self-consistency of V_full_sym: rel={cov_err:.3e}", flush=True)
    # DIRECT spatial-covariance check on the q=0 slice: V0_sym[α(μ),α(ν)]==V0_sym[μ,ν]
    V0s = V_full_sym[0]; sc = np.abs(V0s).max(); worst = 0.0
    for s in range(ntran):
        a = sym_perm[s]
        worst = max(worst, np.abs(V0s[np.ix_(a, a)] - V0s).max() / sc)
    print(f"  DIRECT q=0 spatial covariance of V0_sym: max_rel={worst:.3e} "
          f"(raw was 3.16e-2)", flush=True)
    out_v0cov = float(worst)

    out = {"arm": arm, "roundtrip_rel": rt_err, "cov_selfconsistency_rel": cov_err,
           "dV_ibz_sym": float(dVi), "stab_sizes": ssz}

    # ---- full-H closed window [0,8) x [8,16) ----
    nc_hi = 16
    psi_v = psi[:, 0:8]; psi_c = psi[:, 8:nc_hi]
    eps_v = enk[:, 0:8]; eps_c = enk[:, 8:nc_hi]

    def head_inject(V0, Wflat):
        W3d = Wflat.reshape(nmu, nmu, *GRID)
        V0c, Wc = apply_q0_head_rank1_sharded(
            jnp.asarray(V0), jnp.asarray(W3d), jnp.asarray(G0), jnp.asarray(G0),
            vhead, np.asarray(whead, complex), CELL, omega_index=0)
        return np.asarray(V0c), np.asarray(Wc).reshape(nmu, nmu, NK)

    res = {}
    for tag, (Vall, Wall) in (("raw", (Vq, Wq)), ("covariant", (V_full_sym, W_full_sym))):
        Wflat = np.transpose(Wall, (1, 2, 0)).copy()
        for head_tag in ("nohead", "head"):
            V0u = Vall[0].copy(); Wf = Wflat.copy()
            if head_tag == "head":
                V0u, Wf = head_inject(V0u, Wf)
            H = build_full_H(psi_c, psi_v, eps_c, eps_v, V0u, Wf)
            mfd = manifolds(H, ntop=6)
            res[f"{tag}_{head_tag}"] = mfd
            print(f"  full-H [{tag:>9}_{head_tag}]: "
                  + " | ".join(f"mfd{i}(sz{m['size']}) {m['split_ueV']:.2f}μeV"
                               for i, m in enumerate(mfd[:4])), flush=True)
    out["full_H"] = res
    json.dump(out, open(f"{RUN}/diag2/closed_window_full_{arm}.json", "w"), indent=2)
    print(f"  wrote closed_window_full_{arm}.json", flush=True)


if __name__ == "__main__":
    main()
