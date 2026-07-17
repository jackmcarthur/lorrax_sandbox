#!/usr/bin/env python3
"""RUNG 5 — direct ψ covariance at centroids (the shared-ingredient smoking gun).

For a symmetry-CLOSED band set B (the 4 occupied valence bands — the occupied
manifold is closed under the space group at every k), form the band-overlap Gram
on the centroid grid:
    G^B_k[n,m] = Σ_{s,μ} conj(ψ_nk(r_μ)) ψ_mk(r_μ)          (|B|×|B|, hermitian)
Under a space-group op R with k→Rk and centroids closed (r_μ → r_{π(μ)}), an
EXACTLY covariant ψ gives  G^B_{Rk} = D_R G^B_k D_R†  (D_R unitary) — so the
eigenvalues of G^B_k are R-INVARIANT (no π needed; permutation+band-rotation
similarity).  Thus within each k-star every member must share one sorted
eigenvalue list.  Per-eigenvalue spread within a star measures ψ non-covariance.
  * sym arm: centroids ARE orbit-closed → spread isolates the ψ UNFOLD.
  * old arm: centroids non-closed → spread mixes unfold + centroid-set effects.
Also Tr G^B_k = Σ_{n∈B,s,μ}|ψ_nk(r_μ)|² (band-summed centroid norm) — the
simplest scalar; R-invariant within a star iff covariant.

Usage: python3 rung5_psi.py <arm>
"""
import sys, os, json
import numpy as np
import h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from bse import bse_io
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager")
    sym = symmetry_maps.SymMaps(wfn)
    irr = np.asarray(sym.irr_idx_k)
    stars = [sorted(np.where(irr == u)[0].tolist()) for u in sorted(set(irr.tolist()))]
    nspinor = int(wfn.nspinor)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])   # (nk, nb, s, nmu)
        enk = np.asarray(f["enk_full"][:])
    nk, nb, ns, nmu = psi.shape
    print(f"[{arm}] restart={os.path.basename(restart)} psi{psi.shape} "
          f"WFN nspinor={nspinor}", flush=True)

    # occupied valence = 4 lowest bands for Si (nosoc, 8 electrons).
    B = [0, 1, 2, 3]
    psiB = psi[:, B, :, :]                       # (nk, |B|, s, nmu)
    # G^B_k[n,m] = Σ_{s,μ} conj(ψ_n) ψ_m
    G = np.einsum("knsm,klsm->knl", np.conj(psiB), psiB, optimize=True)  # (nk,|B|,|B|)
    herm = np.max(np.abs(G - np.conj(np.transpose(G, (0, 2, 1)))))
    evals = np.linalg.eigvalsh(0.5*(G + np.conj(np.transpose(G, (0, 2, 1)))))  # (nk,|B|) asc
    trace = np.real(np.einsum("knn->k", G))

    out = {"arm": arm, "nspinor": nspinor, "band_set": B,
           "G_hermiticity_max": float(herm), "stars": [len(s) for s in stars]}
    # per-star eigenvalue spreads
    worst_eig = (0.0, None)
    worst_tr = (0.0, None)
    star_report = []
    for si, st in enumerate(stars):
        ev_star = evals[st]                      # (star,|B|)
        # sorted-eigenvalue spread per component
        eig_spread = (ev_star.max(axis=0) - ev_star.min(axis=0))  # (|B|,)
        eig_scale = np.abs(ev_star).mean()
        tr_star = trace[st]
        tr_spread = float(tr_star.max() - tr_star.min())
        rel_eig = float(eig_spread.max() / (eig_scale + 1e-30))
        rel_tr = float(tr_spread / (np.abs(tr_star).mean() + 1e-30))
        star_report.append({
            "star": si, "size": len(st),
            "eig_spread_abs": float(eig_spread.max()),
            "eig_scale": float(eig_scale), "eig_rel": rel_eig,
            "trace_spread_abs": tr_spread, "trace_mean": float(tr_star.mean()),
            "trace_rel": rel_tr,
        })
        if rel_eig > worst_eig[0]:
            worst_eig = (rel_eig, si)
        if rel_tr > worst_tr[0]:
            worst_tr = (rel_tr, si)
    out["worst_eig_rel"] = {"star": worst_eig[1], "rel": worst_eig[0]}
    out["worst_trace_rel"] = {"star": worst_tr[1], "rel": worst_tr[0]}
    out["per_star"] = star_report
    print(f"  G hermiticity max = {herm:.2e}", flush=True)
    print(f"  WORST valence-Gram eigenvalue R-spread within a star: "
          f"rel={worst_eig[0]:.3e} (star {worst_eig[1]}, size "
          f"{len(stars[worst_eig[1]]) if worst_eig[1] is not None else 0})", flush=True)
    print(f"  WORST band-summed centroid-norm (TrG) R-spread within a star: "
          f"rel={worst_tr[0]:.3e} (star {worst_tr[1]})", flush=True)
    for r in star_report:
        print(f"    star{r['star']} sz={r['size']:2d}  eig_rel={r['eig_rel']:.3e}  "
              f"trace_rel={r['trace_rel']:.3e}  (Tr mean={r['trace_mean']:.4f})",
              flush=True)
    op = f"{RUN}/diag/rung5_{arm}.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"  wrote {op}", flush=True)


if __name__ == "__main__":
    main()
