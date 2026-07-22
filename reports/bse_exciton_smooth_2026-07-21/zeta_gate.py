"""Gate the regenerated FULL-BZ ζ against the production V_qmunu tiles.

The exciton run reuses the converged G0W0 restart unchanged and swaps in only a
full-BZ ``zeta_q.h5`` (``run_zeta_fullbz.sh``).  The statement that has to hold
is: rebuilding the bare exchange tile from that ζ,

    V(q)_{μν} = Σ_G conj(ζ̃_μ(q+G)) v(q+G) ζ̃_ν(q+G),

reproduces the tile the production Σ_x actually used, at every one of the 144 q
— including the 70 that the IBZ-only production ζ never stored.

Everything here is imported from ``bse.vq_interp`` verbatim (``load_zeta_coarse``
→ ``v_sphere_padded`` → ``_batched_vq_relF``); the only local decision is a
small q-chunk, because the 48-q default materialises a 15.9 GB ζ chunk at
n_μ = 2412 / ngkmax = 8603.  Single GPU.
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--zeta", required=True)
    ap.add_argument("--q-chunk", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from bse import vq_interp as vqi

    zx = vqi.load_zeta_coarse(args.restart, args.zeta)
    print(f"[zeta-gate] nq={zx['nq']} nk={zx['nk']} n_mu={zx['n_mu']} "
          f"ngkmax={zx['ngkmax']} zeta_cutoff={zx['zeta_cutoff']} Ry", flush=True)

    v_all = vqi.v_sphere_padded(zx)
    rel = vqi._batched_vq_relF(zx["ZG"], v_all, zx["Vqmunu"],
                               q_chunk=args.q_chunk)
    rel = np.asarray(rel)

    # the sphere itself must sit inside the stored cutoff (vq_interp's own gate)
    def _k2max(q):
        G = zx["gvec"][q][:, :int(zx["ngk"][q])].astype(np.float64)
        K = zx["bvec"].T @ (zx["qfr"][q][:, None] + G)
        return float(np.max(np.sum(K * K, axis=0)))

    k2max = max(_k2max(q) for q in range(zx["nq"]))

    out = {
        "nq": int(zx["nq"]), "nk": int(zx["nk"]), "n_mu": int(zx["n_mu"]),
        "ngkmax": int(zx["ngkmax"]),
        "zeta_cutoff_ry": float(zx["zeta_cutoff"]),
        "sphere_max_k2_minus_cutoff": float(max(0.0, k2max - zx["zeta_cutoff"])),
        "makeVq_vs_disk_max": float(np.max(rel)),
        "makeVq_vs_disk_median": float(np.median(rel)),
        "makeVq_vs_disk_argmax_q": int(np.argmax(rel)),
        "per_q_relF": rel.tolist(),
        "tol": 5e-6,
    }
    out["PASS"] = bool(out["makeVq_vs_disk_max"] <= out["tol"]
                       and out["sphere_max_k2_minus_cutoff"] <= 1e-9)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[zeta-gate] makeVq_vs_disk  max {out['makeVq_vs_disk_max']:.3e}  "
          f"median {out['makeVq_vs_disk_median']:.3e}  "
          f"(tol {out['tol']:.0e})  -> "
          f"{'PASS' if out['PASS'] else '** FAIL **'}", flush=True)
    print(f"[zeta-gate] sphere_max|q+G|^2 - cutoff = "
          f"{out['sphere_max_k2_minus_cutoff']:.3e}", flush=True)
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
