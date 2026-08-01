#!/usr/bin/env python3
"""fastloop runner skeleton — full LORRAX pipeline on a synthetic mini-deck
in ONE process (2x2 mesh from 4 host devices; no MPI, no sbatch).

STATUS: SCAFFOLD (2026-07-31). The deck generator does not exist yet; this
runner refuses at stage 'deck' and says so. See fastloop/PLAN.md for the
plan and tools/probe_w_densifier_hlo.py (in the repo) for the proven
single-routine version of the technique.

Run in-container on a compute node:

    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
        python3 fastloop/run_minideck.py --deck decks/mos2_2x2 [--stop-after STAGE]
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
# HLO/collective tables from this run are only valid cache-cold.
os.environ.setdefault("ISDF_JAX_CACHE_DIR", "")

STAGES = ("deck", "wfn", "centroids", "isdf", "chi_w", "sigma", "parity")


def _mesh():
    import numpy as np
    import jax
    from jax.sharding import Mesh

    devs = jax.devices()
    if len(devs) < 4:
        raise SystemExit(
            "REFUSING: need >= 4 host devices for the 2x2 mesh; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=4")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))


def stage_deck(args):
    raise NotImplementedError(
        "deck generator not implemented (PLAN.md item 1): produce a tiny "
        "2x2-kgrid WFN.h5 + cohsex.in under --deck")


def stage_wfn(args):
    raise NotImplementedError("wfn ingest stage (PLAN.md item 2)")


def stage_centroids(args):
    raise NotImplementedError("centroid selection stage (PLAN.md item 2)")


def stage_isdf(args):
    raise NotImplementedError("ISDF fit stage (PLAN.md item 2)")


def stage_chi_w(args):
    raise NotImplementedError("chi/W stage (PLAN.md item 2)")


def stage_sigma(args):
    raise NotImplementedError("sigma stage (PLAN.md item 2)")


def stage_parity(args):
    raise NotImplementedError(
        "parity harness (PLAN.md item 3): per-stage references + HLO "
        "forbid gate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deck", required=True, help="mini-deck directory")
    ap.add_argument("--stop-after", choices=STAGES, default="parity")
    args = ap.parse_args()

    mesh = _mesh()
    print(f"[fastloop] mesh {mesh.shape} on {len(mesh.devices.flat)} host devices")

    for name in STAGES:
        fn = globals()[f"stage_{name}"]
        try:
            fn(args)
        except NotImplementedError as e:
            print(f"[fastloop] SCAFFOLD STOP at stage '{name}': {e}")
            return 3
        print(f"[fastloop] stage '{name}' done")
        if name == args.stop_after:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
