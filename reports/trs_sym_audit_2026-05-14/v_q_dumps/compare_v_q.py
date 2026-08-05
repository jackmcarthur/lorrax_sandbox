#!/usr/bin/env python3
"""Direct V_q^full tensor comparison: sym vs nosym CrI3 6×6 30Ry.

Loads Vqmunu_sym.h5 (after IBZ→full unfold under HEAD 80edbe8 with both
R_cart + centroid_perm forward-direction fixes) and Vqmunu_nosym.h5 (ntran=1
direct full-BZ computation, no unfold) and compares element-wise.

Goal: localize the residual 4 eV |ΔΣ_X| failure to V_q level (unfold bug)
vs Σ_X-kernel consumption.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import h5py


def main():
    dump_dir = Path('/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/v_q_dumps')
    sym_path   = dump_dir / 'Vqmunu_sym.h5'
    nosym_path = dump_dir / 'Vqmunu_nosym.h5'
    ibz_sym_path = dump_dir / 'Vq_ibz_sym.h5'

    if not sym_path.exists():
        print(f"ERROR: missing {sym_path}")
        return 2
    if not nosym_path.exists():
        print(f"ERROR: missing {nosym_path}")
        return 2

    with h5py.File(sym_path, 'r') as f:
        V_sym = f['V_qmunu'][:]
    with h5py.File(nosym_path, 'r') as f:
        V_nosym = f['V_qmunu'][:]

    print(f"sym   V_qmunu shape: {V_sym.shape}, dtype={V_sym.dtype}")
    print(f"nosym V_qmunu shape: {V_nosym.shape}, dtype={V_nosym.dtype}")

    assert V_sym.shape == V_nosym.shape, "shape mismatch"
    n_q, n_mu, n_mu2 = V_sym.shape
    assert n_mu == n_mu2

    # Per-q residual.
    dV = V_sym - V_nosym
    abs_dV = np.abs(dV)
    abs_V_nosym = np.abs(V_nosym)
    norm_V_nosym = float(np.linalg.norm(V_nosym))
    norm_dV = float(np.linalg.norm(dV))

    # Per-q max |ΔV|
    per_q_max = abs_dV.max(axis=(1, 2))   # (n_q,)
    per_q_l2  = np.linalg.norm(abs_dV, axis=(1, 2))
    per_q_rel = per_q_l2 / (np.linalg.norm(V_nosym, axis=(1, 2)) + 1e-30)

    print()
    print("===== V_q^full sym-vs-nosym direct tensor comparison =====")
    print(f"  shape: {V_sym.shape}")
    print(f"  global max |ΔV| = {abs_dV.max():.6e}")
    print(f"  global max |V_nosym| = {abs_V_nosym.max():.6e}")
    print(f"  rel max = {abs_dV.max() / abs_V_nosym.max():.6e}")
    print(f"  L2(ΔV) / L2(V_nosym) = {norm_dV / norm_V_nosym:.6e}")
    print()
    print("--- Per-q breakdown ---")
    print("   q   max |ΔV_q|        L2 |ΔV_q|       rel L2     status")
    for q in range(n_q):
        status = "MATCH" if per_q_max[q] < 1e-6 else ("MARGINAL" if per_q_max[q] < 1e-3 else "FAIL")
        print(f"  {q:2d}   {per_q_max[q]:.6e}   {per_q_l2[q]:.6e}   {per_q_rel[q]:.6e}   {status}")

    # If we have IBZ pre-unfold dump for sym, decode the q→(parent IBZ, sym_idx) map.
    if ibz_sym_path.exists():
        with h5py.File(ibz_sym_path, 'r') as f:
            V_ibz   = f['V_q_ibz'][:]
            full_to_irr_idx = f['full_to_irr_idx'][:]
            full_to_irr_sym = f['full_to_irr_sym'][:]
            sym_perm = f['sym_perm'][:]
        print()
        print(f"--- IBZ V_q has shape {V_ibz.shape} ---")
        print("    q_full  parent_q_ibz  sym_idx  is_trs  max|ΔV|       L2 |ΔV|")
        n_sym_spatial = sym_perm.shape[0] // 2
        for q in range(n_q):
            pq = int(full_to_irr_idx[q])
            sx = int(full_to_irr_sym[q])
            is_trs = sx >= n_sym_spatial
            print(f"    {q:2d}      {pq:2d}            {sx:2d}      {int(is_trs)}      "
                  f"{per_q_max[q]:.6e}   {per_q_l2[q]:.6e}")

    # Worst row pinpointing
    print()
    print("--- Worst 10 (q, μ, ν) elements ---")
    idx_flat = np.argsort(-abs_dV.flatten())[:10]
    for i in idx_flat:
        q, mu, nu = np.unravel_index(i, abs_dV.shape)
        print(f"   q={q:2d} μ={mu:4d} ν={nu:4d}  V_sym={V_sym[q,mu,nu]:.6e}  "
              f"V_nosym={V_nosym[q,mu,nu]:.6e}  ΔV={dV[q,mu,nu]:.6e}")

    # Sanity: trace at q=0 (should be a real positive number ~ Hartree)
    print()
    print(f"  V_sym[q=0] trace.real   = {np.trace(V_sym[0]).real:.6f}")
    print(f"  V_nosym[q=0] trace.real = {np.trace(V_nosym[0]).real:.6f}")

    # Verdict
    print()
    if abs_dV.max() < 1e-6:
        print("Verdict: V_q^full sym ≡ V_q^full nosym (sub-ulp).")
        print("  → ΔΣ_X failure originates in Σ_X kernel's CONSUMPTION of V_q.")
        return 0
    elif abs_dV.max() < 1e-3:
        print("Verdict: V_q^full sym ≈ V_q^full nosym at ISDF noise floor.")
        print("  → ΔΣ_X failure not at V_q level.")
        return 0
    else:
        print("Verdict: V_q^full DISAGREES sym vs nosym at large absolute scale.")
        print("  → V_q unfold itself is wrong.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
