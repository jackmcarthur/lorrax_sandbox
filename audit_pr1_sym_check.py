"""Bit-equality / correctness check for PR1 SymMaps attrs.

For MoS2 3x3 the scope report predicts:
  - q_full_to_irr_sym (= sym_idx_q) = [0,0,2,0,0,0,2,2,2]
  - irr_idx_q indicating 6 IBZ q's? No — scope says 5 IBZ q's:
    (0,0,0), (0,1,0), (1,0,0), (1,1,0), (1,2,0)
    [(0,2,0)->1, (2,0,0)->2, (2,1,0)->5, (2,2,0)->4? -- scope row says (2,2,0)->(1,1,0)]
  - irr_idx_q (each row indexes into IBZ): [0,1,1,2,3,4,2,4,3] (matches commit msg)
For CrI3 6x6: 36 full-BZ q, 8 IBZ q, 0 TRS folds.
"""
import sys
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
import numpy as np
from common.symmetry_maps import SymMaps
from file_io import WfnLoader

def check(wfn_path, label, expected_irr_idx_q=None, expected_sym_idx_q=None,
          expected_n_ibz=None, expected_n_trs=None):
    print(f"\n=== {label} ===  {wfn_path}")
    wfn = WfnLoader(wfn_path)
    sym = SymMaps(wfn)
    print(f"ntran           = {len(sym.sym_matrices)}")
    print(f"sym_mats_k.shape= {np.asarray(sym.sym_mats_k).shape}")
    print(f"kvecs_asints[N] = {sym.kvecs_asints.shape[0]}")
    print(f"irr_idx_q       = {np.asarray(sym.irr_idx_q).tolist()}")
    print(f"sym_idx_q       = {np.asarray(sym.sym_idx_q).tolist()}")
    print(f"q_irr_kgrid_int = {np.asarray(sym.q_irr_kgrid_int).tolist()}")
    print(f"q_irr_full_idx  = {np.asarray(sym.q_irr_full_idx).tolist()}")
    n_ibz = int(sym.q_irr_kgrid_int.shape[0])
    ntran = len(sym.sym_matrices)
    n_trs = int(np.sum(np.asarray(sym.sym_idx_q) >= ntran))
    print(f"#IBZ q          = {n_ibz}, #TRS folds = {n_trs}")
    ok = True
    if expected_irr_idx_q is not None:
        m = np.array_equal(np.asarray(sym.irr_idx_q), np.asarray(expected_irr_idx_q))
        print(f"irr_idx_q match expected: {m}")
        if not m: ok = False
    if expected_sym_idx_q is not None:
        m = np.array_equal(np.asarray(sym.sym_idx_q), np.asarray(expected_sym_idx_q))
        print(f"sym_idx_q match expected: {m}")
        if not m: ok = False
    if expected_n_ibz is not None:
        print(f"#IBZ matches {expected_n_ibz}: {n_ibz == expected_n_ibz}")
        if n_ibz != expected_n_ibz: ok = False
    if expected_n_trs is not None:
        print(f"#TRS matches {expected_n_trs}: {n_trs == expected_n_trs}")
        if n_trs != expected_n_trs: ok = False

    # Reconstruction check: sym_mats_k[sym_idx_q[i]] @ q_ibz[irr_idx_q[i]] (mod kgrid) == kvecs_asints[i]
    kg = (sym.kvecs_asints.max(axis=0) + 1).astype(np.int64)
    Smk = np.asarray(sym.sym_mats_k, dtype=np.int64)
    qibz = np.asarray(sym.q_irr_kgrid_int, dtype=np.int64)
    full = np.asarray(sym.kvecs_asints, dtype=np.int64)
    recon_ok = True
    for i in range(full.shape[0]):
        s = int(sym.sym_idx_q[i]); ir = int(sym.irr_idx_q[i])
        rec = (Smk[s] @ qibz[ir]) % kg
        if not np.array_equal(rec, full[i]):
            print(f"  RECON FAIL i={i} full={full[i]} got={rec}")
            recon_ok = False
    print(f"Reconstruction (all {full.shape[0]} full points): {recon_ok}")
    if not recon_ok: ok = False
    return ok


# MoS2: commit msg + scope predicts irr_idx_q=[0,1,1,2,3,4,2,4,3], 5 IBZ q, 4 TRS folds
# Scope report predicts sym_idx_q=[0,0,2,0,0,0,2,2,2]
ok1 = check(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5",
    "MoS2 3x3",
    expected_irr_idx_q=[0,1,1,2,3,4,2,4,3],
    expected_sym_idx_q=[0,0,2,0,0,0,2,2,2],
    expected_n_ibz=5,
    expected_n_trs=4,
)

# CrI3: 8 IBZ q from 36 full-BZ, 0 TRS folds (has inversion)
ok2 = check(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/qe/nscf/WFN.h5",
    "CrI3 6x6 80Ry",
    expected_n_ibz=8,
    expected_n_trs=0,
)

print("\n=== SUMMARY ===")
print(f"MoS2 OK: {ok1}; CrI3 OK: {ok2}")
