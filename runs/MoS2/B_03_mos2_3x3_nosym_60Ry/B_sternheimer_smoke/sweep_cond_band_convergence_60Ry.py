"""60 Ry version of the cond-band convergence sweep.

Goes up to N=150 (out of 174 cond bands available with mnband=200).
"""
from __future__ import annotations
import os, sys, subprocess, time
from pathlib import Path
import numpy as np
import h5py

# Sweep configuration — extended N_list since more bands are available.
N_LIST   = [4, 8, 16, 32, 64, 128, 150]
NG_OUT   = 32
IQ       = 4              # signed q = (1/3, 1/3, 0)
TOL      = 1e-6
MAX_ITER = 80
WORK     = Path('cond_sweep_60Ry')
WORK.mkdir(exist_ok=True)

env_in = dict(os.environ)
env_in.setdefault('LORRAX_NGPU', '1')

def _run(label: str, *args: str) -> str:
    out = WORK / f"{label}.h5"
    cmd = [
        'bash', '-lc',
        ('module use /global/u2/j/jackm/modulefiles && '
         'module load lorrax_B && '
         'LORRAX_NGPU=1 lxrun python3 -u -m psp.run_sternheimer '
         '--wfn WFN.h5 --pseudo_dir . '
         f'--iq-list {IQ} --ng-out {NG_OUT} '
         f'--tol {TOL} --max-iter {MAX_ITER} '
         '--truncation-2d ' + ' '.join(args) + f' -o {out}')
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env_in)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"  [FAIL] {label}: {p.returncode}\n{p.stdout}\n{p.stderr}")
        sys.exit(1)
    print(f"  [done] {label}  ({dt:.1f}s)")
    return str(out)

print("══ Reference: full Sternheimer (n_cond_bands=150 for warm-start) ══")
ref = _run('ref_full_n150', '--n-cond-bands', '150')

print("\n══ sos_only sweep ══")
sos_paths = {}
for N in N_LIST:
    sos_paths[N] = _run(f'sos_n{N:03d}', '--sos-only', '--n-cond-bands', str(N))

def _read_chi(path):
    with h5py.File(path, 'r') as f:
        chi = np.asarray(f[f'q_0/chi_col'])
        Gint = np.asarray(f[f'q_0/G_int'])
        q = np.asarray(f[f'q_0'].attrs['q_crys'])
    return chi, Gint, q

chi_ref, Gint, qcrys = _read_chi(ref)
chi_sos = {N: _read_chi(sos_paths[N])[0] for N in N_LIST}

from file_io import WFNReader
wfn = WFNReader('WFN.h5')
B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
qG_cart = (qcrys[None, :] + Gint) @ B
qG_sq = np.sum(qG_cart ** 2, axis=1)
order = np.argsort(qG_sq)
Gint = Gint[order]; qG_sq = qG_sq[order]
chi_ref = chi_ref[order]
chi_sos = {N: chi_sos[N][order] for N in N_LIST}

print("\n══ χ_{G'0}(q=4) vs n_cond_bands  (real parts) ══")
hdr = "  |G'|²/B²    " + "    ".join(f"N={N:<4d}" for N in N_LIST) + "    full"
print(hdr)
for ig in range(NG_OUT):
    row = [f"{qG_sq[ig]:8.3f}"]
    for N in N_LIST:
        row.append(f"{chi_sos[N][ig].real:+.3e}")
    row.append(f"{chi_ref[ig].real:+.3e}")
    print("    " + "  ".join(row))

np.savez(WORK / 'sweep.npz',
         N_list=np.asarray(N_LIST),
         chi_ref=chi_ref, chi_sos=np.stack([chi_sos[N] for N in N_LIST]),
         Gint=Gint, qG_sq=qG_sq, q_crys=qcrys, ng_out=NG_OUT, ecutwfc=60.0)
print(f"\n  Saved {WORK / 'sweep.npz'}")
