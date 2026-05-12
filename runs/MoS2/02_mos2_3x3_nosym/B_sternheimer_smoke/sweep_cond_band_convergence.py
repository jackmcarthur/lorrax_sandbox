"""Sweep over n_cond_bands using explicit cond-N projector path
(``run_sternheimer --sos-only``) and study the convergence of every
χ_{G' 0}(q) component (head + wings) vs N.

Compares against the full Sternheimer reference (which is exact in the
max_iter limit, regardless of n_cond_bands).
"""
from __future__ import annotations
import os, sys, subprocess, time
from pathlib import Path
import numpy as np
import h5py

# Sweep configuration
N_LIST   = [4, 8, 16, 32, 50]
NG_OUT   = 32
IQ       = 4              # signed q = (1/3, 1/3, 0)
TOL      = 1e-6
MAX_ITER = 80
WORK     = Path('cond_sweep')
WORK.mkdir(exist_ok=True)

env_in = dict(os.environ)
env_in.setdefault('LORRAX_NGPU', '1')

def _run(label: str, *args: str) -> str:
    """Invoke `psp.run_sternheimer` via lxrun.  Args are appended to the
    base argv.  Returns the path of the output h5."""
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

# 1) Reference: full Sternheimer, large n_cond_bands (just for warm-start;
#    the answer is independent of n_cond_bands once it converges).
print("══ Reference: full Sternheimer ══")
ref = _run('ref_full_n50', '--n-cond-bands', '50')

# 2) sos_only sweep
print("\n══ sos_only sweep ══")
sos_paths = {}
for N in N_LIST:
    sos_paths[N] = _run(f'sos_n{N:02d}', '--sos-only', '--n-cond-bands', str(N))

# Read and tabulate results
def _read_chi(path):
    with h5py.File(path, 'r') as f:
        chi = np.asarray(f[f'q_0/chi_col'])      # (ng_out,) complex
        Gint = np.asarray(f[f'q_0/G_int'])       # (ng_out, 3) int
        q = np.asarray(f[f'q_0'].attrs['q_crys'])
    return chi, Gint, q

chi_ref, Gint, qcrys = _read_chi(ref)
chi_sos = {N: _read_chi(sos_paths[N])[0] for N in N_LIST}

# |q+G'|² in cartesian for ordering / wing labeling
import jax.numpy as jnp  # noqa
from file_io import WFNReader
wfn = WFNReader('WFN.h5')
B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
qG_cart = (qcrys[None, :] + Gint) @ B
qG_sq = np.sum(qG_cart ** 2, axis=1)
# Sort by |q+G'|^2 (already sorted by build_Gprime_list, but be explicit)
order = np.argsort(qG_sq)
Gint = Gint[order]; qG_sq = qG_sq[order]
chi_ref = chi_ref[order]
chi_sos = {N: chi_sos[N][order] for N in N_LIST}

print("\n══ χ_{G'0}(q=4) vs n_cond_bands  (real parts) ══")
hdr = "  |G'|²/B²    " + "    ".join(f"N={N:<3d}" for N in N_LIST) + "    full"
print(hdr)
for ig in range(NG_OUT):
    row = [f"{qG_sq[ig]:8.3f}"]
    for N in N_LIST:
        row.append(f"{chi_sos[N][ig].real:+.3e}")
    row.append(f"{chi_ref[ig].real:+.3e}")
    print("    " + "  ".join(row))

# Save tables
np.savez(WORK / 'sweep.npz',
         N_list=np.asarray(N_LIST),
         chi_ref=chi_ref, chi_sos=np.stack([chi_sos[N] for N in N_LIST]),
         Gint=Gint, qG_sq=qG_sq, q_crys=qcrys, ng_out=NG_OUT)
print(f"\n  Saved tables to {WORK / 'sweep.npz'}")
