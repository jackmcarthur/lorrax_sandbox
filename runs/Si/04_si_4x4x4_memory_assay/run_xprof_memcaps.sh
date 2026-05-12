#!/bin/bash
# Run the pipeline at several memory caps with HLO dump + fresh-process peak.
# Each cap runs as a SEPARATE srun (fresh process = clean peak counter).
set -euo pipefail

JOBID=${SLURM_JOB_ID:-$1}
SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$BASEDIR"

for MEMCAP in 2 4 8 28; do
    DUMP_DIR="$BASEDIR/profiles/hlo_memcap${MEMCAP}"
    rm -rf "$DUMP_DIR"
    mkdir -p "$DUMP_DIR"
    OUTFILE="$BASEDIR/xprof_memcap${MEMCAP}.out"

    echo "=== memcap=${MEMCAP} GB ==="

    srun --jobid=$JOBID --gres=gpu:4 -N 1 -n 4 \
        shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
        --env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \
        --env=JAX_ENABLE_X64=1 \
        --env=HDF5_USE_FILE_LOCKING=FALSE \
        --env=LORRAX_MEM_PROFILE=1 \
        --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
        --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python3 -u -c "
import os, gc, sys
MODULE_RE = 'fft_and_rslice|reshard_rchunk|compute_P_traced|left_ifft_conj|right_ifft_mul_fft|solve_batch|_update_accum'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS','') + ' --xla_dump_to=$DUMP_DIR --xla_dump_hlo_as_text=true --xla_dump_hlo_module_re=' + MODULE_RE
os.environ['LORRAX_MEM_PROFILE'] = '1'
import jax
pc = int(os.environ.get('SLURM_NTASKS','1'))
if pc > 1: jax.distributed.initialize()
import jax.numpy as jnp, numpy as np, configparser
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.isdf_fitting import fit_zeta_chunked_to_h5
from gw.gw_init import compute_optimal_chunks

wfn = WFNReader('WFN.h5'); sym = SymMaps(wfn)
cfg = configparser.ConfigParser(); cfg.read('cohsex.in'); params = dict(cfg['cohsex'])
fg = tuple(int(x) for x in wfn.fft_grid); kg = np.array(wfn.kgrid)
c = np.loadtxt(params['centroids_file'],dtype=np.float64)
ci = np.round(c*np.array(fg)[None,:]).astype(np.int32)%np.array(fg)[None,:]
d = np.array(jax.devices()); nd=len(d); px=int(np.sqrt(nd))
while nd%px: px-=1
py=nd//px; mesh=jax.sharding.Mesh(d.reshape(px,py),('x','y'))
nk=sym.nk_tot; nb=35; ns=2; n_rmu=len(c); nq=kg[0]*kg[1]*kg[2]
n_rtot=fg[0]*fg[1]*fg[2]; memcap=$MEMCAP

chunks = compute_optimal_chunks(
    n_k=nk, n_b=nb, n_s=ns, n_rmu=n_rmu, n_r=n_rtot, n_q=nq,
    fft_grid=fg, n_devices=nd, memory_budget_gb=memcap,
    p_x=px, p_y=py, n_b_left=nb, n_b_right=nb)
bc = chunks['band_chunk']; rc = chunks['chunk_r']
qc = chunks['q_chunk']; qg = chunks.get('q_gather', 1)
mem = chunks['memory_estimate']

if jax.process_index()==0:
    print(f'memcap={memcap} GB, mesh={px}x{py}')
    print(f'Solver: bc={bc}, rc={rc}, qc={qc}, qg={qg}')
    print(f'Predicted peak: {mem[\"peak_estimate_gb\"]:.3f} GB')
    print(f'Bottleneck: {mem[\"bottleneck\"]}')
    sp = mem.get('stage_peaks_gb', {})
    for s,v in sorted(sp.items(), key=lambda x:-x[1]):
        print(f'  stage_{s}: {v:.3f} GB')
    print()

meta=type('M',(),{'nk_tot':nk,'nspinor':ns,'nspinor_wfnfile':2,'fft_grid':fg,
    'n_rtot':n_rtot,'n_rmu':n_rmu,'kgrid':kg,'memory_per_device_gb':memcap,
    'b_id_0':0,'b_id_3':nb,'b_id_4':nb})()
out=fit_zeta_chunked_to_h5(wfn=wfn,sym=sym,meta=meta,centroid_indices=jnp.asarray(ci),
    bispinor=False,mesh_xy=mesh,band_range_left=(0,nb),band_range_right=(0,nb),
    chunk_r=rc,output_file='tmp/zeta_memcap${MEMCAP}.h5',
    band_chunk_size=bc,q_chunk_size=qc,q_gather_size=qg,
    use_gspace_cache=True,isdf_pair_mode='spin_traced')
out[0].block_until_ready()
if jax.process_index()==0:
    s = jax.local_devices()[0].memory_stats()
    peak = s['peak_bytes_in_use']/1e9
    print(f'\\nObserved peak_bytes_in_use: {peak:.3f} GB')
    print(f'Predicted: {mem[\"peak_estimate_gb\"]:.3f} GB')
    print(f'Error: {peak - mem[\"peak_estimate_gb\"]:+.3f} GB ({100*abs(peak-mem[\"peak_estimate_gb\"])/peak:.1f}%)')
    import glob
    reports = sorted(glob.glob('$DUMP_DIR/*memory-usage-report.txt'))
    print(f'\\nHLO memory reports ({len(reports)} files):')
    for r in reports:
        name = os.path.basename(r).split('.')[1]
        with open(r) as f:
            first_line = f.readline().strip()
        print(f'  {name}: {first_line}')
" 2>&1 | tee "$OUTFILE"

    echo
done
echo "=== All done ==="
