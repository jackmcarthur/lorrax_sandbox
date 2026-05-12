"""Sanity-check the G-vector scatter pipeline used by load_wfns.

For every k in the IBZ slab and every full-BZ k:
1. Verify all G-vectors fit in [-N/2+1, N/2] (no FFT-grid wrap aliasing).
2. Verify that no two distinct G's in a single k's sphere collide
   modulo N (which would mean two coefficients alias to the same FFT cell).
3. Verify the rotated full-BZ G-set has the same property.
4. Verify ngk consistency between header and gvec slab.

If any of these fail, the bare-X centroid values can be slightly wrong.
"""
import os, sys, numpy as np, h5py

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5"

with h5py.File(WFN, 'r') as f:
    fft_grid = tuple(int(v) for v in f["mf_header/gspace/FFTgrid"][:])
    nx, ny, nz = fft_grid
    print(f"FFT grid: {fft_grid}")
    half = np.array([nx//2, ny//2, nz//2])

    ngk = f["mf_header/kpoints/ngk"][:].astype(int)
    nk_ibz = ngk.shape[0]
    print(f"nk_ibz={nk_ibz}, ngk: min={ngk.min()}, max={ngk.max()}")

    # Per-k slab starts
    starts = np.zeros(nk_ibz+1, dtype=int)
    starts[1:] = np.cumsum(ngk)

    gvecs_flat = f["wfns/gvecs"][:].astype(np.int32)
    print(f"gvecs total: {gvecs_flat.shape}, expected {starts[-1]}")
    assert gvecs_flat.shape[0] == starts[-1], "ngk total mismatch with gvec slab!"

    # IBZ checks
    print("\n=== IBZ k slabs ===")
    bad_oob = []   # out-of-box-half
    bad_alias = [] # mod-N collision
    for ik in range(nk_ibz):
        gv = gvecs_flat[starts[ik]:starts[ik+1]]
        amax = np.max(np.abs(gv), axis=0)
        # Allowed range is [-N//2+1, N//2] for even N; [-(N-1)//2, (N-1)//2] for odd
        # Equivalently, |G_i| <= N_i//2 always works without aliasing
        oob_mask = (np.abs(gv[:,0]) > half[0]) | \
                   (np.abs(gv[:,1]) > half[1]) | \
                   (np.abs(gv[:,2]) > half[2])
        if oob_mask.any():
            bad_oob.append((ik, int(oob_mask.sum()), amax.tolist()))

        # mod-N collision check
        wrapped = gv % np.array([nx, ny, nz])
        flat = wrapped[:,0]*ny*nz + wrapped[:,1]*nz + wrapped[:,2]
        if len(np.unique(flat)) != len(flat):
            bad_alias.append((ik, len(flat) - len(np.unique(flat))))

    print(f"  IBZ k's with |G_i|>N_i/2: {len(bad_oob)}")
    if bad_oob:
        for ik, nbad, amax in bad_oob[:5]:
            print(f"    ik={ik} nbad={nbad} max|G|={amax} half={half.tolist()}")
    print(f"  IBZ k's with mod-N collision: {len(bad_alias)}")
    if bad_alias:
        for ik, nc in bad_alias[:5]:
            print(f"    ik={ik} collisions={nc}")

    print("\n=== Full-BZ k slabs (rotated) ===")

# Build fake unfolded kpts via symmetry_maps to be authoritative
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps

wfn = WFNReader(WFN)
sym = SymMaps(wfn)
print(f"  nk_ibz={wfn.nkpts}, nk_full={sym.nk_tot}, ntran={wfn.ntran}, sym_matrices={sym.sym_matrices.shape}")

oob_full = []
alias_full = []
for nk in range(sym.nk_tot):
    gv_rot = np.asarray(sym.get_gvecs_kfull(wfn, nk))
    amax = np.max(np.abs(gv_rot), axis=0)
    oob_mask = (np.abs(gv_rot[:,0]) > half[0]) | \
               (np.abs(gv_rot[:,1]) > half[1]) | \
               (np.abs(gv_rot[:,2]) > half[2])
    if oob_mask.any():
        oob_full.append((nk, int(oob_mask.sum()), amax.tolist()))
    wrapped = gv_rot % np.array([nx, ny, nz])
    flat = wrapped[:,0]*ny*nz + wrapped[:,1]*nz + wrapped[:,2]
    if len(np.unique(flat)) != len(flat):
        alias_full.append((nk, len(flat) - len(np.unique(flat))))

print(f"  full-BZ k's with |G_i|>N_i/2: {len(oob_full)}")
for nk, nb, amax in oob_full[:5]:
    print(f"    nk={nk} nbad={nb} max|G|={amax} half={half.tolist()}")
print(f"  full-BZ k's with mod-N collision: {len(alias_full)}")
for nk, nc in alias_full[:5]:
    print(f"    nk={nk} collisions={nc}")

# Final scatter sanity: actually do the scatter and count nnz
print("\n=== Scatter into FFT box, count nnz ===")
psi_box_test = np.zeros((nx, ny, nz), dtype=np.complex128)
nk_check = min(8, sym.nk_tot)
for nk in range(nk_check):
    psi_box_test[:] = 0.0
    gv_rot = np.asarray(sym.get_gvecs_kfull(wfn, nk))
    # Use a unit ψ_G (one coeff per G) so scatter populates each cell
    psi_box_test[gv_rot[:,0], gv_rot[:,1], gv_rot[:,2]] = 1.0
    nnz = int(np.count_nonzero(psi_box_test))
    expected = gv_rot.shape[0]
    print(f"  nk={nk}  ngk={expected}  scattered_nnz={nnz}  diff={expected-nnz}")
    if nnz != expected:
        print(f"    !! lost {expected-nnz} G-vectors to mod-N aliasing")
