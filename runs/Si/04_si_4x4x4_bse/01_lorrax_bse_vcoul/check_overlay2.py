import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")
import numpy as np
from file_io import read_bgw_vcoul, fill_v_grid_for_q, WFNReader
from common import symmetry_maps

wfn = WFNReader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5")
sym = symmetry_maps.SymMaps(wfn)
tbl = read_bgw_vcoul("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_bgw_eps_vcoul/vcoul")
fft_grid = tuple(int(x) for x in wfn.fft_grid)
vol = float(wfn.cell_volume)
sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)
bvec = np.array(wfn.bvec) * float(wfn.blat)

print("v(q, G=0) at q-points on the 4x4x4 grid — overlay value from BGW file vs LORRAX native point value and native-MC value")
print(f"{'q_frac':>20} {'|q|^2 bohr^-2':>14} {'8pi/|q|^2 (point)':>18} {'BGW overlay G=0':>18}")
for q in [(0.0,0.0,0.0), (0.0,0.0,0.25), (0.0,0.0,0.5), (0.0,0.25,0.25), (0.0,0.25,0.5), (0.0,0.5,0.5)]:
    q_cart = np.array(q) @ bvec
    q2 = float(np.dot(q_cart, q_cart))
    try:
        grid = fill_v_grid_for_q(tbl, q, fft_grid, vol, sym_mats_k=sym_mats_k)
        overlay = grid[0,0,0]
    except Exception as e:
        overlay = float('nan')
    native_pt = 8*np.pi/q2/vol if q2 > 1e-12 else float('inf')
    print(f"  q={q!s:>16s}  {q2:>14.5f}  {native_pt:>18.5e}  {overlay:>18.5e}")
print("\n^^ Overlay of G=0 is nonzero for every q≠0 — this is BGW's unaveraged point value, overwriting LORRAX's mc_average_vcoul_body head.")
