"""Validate the AOT chooser: enumerate candidate (chunk_r, band_chunk)
and print their memory + cost predictions so we can eyeball the pick."""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")
from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs,
    get_kernel, load_fit, load_cost_fit,
    predict_peak, predict_flops_per_call,
    choose_chunks_aot, describe_chunks,
)
from gw.aot_memory_model.chooser import _enumerate_candidates, _ceil_div

# MoS2 3×3 baseline (matches the smoke test)
sys_mos2 = SysDims(
    kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
    n_rmu=640, n_s=2, n_b=80, n_r=80*72*8,
)
mesh = MeshSpec(2, 2)
budget = 28e9  # A100 28 GB budget

kernel = get_kernel("fit_one_rchunk")
mem_fit = load_fit("fit_one_rchunk", tag="current")
cost_fit = load_cost_fit("fit_one_rchunk", tag="current")

n_rtot = sys_mos2.n_r
print(f"System: MoS2 3×3 nosym, n_rtot={n_rtot}, n_b={sys_mos2.n_b}, mesh={mesh.p_x}x{mesh.p_y}")
print(f"Budget: {budget/1e9:.1f} GB\n")

print(f"{'chunk_r':>7} {'bc':>3} {'num_r':>5} {'peak_GB':>8} {'%':>5} "
      f"{'per_call':>9} {'total_GF':>9} {'feas':>4}")
rows = []
for cr, bc in _enumerate_candidates(sys_mos2, mesh):
    kn = Knobs.of(chunk_r=cr, band_chunk=bc)
    peak = predict_peak(mem_fit, kernel, sys_mos2, kn, mesh)
    pc = predict_flops_per_call(cost_fit, kernel, sys_mos2, kn, mesh)
    nr = _ceil_div(n_rtot, cr)
    total = nr * pc
    feas = peak <= budget
    rows.append((cr, bc, nr, peak, pc, total, feas))
    print(f"{cr:>7} {bc:>3} {nr:>5} {peak/1e9:>7.2f} "
          f"{100*peak/budget:>4.0f}% {pc/1e9:>8.2f}G {total/1e9:>7.2f}G "
          f"{'Y' if feas else 'N':>4}")

print()
choice = choose_chunks_aot(
    sys_mos2, mesh, budget_bytes=budget,
    kernel_name="fit_one_rchunk", tag="current",
)
print(describe_chunks(choice))

print(f"\n[Compare with the current heuristic at MoS2 3×3: "
      f"chunk_r=46080, band_chunk=80]")
