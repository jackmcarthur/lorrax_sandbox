"""One-off: fit the fit_one_rchunk FLOPs cost model from existing
samples.json and print the coefficients.  Re-runnable after new sweeps."""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")
from gw.aot_memory_model.cost import fit_cost_from_saved, predict_flops_per_call
from gw.aot_memory_model.core import get_kernel, load_samples, MeshSpec, SysDims, Knobs

fit = fit_cost_from_saved("fit_one_rchunk", tag="current")
print(f"\nCost fit: {fit.kernel} ({fit.n_samples} samples)")
print(f"  intercept = {fit.intercept:.3e} FLOPs")
for name, coef in zip(fit.feature_names, fit.coefs):
    print(f"  β[{name:13s}] = {coef:.3e}")
print(f"  residual RMS = {fit.residual_rms / 1e9:.3f} GFlops")

# Sanity check: predict per-call FLOPs at the DoE points and compare
kernel = get_kernel("fit_one_rchunk")
raw = load_samples("fit_one_rchunk", tag="current")
print(f"\nPer-call prediction vs actual (first 8 points):")
print(f"  {'chunk_r':>7} {'bc':>3} {'n_rmu':>5} {'n_b':>4} {'pred':>8} {'actual':>8} {'err%':>6}")
for s in raw[:8]:
    sys_kw = dict(s["sys"])
    sys_kw["kgrid"] = tuple(sys_kw["kgrid"])
    if sys_kw.get("fft_grid") is not None:
        sys_kw["fft_grid"] = tuple(sys_kw["fft_grid"])
    sys_kw.pop("n_k", None)
    sd = SysDims(**sys_kw)
    kn = Knobs.of(**s["knobs"])
    ms = MeshSpec(**s["mesh"])
    pred = predict_flops_per_call(fit, kernel, sd, kn, ms)
    actual = s["meas"]["flops"]
    err = 100 * (pred - actual) / actual
    print(f"  {kn.get('chunk_r'):>7} {kn.get('band_chunk'):>3} "
          f"{sd.n_rmu:>5} {sd.n_b:>4} {pred/1e9:>7.2f}G {actual/1e9:>7.2f}G "
          f"{err:>+5.1f}%")
