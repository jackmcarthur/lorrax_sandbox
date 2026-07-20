"""study2_verify_npz — cross-check the arbitrary_q_bse.md sec-15 table numbers
against the on-disk result arrays (phantom-table rule)."""
import numpy as np

B = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
     "A_bse_w0_resolvent_2026-07-16/primer_response_study/")

print("== STUDY 1 ladder (study2_study1_MoS2_6x6_results.npz) ==")
d1 = np.load(B + "study2_study1_MoS2_6x6_results.npz")
for r in ("monomial_b26p", "zernike_b26p", "bessel_b26p", "bessel3_b26p",
          "zernike3_b26p", "monomial_b45p", "bessel_b45p"):
    Bs = d1[f"B__{r}"]
    Es = d1[f"exc__{r}"]
    print(f"  {r:<15s} B med {np.median(Bs):.3e} max {np.max(Bs):.3e}  "
          f"exc med {np.median(Es):.3f} max {np.max(Es):.3f}")
print("  angular frac(m==0 mod3):", float(d1["angular_frac3"][0]),
      " pw_m[0:4]:", np.round(d1["angular_pw_m"][:4], 3))
for r in ("monomial_b26p", "bessel_b26p", "zernike_b26p"):
    print(f"  transfer {r:<15s} B med "
          f"{np.median(d1[f'transfer__{r}']):.3e}")

print("\n== STUDY 2 (study2_study2_MoS2_6x6_results.npz) ==")
d2 = np.load(B + "study2_study2_MoS2_6x6_results.npz")
print("  cleaning-eps B med:", {k.split("_")[-1]:
      round(float(np.median(d2[k])), 6)
      for k in d2 if k.startswith("B_eps_")})
print("  ridge B med:", {k.split("_")[-1]:
      round(float(np.median(d2[k])), 6)
      for k in d2 if k.startswith("B_ridge_")})
for smode in ("uniform", "tail"):
    for pol in ("fixed_1e-4", "track_c^-.5"):
        row = [round(float(np.median(d2[f"stretch_{smode}_{pol}_s{s}"])), 5)
               for s in (1.0, 1.25, 1.5, 1.75, 2.0)]
        print(f"  stretch {smode:<8s} {pol:<12s} B med vs sigma: {row}")
