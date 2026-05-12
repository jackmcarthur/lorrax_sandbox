# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/profile_sharded/xla_dump`
**Modules dumped:** 51
**Sum of per-module peak live HBM:** 3.43 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0039.jit_fft` | 618.75 MiB | 337.50 MiB — preallocated-temp: |
| `module_0041.jit_fft` | 618.75 MiB | 337.50 MiB — preallocated-temp: |
| `module_0047.jit_multiply` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0049.jit_multiply` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0049.jit__moveaxis` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0051.jit__moveaxis` | 450.00 MiB | 225.00 MiB — output shape is \|c128[480,480,4,4,4]\|, maybe-live-out: |
| `module_0035.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0037.jit_scatter-add` | 113.38 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0037.jit__moveaxis` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0039.jit__moveaxis` | 112.50 MiB | 56.25 MiB — output shape is \|c128[240,240,4,4,4]\|, maybe-live-out: |
| `module_0001.jit__identity_fn` | 3.75 MiB | 1.88 MiB — output shape is \|c128[64,4,2,240]\|, maybe-live-out: |
| `module_0019.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0021.jit_add` | 2.64 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0017.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0019.jit_multiply` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0033.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0035.jit__squeeze` | 1.76 MiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0013.jit_multiply` | 907.50 KiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0015.jit_multiply` | 907.50 KiB | 900.00 KiB — output shape is \|c128[240,240]\|, maybe-live-out: |
| `module_0053.jit_reshape` | 320.00 KiB | 160.00 KiB — output shape is \|c128[20,1,2,4,64]\|, maybe-live-out: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0039.jit_fft` | `all-gather-start` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:149` | `(c128[240,480,4,4,4]{4,3,2,1,0}, c128[480,480,4,4,4]{4,3,2,1` |
| `module_0041.jit_fft` | `all-gather-start` | 337.50 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:149` | `(c128[240,480,4,4,4]{4,3,2,1,0}, c128[480,480,4,4,4]{4,3,2,1` |
| `module_0039.jit_fft` | `all-gather-start` | 168.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:149` | `(c128[240,240,4,4,4]{4,3,2,0,1}, c128[240,480,4,4,4]{4,3,2,0` |
| `module_0041.jit_fft` | `all-gather-start` | 168.75 MiB | `/global/homes/j/jackm/software/lorrax_C/src/bse/bse_lanczos.py:149` | `(c128[240,240,4,4,4]{4,3,2,0,1}, c128[240,480,4,4,4]{4,3,2,0` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_multiply` | 10 | 450.00 MiB | 905.29 MiB |
| `jit_convert_element_type` | 6 | 96.00 B | 272.00 B |
| `jit_broadcast_in_dim` | 5 | 7.50 KiB | 22.52 KiB |
| `jit__moveaxis` | 4 | 450.00 MiB | 1.10 GiB |
| `jit_fft` | 2 | 618.75 MiB | 1.21 GiB |
| `jit_scatter-add` | 2 | 113.38 MiB | 226.76 MiB |
| `jit_add` | 2 | 2.64 MiB | 5.27 MiB |
| `jit__squeeze` | 2 | 1.76 MiB | 3.52 MiB |
| `jit_reshape` | 2 | 320.00 KiB | 640.00 KiB |
| `jit__multi_slice` | 2 | 11.25 KiB | 22.50 KiB |
| `jit_conjugate` | 2 | 7.50 KiB | 15.00 KiB |
| `jit__reduce_prod` | 2 | 112.00 B | 224.00 B |
| `jit_dynamic_slice` | 2 | 40.00 B | 80.00 B |
| `jit_squeeze` | 2 | 32.00 B | 64.00 B |
| `jit_sqrt` | 2 | 32.00 B | 64.00 B |
| `jit_concatenate` | 2 | 24.00 B | 48.00 B |
| `jit__identity_fn` | 1 | 3.75 MiB | 3.75 MiB |
| `jit__psum` | 1 | 15.05 KiB | 15.05 KiB |

