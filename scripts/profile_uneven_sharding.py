"""profile_uneven_sharding.py — what does XLA SPMD do on an uneven product-axis
sharding constraint?

Three test cases, each compiled with HLO text dump.  All run on a 4×4
device mesh (16 GPUs) with axes ('x', 'y').  Input array shape is
(Q, n_G, μ) where μ is the variable.  Sharding under test is
``P(None, None, ('x','y'))`` on μ — the V_q tile's ζ-disk read.

  A. μ = 672 = 16 · 42.   Mesh-divisible product.  Baseline.
  B. μ = 668 = 4 · 167.   Divisible by p_x and p_y separately, NOT by
                          p_x · p_y = 16.  This is the "user's worry"
                          configuration.
  C. μ = 661 (prime).      Divisible by neither 4 nor 16.  Worst case.

For each μ value we compile two JIT signatures:
  1. ``jit(f)``           where f's *input* arrives on
                          ``P(None, None, ('x','y'))``.  Tests whether
                          uneven top-level sharding errors / works /
                          rematerializes.
  2. ``jit(f_wsc)``       where f_wsc's input arrives on
                          ``P(None, None, 'x')`` (single-axis, divides
                          μ_C and μ_T cleanly for B), then inside the
                          jit a ``with_sharding_constraint`` to the
                          uneven product layout.  Tests whether
                          intermediates can be uneven without
                          all-gather inflation.

We dump HLO text for each compiled module and count
``all-gather`` / ``all-to-all`` / ``collective-permute`` occurrences.
A clean reshard from single-axis → product-axis-on-divisible-extent
is a single ``all-to-all`` per mesh axis we add (one); a "fall back to
rematerialize" is a much larger ``all-gather``.

Usage::

    LORRAX_NNODES=4 lxrun python3 -u \
        scripts/profile_uneven_sharding.py --out ./uneven_test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./uneven_test",
                    help="Artifacts dir for HLO dump + summary")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # XLA HLO dump must be set before jax import.
    xla_dump_dir = out / "xla_dump"
    xla_dump_dir.mkdir(exist_ok=True)
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={xla_dump_dir} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_include_timestamp=false "
        "--xla_dump_hlo_pass_re=spmd-partitioner|sharding-propagation"
    )
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    # Lazy: defer to LORRAX's distributed init for the Cray-MPICH GPU detection.
    sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_A/src")
    from runtime import init_jax_distributed
    init_jax_distributed()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as np

    n_devices = jax.device_count()
    if n_devices != 16:
        print(f"WARNING: expected 16 devices, got {n_devices}.  "
              "Adjust LORRAX_NNODES for a 4x4 mesh.", file=sys.stderr)

    devices_2d = np.asarray(jax.devices()).reshape(4, 4)
    mesh = Mesh(devices_2d, axis_names=("x", "y"))

    # Constants matching the V_q tile's ζ-disk slab shape on a 4x4 mesh.
    Q = 9
    n_G = 2419

    rep = NamedSharding(mesh, P())
    sx_only = NamedSharding(mesh, P(None, None, "x"))
    syx_prod = NamedSharding(mesh, P(None, None, ("x", "y")))

    def _make_input(mu: int, target_sh):
        """Build a deterministic c128 array of shape (Q, n_G, μ) sharded ``target_sh``."""
        # Build replicated, then shard.  device_put with a NamedSharding
        # that doesn't divide → JAX-level error today.
        arr_np = (np.arange(Q * n_G * mu, dtype=np.int64)
                  .reshape(Q, n_G, mu) * 1e-3).astype(np.complex128)
        return jax.device_put(jnp.asarray(arr_np), target_sh)

    def _summarize_hlo(hlo_text: str) -> dict:
        return {
            "all-gather":           len(re.findall(r"\ball-gather\b(?!-start|-done)", hlo_text)),
            "all-gather-start":     hlo_text.count("all-gather-start"),
            "all-to-all":           hlo_text.count("all-to-all"),
            "collective-permute":   hlo_text.count("collective-permute"),
            "involuntary remat":    len(re.findall(r"Involuntary full rematerialization",
                                                    hlo_text, re.IGNORECASE)),
        }

    cases = [
        ("A_mu672_div16",  672),
        ("B_mu668_div4",   668),
        ("C_mu661_prime",  661),
    ]

    summary_lines = ["# Uneven-sharding XLA partitioner microbench",
                     "",
                     f"Mesh: 4x4 ({n_devices} devices)",
                     "Input shape: (9, 2419, μ).  Tested μ: 672, 668, 661.",
                     "",
                     "## Test 1 — top-level input on `P(None, None, ('x','y'))`",
                     "",
                     "```",
                     f"{'case':<20} {'mu':>4}  {'mu%16':>5}  result",
                     "```",
                     ""]

    for tag, mu in cases:
        try:
            arr = _make_input(mu, syx_prod)
            result = "device_put OK"
            ag = ato = 0
            try:
                @jax.jit
                def f1(x):
                    return x.sum(axis=-1)

                lowered = f1.lower(arr).compile()
                hlo = lowered.as_text()
                summary = _summarize_hlo(hlo)
                ag = summary["all-gather"] + summary["all-gather-start"]
                ato = summary["all-to-all"]
                result = (f"compiled.  ag={ag}  ag-start={summary['all-gather-start']}  "
                          f"a2a={ato}  cperm={summary['collective-permute']}  "
                          f"remat={summary['involuntary remat']}")
                (out / f"hlo_test1_{tag}.txt").write_text(hlo)
            except Exception as e2:
                result = f"compile error: {type(e2).__name__}: {e2}"
        except Exception as e:
            result = f"device_put error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<20} {mu:>4}  {mu % 16:>5}  {result}")

    summary_lines.append("")
    summary_lines.append("## Test 2 — input on `P(None, None, 'x')`, "
                         "wsc inside jit to `P(None, None, ('x','y'))`")
    summary_lines.append("")
    summary_lines.append("```")
    summary_lines.append(f"{'case':<20} {'mu':>4}  {'mu%16':>5}  result")
    summary_lines.append("```")

    for tag, mu in cases:
        try:
            arr_x = _make_input(mu, sx_only)
            try:
                @jax.jit
                def f2(x):
                    y = jax.lax.with_sharding_constraint(x, syx_prod)
                    return y.sum(axis=-1)

                lowered = f2.lower(arr_x).compile()
                hlo = lowered.as_text()
                summary = _summarize_hlo(hlo)
                ag = summary["all-gather"] + summary["all-gather-start"]
                ato = summary["all-to-all"]
                result = (f"compiled.  ag={ag}  ag-start={summary['all-gather-start']}  "
                          f"a2a={ato}  cperm={summary['collective-permute']}  "
                          f"remat={summary['involuntary remat']}")
                (out / f"hlo_test2_{tag}.txt").write_text(hlo)
            except Exception as e2:
                result = f"compile error: {type(e2).__name__}: {e2}"
        except Exception as e:
            result = f"device_put error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<20} {mu:>4}  {mu % 16:>5}  {result}")

    summary_lines.append("")
    summary_lines.append("## Test 3 — input padded inside jit, then wsc to product")
    summary_lines.append("")
    summary_lines.append("Padded path: input on `P(None, None, 'x')` at logical μ; "
                         "inside jit jnp.pad to mesh-divisible padded μ, "
                         "wsc to `P(None, None, ('x','y'))`, sum.  Compares "
                         "Test 2's uneven-wsc cost against an explicit pad.")
    summary_lines.append("")
    summary_lines.append("```")
    summary_lines.append(f"{'case':<20} {'mu':>4}  {'mu%16':>5}  result")
    summary_lines.append("```")

    for tag, mu in cases:
        mu_padded = ((mu + 15) // 16) * 16
        try:
            arr_x = _make_input(mu, sx_only)
            try:
                @jax.jit
                def f3(x):
                    pad_after = mu_padded - mu
                    y = jnp.pad(x, ((0, 0), (0, 0), (0, pad_after)))
                    y = jax.lax.with_sharding_constraint(y, syx_prod)
                    return y.sum(axis=-1)

                lowered = f3.lower(arr_x).compile()
                hlo = lowered.as_text()
                summary = _summarize_hlo(hlo)
                ag = summary["all-gather"] + summary["all-gather-start"]
                ato = summary["all-to-all"]
                result = (f"compiled  μ_pad={mu_padded}.  "
                          f"ag={ag}  ag-start={summary['all-gather-start']}  "
                          f"a2a={ato}  cperm={summary['collective-permute']}  "
                          f"remat={summary['involuntary remat']}")
                (out / f"hlo_test3_{tag}.txt").write_text(hlo)
            except Exception as e2:
                result = f"compile error: {type(e2).__name__}: {e2}"
        except Exception as e:
            result = f"device_put error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<20} {mu:>4}  {mu % 16:>5}  {result}")

    # =========================================================================
    # Tests 4-6: V_q-tile-shaped kernel body — closer to the real LORRAX kernel
    # so we know the simple-sum result generalizes.  Body mimics
    # ``_make_V_q_tile_kernel._kernel_body`` at gw/v_q_tile.py:692 — a
    # multi-WSC chain culminating in a 3-D einsum and dynamic_update_slice.
    # =========================================================================

    blk_xy_sh = NamedSharding(mesh, P(None, ("x", "y"), None))    # μ on product axis
    blk_x_sh  = NamedSharding(mesh, P(None, "x", None))           # μ on 'x' single
    blk_y_sh  = NamedSharding(mesh, P(None, "y", None))           # μ on 'y' single
    V_sh      = NamedSharding(mesh, P(None, "x", "y"))            # V_block layout

    summary_lines.append("")
    summary_lines.append("## Test 4 — V_q-tile-like kernel body, μ=672 (mesh-divisible)")
    summary_lines.append("")
    summary_lines.append("Body: WSC chain `P(None,None,('x','y'))` → `P(None,('x','y'),None)` → "
                         "`P(None,'x',None)` & `P(None,'y',None)` → einsum → "
                         "`P(None,'x','y')` → DUS into V_acc.  Mirrors the V_q tile inner kernel.")
    summary_lines.append("")
    summary_lines.append("```")
    summary_lines.append(f"{'case':<24} {'mu':>4}  {'mu%16':>5}  result")
    summary_lines.append("```")

    def _vq_body_factory(mu_eff: int):
        """Build a function f(zeta_disk, v_per_G, V_acc) → V_acc that mirrors
        gw/v_q_tile.py:_kernel_body for one (μ × ν) block.  ``mu_eff`` is
        the size used in WSCs."""
        def f(zeta_disk, v_per_G, V_acc):
            # zeta_disk: (Q, n_G, mu_eff) sharded P(None,None,('x','y'))
            zeta_T = jax.lax.with_sharding_constraint(
                jnp.transpose(zeta_disk, (0, 2, 1)), blk_xy_sh)
            # zeta_T: (Q, mu_eff, n_G) on P(None,('x','y'),None)
            zeta_mu_X = jax.lax.with_sharding_constraint(zeta_T, blk_x_sh)
            zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_T, blk_y_sh)
            zeta_mu_X = jax.lax.with_sharding_constraint(
                zeta_mu_X * v_per_G[:, None, :], blk_x_sh)
            V_block = jnp.einsum("qmG,qnG->qmn",
                                 jnp.conj(zeta_mu_X), zeta_nu_Y, optimize=True)
            V_block = jax.lax.with_sharding_constraint(V_block, V_sh)
            V_new = jax.lax.dynamic_update_slice(
                V_acc, V_block, (jnp.int32(0), jnp.int32(0), jnp.int32(0)))
            return jax.lax.with_sharding_constraint(V_new, V_sh)
        return f

    # Test 4: μ=672 — fully divisible, no padding gymnastics
    for tag, mu in cases:
        if mu != 672:
            continue
        try:
            zeta = _make_input(mu, syx_prod)                      # uneven errors here for 668
            v    = jax.device_put(jnp.ones((Q, n_G), dtype=jnp.complex128) * 0.5,
                                  NamedSharding(mesh, P(None, None)))
            V0   = jax.device_put(jnp.zeros((Q, mu, mu), dtype=jnp.complex128), V_sh)
            f4 = jax.jit(_vq_body_factory(mu))
            lowered = f4.lower(zeta, v, V0).compile()
            hlo = lowered.as_text()
            summary = _summarize_hlo(hlo)
            ag = summary["all-gather"] + summary["all-gather-start"]
            ato = summary["all-to-all"]
            result = (f"compiled.  ag={ag}  ag-start={summary['all-gather-start']}  "
                      f"a2a={ato}  cperm={summary['collective-permute']}  "
                      f"remat={summary['involuntary remat']}")
            (out / f"hlo_test4_{tag}.txt").write_text(hlo)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<24} {mu:>4}  {mu % 16:>5}  {result}")

    summary_lines.append("")
    summary_lines.append("## Test 5 — V_q-tile-like body, INPUT padded inside jit (logical 668 → 672)")
    summary_lines.append("")
    summary_lines.append("Input arrives single-axis sharded at logical μ; jit pads to 672; "
                         "kernel body runs on padded shape.  Tests the 'pad-inside-jit + clean kernel' "
                         "pattern the helper module would expose at top-level boundaries.")
    summary_lines.append("")
    summary_lines.append("```")
    summary_lines.append(f"{'case':<24} {'mu':>4}  {'mu%16':>5}  result")
    summary_lines.append("```")

    for tag, mu in cases:
        if mu == 672:
            continue
        mu_padded = ((mu + 15) // 16) * 16
        try:
            zeta_x = _make_input(mu, sx_only)                     # 661 errors here
            v      = jax.device_put(jnp.ones((Q, n_G), dtype=jnp.complex128) * 0.5,
                                    NamedSharding(mesh, P(None, None)))
            V0     = jax.device_put(jnp.zeros((Q, mu_padded, mu_padded), dtype=jnp.complex128),
                                    V_sh)

            body = _vq_body_factory(mu_padded)

            @jax.jit
            def f5(z_logical, v, V_acc):
                pad_after_n = mu_padded - mu
                z_padded = jnp.pad(z_logical, ((0, 0), (0, 0), (0, pad_after_n)))
                z_padded = jax.lax.with_sharding_constraint(z_padded, syx_prod)
                return body(z_padded, v, V_acc)

            lowered = f5.lower(zeta_x, v, V0).compile()
            hlo = lowered.as_text()
            summary = _summarize_hlo(hlo)
            ag = summary["all-gather"] + summary["all-gather-start"]
            ato = summary["all-to-all"]
            result = (f"compiled  μ_pad={mu_padded}.  "
                      f"ag={ag}  ag-start={summary['all-gather-start']}  "
                      f"a2a={ato}  cperm={summary['collective-permute']}  "
                      f"remat={summary['involuntary remat']}")
            (out / f"hlo_test5_{tag}.txt").write_text(hlo)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<24} {mu:>4}  {mu % 16:>5}  {result}")

    summary_lines.append("")
    summary_lines.append("## Test 6 — V_q-tile-like body with INTERMEDIATES on uneven product spec")
    summary_lines.append("")
    summary_lines.append("This is the user's worry case: WSC inside jit to product spec on logical 668 "
                         "(NOT padded).  XLA must reshape an uneven product axis on every "
                         "intermediate.  Compares HLO collective count to Test 5 (padded path).")
    summary_lines.append("")
    summary_lines.append("```")
    summary_lines.append(f"{'case':<24} {'mu':>4}  {'mu%16':>5}  result")
    summary_lines.append("```")

    for tag, mu in cases:
        if mu == 672:
            continue
        try:
            zeta_x = _make_input(mu, sx_only)                     # 661 errors here
            v      = jax.device_put(jnp.ones((Q, n_G), dtype=jnp.complex128) * 0.5,
                                    NamedSharding(mesh, P(None, None)))
            # V_acc on the uneven product spec: device_put errors top-level for 668.  Use
            # a single-axis V_acc input and let the body's WSC migrate it inside the jit.
            V0     = jax.device_put(jnp.zeros((Q, mu, mu), dtype=jnp.complex128), blk_x_sh)

            body = _vq_body_factory(mu)

            @jax.jit
            def f6(z_logical, v, V_acc):
                # Move z to product spec INSIDE jit on uneven extent
                z_prod = jax.lax.with_sharding_constraint(
                    jnp.transpose(z_logical, (0, 2, 1)), syx_prod)  # NB: (Q, n_G, μ) shape needed
                # Wait — z_logical is (Q, n_G, μ) on sx_only.  Transpose to (Q, μ, n_G) is wrong.
                # Pass the unsharded view through: just WSC to syx_prod (no transpose).
                z_prod = jax.lax.with_sharding_constraint(z_logical, syx_prod)
                # Build a fresh V_acc with the layout the body expects.
                V_acc_target = jax.lax.with_sharding_constraint(V_acc, V_sh)
                return body(z_prod, v, V_acc_target)

            lowered = f6.lower(zeta_x, v, V0).compile()
            hlo = lowered.as_text()
            summary = _summarize_hlo(hlo)
            ag = summary["all-gather"] + summary["all-gather-start"]
            ato = summary["all-to-all"]
            result = (f"compiled  no-pad path.  "
                      f"ag={ag}  ag-start={summary['all-gather-start']}  "
                      f"a2a={ato}  cperm={summary['collective-permute']}  "
                      f"remat={summary['involuntary remat']}")
            (out / f"hlo_test6_{tag}.txt").write_text(hlo)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        summary_lines.append(f"  {tag:<24} {mu:>4}  {mu % 16:>5}  {result}")

    summary_text = "\n".join(summary_lines) + "\n"
    if jax.process_index() == 0:
        print(summary_text)
        (out / "summary.md").write_text(summary_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
