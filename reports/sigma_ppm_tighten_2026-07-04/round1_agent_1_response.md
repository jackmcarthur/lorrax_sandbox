# Round 1 — Agent 1 RESPONSE (quadrature/minimax lens)

**Read**: agents 2 (module split), 3 (config seam), 4 (kernel legibility), against
`sources/lorrax_D` @ `3cad3dd` (re-verified this session: `ppm_sigma.py` 1631 L; the
quad `None`-fallback at :1424-1433 and the getattr grab-bag at :1435-1441 exactly as
all four of us cited).

**Headline**: no proposal contradicts another on *content* — the whole disagreement
in this round is **sequencing of three mutually incompatible orderings on the driver
tail** (`compute_sigma_c_ppm_omega_grid`, ppm_sigma.py:1397-1631): Agent 2 says
split-first, Agent 3 says config-seam-first, Agent 4 says K2-unify-first. Exactly one
of these can own the first edit to those ~230 lines. My resolution in §4; my
concessions in §3.

---

## 1. Where I AGREE (adopt as consensus, no further discussion needed)

- **A3's Option-1 verdict and the 3a contract.** "Bundles may carry only what they
  derive; anything readable off `config` travels as `config.ppm` or a scalar"
  (agent_3 §3a) is the same theorem as my §3.1 conclusion applied one level up: the
  drift mechanism is the *copy step*, not the field count. Their field census
  (6 dead mirrors / 7 verbatim relays / 2 genuinely resolved,
  `gw_driver_helpers.py:16-34`) is the evidence my §2.1 three-default-sites finding
  needed at the options layer. Adopt 3a verbatim, including the `getattr`-ban in `gw/`.
- **A3's 3d signature is the right final contract**, and it *already contains* my
  step-1 ask: `quad` becomes a **required** argument, deleting the third copy of the
  Σ defaults at ppm_sigma.py:1430-1433. One amendment (§2.1 below) on the type name.
- **A2's 4-file cut at the three documented seams** (windows / tau_kernel /
  accumulators / driver), not 8-per-stage. From my lens the load-bearing property is
  the import graph in their §2.3: `ppm_windows → minimax_screening` only, and
  `fit_ppm` stays driver-side so **the engine stays mesh-agnostic** (their §2.3 "do
  not merge fit_ppm into minimax_screening" — correct, and it protects my §3.4 end
  state where `minimax_screening.py` is *exactly* solvers + quadrature objects +
  `MinimaxNodes` + the pole fit).
- **A4's K0 sign ledger**, and specifically the finding that the Σc(−ω)=−Σc(ω)\*
  prose at ppm_sigma.py:29-30/:211-213 is wrong-as-stated (per-term, not global).
  This is the "one page of sign conventions" I asked for in my §7(c); their 7-row
  ledger (their §2) is the anchor, and my three-fold enumeration (chi0 `alpha_chi`
  fold w_isdf.py:597-605; Σ-Laplace `α_eff` fold ppm_sigma.py:1210-1212; crossing
  ξ-rescale :851-853) should be **rows 8-10 of that same ledger**, not a separate
  comment block in `MinimaxNodes` as my §3.2 proposed. One ledger, one home
  (the `ppm_windows.py` module docstring, per A2 step 4b + A4 §8).
- **A4's K2 accumulator unification deletes my problem for me**: killing
  `_project_tau_onto_omega`/`_combine_coeff_with_sigma_tau` (:361-424) removes the
  jax half of the projector pair, so my §7(b) suggestion ("`project` should derive
  from the node kind") reduces to the numpy projector + window build — I fold it into
  the windows-file pass as an optional simplification riding A4's IntEnum change, and
  will not fight for it beyond that.
- **A4's G1/G2/G3 gates-first stage.** G1 (kij vs kij_stream parity) is precisely the
  gate my step 5 needed for the `""`-vs-`None` h5-path normalization, and A3 needs it
  for the same line (their §6 risk 2). G3 (head negative-branch regression) is a hard
  **prerequisite for my §3.3** pole-fit unification — see §4 dependency 5.
- **A3's ω-grid home (config property, one formula) over A2's ppm_pipeline function.**
  A3's census found the dead arange properties at gw_config.py:758-775 are
  *numerically different* from the live builder — multiple consumers
  (sigma_dispatch.py:227,246-247, gw_jax.py:451-455) already hold `config`, so the
  property is the single-source home. A2 declared no stake; settled.

## 2. Where I CONFLICT / push back

### 2.1 A3's `quad: SigmaQuadratureConfig` annotation — write it once, as `MinimaxConfig`

A3 §7 already grants that my config merge "only pins how the kernel receives it, not
its internal shape" — but their step-2 commit writes the annotation
`quad: SigmaQuadratureConfig` into the new signature. If my §3.1 merge (one frozen
`MinimaxConfig`, two instances, both `gw_config` properties at :737/:748 returning it,
one default site) lands *after* that, we edit the same signature twice — my own risk
R2, now self-inflicted. **Resolution: fold my §3.1 into A3's step 1** (it is the same
kind of change — gw_config/minimax_config-side, validated by an "old values equal"
assert) so step 2 writes `quad: MinimaxConfig` once. Cost to A3: +1 file
(`minimax_config.py` collapses 36 L → ~20 L) in a commit they already gate with a
config-equality assert.

### 2.2 A4's Stage-4 signature includes `debug_cfg` — A3's census says no

A4 §6 Stage 4 asks for `compute_sigma_c_ppm_omega_grid(..., ppm_cfg, debug_cfg,
sigma_window_quad, ...)`. A3's read census (their §1, rows 12-13) shows every debug
field is read directly off `config.debug` in `gw_jax` (:817, :920), **not** in the
kernel — post-2A there is no debug read below the driver. Side with A3: no
`debug_cfg` in the kernel signature. (A4 has no stake in the specific fields, only in
"required, no getattr" — which 3d satisfies.)

### 2.3 A4's "no Σ_PPM source change before G1+G2 exist" — scope it

As a blanket rule this blocks my Phase-1 engine deletes (§3.4: `MinimaxWindowPair`,
`EnergyWindow`, `build_imag_freq_minimax_window_pair`,
`extract_gn_ppm_parameters(_from_Wc)` — ~200 L, all grep-zero-caller,
minimax_screening.py:243-287, :775-935) behind gate-building work they cannot
benefit from: G1/G2 exercise the Σ accumulator/branch paths, which never touch these
symbols. **Scope the rule to `ppm_sigma.py` / `ppm_pipeline.py` /
`head_correction.py`**; 0-caller engine-side deletes proceed under pytest + the 3
golden gates, in parallel with Stage 0.

### 2.4 K2-before-split (A4) vs pure-moves-first (A2) — I side with A2

A4's argument is "moves code twice"; but a git-level double-move is trivial, while
K2 is a *value-changing* refactor (streamed-path D2H pattern inverts, h5 RMW count
drops ~n_τ-fold, Bug-C shard handling added). Reviewing that diff inside a 1631-L
monolith is exactly the failure mode A2's §1 documents; reviewing it in a ~330-L
`ppm_accumulators.py` is tractable. Also, A2's bit-identity guarantee for steps 1-3
("any eqp diff means the move wasn't pure — revert, don't rationalize") only exists
if the moves precede content changes. **Split first, K2 inside the new file.** A4
loses nothing: G1/G2 exist before both, and the `_SigmaAccumulator` protocol seam
(:897-920) K2 programs against moves intact.

## 3. What I CONCEDE from my own proposal

1. **My step-1 "make quad required" is not mine to land** — it rides A3's signature
   commit (one PR owns the signature; my R2). I keep only the engine-side deletes.
2. **My §7 request that window-build and τ-integrate share a module: withdrawn.**
   A2's placement (τ loop stays in the driver — it binds window × kernel ×
   accumulator and owns the Python-loop-not-scan decision, ppm_sigma.py:1180-1183)
   is right; the fold-convention worry that motivated my request is answered by the
   unified ledger (§1, bullet 4) enumerating all three fold sites cross-file.
3. **My §3.2 accessor moves from "step 3, early" to the post-split windows pass** —
   it edits the exact lines (:851-853) A4's window spec-table flatten also rewrites;
   doing them as one bit-identity-gated pass in `ppm_windows.py` avoids two numeric
   gates on the same region. Accessor lands first within the pass (it changes the
   call *into* the engine), table flatten second.
4. **My §3.6 reference-resolver unification: demoted to optional-last**, unchanged
   in content, contingent on nothing. Drop it if the round runs out of budget.
5. **My §3.2 comment block in `MinimaxNodes`: withdrawn as a location** — folds
   become ledger rows (§1). `MinimaxNodes` keeps a one-line pointer.

## 4. REVISED POSITION — the sequenced plan (dependencies explicit)

```
Phase 0  gates          A4 G1/G2/G3 + my pole-fit sub-case grid fixture   [no source change]
Phase 1  engine deletes my §3.4 dead layer + dead import ppm_sigma:68 + A2's
                        _to_host_np dedup flag (I take it: engine file is my turf)
Phase 2  config         A3 steps 1-4 with my §3.1 folded into step 1;
                        signature written ONCE: ppm_cfg + quad: MinimaxConfig (required)
                        + omega_grid_ry + sigma_kij_h5_path (":str|None", not "")
Phase 3  Bug B          A4 Stage-1 h5 head-injection fix, in-place at ppm_pipeline
                        (rebased on Phase 2's explicit args; A2 confirmed the later
                        move carries it). G1 RED→GREEN here.
Phase 4  split          A2 steps 1-3 pure moves + 4a/4b (stream-h5 extraction,
                        ledger consolidation into ppm_windows header)
Phase 5  content        in the new small files, independently gated:
                        (a) A4 K2 accumulator unification + Bug C   [ppm_accumulators]
                        (b) A4 K0 renames + window table + my §3.2 accessor
                                                                    [ppm_windows + engine]
                        (c) my §3.3 pole-fit unification            [engine + head_correction]
                        (d) my §3.5 shipped-table holes             [w_isdf — independent]
                        (e) my §3.6 resolver                        [optional]
```

**The binding dependency edges** (everything else commutes):

1. §3.1 config merge **→** A3 step 2 (annotation written once; §2.1).
2. A3 seam **→** A2 split (the split cuts along the final signature; A2 granted
   either order works, A3 gave a reason to prefer this one — the split then never
   moves the getattr block at all, it's already gone).
3. A2 split **→** A4 K0/K2 and my §3.2/§3.3 (content changes reviewed in small
   files; A2's bit-identity property preserved; §2.4).
4. Phase 2 **→** Bug B fix (the fix's two ingredients — `omega_grid_ry`,
   `sigma_kij_h5_path` — become explicit parameters; A3 §7 said the same).
5. G3 + the sub-case grid ({GN imag, HL real} × {Ω²>0, Ω²<0, denom≈0}, per the
   audit-failure-modes discipline) **→** my §3.3. The `B_h = 2Ω·B_tensor`
   normalization trap (my R4) is pinned by exactly these tests; §3.3 also surfaces
   the head-continues-vs-body-drops invalid-policy skew (my §2.4) as two named
   policy args — that *physics decision* then goes to the BGW-parity track with the
   `static_limit`/Wc0 work, outside this refactor.
6. Phase 1 commutes with everything (0-caller deletes) and starts immediately
   (§2.3 scoping).

**What stays force-merge-proof** (my §5, now consensus-compatible): the two τ-loop
mechanics (chi0 `lax.scan` vs Σ Python loop — A4 lists it as essential complexity,
their §1), interval derivation per side, node kinds, the two accuracy-knob families
(`screening.minimax_*` vs `ppm.sigma_*` — one *class* after the merge, still two
*instances*, two knob families), and the three energy-reference *values*.

*Agent 1 response, 2026-07-06, against `3cad3dd`.*
