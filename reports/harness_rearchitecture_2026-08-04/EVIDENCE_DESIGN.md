# Evidence-record design — synthesis of non-Weng sources, 2026-08-05

Owner question: the CLAIMS.md suggestion feels bespoke — what do other
measurement-heavy communities do? Three parallel research sweeps: (1) ML
experiment tracking (MLflow, W&B, Sacred, Aim, DVC exp, practitioner
guides), (2) computational-science provenance (AiiDA, signac, Sumatra,
Snakemake/Nextflow/FireWorks, Wilson et al.), (3) continuous benchmarking
and HPC regression practice (asv, Conbench, CodSpeed, Bencher, ReFrame,
buildtest, XDMoD, Caliper/Thicket, GROMACS/LAMMPS/QMCPACK/BerkeleyGW).
Full sweep reports with per-source URLs are preserved in the session
transcript; key sources inline below.

## 1. Verdict on the current design

**The CLAIMS.md pattern is NOT bespoke in its bones.** Three independent
literatures validate it directly:

- The lab-notebook tradition and Wilson et al.'s "Good enough practices in
  scientific computing" (PLOS Comp Bio 2017) prescribe exactly an
  append-only, dated, never-delete notebook of hypothesis → experiment →
  interpretation, with failures as first-class entries.
- Sumatra (the canonical simulation-tracking tool) is structurally a
  CLAIMS row generator: auto-captured context + a manual "reason" at
  launch and "outcome" after — intent→verdict per run.
- W&B's finished-run immutability ("corrections are new runs, never
  edits") is the append-only discipline at industrial scale.
- Most striking: **no mature system anywhere automates the verdict.**
  Every community keeps conclusions, tolerances, baseline blessing, and
  the narrative "why" as human judgment; what they automate is everything
  around it so the judgment layer is short and trustworthy. And
  conclusion-to-evidence linking — CLAIMS' strongest feature — is the
  WEAKEST area of every surveyed tool. The ledger is ahead of the tools
  there, not behind.

**What IS nonstandard:** using hand-written prose as the PRIMARY data
store for machine-capturable facts. All three sweeps converge on the same
split: machine-parseable evidence written BY THE HARNESS, prose reserved
for hypothesis and verdict. The recurring phrase across sources:
"anything logged by hand will eventually be wrong or missing" — which
bites hardest when the authors are agents of varying diligence, and the
agent-era guidance (MLflow agent autologging; SC'25 LLM-provenance work)
says it explicitly: **the harness logs, not the agent**, so no run can be
unrecorded or self-flatteringly recorded.

## 2. The convergent prescription (practices recurring across ≥2 sweeps)

1. **One structured, immutable record per run, written at launch/
   completion by the submission machinery** (Sumatra wrapper, AiiDA
   engine, Nextflow trace, ReFrame perflog, asv results JSON). Fields
   that every schema shares: run/job id, code version (with dirty-tree
   handling — Sumatra REFUSES to run a dirty tree), full config, machine +
   environment fingerprint, start/end, exit status, resource stats,
   artifact inventory. SLURM jobid as the run key is the standard bridge
   (Mila/W&B convention; ReFrame perflogs carry `%(check_jobid)s`).
2. **Baselines and tolerances are versioned artifacts scoped per
   machine.** ReFrame's reference dict — `(value, -tol, +tol, unit)` per
   `system:partition`, in the repo next to the test — is the mature form
   of GATES.md; Bencher's Branch×Testbed×Measure and GROMACS's
   ULP-tolerances-per-precision agree. Tolerance CHOICE stays a physics
   judgment (BerkeleyGW's docs say so verbatim).
3. **Regression detection is automatic over the structured history** —
   either statistics (asv's noise-weighted step detection, Conbench's
   lookback z-score, XDMoD anomaly detection) or engineered determinism +
   tolerance bands (CodSpeed instruction counts; ReFrame; QMCPACK's
   `deterministic` test label). Nobody greps prose to find regressions.
   Corollary: prefer deterministic proxy metrics (op counts, collective
   counts, bytes moved) over wall time where possible — the cache-cold
   HLO table practice, independently validated.
4. **Environment changes are declared events, not discovered surprises.**
   Conbench requires flagging distribution breaks; asv keys on
   machine.json; XDMoD runs scheduled sentinel kernels to detect SYSTEM
   drift independently of code drift. LORRAX translation: fastloop on a
   cron is an application kernel; transport/mesh/gate rulings are
   declared distribution breaks in the run record's context field.
5. **Artifacts carry a durability class, decided at job-definition time.**
   AiiDA's retrieve (small, copied into the permanent record) / stash
   (moved to persistent storage before purge) / expendable (scratch)
   triage is the only first-class answer found to purgeable-scratch
   evidence. The load-bearing numbers supporting a claim are excerpted
   into the permanent record AT COMPLETION, while the artifacts exist.
6. **Content-keying enables "have we already measured this?"** — signac
   statepoint hashes, AiiDA/Nextflow input-hash caching, FireWorks
   duplicate detection. A deck+config+src hash in the run record turns
   duplicate-work detection from memory into a query.
7. **Decisions live next to the code as executable policy.** The
   community version of GATES.md rows is a reference dict a test
   framework READS — "gloo banned at distributed tiers" becomes a check
   that fails a run, with `git log` as its history.

## 3. Revised evidence design for LORRAX

Keep, unchanged: append-only CLAIMS, REFUTED rows forever, human/agent
verdicts against on-disk artifacts, jobid discipline, prose "why".

Change (each item names its outside precedent):

1. **`harness/run_record.py`, called from the sbatch template** (not by
   agents): at submit + completion, append one JSON object per job to
   `runs/records/<year-month>.jsonl` — jobid, bundle/src sha (+ dirty
   refusal, Sumatra-style), deck path + content hash (signac statepoint),
   `system:partition` + env fingerprint (Conbench context), exit status,
   sacct + /proc VmHWM, per-stage walls parsed from the timing table
   (ReFrame perf_patterns), artifact paths each tagged
   retrieve/stash/expendable (AiiDA), and a free `context_break` field
   for declared environment changes. Schema change ⇒ rotate file, never
   mix (ReFrame v4 rule).
2. **CLAIMS rows shrink to judgment**: hypothesis/claim, verdict, and
   jobid(s) — the record file carries everything else; the claim file
   (claims/NNNN.md) embeds the retrieve-class excerpts captured at
   completion. This is the §I proposal from RULES_v2, now grounded:
   excerpt-at-landing = AiiDA retrieve; the structured sidecar = ReFrame
   perflog; prose-on-top = lab-notebook tradition.
3. **GATES.md numeric rows become a reference dict** the fastloop/parity
   harness reads: metric, baseline, ±tolerance, unit, scope
   (deck × system:partition), blessed-by (jobid + CLAIMS row). Re-pinning
   is the scripted, reviewed golden-file blessing (GROMACS pattern) it
   already almost is.
4. **`tools/trend.py` over the jsonl** starts as `asv compare`-style A/B
   plus a per-stage history table; add asv-style step detection only if
   eyeballing the table ever misses a real regression.
5. **Sentinel cadence**: fastloop deck on a schedule (queue permitting,
   weekly is enough) with results in the same jsonl, tagged sentinel —
   separates "Frontera moved" from "the code moved" (XDMoD).

Explicitly rejected for this context, with reasons:
- **AiiDA / FireWorks / MLflow server / any daemon+DB stack**: Frontera
  login nodes forbid daemons and cap processes (RLIMIT_NPROC 300); the
  plugin tax for a bespoke JAX code is weeks; and hosted trackers are a
  liability (Neptune SaaS shut down 2026-03). The environment itself
  disqualifies them — files only.
- **Full workflow-manager adoption** (Snakemake/Nextflow): the chain is 6
  drivers, already orchestrated by certified sbatch templates; the
  migration buys telemetry the jsonl gets for ~100 lines of Python.
- **Automating verdicts**: no surveyed community does it; the judgment
  layer stays prose, by design.

## 4. Sources (primary)

ReFrame (CSCS) reference/perflog model: reframe-hpc.readthedocs.io; HUST'19
paper. asv results/step-detection: asv.readthedocs.io. Conbench lookback
z-score + context: conbench.github.io. Bencher thresholds: bencher.dev.
XDMoD app kernels: appkernels.xdmod.org; Simakov 2015. Caliper/Adiak/
Thicket: software.llnl.gov/Caliper. AiiDA provenance + stashing:
aiida.readthedocs.io; Pizzi et al. arXiv:1504.01163; Sci Data 2020.
signac: docs.signac.io. Sumatra: sumatra.readthedocs.io. Wilson et al.
PLOS Comp Bio 2017 (Good enough practices); PLOS Ten Simple Rules (lab
notebooks). MLflow tracking schema + agent autologging: mlflow.org. W&B
run immutability: docs.wandb.ai. Sacred observers: sacred.readthedocs.io.
DVC exp: dvc.org. Trackio JSONL-on-NFS: github.com/gradio-app/trackio.
GROMACS refdata (ULP tolerances): manual.gromacs.org. QMCPACK
deterministic tests + CDash: qmcpack.org. BerkeleyGW testsuite:
manual.berkeleygw.org. Jacamar CI (SLURM CI plumbing): ecp-ci.gitlab.io.
LLM-agent provenance: MLflow blog; SC'25 workshop 10.1145/3731599.3767582.
