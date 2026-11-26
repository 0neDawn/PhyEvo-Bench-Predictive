# PhyEvo-Bench: predictive evaluation and HSR

This release packages the final predictive-model experiments: nine models,
300 State Persistence (SP) pairs and 300 Interaction Evolution (IE) pairs,
baseline/HSR outputs, and one recorded HSR configuration per model and criterion.

**Validation: 36/36 native runs completed.** All 180 displayed metric values and
90 displayed differences match the archived table. See [VALIDATION.md](VALIDATION.md)
for the independent checks and the small, metric-neutral DINO error differences.

## Hardware and environment

The release validation ran serially on **one NVIDIA GeForce RTX 4060 Ti (16 GB)**,
with an Intel Core i5-12600KF and approximately 32 GB system RAM. The host used
Ubuntu 26.04.1 LTS and NVIDIA driver 580.178.04. Most models used Python 3.10.19,
PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124; DDLP used a separate Python
3.10.20 environment with PyTorch 2.6.0+cu126 and torchvision 0.21.0+cu126.

Installation commands, model-specific dependencies and memory settings are in
[ENVIRONMENTS.md](ENVIRONMENTS.md). The machine-readable record is
[environments/validated_runtime.json](environments/validated_runtime.json).
These are the tested host settings, not minimum hardware requirements.

## Reported results

These tables were independently recomputed from fresh native predictions and
verified against the archived per-pair errors. Improvements and degradations
are both shown. Verification details are in [VALIDATION.md](VALIDATION.md).

![State Persistence results](results/figures/SP_results.png)
![Interaction Evolution results](results/figures/IE_results.png)

Vector PDFs: [SP](results/figures/SP_results.pdf),
[IE](results/figures/IE_results.pdf).

## Reproduce the table without running a model

```bash
python -m pip install -r requirements-metrics.txt
bash reproduce.sh --mode verify
```

This recomputes PairAcc, VSA, AUC, Localization MAE and the joint SP/IE pass rate
from the archived per-pair error arrays. It checks all 180 reported values against
the archive, verifies the same pair IDs and temporal grids, and writes readable
tables to `recomputed/SP.csv`, `recomputed/IE.csv` and their Markdown counterparts.
It does not load weights or perform inference.
When the package includes completed native reruns, this command also verifies
their per-pair outputs and case traces in `audit/native_reproduction/`; all 36
native outputs must pass. The result is `recomputed/native_verification.json`.

## Run all models from the original frames

Use the inference environments described in [ENVIRONMENTS.md](ENVIRONMENTS.md).
Download the two original-data files and `assets_manifest.json` from the
repository Release. Restore the data, obtain the 13 upstream checkpoints at the
paths listed in [ASSETS.md](ASSETS.md), and fetch the pinned OCVP code:

```bash
python scripts/restore_assets.py /path/to/downloads --only data_sp data_ie
python scripts/fetch_ocvp.py
```

Checkpoint hashes are fixed in `configs/weights.json`. Checkpoints are obtained
from their original distributions and are not included in the Git repository.
If you have the complete local data/checkpoint asset bundle, omit `--only` to
restore all 15 assets together.

Run the following once to execute all 36 evaluations, sequentially:

```bash
bash reproduce.sh --mode full --wait-for-resources
```

The launcher verifies all input-frame hashes and all 13 checkpoints before
inference. It uses the exact settings in `configs/final_configs.json`, checks
the archived per-case neural call graph, keeps the original prediction horizon,
and compares every metric with the archived result. It never tunes parameters
or substitutes archived errors for fresh predictions. Each model/criterion has
a separate baseline and HSR run. DINO is included with its diagnostic status.

For separate Python environments:

```bash
export PHYEVO_PYTHON=/path/to/main-environment/bin/python
export PHYEVO_DDLP_PYTHON=/path/to/ddlp-environment/bin/python
bash reproduce.sh --mode full --wait-for-resources
```

Restarting the same command resumes completed cases and skips completed runs.
Concurrent invocations targeting the same run directory are rejected.
Results, exact-match flags and logs are retained under `runs/full_reproduction/`.
`COMPLETE.json` is written only after all 300 cases and all metric checks pass.
A failed check remains a failed check; it is not replaced by the paper value.
The final summary independently reloads the new per-pair errors, recomputes all
overall and per-type metrics, and checks the displayed values and deltas against
the table. Its report is `independent_metric_verification.json` in the run directory.
Run a single configuration with:

```bash
python scripts/run_evaluation.py --model ocvp_seq --task IE --variant hsr
```

The default GPU allocator fraction is 0.45. Run one model per GPU; the launcher
waits when the GPU is occupied or system memory is low. This fraction limits
PyTorch's GPU allocator, not total host RAM. On Linux, a process/container memory
limit is recommended for long runs. See [VALIDATION.md](VALIDATION.md) for the
release validation status and the limits of the checks performed.

## Data and protocol

The final manifest is `data/pairs_8to6.csv`. It concatenates the actual frozen SP
and IE evaluation manifests, not the older generic runner manifest. In
particular, IE uses `motion_Animaly2`, `penetration` and `contact_response` data;
the older `no_contact` subset is not used. Each criterion has three types with
100 pairs each. All models receive the same eight observed original frames
(indices 12–19) and evaluate six future original frames (20–25), with a shared
history for the plausible and implausible candidate.

V-JEPA 2 and VideoMAE-V2 represent time in two-frame blocks: four context blocks
and three future error blocks. VSA reads the block containing the onset;
localization assigns its error to both covered frames. Native temporal context
inside a model is retained (including DINO's five encoded history frames).

PairAcc and VSA use strict `>` comparisons; ties fail. SP/IE is the fraction
passing both. AUC pools pair-normalized mean-error scores. Loc. MAE is in
original-frame units and is conditioned on correct PairAcc and a detected onset;
its denominator is provided with each result. `benchmark_metrics.py` is copied
unchanged from the final audit. Displayed deltas subtract the displayed baseline
value from the displayed HSR value, using two decimals for percentages and three
for MAE.

## HSR configurations and interpretation

With historical mean `c`, maximum Euclidean deviation `R`, and `d = ||y-c||`,
the implementation uses `beta = eta * max(0, 1-alpha*R/d)` and
`a = (1-beta)*y + beta*c`, with `beta=0` for `d=0` or an invalid reference.
The exact representation, filtering constants, insertion mode and layer are
recorded in `configs/final_configs.json` and per-model result provenance.
HSR does not train or change weights and adds no neural forward passes relative
to the corresponding native baseline. Models have different native forward
budgets; G-SWM retains its eight native posterior rollouts.

Configurations were selected separately for SP and IE on the same 300 pairs used
for reporting, without a held-out selection set. Every row uses one configuration
for all five metrics. The complete 5,733-setting record is retained in
`audit/all_parameter_results.json.gz`. VideoMAE's reported setting was selected
post hoc for 1–2 percentage-point PairAcc/joint gains; SlotFormer IE retains the
previously selected setting. These are the reported configurations, not a claim
of an unbiased held-out optimum.

**DINO-Foresight is diagnostic only.** The first masked output reconstructs
the last observed frame but is scored against the next frame. Both baseline and
HSR retain that temporal mismatch. Its scores are excluded from formal rankings,
aggregate improvement statistics and claims of HSR effectiveness.

## Repository layout

- `scripts/`: portable reproduction, input checks, asset restoration and tables.
- `configs/`: exact final configurations, per-case execution schedules, weights.
- `results/SP/` and `results/IE/`: baseline/HSR metrics, raw errors, case records
  and compressed provenance for each model.
- `predictive_tracks/`: archived evaluation adapters and correction implementations.
  The dated experiment drivers are retained for provenance; the supported release
  entry point is `reproduce.sh`, not those historical drivers.
- `third_party/`, `ddlp/`, `VideoMAEv2/`, `jepa-intuitive-physics/`: runtime code with
  its original notices. OCVP is fetched separately at an exact revision.
- `audit/`: source checksums, table correspondence and parameter-selection record.
- `data_generation/`: original rendering scripts for reference. Reproduction uses
  the supplied original frames and does not regenerate the benchmark.

The package contains predictive experiments only; it does not include the separate
generative-video HSR project. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream
sources and licenses. No third-party checkpoint is relicensed by this package.
