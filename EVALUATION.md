# Evaluation

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
