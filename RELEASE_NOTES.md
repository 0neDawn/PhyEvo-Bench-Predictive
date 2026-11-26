# Native validation release

Nine predictive models, SP/IE, baseline/HSR: all 36 original-frame evaluations
completed with 300 pairs per run. Independent audits verified all 180 displayed
metric values, 90 displayed differences, per-pair decisions and input hashes.
The code includes exact configurations, raw errors, case traces, tables and figures.

Hardware: one NVIDIA GeForce RTX 4060 Ti 16 GB, Intel Core i5-12600KF,
approximately 32 GB RAM. See ENVIRONMENTS.md for both Python/PyTorch environments
and the serial execution/memory settings.

CPU-only table verification:

```bash
bash reproduce.sh --mode verify
```

Full inference after the documented environment and asset setup:

```bash
bash reproduce.sh --mode full --wait-for-resources
```

The release data attachments contain the original SP and IE frames. Third-party
checkpoints are obtained from their original distributions and validated against
configs/weights.json; ASSETS.md gives the mapping. They are not included in Git.

DINO-Foresight remains diagnostic only because the archived evaluation retains
a temporal mismatch. Its tiny raw-error differences change no reported metric.
Settings were selected on the reporting pairs, without a held-out selection set.
All values and negative changes are retained.
