# Release validation

All 36 native evaluations (nine models, SP/IE, baseline/HSR) have completed from the original frames. All reported metrics exactly match their archived values. Each run verified the original per-case neural call graph and input pixel hashes. Per-pair maximum error differences, exact settings and available runtime records are recorded in audit/native_reproduction.json and its accompanying files. DINO remains diagnostic only. No held-out selection claim is made.

All 180 values also passed independent recomputation from archived errors. The 600 input-pair hashes and all 13 checkpoint hashes match their final-run records. All split assets passed read-back SHA-256 checks. A clean source extraction reproduced the metric tables without private paths. A fresh dependency installation has not been tested.

## Independent checks

- 36/36 native runs, 300 pairs each; no incomplete or mismatched run.
- All 180 displayed metric values and 90 displayed differences match; all
  unrounded overall and per-type metrics also match.
- All per-pair PairAcc, VSA, joint decisions, detected onsets and localization
  contributing subsets match. Displayed differences use displayed HSR minus
  displayed baseline values.
- All 32 non-DINO runs have exactly equal per-pair error arrays. DINO's maximum
  absolute raw-error differences are 4.172325134277344e-7 for baseline and
  1.1920928955078125e-7 for HSR. They change no decision, localization or metric;
  the original differences are retained.
- Input pixel hashes, checkpoint hashes and per-case neural call count/order/
  input-shape traces are checked. The independent metric audit adds no model
  forward calls.

The machine and Python environments are documented in [ENVIRONMENTS.md](ENVIRONMENTS.md).
The complete independent report is
[audit/native_metric_recomputation.json](audit/native_metric_recomputation.json).
To repeat the CPU-only audit, run `bash reproduce.sh --mode verify`.
