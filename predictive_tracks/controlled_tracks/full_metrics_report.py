"""Post-run metric aggregation only; never loads or invokes a neural model."""

import argparse
import json
from pathlib import Path

import numpy as np

from .confirmation_report import combine
from .registry import NATIVE_TOLERANCE, OUTPUT
from .run import records_and_split, write_json


CATEGORIES = {
    "sp": ["identity_change", "occluded_wrong_reappearance", "sudden_disappearance"],
    "ie": ["motion_anomaly", "penetration", "contact_response"],
}


def aggregate(rows, model):
    result = {}
    tolerance = str(NATIVE_TOLERANCE[model])
    for baseline, name in [(True, "native_baseline"), (False, "corrected")]:
        overall = combine(rows, model, baseline)
        by_type = {
            category: combine([r for r in rows if r["category"] == category], model, baseline)
            for category in CATEGORIES[rows[0]["benchmark"]]
        }
        weighted = sum(v["num_pairs"] * v["joint_hit"][tolerance] for v in by_type.values())
        assert np.isclose(weighted / len(rows), overall["joint_hit"][tolerance])
        assert sum(v["localization_mae_denominator"] for v in by_type.values()) == overall["localization_mae_denominator"]
        result[name] = {"overall": overall, "by_type": by_type}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("A simple exclusive output name is required")
    plan = json.loads(Path(__file__).with_name("confirmation_plan.json").read_text())
    previous = json.loads((OUTPUT / "confirmation_report_001.json").read_text())
    payload = {
        "scope": "Original SP 300 and IE 300 pairs per model; 30 development + 270 confirmation. Full results include development and are descriptive.",
        "method": "parallel_boundary",
        "metric_source": "Unmodified sp_benchmark.protocol_metrics.summarize via confirmation_report.combine",
        "baseline": "Matched native prediction from the original forward, not a selected archived baseline.",
        "additional_neural_forwards": 0,
        "units": {"paired_accuracy": "fraction", "auc": "0..1", "localization_mae": "original video frames; conditional on correct and localized", "joint_hit": "fraction"},
        "category_order": CATEGORIES,
        "dino": "Excluded from positive evidence due to original temporal alignment; original model denominator remains nine.",
        "models": {},
    }
    for model in plan["models"]:
        rows = []
        for split in ["dev", "confirmation"]:
            directory = OUTPUT / f"{model}_parallel_{split}_001"
            assert (directory / "summary.json").exists()
            rows.extend(json.loads(p.read_text()) for p in sorted((directory / "pairs").glob("*.json")))
        entry = {"native_joint_tolerance_frames": NATIVE_TOLERANCE[model], "benchmarks": {}}
        for task in ["sp", "ie"]:
            selected = [r for r in rows if r["benchmark"] == task]
            assert len(selected) == 300
            expected_ids = {r["pair_id"] for r in records_and_split(task)}
            assert {r["pair_id"] for r in selected} == expected_ids
            assert set(r["category"] for r in selected) == set(CATEGORIES[task])
            assert all(sum(r["category"] == c for r in selected) == 100 for c in CATEGORIES[task])
            full = aggregate(selected, model)
            prior = previous["models"][model]["benchmarks"][task]["full300_descriptive_only_includes_development"]
            for variant in ["native_baseline", "corrected"]:
                assert full[variant]["overall"] == prior[variant], (model, task, variant)
            confirmation = [r for r in selected if r["split"] == "confirmation"]
            assert len(confirmation) == 270
            entry["benchmarks"][task] = {"full300": full, "confirmation270": aggregate(confirmation, model)}
        payload["models"][model] = entry
    write_json(OUTPUT / args.output_name, payload)
    for task in ["sp", "ie"]:
        print(task.upper(), "FULL300; baseline -> corrected; PA/JH in percent, AUC in 0..1, MAE in frames")
        for model, entry in payload["models"].items():
            data = entry["benchmarks"][task]["full300"]
            b, c = data["native_baseline"]["overall"], data["corrected"]["overall"]
            tolerance = str(NATIVE_TOLERANCE[model])
            fields = [model]
            for key, scale, digits in [("paired_accuracy", 100, 2), ("auc", 1, 4), ("localization_mae", 1, 3)]:
                fields.append(f"{b[key]*scale:.{digits}f} → {c[key]*scale:.{digits}f}")
            for category in CATEGORIES[task]:
                values = [data[v]["by_type"][category]["joint_hit"][tolerance]*100 for v in ["native_baseline", "corrected"]]
                fields.append(f"{values[0]:.2f} → {values[1]:.2f}")
            fields.append(f"{100*b['joint_hit'][tolerance]:.2f} → {100*c['joint_hit'][tolerance]:.2f}")
            print(" | ".join(fields))
    print("Saved:", OUTPUT / args.output_name)


if __name__ == "__main__":
    main()
