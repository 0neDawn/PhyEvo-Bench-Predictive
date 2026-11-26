"""CPU-only PPR tables and paired uncertainty, reusing unchanged metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from predictive_tracks.controlled_tracks.full_metrics_report import aggregate, CATEGORIES
from predictive_tracks.controlled_tracks.report import indicators, paired_stats
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE

ROOT = Path(__file__).resolve().parents[2]


def comparison(rows, model, before_rows=None):
    rows = sorted(rows, key=lambda r: r["pair_id"])
    before_rows = sorted(before_rows, key=lambda r: r["pair_id"]) if before_rows is not None else rows
    if [r["pair_id"] for r in rows] != [r["pair_id"] for r in before_rows]:
        raise RuntimeError("Unpaired comparison")
    args = (np.asarray(rows[0]["frame_spans"]), np.asarray([r["t_ref"] for r in rows]), NATIVE_TOLERANCE[model])
    current = indicators(np.asarray([r["possible_errors"] for r in rows]),
                         np.asarray([r["impossible_errors"] for r in rows]), *args)
    prefix = "matched_identity_" if before_rows is rows else ""
    before = indicators(np.asarray([r[prefix + "possible_errors"] for r in before_rows]),
                        np.asarray([r[prefix + "impossible_errors"] for r in before_rows]), *args)
    return {endpoint: paired_stats(c, b, [r["category"] for r in rows], seed=20261028, repetitions=10000)
            for endpoint, c, b in zip(("paired_accuracy", "joint_hit"), current, before)}


def number(value, scale=1, digits=3):
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def feedback_configuration_check(feedback, output_only, model):
    """Compare native configuration outputs, with G-SWM ids aligned per draw."""
    left = {r["pair_id"]: r for r in feedback}
    right = {r["pair_id"]: r for r in output_only}
    if left.keys() != right.keys():
        raise RuntimeError("Configuration diagnostic requires the same cases")
    by_step = np.zeros(6)
    changed_later_cases = 0
    for key in sorted(left):
        a, b = left[key]["projection_audits"], right[key]["projection_audits"]
        if len(a) != len(b):
            raise RuntimeError("Native MC count changed between tracks")
        case_delta = np.zeros(6)
        for current, original in zip(a, b):
            if model == "ddlp":
                x, y = np.asarray(current["future_position"]), np.asarray(original["future_position"])
                delta = np.max(np.abs(x - y), axis=(1, 2))
            else:
                delta = []
                for t in range(6):
                    ids_x, ids_y = current["future_render_ids"][t], original["future_render_ids"][t]
                    if set(ids_x) != set(ids_y) or len(set(ids_x)) != len(ids_x):
                        raise RuntimeError("G-SWM native identity set changed")
                    x = dict(zip(ids_x, current["future_where"][t]))
                    y = dict(zip(ids_y, original["future_where"][t]))
                    delta.append(max((float(np.max(np.abs(np.asarray(x[i]) - np.asarray(y[i])))) for i in x), default=0.))
                delta = np.asarray(delta)
            case_delta = np.maximum(case_delta, delta)
        by_step = np.maximum(by_step, case_delta)
        changed_later_cases += bool(np.any(case_delta[1:] > 1e-6))
    return {"cases": len(left), "max_configuration_abs_difference_by_step": by_step.tolist(),
            "first_step_matches_within_1e_6": bool(by_step[0] <= 1e-6),
            "cases_with_later_configuration_difference_over_1e_6": changed_later_cases,
            "tolerance_purpose": "numeric diagnostic only, not a method hyperparameter or scoring change",
            "interpretation": "First-step configuration precedes correction and should match. Later change is evidence of propagation, not evidence of improved physics."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("Simple run name required")
    source = ROOT / "predictive_tracks/ppr_20261028/runs" / args.run
    report = {"run": args.run, "created_at": time.time(), "scope": "Exploratory, original SP300 and IE300; previously exposed data",
              "original_model_denominator": 9, "applicable_PPR_models": ["ddlp", "gswm"],
              "not_implemented": {"ocvp_seq": "No authorized identified persistent-state read/write interface",
                                  "savi_transformer": "No authorized identified persistent-state read/write interface",
                                  "slotformer": "No authorized identified persistent-state read/write interface",
                                  "vjepa2_vitl": "No native per-future-step feedback in frozen adapter",
                                  "vjepa2_vitg": "No native per-future-step feedback in frozen adapter",
                                  "videomaev2": "No native per-future-step feedback in frozen adapter",
                                  "dino_foresight": "Original temporal alignment issue; baseline fix is not method gain"},
              "jobs": {}, "metrics": {}, "paired_comparisons": [], "configuration_diagnostics": {}}
    loaded = {}
    for model in ("ddlp", "gswm"):
        for mode in ("audit", "feedback", "output_only"):
            directory = source / model / mode
            name = f"{model}/{mode}"
            files = sorted((directory / "pairs").glob("*.json"))
            if (directory / "FAILED.json").exists():
                report["jobs"][name] = {"status": "failed", "completed_pairs": len(files), "failure": json.loads((directory / "FAILED.json").read_text())}
                continue
            if not (directory / "summary.json").exists():
                report["jobs"][name] = {"status": "incomplete", "completed_pairs": len(files)}
                continue
            report["jobs"][name] = {"status": "complete", "completed_pairs": len(files)}
            if mode == "audit":
                continue
            rows = [json.loads(p.read_text()) for p in files]
            if len(rows) != 600:
                raise RuntimeError("A completed full run must contain 600 pairs")
            loaded[model, mode] = rows
            report["metrics"][name] = {}
            for task in ("sp", "ie"):
                subset = [r for r in rows if r["benchmark"] == task]
                assert len(subset) == 300 and len({r["pair_id"] for r in subset}) == 300
                assert all(sum(r["category"] == c for r in subset) == 100 for c in CATEGORIES[task])
                report["metrics"][name][task] = aggregate(subset, model)
                for endpoint, value in comparison(subset, model).items():
                    report["paired_comparisons"].append(dict(value, model=model, benchmark=task,
                          comparison=f"{mode}_vs_native", endpoint=endpoint))
    for model in ("ddlp", "gswm"):
        if all((model, mode) in loaded for mode in ("feedback", "output_only")):
            for task in ("sp", "ie"):
                a = [r for r in loaded[model, "feedback"] if r["benchmark"] == task]
                b = [r for r in loaded[model, "output_only"] if r["benchmark"] == task]
                report["configuration_diagnostics"][model + "/" + task] = feedback_configuration_check(a, b, model)
                for endpoint, value in comparison(a, model, b).items():
                    report["paired_comparisons"].append(dict(value, model=model, benchmark=task,
                          comparison="feedback_vs_output_only", endpoint=endpoint))
    ordered = sorted(report["paired_comparisons"], key=lambda r: r["exact_mcnemar_p"])
    previous = 0.
    for rank, value in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * value["exact_mcnemar_p"]))
        value["holm_p_all_reported_comparisons"] = previous
    report["uncertainty"] = "Paired category-stratified 10,000-bootstrap marginal 95% CIs for accuracy/JointHit differences; exact McNemar and Holm over all reported comparisons. Not a new heldout test."
    report["localization_note"] = "Original conditional MAE; compare with its denominator and detection coverage."
    report["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(__file__).parent / "runs" / args.run / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["PPR: original full300/task; PA/JointHit/Coverage in %, AUC in [0,1], MAE in original frames.",
             "Exploratory exposed-data evaluation. Unimplemented models are not counted as successes.", ""]
    for task in ("sp", "ie"):
        lines.extend([task.upper(), "Model/Track | Paired Acc. | AUC | Loc. MAE | Coverage | MAE N | " + " | ".join(CATEGORIES[task]) + " | Overall Joint Hit"])
        for model in ("ddlp", "gswm"):
            native_written = False
            for mode in ("feedback", "output_only"):
                data = report["metrics"].get(f"{model}/{mode}", {}).get(task)
                if data is None:
                    lines.append(f"{model}/{mode}: {report['jobs'][model+'/'+mode]['status']}")
                    continue
                for variant, label in (("native_baseline", "native"), ("corrected", mode)):
                    if label == "native" and native_written:
                        continue
                    native_written |= label == "native"
                    overall = data[variant]["overall"]
                    tol = str(NATIVE_TOLERANCE[model])
                    values = [model + "/" + label, number(overall["paired_accuracy"], 100, 2), number(overall["auc"], 1, 4),
                              number(overall["localization_mae"]), number(overall["detection_coverage"], 100, 2), str(overall["localization_mae_denominator"])]
                    values.extend(number(data[variant]["by_type"][c]["joint_hit"][tol], 100, 2) for c in CATEGORIES[task])
                    values.append(number(overall["joint_hit"][tol], 100, 2))
                    lines.append(" | ".join(values))
        lines.append("")
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
