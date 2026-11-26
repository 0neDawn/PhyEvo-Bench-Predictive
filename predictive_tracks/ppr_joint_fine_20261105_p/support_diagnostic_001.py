"""Diagnose zero-gain output corrections; no model, score or grid changes."""
import time
from pathlib import Path
import numpy as np
from . import common as c


def main():
    c.cpu_guard()
    result = dict(at=time.time(), neural_forwards=0, official_scores_changed=False, validation_grid_changed=False)
    index = c.read(c.HERE / "analysis/vjepa2_vitl/INDEX.json")
    with np.load(c.HERE / "analysis/vjepa2_vitl/errors.npz", allow_pickle=False) as values:
        native = values["native"][300:]
    wrong = np.flatnonzero(native[:, 1].mean(1) <= native[:, 0].mean(1))
    cases = []
    for n in wrong:
        path = c.OLD / "cache/vjepa2_vitl/pairs" / f"sp_{n:03d}.json"
        value = c.read(path)
        audit = value["projection_audits"]["1.0"]
        outside = [step["outside_units"] for step in audit["steps"]]
        cases.append(dict(key=f"sp_{n:03d}", outside_at_alpha1=outside, inactive_at_all_steps=not any(outside), source_sha256=c.digest(path)))
    result["vjepa2_vitl_SP_output"] = dict(original_wrong_pairs=len(wrong), wrong_cases=cases,
        all_wrong_cases_inactive=all(x["inactive_at_all_steps"] for x in cases),
        implication="If all wrong predictions are already inside the alpha1 ball, alpha>=1 and eta<=1 leave those predictions unchanged. Scope is this output insertion only; internal insertion requires actual validation.")
    source = c.ROOT / "predictive_tracks/ppr_eta_radius_20261104_n/MSE_IDEAL_COEFFICIENTS_002.json"
    data = c.read(source)
    cases = []
    for row in data["rows"]:
        base, slope = np.asarray(row["base"]), np.asarray(row["slope"])
        ratio = np.asarray([np.inf if x is None else x for x in row["ratio"]])
        if base.mean() > 0:
            continue
        # On each alpha activation interval, mean margin is affine in alpha
        # and eta. A per-case relaxed maximum is attained at eta0/eta1 and
        # alpha endpoints or one of the three frame activation radii.
        alpha = np.unique(np.r_[1., 4., ratio[(ratio > 1) & (ratio < 4)]])
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.maximum(0., 1.-alpha[:, None]/ratio[None])
        factor[:, ratio == 0] = 0.
        margins = base.mean() + (factor*slope).mean(1)
        best = max(float(base.mean()), float(margins.max()))
        cases.append(dict(key=row["key"], native_mean_margin=float(base.mean()), relaxed_maximum_mean_margin=best,
                          potentially_correctable_in_ideal_arithmetic=best > 0))
    result["videomaev2_IE_output"] = dict(original_wrong_pairs=len(cases), potentially_correctable_pairs=sum(x["potentially_correctable_in_ideal_arithmetic"] for x in cases),
        cases=cases, alpha_range=[1., 4.], eta_range=[0., 1.], coefficient_source=str(source), coefficient_source_sha256=c.digest(source),
        diagnostic_only=True, float32_CUDA_scores_remain_authoritative=True,
        warning="Real-arithmetic, per-case relaxed upper bound; not a measured performance, global optimum, or a bound for internal-state insertion.")
    c.save(c.HERE / "SUPPORT_DIAGNOSTIC_001.json", result)
    print("L SP wrong pairs", len(wrong), "all inactive", result["vjepa2_vitl_SP_output"]["all_wrong_cases_inactive"], flush=True)
    print("Video IE ideal potentially correctable", result["videomaev2_IE_output"]["potentially_correctable_pairs"], "/", len(cases), flush=True)


if __name__ == "__main__":
    main()
