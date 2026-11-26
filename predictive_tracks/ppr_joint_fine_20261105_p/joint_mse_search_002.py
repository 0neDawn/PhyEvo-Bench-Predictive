"""Search VideoMAE output eta/alpha sign cells; proposals are not scores."""
import time
import numpy as np
from . import common as c


def main():
    c.cpu_guard()
    source = c.ROOT / "predictive_tracks/ppr_eta_radius_20261104_n/MSE_IDEAL_COEFFICIENTS_002.json"
    records = c.read(source)["rows"]
    assert len(records) == 300 and [r["key"] for r in records] == [f"ie_{i:03d}" for i in range(300)]
    base = np.asarray([r["base"] for r in records])
    scale = np.asarray([r["slope"] for r in records])
    ratio = np.asarray([[np.inf if v is None else v for v in r["ratio"]] for r in records])
    refs = np.asarray([r["t_ref"] for r in records])
    spans = np.asarray(records[0]["frame_spans"])
    assert all(r["frame_spans"] == records[0]["frame_spans"] for r in records)
    mean = base.mean(1)
    inverse = np.divide(1., ratio, out=np.zeros_like(ratio), where=np.isfinite(ratio) & (ratio > 0))
    segments = []
    for i in range(300):
        cuts = np.unique(np.r_[1., 4., ratio[i][(ratio[i] > 1) & (ratio[i] < 4)]])
        for lo, hi in zip(cuts, cuts[1:]):
            active = ratio[i] > (lo+hi)/2
            segments.append((i, lo, hi, np.mean(scale[i]*active), np.mean(scale[i]*inverse[i]*active), mean[i]))
    ss = np.asarray(segments)
    ident, lo, hi, aa, bb, cc = ss.T
    denominator = cc[:, None]*bb[None] - cc[None]*bb[:, None]
    numerator = cc[:, None]*aa[None] - cc[None]*aa[:, None]
    overlap_lo = np.maximum(lo[:, None], lo[None])
    overlap_hi = np.minimum(hi[:, None], hi[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        crossings = numerator/denominator
        eta = -cc[:, None]/(aa[:, None]-crossings*bb[:, None])
        eta1 = (cc+aa)/bb
    valid = (ident[:, None] < ident[None]) & (overlap_lo < crossings) & (crossings < overlap_hi) & (eta > 0) & (eta < 1)
    cuts = np.unique(np.r_[1., 4., lo, hi, crossings[valid], eta1[(eta1 > lo) & (eta1 < hi)]])
    cuts = cuts[np.isfinite(cuts) & (cuts >= 1) & (cuts <= 4)]
    baseline = int((mean > 0).sum())
    max_pa, max_jh = baseline, baseline
    proposals = []
    from predictive_tracks.ppr_ie_eta_20261103_f.loss_curves import decisions
    for n, (left, right) in enumerate(zip(cuts, cuts[1:])):
        alpha = float((left+right)/2)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.maximum(0., 1.-alpha/ratio)
        factor[ratio == 0] = 0.
        step_slope = scale*factor
        slope = step_slope.mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            roots = -mean/slope
        valid = np.isfinite(roots) & (roots > 0) & (roots < 1)
        order = np.argsort(roots[valid])
        root_values = roots[valid][order]
        weights = np.where(slope[valid][order] > 0, 1, -1)
        unique, first = np.unique(root_values, return_index=True)
        initial = int(((mean > 0) | ((mean == 0) & (slope > 0))).sum())
        counts = np.r_[initial, initial+np.cumsum(np.add.reduceat(weights, first))] if len(first) else np.asarray([initial])
        max_pa = max(max_pa, int(counts.max()))
        edges = np.r_[0., unique, 1.]
        choices = np.flatnonzero(counts > baseline)
        if len(choices):
            etas = (edges[choices]+edges[choices+1])/2
            margins = base[None]+etas[:, None, None]*step_slope[None]
            pa, hit, loc = decisions(margins.transpose(2, 1, 0).reshape(base.shape[1], -1), spans, np.repeat(refs, len(etas)), 2)
            pa, hit = (v.reshape(300, len(etas)).sum(0) for v in (pa, hit))
            max_jh = max(max_jh, int(hit.max()))
            for j in np.flatnonzero((hit >= baseline) & ((hit > baseline) | (pa > baseline))):
                e = int(choices[j])
                rounded_eta, rounded_alpha = round(float(etas[j]), 8), round(alpha, 8)
                if not edges[e] < rounded_eta < edges[e+1] or not left < rounded_alpha < right:
                    continue
                proposals.append(dict(eta=rounded_eta, alpha=rounded_alpha, predicted_PA=int(pa[j]), predicted_JH=int(hit[j]),
                    eta_interval=[float(edges[e]), float(edges[e+1])], alpha_interval=[float(left), float(right)],
                    interval_area=float((edges[e+1]-edges[e])*(right-left)), proposal_only=True))
        if n % 250 == 0:
            c.cpu_guard()
            c.status(c.HERE / "mse_search_progress_002.json", dict(at=time.time(), completed_alpha_intervals=n+1, total=len(cuts)-1, max_ideal_PA=max_pa, max_ideal_JH=max_jh))
    proposals.sort(key=lambda x: (-x["predicted_JH"], -x["predicted_PA"], -x["interval_area"], x["eta"], x["alpha"]))
    distinct = []
    for value in proposals:
        if any(abs(value["eta"]-x["eta"]) < .002 and abs(value["alpha"]-x["alpha"]) < .02 for x in distinct):
            continue
        distinct.append(value)
        if len(distinct) == 12:
            break
    c.save(c.HERE / "JOINT_MSE_DIAGNOSTIC_002.json", dict(at=time.time(), model="videomaev2", task="ie", insertion="existing_output_only",
        eta_range=[0., 1.], alpha_range=[1., 4.], coefficient_sha256=c.digest(source), source=str(source), alpha_sign_intervals=len(cuts)-1,
        baseline_ideal_PA=baseline, maximum_ideal_PA_found=max_pa, maximum_ideal_JH_found=max_jh, candidates=distinct,
        neural_forwards=0, official_scores_changed=False, current_validation_grid_changed=False,
        scope="Positive-width real-arithmetic sign-cell proposal search; not a measured result or a certified float32 bound. Native CUDA full600 validation required. Internal-state insertion excluded."))
    print("VIDEO JOINT MSE PROPOSALS", "alpha cells", len(cuts)-1, "ideal PA", max_pa, "ideal JH", max_jh, "candidates", distinct, flush=True)


if __name__ == "__main__":
    main()
