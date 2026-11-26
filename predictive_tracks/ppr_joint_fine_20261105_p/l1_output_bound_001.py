"""Ideal L1 upper bound, diagnostics only; no model or official-score changes.

In the existing pooled output correction, frame h is p_h+s_h*v_h, where
v_h=c-mean_spatial(p_h), and 0<=s_h<=max(0,1-R/||v_h||) for eta<=1,alpha>=1.
Relax the shared eta/alpha to independent s_h. Each frame's possible-versus-
impossible L1 margin is piecewise linear; its maximum is at an endpoint or
an absolute-value breakpoint. The sum of independent maxima bounds every
shared-parameter margin in ideal arithmetic. Future targets enter ONLY this
oracle diagnostic, never the correction or a reported benchmark score.
"""
import time
from pathlib import Path
import numpy as np
from . import common as c


def frame_bound(prediction, possible, impossible, direction, limit):
    ap = (prediction-possible).reshape(-1)
    ai = (prediction-impossible).reshape(-1)
    v = np.broadcast_to(direction, prediction.shape).reshape(-1)
    size = len(v)
    initial = np.abs(ai).sum()-np.abs(ap).sum()
    def derivative(a):
        return np.where(a > 0, v, np.where(a < 0, -v, np.abs(v))).sum()
    slope = derivative(ai)-derivative(ap)
    times, jumps = [], []
    for a, sign in ((ai, 1.), (ap, -1.)):
        roots = np.full_like(v, np.inf)
        np.divide(-a, v, out=roots, where=v != 0)
        keep = (roots > 0) & (roots < limit)
        times.append(roots[keep])
        jumps.append(sign*2*np.abs(v[keep]))
    times, jumps = np.concatenate(times), np.concatenate(jumps)
    order = np.argsort(times, kind='stable')
    times, jumps = times[order], jumps[order]
    nodes = np.r_[0., times, limit]
    slopes = np.r_[slope, slope+np.cumsum(jumps)]
    values = initial+np.r_[0., np.cumsum(np.diff(nodes)*slopes)]
    winner = int(np.argmax(values))
    maximum, where = float(values[winner]/size), float(nodes[winner])
    def direct(s):
        return float((np.abs(ai+s*v).sum()-np.abs(ap+s*v).sum())/size)
    error = abs(direct(where)-maximum)
    assert error < 1e-10, error
    probes = [direct(0.), direct(limit/2), direct(limit)]
    assert max(probes) <= maximum+1e-10
    return dict(maximum_margin=maximum, oracle_s=where, s_max=float(limit),
        native_margin=direct(0.), breakpoint_count=len(times), direct_check_error=error,
        explanation='Independent-frame, target-aware upper bound, not a usable inference rule or official measurement')


def main():
    c.cpu_guard()
    frozen = c.read(c.HERE / 'L1_DIAGNOSTIC_FROZEN_001.json')
    for path, wanted in frozen['files'].items():
        assert c.digest(path) == wanted, path
    import sys
    sys.path.insert(0, str(c.ROOT / 'predictive_tracks/sp/code'))
    import torch
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import load_data, forbid
    from predictive_tracks.ppr_all_20261029.projection import reference
    torch.set_num_threads(1)
    hook = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    support = c.read(c.HERE / 'SUPPORT_DIAGNOSTIC_001.json')['vjepa2_vitl_SP_output']
    assert support['original_wrong_pairs'] == 13
    results = []
    try:
        with torch.inference_mode():
            for entry in support['wrong_cases']:
                c.cpu_guard()
                path = c.OLD / 'cache/vjepa2_vitl/pairs' / (entry['key']+'.json')
                assert c.digest(path) == entry['source_sha256']
                cached = c.read(path)
                meta_path = Path(cached['native_meta'])
                assert c.digest(meta_path) == cached['source_meta_sha256']
                meta = c.read(meta_path)
                data = load_data(meta)
                assert meta['sha256'] == cached['source_state_sha256']
                p, history = data['prediction'], data['history']
                center, radius, valid = reference(history.mean(2, keepdim=True))
                assert p.ndim == 4 and p.shape[0] == 1 and center.shape[0] == 1 and bool(valid.all())
                center = center[0].double().numpy()
                radius = float(radius[0])
                pm = p.mean(2)[0].double().numpy()
                pp, tp, ti = [data[k][0].double().numpy() for k in ('prediction','possible_target','impossible_target')]
                frames = []
                gpu_margin = np.asarray(meta['matched_identity_impossible_errors'])-np.asarray(meta['matched_identity_possible_errors'])
                assert gpu_margin.mean() <= 0
                for h in range(p.shape[1]):
                    direction = center-pm[h]
                    distance = float(np.linalg.norm(direction))
                    limit = max(0., 1.-radius/distance) if distance else 0.
                    bound = frame_bound(pp[h], tp[h], ti[h], direction, limit)
                    bound['float64_native_minus_original_GPU_margin'] = bound['native_margin']-float(gpu_margin[h])
                    frames.append(bound)
                upper = float(np.mean([v['maximum_margin'] for v in frames]))
                results.append(dict(key=entry['key'], frames=frames, relaxed_upper_bound_mean_margin=upper,
                    could_be_corrected_even_under_independent_frame_oracle=upper > 0,
                    original_GPU_mean_margin=float(gpu_margin.mean()),
                    source_meta=str(meta_path), source_meta_sha256=cached['source_meta_sha256'],
                    state_path=meta['state'], state_sha256=meta['sha256']))
                print(entry['key'], 'ideal relaxed upper margin', upper, flush=True)
    finally:
        hook.remove()
    assert len(results) == 13 and not torch.cuda.is_initialized()
    for path, wanted in frozen['files'].items():
        assert c.digest(path) == wanted, path
    result = dict(at=time.time(), model='vjepa2_vitl', task='sp', neural_forwards=0, CUDA_initialized=False,
        original_wrong_cases=13, original_correct_pairs=287, all_original_wrong_cases_included=True,
        alpha_domain=[1., 'infinity'], eta_domain=[0.,1.],
        future_targets_used_only_for_diagnostic_bound=True, inference_operator_unchanged=True,
        validation_grid_unchanged=True, official_scores_unchanged=True,
        ideal_original_wrong_cases_not_ruled_out=sum(x['could_be_corrected_even_under_independent_frame_oracle'] for x in results),
        maximum_direct_piecewise_check_error=max(f['direct_check_error'] for x in results for f in x['frames']),
        cases=results, scientific_scope='Optimistic independent-frame real-arithmetic bound for pooled output insertion only. Not a float32 CUDA proof, measured performance, or a bound for internal insertion. Positive bounds do not establish a feasible shared eta/alpha.')
    c.save(c.HERE / 'L1_OUTPUT_BOUND_001.json', result)
    print('IDEAL relaxed correctable wrong cases', result['ideal_original_wrong_cases_not_ruled_out'], '/13; NN0 CUDAfalse', flush=True)


if __name__ == '__main__':
    main()
