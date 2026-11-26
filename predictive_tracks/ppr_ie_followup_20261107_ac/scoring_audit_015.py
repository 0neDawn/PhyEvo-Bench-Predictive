"""Diagnose old DINO cache/adapter reductions; never choose or change a metric."""
import time
import numpy as np
from . import common as c


def main():
    import sys
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.controlled_tracks.registry import archive
    from predictive_tracks.ppr_existing_20261102.metrics import make_summary
    all_rows = c.dino_rows()
    results = {}
    sources = {}
    all_original,all_cached = [],[]
    for task in ('sp','ie'):
        path = archive(task)/'models/dino_foresight/raw/error_arrays.npz'
        original,cached = [],[]
        with np.load(path,allow_pickle=False) as values:
            for row,mp,meta in all_rows:
                if row['benchmark'] != task:continue
                i = row['archive_index']
                pair_id = values['pair_ids'][i]
                if isinstance(pair_id,bytes):pair_id=pair_id.decode()
                assert str(pair_id) == row['pair_id']
                base = dict(benchmark=task,archive_index=i,pair_id=row['pair_id'],category=row['category_directory'],
                    split=row['split'],t_ref=int(row['t_ref']),frame_spans=meta['frame_spans'],
                    matched_identity_possible_errors=meta['matched_identity_possible_errors'],
                    matched_identity_impossible_errors=meta['matched_identity_impossible_errors'])
                a = dict(base,possible_errors=np.asarray(values['possible_errors'][i]).tolist(),impossible_errors=np.asarray(values['impossible_errors'][i]).tolist())
                b = dict(base,possible_errors=meta['matched_identity_possible_errors'],impossible_errors=meta['matched_identity_impossible_errors'])
                original.append(a);cached.append(b)
                sources[str(mp)] = c.digest(mp)
        all_original.extend(original)
        all_cached.extend(cached)
        difference = max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for a,b in zip(original,cached) for k in ('possible_errors','impossible_errors'))
        results[task] = dict(max_error_array_abs_difference=difference,
            cases=300,source_archive=str(path),source_archive_sha256=c.digest(path))
    raw = make_summary(all_original,'dino_foresight')['metrics']
    existing = make_summary(all_cached,'dino_foresight')['metrics']
    for task in ('sp','ie'):
        a,b = raw[task]['corrected'],existing[task]['corrected']
        results[task].update(original_archival_perframe_metrics=a,existing_fixed_cache_metrics=b,all_metrics_exactly_equal=a==b)
    c.save(c.HERE/'DINO_BASELINE_SCORING_AUDIT_015.json',dict(at=time.time(),tasks=results,
        source_hashes=sources,no_model_calls=True,no_metric_or_baseline_changed=True,
        existing_cache_formula_remains_fixed_for_all_new_comparisons=True,diagnostic_only=True))
    for task,value in results.items():print(task,'all baseline metrics equal',value['all_metrics_exactly_equal'],'max array diff',value['max_error_array_abs_difference'],flush=True)


if __name__=='__main__':main()
