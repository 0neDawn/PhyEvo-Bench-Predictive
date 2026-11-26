"""Original full-budget inference for every newly selected parameter setting."""
import argparse
import hashlib
import time
from pathlib import Path
from . import common as c


def dino():
    import sys
    import numpy as np
    import torch
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.ppr_dino_feedback_20261106_aa.worker import adapter_for
    from predictive_tracks.ppr_position_refine_20261106_s.worker import load_data,exact
    from predictive_tracks.ppr_existing_20261102.metrics import make_summary
    from sp_core.common import PairRecord,read_pair_rgb
    from .dino_reference_005 import score
    old = c.install()
    adapter = adapter_for('dino_foresight')
    reference = c.read(c.HERE/'NATIVE_TRACE_REFERENCE_005.json')['trace']
    jobs = [v for v in c.read(c.HERE/'grids/validation.json')['jobs'] if v['model'] == 'dino_foresight']
    for item in jobs:
        directory = c.directory(item)
        if (directory/'summary.json').exists():continue
        search = c.read(c.HERE/'search_settings'/'dino_foresight'/item['config_id']/'summary.json')
        rows = []
        for row,mp,meta in c.dino_rows():
            destination = directory/'pairs'/f'{row["key"]}.json'
            if destination.exists():
                rows.append(c.read(destination));continue
            old.before_native('dino_foresight')
            cached = load_data(meta)
            pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible,impossible = read_pair_rgb(pair)
            pixels = hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
            assert pixels == meta['source_decoded_pixels_sha256']
            adapter.prediction_calls = row['archive_index']
            native,errors,actual,trace,audits = score(adapter,possible,impossible,item)
            assert trace == reference and adapter.prediction_calls == row['archive_index']+1
            for key in ('history','possible_target','impossible_target'):
                assert actual[key].dtype == cached[key].dtype and torch.equal(actual[key],cached[key]),key
            expected = c.read(Path(search['compact_case_source'])/f'{row["key"]}.json')['settings'][item['config_id']]
            exact(errors,[expected[b+'_errors'] for b in ('possible','impossible')])
            result = dict(benchmark=row['benchmark'],archive_index=row['archive_index'],pair_id=row['pair_id'],category=row['category_directory'],
                split=row['split'],t_ref=int(row['t_ref']),frame_spans=native.frame_spans.tolist(),
                possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist(),
                matched_identity_possible_errors=meta['matched_identity_possible_errors'],matched_identity_impossible_errors=meta['matched_identity_impossible_errors'],
                job=item,model='dino_foresight',mode=item['mode'],eta=item['eta'],radius_multiplier=item['alpha'],
                neural_call_trace=trace,trace_matches_original=True,trace_reference='Fresh unchanged adapter and existing cache-generation graph',
                original_archived_deployment_trace_unverified=not meta.get('original_trace_match',False),
                extra_forwards=0,forecaster_calls=6,encoded_frames=17,weight_identity_and_version_unchanged=True,
                projection_audits=audits,source_decoded_pixels_sha256=pixels,
                original_adapter_possible_errors=native.possible.tolist(),original_adapter_impossible_errors=native.impossible.tolist(),
                existing_benchmark_scoring_expression_unchanged=True,raw_history_and_targets_exact=True,
                original_alignment_unchanged=True,diagnostic_only=True)
            c.save(destination,result);rows.append(result)
            c.status(directory/'progress.json',dict(at=time.time(),completed=len(rows),total=600))
            del cached,actual,possible,impossible
        summary = make_summary(rows,'dino_foresight')
        assert summary['metrics'] == search['metrics']
        summary.update(job=item,all_native_traces_matched=True,extra_forwards=0,
            existing_benchmark_scoring_exact=True,original_perframe_scalar_rounding_separately_reported=True,diagnostic_only=True)
        c.save(directory/'summary.json',summary)
    c.save(c.HERE/'models_complete/validation_dino_foresight.json',dict(at=time.time(),settings=len(jobs),cases_each=600,diagnostic_only=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=c.MODELS,required=True)
    args = parser.parse_args()
    from .worker_008 import install_namespace
    install_namespace();c.verify()
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    handle = install(lambda v:c.save(c.HERE/'events'/f'thermal_native009_{time.time_ns()}.json',v))
    try:
        if args.model == 'dino_foresight':
            dino()
        else:
            from predictive_tracks.ppr_feedback_refine_20261106_t import native_worker as original
            original.c = c
            original.main()
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
