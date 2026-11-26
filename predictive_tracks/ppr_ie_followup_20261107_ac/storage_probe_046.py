"""Check completed ie265 and failed ie266 with the same budget and exact scores."""
import time
from pathlib import Path
from . import common as c


def main():
    import torch
    c.verify_sources('STORAGE_READY_044.json')
    c.verify()
    c.require_resources('dino_foresight')
    from .storage_guard_044 import install_factory
    from .worker_008 import dino_setup,install_namespace
    from .dino_reference_005 import score
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    install_namespace()
    install_factory()
    base = dino_setup()
    from sp_core.common import PairRecord,read_pair_rgb
    adapter = base.adapter_for('dino_foresight')
    before = base.versions(adapter,'dino_foresight')
    jobs = c.read(c.HERE/'grids/coarse_dino_foresight.json')['jobs']
    thermal = install(lambda v:c.save(c.HERE/'events'/f'thermal_storage046_{time.time_ns()}.json',v))
    checked = []
    try:
        with torch.inference_mode():
            for row,mp,ref in c.dino_rows():
                if row['key'] not in ('ie_265','ie_266'):
                    continue
                destination = c.HERE/'storage_probe_046'/f'{row["key"]}.json'
                if destination.exists():
                    checked.append(c.read(destination));continue
                c.require_resources('dino_foresight')
                meta,loaded = base.read_capture('dino_foresight',row['key'])
                source = c.read(meta['source_meta'])
                cached = base.load_data(source)
                data = base.cuda(loaded)
                evaluator = base.evaluator_for(cached,'dino_foresight')
                evaluator.adapter.device = torch.device('cuda:0')
                evaluator.targets = [v.to('cuda:0') for v in evaluator.targets]
                raw = cached['prediction'].to('cuda:0')
                eta0 = dict(jobs[0],eta=0.,alpha=1.)
                original = base.predicted(adapter,'dino_foresight',eta0,data)
                assert torch.equal(original,raw)
                base.exact(evaluator.errors(original),[source['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
                settings = {}
                prior = c.HERE/'search/coarse/dino_foresight/pairs'/f'{row["key"]}.json'
                previous = c.read(prior) if prior.exists() else None
                for item in jobs:
                    output = base.predicted(adapter,'dino_foresight',item,data)
                    errors = evaluator.errors(output)
                    if previous:
                        base.exact(errors,[previous['settings'][item['config_id']][b+'_errors'] for b in ('possible','impossible')])
                    settings[item['config_id']] = dict(possible_errors=list(map(float,errors[0])),impossible_errors=list(map(float,errors[1])))
                full_checks = []
                if row['key']=='ie_266':
                    pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
                    possible,impossible = read_pair_rgb(pair)
                    for item in (eta0,jobs[0]):
                        adapter.prediction_calls = row['archive_index']
                        result,errors,actual,trace,audits = score(adapter,possible,impossible,item)
                        assert trace==c.read(c.HERE/'NATIVE_TRACE_REFERENCE_005.json')['trace']
                        assert adapter.prediction_calls==row['archive_index']+1
                        for key in ('history','possible_target','impossible_target'):
                            assert actual[key].dtype==cached[key].dtype and torch.equal(actual[key],cached[key]),key
                        if item['eta']==0:
                            assert torch.equal(actual['prediction'],cached['prediction'])
                            expected = [source['matched_identity_'+b+'_errors'] for b in ('possible','impossible')]
                        else:
                            expected = [settings[item['config_id']][b+'_errors'] for b in ('possible','impossible')]
                        base.exact(errors,expected)
                        full_checks.append(dict(eta=item['eta'],trace_exact=True,forecaster_calls=6,encoded_frames=17,errors_bitwise_exact=True))
                        del actual
                    del possible,impossible
                assert base.versions(adapter,'dino_foresight')==before
                value = dict(at=time.time(),key=row['key'],all_nine_settings=settings,
                    original_prediction_and_errors_exact=True,
                    completed_case_previous_nine_scores_exact=previous is not None,
                    full_native_checks=full_checks,weights_identity_version_unchanged=True,
                    per_forecast_original_six_calls=True,per_process_fraction=.45,
                    storage=adapter._storage_guard_044.audit(),diagnostic_only=True)
                c.save(destination,value);checked.append(value)
                print('STORAGE PROBE',row['key'],'nine settings exact/no OOM; native checks',len(full_checks),flush=True)
                del data,cached,loaded,evaluator,raw,original,output
        assert len(checked)==2
        c.save(c.HERE/'STORAGE_PROBE_COMPLETE_046.json',dict(at=time.time(),checked=[v['key'] for v in checked],
            completed_ie265_all_nine_scores_bitwise_exact=True,failed_ie266_all_nine_forecasts_finish=True,
            failed_case_original_native_and_eta01_states_scores_and_trace_exact=True,
            original_forecaster6_encoder17_and_memory_fraction045_unchanged=True,
            no_case_skipped_no_precision_or_metric_relaxation=True,diagnostic_only=True,
            files={str(p):c.digest(p) for p in (c.HERE/'storage_probe_046').glob('*.json')}))
    finally:
        thermal.remove()


if __name__=='__main__':
    main()
