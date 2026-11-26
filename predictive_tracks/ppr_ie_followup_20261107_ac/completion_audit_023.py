"""Independent CPU-only audit of new results; never rerank or run a model."""
import json,os,time
from pathlib import Path
from . import common as c


def check_manifest(path):
    value=c.read(path)
    for p,wanted in value.get('files',{}).items():assert c.digest(p)==wanted,p
    return c.digest(path)


def audit(root,models):
    import torch
    def forbid(*args,**kwargs):raise RuntimeError('Neural model forbidden in completion audit')
    torch.nn.Module._call_impl=forbid
    from predictive_tracks.ppr_existing_20261102.metrics import make_summary
    for p,wanted in c.read(root/'SOURCE_READY_001.json')['files'].items():assert c.digest(p)==wanted,p
    for p,wanted in c.read(root/'PARENT_MANIFESTS.json').items():
        assert c.digest(p)==wanted,p
        check_manifest(Path(p))
    directory=Path(c.read(root/'FINAL_DELIVERY.json')['directory'])
    manifest=check_manifest(directory/'DELIVERY.json')
    for p,wanted in c.read(directory/'input_hashes.json').items():assert c.digest(p)==wanted,p
    report=c.read(directory/'report.json')
    parent=c.read(report['source_report'])
    assert c.digest(report['source_report'])==report['source_report_sha256']
    for model,values in parent['all_configurations'].items():assert report['all_configurations'][model][:len(values)]==values
    planned=[]
    for phase in ('coarse','fine'):
        for model in models:
            path=root/'grids'/f'{phase}_{model}.json'
            if path.exists():
                frozen=c.read(path.with_suffix('.frozen.json'));assert c.digest(path)==frozen['sha256']
                planned.extend(c.read(path)['jobs'])
    current=report['current_round_settings']
    assert len(current)==len(planned)
    lookup={v['config_id']:v for v in planned}
    assert set(lookup)=={v['config_id'] for v in current}
    cache={};native_count=0
    for item in current:
        source=item['compact_case_source']
        if source not in cache:
            paths=sorted(Path(source).glob('*.json'));assert len(paths)==600
            cache[source]=[c.read(p) for p in paths]
        payloads=cache[source]
        rows=[dict(v['native'],**v['settings'][item['config_id']]) for v in payloads]
        assert make_summary(rows,item['model'])['metrics']==item['metrics']
        assert all(v['original_native_errors_exact'] and v['original_suffix_output_bitwise_equal'] for v in payloads)
        if item['full600_native_validated']:
            native_count+=1
            expected={f'{v["benchmark"]}_{v["archive_index"]:03d}':v for v in rows}
            paths=sorted((Path(item['directory'])/'pairs').glob('*.json'));assert len(paths)==600
            for path in paths:
                row=c.read(path);key=f'{row["benchmark"]}_{row["archive_index"]:03d}'
                assert row['job']==lookup[item['config_id']]
                assert row['possible_errors']==expected[key]['possible_errors'] and row['impossible_errors']==expected[key]['impossible_errors']
                assert row['trace_matches_original'] and row['extra_forwards']==0 and row['weight_identity_and_version_unchanged']
                if item['model']=='dino_foresight':
                    assert row['forecaster_calls']==6 and row['encoded_frames']==17
                    assert row['existing_benchmark_scoring_expression_unchanged'] and row['raw_history_and_targets_exact']
                    assert row['diagnostic_only'] and row['original_alignment_unchanged']
                    assert len(row['original_adapter_possible_errors'])==len(row['possible_errors'])==6
        if item.get('hard_limit_control'):
            assert item['eta']==1. and item['mode']=='buffer_read'
            assert not item['eligible_for_selection']
    required={v['config_id'] for v in c.read(root/'grids/validation.json')['jobs']}
    assert required=={v['config_id'] for v in current if v['full600_native_validated']}
    for task in ('sp','ie'):
        assert len(report['selected'][task])==9
        for row in report['selected'][task].values():
            if row['config_id'] in lookup:
                assert row['full600_native_validated'] and row['eligible_for_selection']
                assert not row.get('hard_limit_control',False)
    if 'MATCHED_INSERTION_CONTROLS.json' in {p.name for p in directory.iterdir()}:
        controls=c.read(directory/'MATCHED_INSERTION_CONTROLS.json')
        assert controls['same_eta_and_radius_for_each_pair'] and controls['hard_read_eta1_is_a_nonselectable_limiting_control']
    assert not torch.cuda.is_initialized()
    return dict(at=time.time(),root=str(root),directory=str(directory),settings=len(current),native600=native_count,
        independent_pairs=600,all_original_metrics_recomputed=True,parent_and_all_negative_results_preserved=True,
        new_primary_original_budget_weight_and_errors_exact=True,DINO_cache_recipe_and_diagnostic_status_unchanged=True,
        core_manifest_sha256=manifest,no_neural_or_CUDA_calls=True,root_visual_inspection_pending=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    c.verify_sources('AUDIT_READY_023.json')
    stages=[('AC',c.HERE,c.MODELS,'COMPLETED_012.json'),
        ('AE',c.ROOT/'predictive_tracks/ppr_buffer_read_20261107_ae',c.SLOTS,'COMPLETED_001.json')]
    for label,root,models,complete in stages:
        marker=c.HERE/'reviews'/f'REVIEWED_{label}_023.json'
        if marker.exists():
            previous=c.read(marker);assert c.digest(Path(previous['directory'])/'DELIVERY.json')==previous['core_manifest_sha256'];continue
        while not (root/complete).exists():
            state=c.read(root/'monitor/queue.json') if (root/'monitor/queue.json').exists() else {}
            c.status(c.HERE/'monitor/audit_023.json',dict(at=time.time(),state='waiting',stage=label,observed_state=state.get('state'),no_GPU=True))
            if state.get('state')=='failed':raise RuntimeError(f'{label} failed; no audit success fabricated')
            time.sleep(30)
        c.status(c.HERE/'monitor/audit_023.json',dict(at=time.time(),state='auditing',stage=label,no_GPU=True))
        c.save(marker,audit(root,models))
    c.save(c.HERE/'ALL_REVIEWS_COMPLETE_023.json',dict(at=time.time(),stages=['AC','AE'],no_model_calls=True,
        latest_delivery=c.read(stages[-1][1]/'FINAL_DELIVERY.json')['directory'],visual_inspection_pending=True))
    c.status(c.HERE/'monitor/audit_023.json',dict(at=time.time(),state='complete',no_GPU=True,visual_inspection_pending=True))


if __name__=='__main__':main()
