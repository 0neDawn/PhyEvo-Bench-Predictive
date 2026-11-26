"""Report all nine IE300 results once available; never promote partial validation."""
import os
import time
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import sys
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.controlled_tracks.full_metrics_report import aggregate,CATEGORIES
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
    c.verify_sources('IE_INTERIM_READY_051.json')
    c.verify()
    directory = c.HERE/'search/coarse/dino_foresight/pairs'
    while True:
        paths = sorted(directory.glob('ie_*.json'))
        if len(paths)==300:
            break
        state = c.read(c.HERE/'monitor/queue.json')
        if state.get('state')=='failed':
            raise RuntimeError('DINO queue failed; do not invent IE300 results')
        c.status(c.HERE/'monitor/IE_interim_051.json',dict(at=time.time(),state='waiting',
            completed_IE=len(paths),required_IE=300,no_GPU=True))
        time.sleep(30)
    payloads = [c.read(p) for p in paths]
    assert {v['native']['archive_index'] for v in payloads}==set(range(300))
    assert all(v['original_native_errors_exact'] and v['original_suffix_output_bitwise_equal'] for v in payloads)
    assert all(sum(v['native']['category']==cat for v in payloads)==100 for cat in CATEGORIES['ie'])
    parent = c.read(c.source_path())
    baseline = parent['all_configurations']['dino_foresight'][0]['metrics']['ie']['native_baseline']
    jobs = c.read(c.HERE/'grids/coarse_dino_foresight.json')['jobs']
    candidates = []
    for job in jobs:
        rows = [dict(v['native'],**v['settings'][job['config_id']]) for v in payloads]
        metrics = aggregate(rows,'dino_foresight')
        assert metrics['native_baseline']==baseline
        candidates.append(dict(job,parameters=dict(eta=job['eta'],alpha=job['alpha']),
            metrics=dict(ie=metrics),full600_native_validated=False,eligible_for_final_selection=False,
            diagnostic_only=True))
    native = dict(model='dino_foresight',mode='native',config_id='native_eta0',parameters=dict(eta=0.,alpha=1.),
        metrics=dict(ie=dict(native_baseline=baseline,corrected=baseline)),
        diagnostic_only=True,eligible_for_final_selection=False)
    selected = min(candidates+[native],key=lambda v:rank(v,'ie'))
    output = c.HERE/'task_interims/dino_IE300_storage051.json'
    c.save(output,dict(at=time.time(),task='ie',pairs=300,cases_each_type=100,settings=9,
        rows=candidates,original_baseline=native,provisional_selected=selected,
        selection_rule_unchanged=True,all_metrics_one_actual_setting=True,
        complete_IE300_only_not_SP600_completion=True,all_new_full_native600_validation_pending=True,
        original_time_alignment_unchanged_diagnostic_only=True,same_exposed_data_exploratory=True,
        no_neural_or_GPU_calls=True,parent_report_sha256=c.digest(c.source_path()),
        input_files={str(p):c.digest(p) for p in paths},source_sha256=c.digest(__file__)))
    c.status(c.HERE/'monitor/IE_interim_051.json',dict(at=time.time(),state='complete',no_GPU=True,
        output=str(output),full_native600_validation_pending=True))
    print('All nine DINO IE300 rows saved; diagnostic, full-native600 validation pending',output,flush=True)


if __name__=='__main__':
    main()
