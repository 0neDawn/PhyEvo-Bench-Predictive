"""Describe completed grid sensitivity; never choose parameters per example."""
import os
import time
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    import numpy as np
    torch.set_num_threads(1)
    def forbid(*args, **kwargs):
        raise RuntimeError('No neural execution in stored-score diagnostic')
    torch.nn.Module._call_impl = forbid
    from predictive_tracks.ppr_existing_20261102.metrics import indicators, NATIVE_TOLERANCE
    result = {}
    for model in c.SLOTS:
        root = c.HERE/'search/coarse'/model
        paths = sorted((root/'pairs').glob('*.json'))
        assert len(paths) == 600
        payloads = [c.read(p) for p in paths]
        assert all(p['original_native_errors_exact'] and p['original_suffix_output_bitwise_equal'] for p in payloads)
        jobs = c.read(c.HERE/'grids'/f'coarse_{model}.json')['jobs']
        result[model] = {}
        for task in ('sp','ie'):
            data = [v for v in payloads if v['native']['benchmark']==task]
            assert len(data)==300
            args = (np.asarray(data[0]['native']['frame_spans']),
                np.asarray([v['native']['t_ref'] for v in data]), NATIVE_TOLERANCE[model])
            native_errors = [np.asarray([v['native'][name+'_errors'] for v in data])
                for name in ('possible','impossible')]
            native = indicators(*native_errors, *args)
            result[model][task] = {}
            for mode in ('feedback','state_only'):
                group = [j for j in jobs if j['mode']==mode]
                changed = np.zeros(300,dtype=bool)
                rescued = [np.zeros(300,dtype=bool) for _ in range(2)]
                harmed = [np.zeros(300,dtype=bool) for _ in range(2)]
                settings = []
                for job in group:
                    errors = [np.asarray([v['settings'][job['config_id']][name+'_errors'] for v in data])
                        for name in ('possible','impossible')]
                    flags = indicators(*errors, *args)
                    changed |= np.any(errors[0]!=native_errors[0],axis=1) | np.any(errors[1]!=native_errors[1],axis=1)
                    item = dict(config_id=job['config_id'],eta=job['eta'],alpha=job['alpha'])
                    for i,label in enumerate(('paired_accuracy','joint_hit')):
                        gains = flags[i] & ~native[i]
                        losses = ~flags[i] & native[i]
                        rescued[i] |= gains; harmed[i] |= losses
                        item[label] = dict(gained_cases=int(gains.sum()),lost_cases=int(losses.sum()),
                            net_cases=int(gains.sum())-int(losses.sum()))
                    settings.append(item)
                group_result = dict(cases=300,settings_tested=len(group),
                    score_unchanged_for_every_tested_setting=int((~changed).sum()),settings=settings)
                for i,label in enumerate(('paired_accuracy','joint_hit')):
                    group_result[label] = dict(native_correct=int(native[i].sum()),
                        wrong_cases_rescued_by_at_least_one_setting=int(rescued[i].sum()),
                        correct_cases_harmed_by_at_least_one_setting=int(harmed[i].sum()),
                        maximum_single_global_setting_net_cases=max(v[label]['net_cases'] for v in settings),
                        minimum_single_global_setting_net_cases=min(v[label]['net_cases'] for v in settings),
                        rescued_case_keys=[f"{task}_{data[k]['native']['archive_index']:03d}" for k in np.flatnonzero(rescued[i])],
                        harmed_case_keys=[f"{task}_{data[k]['native']['archive_index']:03d}" for k in np.flatnonzero(harmed[i])])
                result[model][task][mode] = group_result
    assert not torch.cuda.is_initialized()
    c.verify()
    c.save(c.HERE/'SLOT_SENSITIVITY_042.json',dict(at=time.time(),results=result,
        completed_AC_coarse_only=True,no_change_to_parameters_or_inference=True,
        union_counts_are_diagnostics_not_achievable_single_setting_metrics=True,
        no_per_case_or_per_type_selection=True,finite_tested_grid_not_continuous_optimality=True,
        no_model_or_CUDA_execution=True,source_sha256=c.digest(__file__)))
    print({model:{mode:{key:result[model]['ie'][mode][key]['maximum_single_global_setting_net_cases']
        for key in ('paired_accuracy','joint_hit')} for mode in ('feedback','state_only')} for model in c.SLOTS},flush=True)


if __name__=='__main__':
    main()
