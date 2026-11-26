"""Finite-grid diagnostic only; never deploy per-case parameter selection."""
import sys,time
import numpy as np
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from sp_benchmark.protocol_metrics import pairwise_results

def main():
    model='dino_foresight';baseline={};details={};hashes={}
    for phase,grid in [('cache',c.HERE/'GRID.json'),('cache_stage2',c.HERE/'STAGE2.json')]:
        jobs=[j for j in c.read(grid)['cached'] if j['model']==model]
        for task in ('sp','ie'):
            paths=sorted((c.HERE/phase/model/'pairs').glob(task+'_*.json'));assert len(paths)==300
            payloads=[c.read(p) for p in paths]
            assert all(p['native_max_abs_error']==0 and p['neural_forwards']==0 for p in payloads)
            b=pairwise_results(np.asarray([p['native']['possible_errors'] for p in payloads]),
                np.asarray([p['native']['impossible_errors'] for p in payloads])).correct
            if task in baseline:assert np.array_equal(baseline[task],b)
            else:baseline[task]=b;details[task]=[]
            for job in jobs:
                arrays=[p['settings'][job['config_id']] for p in payloads]
                flags=pairwise_results(np.asarray([p['possible_errors'] for p in arrays]),
                    np.asarray([p['impossible_errors'] for p in arrays])).correct
                summary=c.read(c.directory(job)/'summary.json')['metrics'][task]['corrected']['overall']
                assert int(flags.sum())==round(300*summary['paired_accuracy'])
                details[task].append(dict(config_id=job['config_id'],eta=job['eta'],alpha=job['alpha'],
                    global_correct_count=int(flags.sum()),joint_hit=summary['joint_hit'],
                    all_original_metrics_at_this_setting=summary,
                    gain_case_ids=[f'{task}_{i:03d}' for i in np.flatnonzero(flags&~b)],
                    loss_case_ids=[f'{task}_{i:03d}' for i in np.flatnonzero(~flags&b)]))
            hashes.update({str(p):c.digest(p) for p in paths})
    tasks={}
    for task,rows in details.items():
        gains=sorted({i for r in rows for i in r['gain_case_ids']});losses=sorted({i for r in rows for i in r['loss_case_ids']})
        tasks[task]=dict(baseline_correct_count=int(baseline[task].sum()),settings=len(rows),
            maximum_global_setting_correct_count=max(r['global_correct_count'] for r in rows),
            wrong_cases_correct_in_any_searched_global_setting=gains,
            correct_cases_lost_in_any_searched_global_setting=losses,
            union_is_a_diagnostic_not_a_deployable_method=True,all_settings=rows)
    assert 'torch' not in sys.modules
    value=dict(at=time.time(),model=model,tasks=tasks,source_hashes=hashes,neural_forwards=0,
        original_GPU_errors_and_strict_pairwise_rule_unchanged=True,
        diagnostic_only_due_to_original_temporal_alignment=True,
        finite_grid_not_a_continuous_optimum=True,no_per_case_parameter_method=True,
        source_sha256=c.digest(__file__))
    out=c.HERE/'dino_response_041'/str(time.time_ns());c.save(out/'response.json',value)
    for task,x in tasks.items():print(task,{k:v for k,v in x.items() if k!='all_settings'},flush=True)
    print(out,flush=True)

if __name__=='__main__':main()
