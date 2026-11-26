"""Describe completed parameter-grid neighborhoods; no new grid or predictions."""
import time
from . import common as c
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank


def main():
    model='videomaev2'
    rows=[]
    hashes={}
    for phase in ('coarse','fine'):
        grid=c.HERE/'grids'/f'{phase}_{model}.json'
        hashes[str(grid)]=c.digest(grid)
        for job in c.read(grid)['jobs']:
            path=c.HERE/'search_settings'/model/job['config_id']/'summary.json'
            summary=c.read(path)
            assert summary['complete'] and summary['cases']==600
            hashes[str(path)]=c.digest(path)
            if not job.get('confirmation_only',False):
                rows.append(dict(job,metrics=summary['metrics']))
    result=dict(at=time.time(),model=model,unique_settings=len(rows),tasks={},
        no_new_model_calls=True,no_new_parameter_proposals=True,
        same_exposed_data_exploratory=True,
        scope='Observed discrete-grid sensitivity, not a continuous-region or independent-test guarantee.',
        source_hashes=hashes)
    for task in ('sp','ie'):
        best=min(rows,key=lambda x:rank(x,task))
        baseline=best['metrics'][task]['native_baseline']['overall']
        def compact(item):
            o=item['metrics'][task]['corrected']['overall']
            return dict(config_id=item['config_id'],eta=item['eta'],alpha=item['alpha'],position=item['position'],
                paired_accuracy=o['paired_accuracy'],joint_hit=o['joint_hit']['2'],auc=o['auc'],localization_mae=o['localization_mae'])
        selected_family=[x for x in rows if x['position']==best['position']]
        improving=[x for x in selected_family
            if x['metrics'][task]['corrected']['overall']['paired_accuracy']>baseline['paired_accuracy']
            and x['metrics'][task]['corrected']['overall']['joint_hit']['2']>baseline['joint_hit']['2']]
        result['tasks'][task]=dict(best=compact(best),positions={
            position:compact(min([x for x in rows if x['position']==position],key=lambda x:rank(x,task)))
            for position in sorted({x['position'] for x in rows})},
            selected_position_settings=len(selected_family),
            sampled_settings_with_both_PA_JH_improvement=len(improving),
            same_alpha_eta_sweep=[compact(x) for x in sorted(selected_family,key=lambda x:x['eta']) if x['alpha']==best['alpha']],
            same_eta_alpha_sweep=[compact(x) for x in sorted(selected_family,key=lambda x:x['alpha']) if x['eta']==best['eta']])
    out=c.HERE/'sensitivity_018'/f'{time.time_ns()}.json'
    c.save(out,result)
    for task,d in result['tasks'].items():
        print(task,'best',d['best'], 'both-improving samples',d['sampled_settings_with_both_PA_JH_improvement'],
              '/',d['selected_position_settings'],flush=True)
        print('same_alpha_eta_sweep',d['same_alpha_eta_sweep'],flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
