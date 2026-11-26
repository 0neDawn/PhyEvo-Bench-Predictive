"""Original metrics on a complete DINO task; diagnostic, never physical repair."""
import argparse
import sys
import time
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate,CATEGORIES
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
from predictive_tracks.ppr_eta_radius_20261104_o.cache_worker import expand


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('sp','ie'),required=True);args=p.parse_args()
    model='dino_foresight'
    paths=sorted((c.HERE/'cache'/model/'pairs').glob(args.task+'_*.json'))
    if len(paths)!=300:
        print('Task incomplete; no metrics written',args.task,len(paths),flush=True);return
    payload=[c.read(path) for path in paths]
    assert all(x['native_max_abs_error']==0 and x['neural_forwards']==0 for x in payload)
    assert sorted(x['native']['archive_index'] for x in payload)==list(range(300))
    assert all(sum(x['native']['category']==cat for x in payload)==100 for cat in CATEGORIES[args.task])
    jobs=[x for x in c.read(c.HERE/'GRID.json')['cached'] if x['model']==model]
    parent=c.read(c.source_path())['selected'][args.task][model]
    measured=[]
    for item in jobs:
        metric=aggregate([expand(x,item) for x in payload],model)
        assert metric['native_baseline']==parent['metrics'][args.task]['native_baseline']
        measured.append(dict(item,parameters=dict(eta=item['eta'],alpha=item['alpha']),
            metrics={args.task:metric},diagnostic_only=True))
    choices=measured+[parent]
    best=min(choices,key=lambda x:rank(x,args.task))
    out=c.HERE/'task_interims'/f'dino_foresight_{args.task}_coarse_{time.time_ns()}.json'
    c.save(out,dict(at=time.time(),model=model,task=args.task,cases=300,best_provisional=best,
        previous_main=parent,all_settings=measured,diagnostic_only=True,
        original_timing_problem_unchanged=True,no_new_model_calls=True,
        no_new_main_until_full600_complete=True,same_exposed_data_exploratory=True,
        source_hashes={str(path):c.digest(path) for path in paths}))
    for item in [parent]+sorted(choices,key=lambda x:rank(x,args.task))[:4]:
        print(item['config_id'],item['parameters'],item['metrics'][args.task]['corrected']['overall'],flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
