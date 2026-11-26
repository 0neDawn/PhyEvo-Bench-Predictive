"""Original metrics for a complete task only; never a validated main-table row."""
import argparse
import sys
import time
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate,CATEGORIES
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
from .worker import expand


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=c.SLOTS+c.FEATURES,required=True)
    parser.add_argument('--task',choices=('sp','ie'),required=True)
    parser.add_argument('--phase',choices=('coarse','fine'),default='coarse')
    args=parser.parse_args()
    folder=c.HERE/'search'/args.phase/args.model/'pairs'
    paths=sorted(folder.glob(args.task+'_*.json'))
    payload=[c.read(p) for p in paths]
    assert len(payload)==300 and len({x['native']['pair_id'] for x in payload})==300
    assert all(sum(x['native']['category']==cat for x in payload)==100 for cat in CATEGORIES[args.task])
    jobs=c.read(c.HERE/'grids'/f'{args.phase}_{args.model}.json')['jobs']
    measured=[]
    for item in jobs:
        rr=[expand(x,item) for x in payload]
        measured.append(dict(item,parameters={k:item[k] for k in ('eta','alpha')}|dict(position=item.get('position')),
            metrics={args.task:aggregate(rr,args.model)},not_full600_native_validated=True))
    parent=c.read(c.source_path())['selected'][args.task][args.model]
    choices=[x for x in measured if not x.get('confirmation_only',False)]+[parent]
    best=min(choices,key=lambda x:rank(x,args.task))
    dest=c.HERE/'task_interims'/f'{args.model}_{args.task}_{args.phase}_{time.time_ns()}.json'
    c.save(dest,dict(at=time.time(),model=args.model,task=args.task,phase=args.phase,cases=300,
        original_metrics_unchanged=True,not_eligible_for_main_table=True,no_new_model_calls=True,
        previous_main=parent,best_provisional=best,all_settings=measured,
        source_record_hashes={str(p):c.digest(p) for p in paths}))
    print('PROVISIONAL complete300 only; original full600 validation still required',flush=True)
    for item in [parent]+sorted(choices,key=lambda x:rank(x,args.task))[:5]:
        o=item['metrics'][args.task]['corrected']['overall']
        print(item['config_id'],item['parameters'],{k:o[k] for k in ('paired_accuracy','auc','localization_mae','joint_hit')},flush=True)
    print(dest,flush=True)


if __name__=='__main__':main()
