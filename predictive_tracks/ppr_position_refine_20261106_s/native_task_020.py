"""Original metrics for one fully completed native task; no forecast calls."""
import argparse
import sys
import time
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate,CATEGORIES
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE
from predictive_tracks.ppr_position_controls_20261103_k.casewise_audit_001 import flags
from sp_benchmark.protocol_metrics import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=c.SLOTS+c.FEATURES,required=True)
    p.add_argument('--config-id',required=True)
    p.add_argument('--task',choices=('sp','ie'),required=True)
    args=p.parse_args()
    folder=c.HERE/'runs'/args.model/args.config_id
    paths=sorted((folder/'pairs').glob(args.task+'_*.json'))
    assert len(paths)==300
    rows=[c.read(path) for path in paths]
    assert sorted(r['archive_index'] for r in rows)==list(range(300))
    assert all(sum(r['category']==cat for r in rows)==100 for cat in CATEGORIES[args.task])
    assert all(r['trace_matches_original'] and r['extra_forwards']==0
               and r['weight_identity_and_version_unchanged'] for r in rows)
    measured=aggregate(rows,args.model)
    search=c.HERE/'search_settings'/args.model/args.config_id/'summary.json'
    assert measured==c.read(search)['metrics'][args.task]
    tolerance=NATIVE_TOLERANCE[args.model]
    def calculate(prefix):
        return summarize(np.asarray([r[prefix+'possible_errors'] for r in rows]),
            np.asarray([r[prefix+'impossible_errors'] for r in rows]),frame_spans=rows[0]['frame_spans'],
            violation_frames=[r['t_ref'] for r in rows],epsilon=1e-12,threshold=0.,
            persistence_frames=2,joint_tolerances=(tolerance,))
    bp,bm,bj,be=flags(calculate('matched_identity_'),tolerance)
    ap,am,aj,ae=flags(calculate(''),tolerance)
    common=bm&am
    def ids(mask):
        return [f'{args.task}_{rows[i]["archive_index"]:03d}' for i in np.flatnonzero(mask)]
    result=dict(at=time.time(),model=args.model,config_id=args.config_id,task=args.task,
        metrics=measured,PA_gain=ids(ap&~bp),PA_loss=ids(~ap&bp),
        JH_gain=ids(aj&~bj),JH_loss=ids(~aj&bj),native_MAE_n=int(bm.sum()),
        corrected_MAE_n=int(am.sum()),common_MAE_n=int(common.sum()),
        common_native_MAE=float(be[common].mean()) if common.any() else None,
        common_corrected_MAE=float(ae[common].mean()) if common.any() else None,
        onset_improved=ids(common&(ae<be)),onset_worsened=ids(common&(ae>be)),
        original300_task_complete=True,full600_summary_complete=(folder/'summary.json').exists(),
        not_eligible_as_new_main_until_full600_validation=True,no_new_model_calls=True,
        original_metrics_unchanged=True,same_exposed_data_exploratory=True,
        source_hashes={str(path):c.digest(path) for path in paths+[search]})
    out=c.HERE/'native_task_020'/f'{args.config_id}_{args.task}_{time.time_ns()}.json'
    c.save(out,result)
    print(measured,flush=True)
    print({k:v for k,v in result.items() if k not in ('metrics','source_hashes')},flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
