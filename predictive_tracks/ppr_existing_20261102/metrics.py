"""Original scoring, exact paired tests, and unchanged full denominators."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate,CATEGORIES
from predictive_tracks.controlled_tracks.report import indicators
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE
from predictive_tracks.ppr_all_20261029.common import rows as original_rows

def make_summary(records,model):
    expected={(r['benchmark'],r['pair_id']) for r in original_rows()}
    if len(records)!=600 or {(r['benchmark'],r['pair_id']) for r in records}!=expected:
        raise RuntimeError('Full original 600 distinct pairs required')
    result={'model':model,'cases':600,'metrics':{},'exact_paired_tests':{},'complete':True}
    for task in ('sp','ie'):
        rs=sorted((r for r in records if r['benchmark']==task),key=lambda r:r['pair_id'])
        if len(rs)!=300 or any(sum(r['category']==c for r in rs)!=100 for c in CATEGORIES[task]):
            raise RuntimeError('Original task/type denominators changed')
        result['metrics'][task]=aggregate(rs,model)
        args=(np.asarray(rs[0]['frame_spans']),np.asarray([r['t_ref'] for r in rs]),NATIVE_TOLERANCE[model])
        a=indicators(np.asarray([r['possible_errors'] for r in rs]),np.asarray([r['impossible_errors'] for r in rs]),*args)
        b=indicators(np.asarray([r['matched_identity_possible_errors'] for r in rs]),np.asarray([r['matched_identity_impossible_errors'] for r in rs]),*args)
        result['exact_paired_tests'][task]={}
        for endpoint,x,y in zip(('paired_accuracy','joint_hit'),a,b):
            x,y=np.asarray(x,dtype=bool),np.asarray(y,dtype=bool)
            gain=int((x&~y).sum());loss=int((~x&y).sum());n=gain+loss
            result['exact_paired_tests'][task][endpoint]={'gain_cases':gain,'loss_cases':loss,'net_cases':gain-loss,
                'num_pairs':300,'exact_mcnemar_p':float(binomtest(gain,n,.5).pvalue) if n else 1.}
    return result
