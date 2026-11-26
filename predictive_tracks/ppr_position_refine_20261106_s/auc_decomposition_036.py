"""Decompose the unchanged normalized pooled AUC; no new metric or selection."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from sp_benchmark.protocol_metrics import pairwise_results, pooled_auc
from predictive_tracks.controlled_tracks.full_metrics_report import CATEGORIES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--task',choices=('sp','ie'),required=True)
    parser.add_argument('--phase',choices=('coarse','fine'),required=True)
    parser.add_argument('--config-id',required=True)
    args=parser.parse_args()
    grid=c.HERE/'grids'/f'{args.phase}_{args.model}.json'
    jobs=c.read(grid)['jobs']
    job=next(x for x in jobs if x['config_id']==args.config_id)
    folder=c.HERE/'search'/args.phase/args.model/'pairs'
    assert len(list(folder.glob('*.json')))==600,'Require the complete original600 experiment'
    paths=sorted(folder.glob(args.task+'_*.json'))
    assert len(paths)==300
    arrays={key:[] for key in ('p','i','bp','bi')};types=[];hashes={}
    for path in paths:
        row=c.read(path);base=row['native'];changed=row['settings'][args.config_id]
        assert row['original_native_errors_exact'] and row['original_suffix_output_bitwise_equal']
        types.append(base['category'])
        for key,value in [('p',changed['possible_errors']),('i',changed['impossible_errors']),
            ('bp',base['matched_identity_possible_errors']),('bi',base['matched_identity_impossible_errors'])]:
            arrays[key].append(value)
        hashes[str(path)]=c.digest(path)
    types=np.asarray(types);categories=CATEGORIES[args.task]
    assert all(int((types==category).sum())==100 for category in categories)
    summary_path=c.HERE/'search_settings'/args.model/args.config_id/'summary.json'
    summary=c.read(summary_path)
    values={}
    for variant,p,i in [('native_baseline','bp','bi'),('corrected','p','i')]:
        pair=pairwise_results(np.asarray(arrays[p]),np.asarray(arrays[i]),epsilon=1e-12)
        matrix=np.asarray([[pooled_auc(pair.score_impossible[types==a],pair.score_possible[types==b])
            for b in categories] for a in categories])
        existing=summary['metrics'][args.task][variant]
        original=pooled_auc(pair.score_impossible,pair.score_possible)
        assert original==existing['overall']['auc']
        assert np.isclose(matrix.mean(),original,rtol=0,atol=1e-15)
        assert all(matrix[j,j]==existing['by_type'][category]['auc'] for j,category in enumerate(categories))
        values[variant]=dict(unchanged_original_metrics=existing,
            impossible_type_rows_possible_type_columns=matrix.tolist(),
            equal_type_weights=[1/3]*3,original_pooled_AUC=original,
            diagonal_contribution_to_original_AUC=float(np.trace(matrix)/9),
            cross_type_contribution_to_original_AUC=float((matrix.sum()-np.trace(matrix))/9))
    native=c.directory(job)/'summary.json'
    if native.exists():assert c.read(native)['metrics']==summary['metrics']
    dest=c.HERE/'auc_diagnostics_036'/str(time.time_ns())
    value=dict(at=time.time(),model=args.model,task=args.task,job=job,
        original_task_pairs=300,complete_experiment_pairs=600,
        original_epsilon_and_normalized_pairwise_scores_unchanged=True,
        source_grid=str(grid),source_grid_sha256=c.digest(grid),
        source_summary=str(summary_path),source_summary_sha256=c.digest(summary_path),
        source_hashes=hashes,values=values,
        native600_already_verified=native.exists(),
        original_pooled_AUC_not_replaced=True,no_parameter_selection=True,
        no_neural_forward=True,CUDA_not_used=True,
        interpretation='AUC compares the original within-pair-normalized scores across different cases and types. The matrix only explains its original components; neither raw scores nor the reported pooled AUC are changed. It is not evidence of repaired interaction dynamics.')
    c.save(dest/'decomposition.json',value)
    print(dest,flush=True)
    for name,row in values.items():
        print(name,'pooled',row['original_pooled_AUC'],'diagonal contribution',row['diagonal_contribution_to_original_AUC'],
            'cross contribution',row['cross_type_contribution_to_original_AUC'],flush=True)
        print(np.asarray(row['impossible_type_rows_possible_type_columns']),flush=True)


if __name__=='__main__':main()
