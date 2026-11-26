"""Report the locked confirmation experiment without selecting models or rules."""

import argparse
import json
from pathlib import Path
import numpy as np
from .registry import OUTPUT, NATIVE_TOLERANCE
from .run import write_json, scalar_metrics
from .report import collect, summarize


def control_metrics(rows,model):
    return scalar_metrics(summarize(
        np.asarray([r['cached_controls']['persistence_possible_errors'] for r in rows]),
        np.asarray([r['cached_controls']['persistence_impossible_errors'] for r in rows]),
        frame_spans=np.asarray(rows[0]['frame_spans']),violation_frames=np.asarray([r['t_ref'] for r in rows]),
        epsilon=1e-12,threshold=0.,persistence_frames=2,joint_tolerances=(NATIVE_TOLERANCE[model],)))


def combine(rows,model,baseline=False):
    prefix='matched_identity_' if baseline else ''
    return scalar_metrics(summarize(
        np.asarray([r[prefix+'possible_errors'] for r in rows]),
        np.asarray([r[prefix+'impossible_errors'] for r in rows]),
        frame_spans=np.asarray(rows[0]['frame_spans']),violation_frames=np.asarray([r['t_ref'] for r in rows]),
        epsilon=1e-12,threshold=0.,persistence_frames=2,joint_tolerances=(NATIVE_TOLERANCE[model],)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',required=True)
    args=parser.parse_args()
    if Path(args.output_name).name!=args.output_name: raise ValueError('Simple output name required')
    plan=json.loads(Path(__file__).with_name('confirmation_plan.json').read_text())
    report=collect(OUTPUT)
    primary=[r for r in report['comparisons'] if r['split']=='confirmation' and r['condition']==plan['selected_correction']]
    expected=4*len(plan['models'])
    if len(primary)!=expected: raise RuntimeError(f'Confirmation incomplete or duplicated: {len(primary)} endpoints, expected {expected}')
    ordered=sorted(primary,key=lambda r:r['exact_mcnemar_p']);previous=0.
    for rank,row in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-rank)*row['exact_mcnemar_p']))
        row['holm_p_confirmation_primary_family']=previous
    models={}
    for model in plan['models']:
        root=OUTPUT/f'{model}_parallel_confirmation_001'
        summary=json.loads((root/'summary.json').read_text())
        rows=[json.loads(p.read_text()) for p in (root/'pairs').glob('*.json')]
        dev=OUTPUT/f'{model}_parallel_dev_001/pairs'
        development=[json.loads(p.read_text()) for p in dev.glob('*.json')]
        passed=True;details={}
        for track in ('sp','ie'):
            selected=[r for r in rows if r['benchmark']==track]
            if len(selected)!=270: raise RuntimeError('Confirmation pair count changed')
            cur=summary[track]['current'];base=summary[track]['matched_identity_same_pairs']
            tolerance=str(NATIVE_TOLERANCE[model])
            pa_gain=round(cur['paired_accuracy']*270)-round(base['paired_accuracy']*270)
            jh_gain=round(cur['joint_hit'][tolerance]*270)-round(base['joint_hit'][tolerance]*270)
            passed &= pa_gain>0 and jh_gain>=0
            merged=selected+[r for r in development if r['benchmark']==track]
            if len(merged)!=300 or len({r['pair_id'] for r in merged})!=300:
                raise RuntimeError('Full-dataset merge is not the original 300 distinct pairs')
            details[track]={'native_baseline':base,'corrected':cur,'paired_correct_count_gain':pa_gain,
                            'joint_hit_count_gain':jh_gain,'persistence_control':control_metrics(selected,model),
                            'full300_descriptive_only_includes_development':{
                                'native_baseline':combine(merged,model,True),'corrected':combine(merged,model)}}
        models[model]={'passes_frozen_descriptive_criterion':bool(passed),'benchmarks':details}
    number=sum(v['passes_frozen_descriptive_criterion'] for v in models.values())
    result={'selected_correction':plan['selected_correction'],
            'original_model_denominator':9,'eligible_models_tested':8,
            'models_passing_confirmation_criterion':number,
            'descriptive_majority_confirmed':number>=5,
            'statistical_caveat':'A positive point estimate is not significance; report paired intervals and adjusted tests. Models sharing backbones are not independent samples.',
            'split_caveat':'Post-hoc fixed confirmation partition; historical full baseline aggregates were already known. No confirmation treatment outputs guided the locked rule.',
            'dino_qualification':'Known original temporal-alignment issue; excluded from positive evidence, denominator not reduced.',
            'models':models,'confirmation_primary_paired_comparisons':primary,
            'confirmation_ablations':[r for r in report['same_forward_cached_ablations'] if r['split']=='confirmation'],
            'all_development_and_confirmation_comparisons':report}
    write_json(OUTPUT/args.output_name,result)
    print(json.dumps({'models_passing_confirmation_criterion':number,'denominator':9,
                      'descriptive_majority_confirmed':number>=5,'output':str(OUTPUT/args.output_name)}))
    for model,value in models.items():
        print(model,{k:{'PA_count_gain':v['paired_correct_count_gain'],'JointHit_count_gain':v['joint_hit_count_gain']}
                     for k,v in value['benchmarks'].items()})


if __name__=='__main__': main()
