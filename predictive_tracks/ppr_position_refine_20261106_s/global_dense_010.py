"""Existing eta/radius midpoints around parent AND new global anchors.

This writes a declared fine grid before it is run. The frozen queue already
accepts immutable predeclared fine grids; no running/source code is changed.
"""
import argparse
import copy
import time
from . import common as c


def main():
    from .report import new_rows,jobs
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=c.SLOTS+c.FEATURES)
    args=parser.parse_args()
    model=args.model
    assert (c.HERE/'models_complete'/f'coarse_{model}.json').exists()
    assert not (c.HERE/'grids'/f'fine_{model}.json').exists()
    parent=c.read(c.source_path())
    architecture=c.read(c.HERE/'capture'/model/'architecture.json')
    existing=[]
    for raw in parent['all_configurations'][model]:
        if raw['mode']!=('output_only' if model in c.SLOTS else 'pre_final_block'):
            continue
        item=copy.deepcopy(raw)
        pp=item['parameters']
        item.update(eta=pp['eta'],alpha=pp.get('alpha',1.),
            position=None if model in c.SLOTS else f'block_{architecture["original_depth"]-1}')
        existing.append(item)
    current=[x for x in new_rows() if x['model']==model and not x.get('confirmation_only',False)]
    lattice=existing+current
    anchors=[x for x in existing if x.get('eligible_for_selection',True)]+current
    assert anchors
    occupied={(x.get('position'),x['eta'],x['alpha']) for x in existing}
    occupied.update((x.get('position'),x['eta'],x['alpha']) for x in jobs() if x['model']==model)
    proposed=[]
    used=[]
    for task in ('sp','ie'):
        for endpoint in ('overall_joint_hit','auc'):
            anchor=min(anchors,key=lambda x:rank(x,task,endpoint))
            used.append(dict(task=task,endpoint=endpoint,id=anchor['config_id'],eta=anchor['eta'],alpha=anchor['alpha'],position=anchor.get('position')))
            group=[x for x in lattice if x.get('position')==anchor.get('position')]
            es=sorted({v['eta'] for v in group})
            aa=sorted({v['alpha'] for v in group})
            e,a=anchor['eta'],anchor['alpha']
            ei,ai=es.index(e),aa.index(a)
            ev=sorted({round(e+(es[k]-e)/(2**level),8) for k in (ei-1,ei+1) if 0<=k<len(es) for level in (1,2,3,4,5)})
            av=sorted({round(a+(aa[k]-a)/(2**level),8) for k in (ai-1,ai+1) if 0<=k<len(aa) for level in (1,2,3,4,5)})
            for eta,alpha in [(v,a) for v in ev]+[(e,v) for v in av]+[(v,w) for v in ev for w in av]:
                token=anchor.get('position'),eta,alpha
                if token in occupied:
                    continue
                occupied.add(token)
                proposed.append(c.job(model,anchor['mode'],eta,alpha,anchor.get('position'),
                    proposal='dyadic eta/radius refinement around combined parent/new global primary or AUC anchor',anchor=anchor['config_id']))
    item=dict(at=time.time(),model=model,jobs=proposed,settings=len(proposed),all_cases_each=600,
        parent_and_new_settings_considered=True,anchors=used,history_formula_distance_unchanged=True,
        source_generator=str(__import__('pathlib').Path(__file__).resolve()),source_generator_sha256=c.digest(__file__),
        parent_report_sha256=c.digest(c.source_path()),coarse_grid_sha256=c.digest(c.HERE/'grids'/f'coarse_{model}.json'))
    item['search_resolution_denominators']=[2,4,8,16,32]
    item['only_existing_eta_and_radius_parameters']=True
    c.freeze_grid(f'fine_{model}',item)
    c.save(c.HERE/'fine_declarations'/f'{model}_010.json',dict(at=time.time(),source_sha256=c.digest(__file__),
        grid_sha256=c.digest(c.HERE/'grids'/f'fine_{model}.json'),settings=len(proposed),
        scope='Only existing eta/radius, anchored to actual comparable parent and current settings; one global configuration per task. Predeclared before fine experiments. No source/metric/forward changes.'))
    print('S COMBINED GLOBAL FINE GRID',model,len(proposed),used,flush=True)


if __name__=='__main__':main()
