"""Refine only the existing eta/radius coordinates around global IE anchors."""
import time
from pathlib import Path
from . import common as c


def neighbours(value, lattice):
    lattice = sorted(set(lattice))
    i = lattice.index(value)
    out = {value}
    for k in (i-1,i+1):
        if 0 <= k < len(lattice):
            out.update(round(value+f*(lattice[k]-value),8) for f in (.25,.5,.75))
    return sorted(out)


def main():
    from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank
    parent = c.read(c.source_path())
    assert c.read(c.PARENT/'FINAL_STATE_001.json')['all_lineage_audits_complete']
    for model in c.SLOTS:
        path = c.HERE/'grids'/f'coarse_{model}.json'
        if path.exists():
            continue
        rows = parent['all_configurations'][model]
        occupied = {(v['mode'],v['parameters']['eta'],v['parameters'].get('alpha',1.)) for v in rows}
        proposed, anchors = [], []
        for mode in ('feedback','state_only'):
            all_mode = [v for v in rows if v['mode'] == mode and not v.get('confirmation_only',False)]
            eligible = [v for v in all_mode if v.get('eligible_for_selection',True)]
            es = sorted({v['parameters']['eta'] for v in all_mode})
            aa = sorted({v['parameters'].get('alpha',1.) for v in all_mode})
            for metric in ('overall_joint_hit','paired_accuracy','auc'):
                anchor = min(eligible,key=lambda v:rank(v,'ie',metric))
                p = anchor['parameters']
                anchors.append(dict(mode=mode,task='ie',metric=metric,config_id=anchor['config_id']))
                for eta in neighbours(p['eta'],es):
                    for alpha in neighbours(p.get('alpha',1.),aa):
                        token = mode,eta,alpha
                        if token in occupied:
                            continue
                        occupied.add(token)
                        proposed.append(c.job(model,mode,eta,alpha,anchor=anchor['config_id'],
                            proposal='Existing eta/radius quarter-neighbour grid; global IE anchor, never per case/type'))
        c.freeze_grid(f'coarse_{model}',dict(at=time.time(),model=model,jobs=proposed,settings=len(proposed),
            cases_each=600,original_history=8,original_forecast=6,anchors=anchors,
            no_new_method_parameters=True,same_exposed_data_exploratory=True,
            source_generator=str(Path(__file__).resolve()),source_generator_sha256=c.digest(__file__)))
        for path in (c.BASE/'capture'/model/'pairs').glob('*.json'):
            c.save(c.HERE/'capture'/model/'pairs'/path.name,c.read(path))
        assert len(list((c.HERE/'capture'/model/'pairs').glob('*.json'))) == 600
        print(model,'new existing-parameter settings',len(proposed),flush=True)
    sources = [c.HERE/n for n in ('common.py','slot_plan_004.py','slot_worker_004.py','slot_queue_004.py','annotation.json')]
    sources += [c.PARENT/n for n in ('worker.py','hooks.py')]
    sources += [c.BASE/n for n in ('worker.py','hooks.py','thermal_guard_032.py')]
    sources += list((c.HERE/'grids').glob('coarse_*.json'))
    c.save(c.HERE/'SLOT_READY_004.json',dict(at=time.time(),parent_report=str(c.source_path()),
        parent_report_sha256=c.digest(c.source_path()),original_operator_and_data_unchanged=True,
        files={str(p.resolve()):c.digest(p) for p in sources}))


if __name__ == '__main__':
    main()
