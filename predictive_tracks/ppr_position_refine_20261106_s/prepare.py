"""Freeze scope and a DINO grid from original history/prediction audits."""
import time
from . import common as c


def main():
    parent = c.read(c.source_path())
    assert c.read(c.PARENT/'FINAL_STATE_001.json')['all_lineage_audits_complete']
    c.save(c.HERE/'PARENT_MANIFESTS.json', {str(p):c.digest(p) for p in (
        c.source_path().parent/name for name in ('DELIVERY.json', 'SUPPLEMENT_MANIFEST_001.json', 'USER_SUMMARY_MANIFEST_001.json', 'FINAL_REVIEW_MANIFEST_001.json'))})
    annotation = c.read(c.PARENT/'annotation.json')
    c.save(c.HERE/'annotation.json', annotation)
    st = __import__('pathlib').Path(annotation['rollout_path']).stat()
    assert annotation['task_complete_verified'] and annotation['turn_status'] == 'completed'
    assert st.st_size == annotation['rollout_size'] and st.st_mtime_ns == annotation['rollout_mtime_ns']
    ratios, sources, per_task = [], {}, {}
    for task in ('sp', 'ie'):
        values = []
        for index in range(300):
            p = c.OLD/'cache/dino_foresight/pairs'/f'{task}_{index:03d}.json'
            a = c.read(p)['projection_audits']['1.0']
            radius = a['base_radii_min_mean_max'][1]
            assert radius > 0 and a['history_steps'] == 5
            values.extend(1.+s['change_l2']/radius if s['outside_units'] else 1. for s in a['steps'])
            sources[str(p)] = c.digest(p)
        per_task[task] = {str(q):c.quantile(values, q) for q in (0., .1, .25, .5, .75, .9, 1.)}
        ratios.extend(values)
    alphas = sorted(set(c.grid_alphas(ratios)+[4., 5., 6., 7., 8., 10., 12., 16., 20., 24.]))
    existing = {(v['parameters']['eta'],v['parameters'].get('alpha',1.)) for v in parent['all_configurations']['dino_foresight'] if v['mode']=='output_only'}
    jobs = [c.job('dino_foresight','output_only',eta,alpha,proposal='history/native prediction activation range, same operator')
            for alpha in alphas for eta in c.ETAS if (eta,alpha) not in existing]
    c.save(c.HERE/'DINO_ACTIVATION_ANALYSIS.json',dict(at=time.time(),per_task=per_task,alphas=alphas,files=sources,
        grid_uses_only_history_and_native_predictions=True, diagnostic_only=True))
    c.save(c.HERE/'GRID.json',dict(at=time.time(),cached=jobs,native_new=[],native_reuse=[],settings=len(jobs),
        independent_pairs=600,cases_each=600,all_neural_forwards=0))
    plan = dict(at=time.time(),parent=str(c.source_path()),parent_sha256=c.digest(c.source_path()),
        formula='raw + eta*(Pi_B(c,alpha*R)(raw)-raw); eta0/raw, eta1/hard, original float expressions',
        eta_domain=[0.,1.],alpha_minimum=1.,metric='original Euclidean',
        history='Original native history, no window/selection changes; DINO original native context remains five encoded states from eight available frames',
        phases=['DINO existing output eta/radius range expansion',
            'OCVP/SAVi capture original slots once and search original output-only decoding with denser eta/radius grid',
            'VideoMAE/V-JEPA-L same projection before other original blocks or original final norm, one insertion per configuration',
            'Global midpoint refinement around actual taskwise primary/AUC candidates',
            'Full original600 validation for every new primary partial-graph candidate',
            'Recompute original metrics, retain all parent/negative/zero results, nine-model table'],
        feature_positions='Video: all nonfinal blocks if depth<=4; otherwise middle and penultimate. L: middle and third-last. Both: before original final norm.',
        coarse_etas=list(c.ETAS),coarse_alphas='1 and pooled native/history d/R quantiles 0,.1,.25,.5,.75,.9,1, minimum1; slot references per original unit',
        refinement='At most two anchors (fixed primary, AUC) per model/task; neighboring eta/alpha midpoints; one global configuration per row, never per-case or per-type',
        no_changes=['original source files','weights','data','scores or thresholds','native forward count/order/shapes','history window','precision/batch/device','distance definition'],
        placement_is_explicit_variant=True,parameter_independent_prefix_reuse_is_offline_search_only=True,
        selected_new_partial_graph_settings_require_original_full600=True,
        adaptive_same_exposed_data_search=True,never_claim_independent_validation=True,
        stop_on_any_exact_equivalence_or_budget_failure=True,other_tasks_unchanged=True)
    c.save(c.HERE/'PLAN.json', plan)
    with (c.HERE/'PLAN.txt').open('x') as stream:
        stream.write('S: same original Euclidean history projection; global eta/radius and explicitly declared original-layer placement comparisons. Original files, data, weights, inference budget and metric fixed. Reuse prefixes only for offline search. Full original600 validates every new primary configuration. All outcomes preserved. Exposed-data exploratory search.\n')
    paths = list(c.HERE.glob('*.py'))+[c.HERE/n for n in ('GRID.json','PLAN.json','PLAN.txt','DINO_ACTIVATION_ANALYSIS.json','PARENT_MANIFESTS.json')]
    paths += [__import__('pathlib').Path(p) for p in c.read(c.PARENT/'frozen.json')['files']]
    c.save(c.HERE/'frozen.json',dict(at=time.time(),files={str(p):c.digest(p) for p in dict.fromkeys(paths)}))
    c.verify(full=True)
    print('S PLAN FROZEN',len(jobs),'DINO configurations; remaining coarse grids derive only from original native states',flush=True)


if __name__=='__main__':
    main()
