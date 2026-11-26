"""Freeze the complete additive follow-up before the remaining GPU phases."""
import time
from pathlib import Path
from . import common as c


def main():
    c.verify_sources('SLOT_READY_004.json')
    c.verify_sources('PROBE_READY_005.json')
    assert c.read(c.PARENT/'FINAL_STATE_001.json')['all_lineage_audits_complete']
    parent = c.source_path().parent
    c.save(c.HERE/'PARENT_MANIFESTS.json',{str(parent/name):c.digest(parent/name) for name in
        ('DELIVERY.json','SUPPLEMENT_MANIFEST_001.json','USER_SUMMARY_MANIFEST_001.json','FINAL_REVIEW_MANIFEST_025.json')})
    values = [c.job('dino_foresight','feedback',eta,alpha,
        proposal='Same CPU Euclidean projection at original-loop writeback; no change in history or model calls')
        for eta in (.1,.5,1.) for alpha in (1.,2.,4.)]
    c.freeze_grid('coarse_dino_foresight',dict(at=time.time(),model='dino_foresight',jobs=values,settings=len(values),cases_each=600,
        diagnostic_only=True,original_alignment_unchanged=True,eta_and_radius_only=True))
    for row,mp,meta in c.dino_rows():
        c.save(c.HERE/'capture/dino_foresight/pairs'/f'{row["key"]}.json',dict(at=time.time(),key=row['key'],model='dino_foresight',
            tensor=meta['state'],tensor_sha256=meta['sha256'],source_meta=str(mp),source_meta_sha256=c.digest(mp),source_state_sha256=meta['sha256'],
            source_pixels_sha256=meta['source_decoded_pixels_sha256'],positions=['feedback'],
            native_cache_reused=True,raw_forecaster_replay_exact_required_before_every_search_case=True,
            original_prediction_bitwise_equal_not_yet_claimed_for_new_replay=True,diagnostic_only=True))
    c.save(c.HERE/'PLAN.json',dict(at=time.time(),models=list(c.MODELS),independent_pairs=600,SP=300,IE=300,
        original_Euclidean_operator_unchanged=True,original_weights_sources_and_per_forecast_budget_unchanged=True,
        slot_history=8,slot_native_buffer=5,dino_history=5,dino_context_not_a_tunable_window=True,future_steps=6,
        original_DINO_cache_scoring_expression_unchanged=True,original_perframe_scalars_saved_separately=True,
        DINO_alignment_not_repaired=True,DINO_diagnostic_only=True,
        search='Declared slot quarter-neighbour grids then one midpoint round; DINO 3x3 then midpoint round; global endpoint anchors only.',
        primary_rank='Inherited original JH/PA/AUC/conditional-MAE ordering; one real setting per model/task row.',
        selected_new_primary_and_PAmax_native600_required=True,all_old_negative_zero_and_failure_results_retained=True,
        same_exposed_data_exploratory_not_held_out=True))
    with (c.HERE/'PLAN.txt').open('x') as stream:
        stream.write('AC: only existing eta/radius and original slot/DINO loop positions. Original weights, Euclidean norm, history, precision and per-inference forward count unchanged. SP300+IE300. DINO old alignment retained, diagnostic only; existing cached scoring unmodified, native per-frame scalar differences separately saved. Full600 validates every new selected setting. All outcomes retained; exposed-data exploratory.\n')
    paths = list(c.HERE.glob('*.py'))+[c.HERE/n for n in ('SLOT_READY_004.json','PROBE_READY_005.json','PLAN.json','PLAN.txt','PARENT_MANIFESTS.json','annotation.json')]
    paths += list((c.HERE/'grids').glob('*.json'))
    paths += list((c.HERE/'capture').glob('*/pairs/*.json'))
    for declaration in ('SLOT_READY_004.json','PROBE_READY_005.json'):
        paths += [Path(p) for p in c.read(c.HERE/declaration)['files']]
    frozen = dict(at=time.time(),files={str(p.resolve()):c.digest(p) for p in dict.fromkeys(paths)})
    c.save(c.HERE/'SOURCE_READY_001.json',frozen)
    c.save(c.HERE/'frozen.json',frozen)
    c.verify(full=True)
    print('AC follow-up frozen: existing slot grids and DINO9, followed by declared midpoint/native600 phases',flush=True)


if __name__ == '__main__':
    main()
