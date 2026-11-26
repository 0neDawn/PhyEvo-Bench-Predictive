"""Quiet CPU-only monitor: persistent status, no model execution or restarts."""
import json
import os
import time
from pathlib import Path
from . import common as c


def load(path):
    return c.read(path) if path.exists() else None


def snapshot():
    roots = c.ROOT/'predictive_tracks'
    ae = roots/'ppr_buffer_read_20261107_ae'
    u = roots/'ppr_vitg_placement_20261106_u'
    z = roots/'ppr_after_norm_20261106_z'
    stages = dict(AC=load(c.HERE/'monitor/queue.json'),
        AE=load(ae/'monitor/queue.json'), AE_priority=load(ae/'monitor/priority_004.json'),
        U_recovery=load(u/'monitor/memory_recovery_003.json'),
        U_priority=load(u/'monitor/memory_priority_004.json'),
        W=load(roots/'ppr_history_controls_20261106_w/monitor/queue.json'),
        Z=load(z/'monitor/queue.json'),
        union=load(c.HERE/'monitor/final_union_035.json'),
        independent_review=load(c.HERE/'monitor/audit_023.json'))
    progress = []
    for root in (c.HERE,ae):
        for path in sorted((root/'search').glob('*/*/progress.json')):
            progress.append(dict(track=root.name, **c.read(path)))
    last = roots/'ppr_feedback_refine_20261106_t/delivery_20261106_001'
    for marker in (c.HERE/'FINAL_DELIVERY.json', ae/'FINAL_DELIVERY.json', c.HERE/'FINAL_ALL_DELIVERY_035.json'):
        value = load(marker)
        if value and value.get('complete'):
            last = Path(value['directory'])
    complete = (c.HERE/'FINAL_ALL_DELIVERY_035.json').exists()
    failures = {key:value for key,value in stages.items() if value and value.get('state')=='failed'}
    # The original three U failures are archived, separate from the new priority recovery.
    return dict(at=time.time(), all_new_experiments_and_automatic_reviews_complete=complete,
        root_visual_inspection_still_required=True,
        priority_order='AC (OCVP/SAVi/DINO) → AE (OCVP/SAVi matched read position) → U → W → Z → final union',
        stages=stages, progress=progress, current_failures=failures,
        last_completed_table=str(last/'table.png'),last_completed_parameters=str(last/'best_parameters.json'),
        hardware=load(c.HERE/'monitor/hardware_latest.json'),
        original_constraints_unchanged=True,all_results_retained=True,DINO_diagnostic_only=True,
        same_exposed_data_exploratory=True,no_model_calls_or_process_intervention=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path,wanted in c.read(c.HERE/'WATCH_READY_038.json')['files'].items():
        assert c.digest(path)==wanted,path
    while True:
        value = snapshot()
        c.status(c.HERE/'EXPERIMENT_STATUS_038.json',value)
        if value['all_new_experiments_and_automatic_reviews_complete']:
            c.save(c.HERE/'AUTOMATIC_COMPLETION_038.json',dict(at=time.time(),
                table=value['last_completed_table'],parameters=value['last_completed_parameters'],
                all_model_experiments_and_automatic_reviews_complete=True,
                root_must_view_table_before_claiming_visual_review=True))
            return
        time.sleep(30)


if __name__=='__main__':
    main()
