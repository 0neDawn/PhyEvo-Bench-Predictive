"""Small read-only status across existing owned queues; no model imports."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    'S': 'ppr_position_refine_20261106_s',
    'T': 'ppr_feedback_refine_20261106_t',
    'U': 'ppr_vitg_placement_20261106_u',
    'W': 'ppr_history_controls_20261106_w',
    'Z': 'ppr_after_norm_20261106_z',
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    stages = {}
    last_table = None
    for key, package in PACKAGES.items():
        folder = ROOT/'predictive_tracks'/package
        queue = read(folder/'monitor/queue.json')
        row = {k:queue.get(k) for k in ('state','active','next','error','faults')}
        row['last_queue_update'] = queue.get('updated_at', queue.get('at'))
        row['model_runs_complete'] = (folder/'MODEL_RUNS_COMPLETE.json').exists()
        row['final_state'] = read(folder/'FINAL_STATE_001.json')
        row['progress'] = []
        active = queue.get('active') or {}
        if active.get('model') and active.get('phase'):
            p = folder/'search'/active['phase']/active['model']/'progress.json'
            if p.exists():
                row['progress'].append(dict(file=str(p), **read(p)))
        delivery = read(folder/'FINAL_DELIVERY.json').get('directory')
        if delivery and key != 'W' and queue.get('state') == 'complete':
            last_table = dict(stage=key, table=str(Path(delivery)/'table.png'),
                              parameters=str(Path(delivery)/'best_parameters.json'),
                              report=str(Path(delivery)/'report.json'))
        stages[key] = row
    v = ROOT/'predictive_tracks/ppr_completion_review_20261106_v'
    result = dict(at=time.time(), stages=stages, independent_CPU_audit=read(v/'status.json'),
                  last_completed_nine_model_table=last_table,
                  all_queued_stages_complete=all(x.get('state') == 'complete' for x in stages.values()),
                  DINO_legacy_diagnostic_only=True,
                  DINO_alignment_GPU_not_queued_without_actual_user_decision=True,
                  no_model_imports_or_foreign_process_changes=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
