"""Read-only full graph status; never import torch or execute a model."""
import json,time
from pathlib import Path
from . import common as c


def optional(path):return c.read(path) if path.exists() else None


def main():
    ae=c.ROOT/'predictive_tracks/ppr_buffer_read_20261107_ae'
    stages={}
    for label,package in [('S','ppr_position_refine_20261106_s'),('T','ppr_feedback_refine_20261106_t'),
        ('AC','ppr_ie_followup_20261107_ac'),('U','ppr_vitg_placement_20261106_u'),
        ('W','ppr_history_controls_20261106_w'),('Z','ppr_after_norm_20261106_z'),('AE','ppr_buffer_read_20261107_ae')]:
        root=c.ROOT/'predictive_tracks'/package
        stages[label]=optional(root/'monitor/queue.json')
    selected=None
    for root,filename,label in [(ae,'FINAL_DELIVERY.json','AE'),(c.HERE,'MERGED_FINAL_DELIVERY_019.json','AC+Z merge'),
        (c.HERE,'FINAL_DELIVERY.json','AC'),(c.ROOT/'predictive_tracks/ppr_after_norm_20261106_z','FINAL_DELIVERY.json','Z'),
        (c.PARENT,'FINAL_DELIVERY.json','T')]:
        path=root/filename
        if path.exists():
            value=c.read(path)
            if value.get('complete'):
                folder=Path(value['directory']);selected=dict(stage=label,table=str(folder/'table.png'),parameters=str(folder/'best_parameters.json'));break
    progress=[]
    for model in c.MODELS:
        for phase in ('coarse','fine'):
            p=c.HERE/'search'/phase/model/'progress.json'
            if p.exists():progress.append(dict(c.read(p),model=model,phase=phase))
    result=dict(at=time.time(),stages=stages,AC_slot_queue=optional(c.HERE/'monitor/slots_004.json'),AC_progress=progress,
        DINO_new_probe_wait=optional(c.HERE/'monitor/probe_006_wait.json'),DINO_latest_probe=optional(c.HERE/'monitor/probe.json'),
        DINO_new_probe_complete=(c.HERE/'PROBE_COMPLETE_005.json').exists(),
        U_memory_recovery=optional(c.ROOT/'predictive_tracks/ppr_vitg_placement_20261106_u/monitor/memory_recovery_003.json'),
        merge=optional(c.HERE/'monitor/merge_019.json'),last_completed_table=selected,
        all_new_phases_complete=(ae/'COMPLETED_001.json').exists(),
        DINO_legacy_alignment_diagnostic_qualification_retained=True,no_model_calls=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
