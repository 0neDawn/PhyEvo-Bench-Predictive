"""CPU-only final provenance supplement; original delivery files never edited."""
import os
from pathlib import Path
import shutil
import subprocess
import time
from . import common as c

UNIT='predict-ppr-eta-radius-final-runtime-n-20261104.service'

def main():
    while not (c.HERE/'PERSISTED.json').exists():
        if (c.HERE/'monitor/queue.json').exists():
            q=c.read(c.HERE/'monitor/queue.json')
            if q.get('state')=='failed':
                state=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],text=True).strip()
                if state=='failed':
                    c.save(c.HERE/'POST_PIPELINE_ATTENTION_001.json',dict(at=time.time(),queue=q,reason='Main queue failed; retained data, no false completion'))
                    raise RuntimeError('Main queue failed; see retained status/logs')
        time.sleep(20)
    d=Path(c.read(c.HERE/'FINAL_DELIVERY.json')['directory']);core=c.read(d/'DELIVERY.json')
    for p,w in core['files'].items():assert c.digest(p)==w
    out=d/'runtime_and_diagnostics_001';out.mkdir(exist_ok=False)
    names=['PLAN.txt','GRID.json','STAGE2.json','STAGE2_FROZEN.json','PREFLIGHT.json','FROZEN_VERIFIED.json',
        'RUNTIME_ADDENDUM_001.json','RUNTIME_ADDENDUM_002.json','RUNTIME_ADDENDUM_003.json','REPORT_FIX_TRANSFORM_001.json',
        'PROPOSAL_TRANSFORM_002.json','PROPOSAL_TRANSFORM_003.json','RESTORED_METRIC_RECHECK_001.json',
        'OCVP_FIRST_GAIN_AUDIT_001.json','VIDEO_LOCALIZATION_TRADEOFF_AUDIT_001.json',
        'MSE_PROPOSAL_ANALYSIS_002.json','MSE_IDEAL_COEFFICIENTS_002.json','SHARED_DISCUSSION_159.json','SHARED_DISCUSSION_160.json',
        'discussion_159.txt','discussion_160.txt','annotation.json']
    for name in names:
        p=c.HERE/name
        if p.exists():
            with (out/name).open('xb') as f:f.write(p.read_bytes())
    sources={}
    for name in ('RUNTIME_ADDENDUM_001.json','RUNTIME_ADDENDUM_002.json','RUNTIME_ADDENDUM_003.json'):
        for p,w in c.read(c.HERE/name)['files'].items():assert c.digest(p)==w;sources[p]=w
    report=c.read(d/'report.json')
    assert len(report['current_round_settings'])==len(c.read(c.HERE/'GRID.json')['neural_new'])+126+len(c.read(c.HERE/'STAGE2.json')['neural_new'])+len(c.read(c.HERE/'STAGE2.json')['cached_video'])
    from PIL import Image
    with Image.open(d/'table.png') as im:im.verify()
    c.save(d/'SUPPLEMENT_MANIFEST_001.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir()},
        runtime_sources=sources,core_manifest_sha256=c.digest(d/'DELIVERY.json'),final_png_decodable=True,
        initial_report_bug_fixed_in_new_files=True,all_old_files_and_saved_scores_preserved=True,
        layout_template_visual_QA='preview_001/table.png inspected; fixed nine-model/eighteen-row layout reused',
        actual_full_settings=len(report['current_round_settings']),all_extra_files_supplement_only=True))
    c.save(c.HERE/'FINAL_STATE_001.json',dict(at=time.time(),all_authorized_grid_and_bounded_refinement_complete=True,
        all_diagnostics_and_runtime_lineage_preserved=True,directory=str(d),all_three_IE_gains_not_assumed=True))
    subprocess.run(['sync','-f',str(c.HERE)],check=True)
    print('FINAL PROVENANCE SUPPLEMENT COMPLETE',d,flush=True)

if __name__=='__main__':main()
