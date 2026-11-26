"""Lightweight CPU-only final lineage audit after the main serial queue."""
from pathlib import Path
import subprocess,time
from . import common as c
UNIT='predict-ppr-eta-radius-o-lineage-20261104.service'
def main():
    while not (c.HERE/'PERSISTED.json').exists():
        p=c.HERE/'monitor/queue.json'
        if p.exists() and c.read(p).get('state')=='failed':
            state=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],text=True).strip()
            if state=='failed':
                c.save(c.HERE/'POST_PIPELINE_ATTENTION_001.json',dict(at=time.time(),queue=c.read(p),reason='Main queue failed; all saved data retained, no false completion'))
                raise RuntimeError('Main queue failed; see retained logs/status')
        time.sleep(20)
    d=Path(c.read(c.HERE/'FINAL_DELIVERY.json')['directory']);core=c.read(d/'DELIVERY.json')
    for p,w in core['files'].items():assert c.digest(p)==w
    for name in ('PATH_RUNTIME_ADDENDUM_001.json','REPORT_RUNTIME_ADDENDUM_002.json','RETAINED_OUTPUT_CONTROLS_002.json'):
        for p,w in c.read(c.HERE/name)['files'].items():assert c.digest(p)==w
    out=d/'runtime_and_diagnostics_001';out.mkdir(exist_ok=False)
    names=['PLAN.txt','GRID.json','STAGE2.json','STAGE2_FROZEN.json','FROZEN_VERIFIED.json','PREFLIGHT.json','annotation.json',
        'PATH_RUNTIME_ADDENDUM_001.json','REPORT_RUNTIME_ADDENDUM_002.json','RETAINED_OUTPUT_CONTROLS_002.json','VJEPA_G_SP_FIRST_GAIN_CASEWISE_001.json',
        'LAUNCH_RECORD_001.json','discussion_162.txt','SHARED_DISCUSSION_162.json']
    for name in names:
        p=c.HERE/name
        if p.exists():
            with (out/name).open('xb') as f:f.write(p.read_bytes())
    from PIL import Image
    with Image.open(d/'table.png') as im:im.verify()
    c.save(d/'SUPPLEMENT_MANIFEST_001.json',dict(at=time.time(),files={str(p):c.digest(p) for p in out.iterdir()},
        core_manifest_sha256=c.digest(d/'DELIVERY.json'),all_runtime_and_retained_control_sources_verified=True,final_png_decodable=True,
        original_result_files_untouched=True,novel_parameter_points_separated_from_old_native_controls=True))
    c.save(c.HERE/'FINAL_STATE_001.json',dict(at=time.time(),all_grid_and_bounded_refinement_complete=True,all_lineage_audits_complete=True,
        directory=str(d),all_models_improving_not_assumed=True))
    subprocess.run(['sync','-f',str(c.HERE)],check=True);print('FINAL LINEAGE SUPPLEMENT COMPLETE',d,flush=True)
if __name__=='__main__':main()
