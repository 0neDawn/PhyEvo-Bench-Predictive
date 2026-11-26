"""Casewise inspection of the complete IE candidate; no new prediction calls."""
import time
import sys
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
from sp_benchmark.protocol_metrics import summarize
from predictive_tracks.ppr_position_controls_20261103_k.casewise_audit_001 import flags
from .worker import expand


def main():
    task='ie'
    model='videomaev2'
    interim=max((c.HERE/'task_interims').glob('videomaev2_ie_coarse_*.json'),key=lambda p:p.name)
    state=c.read(interim)
    item=state['best_provisional']
    assert item['config_id'].startswith('S_')
    files=sorted((c.HERE/'search/coarse/videomaev2/pairs').glob('ie_*.json'))
    assert len(files)==300
    rows=[expand(c.read(p),item) for p in files]
    def calculate(prefix):
        return summarize(np.asarray([r[prefix+'possible_errors'] for r in rows]),
            np.asarray([r[prefix+'impossible_errors'] for r in rows]),frame_spans=rows[0]['frame_spans'],
            violation_frames=[r['t_ref'] for r in rows],epsilon=1e-12,threshold=0.,persistence_frames=2,joint_tolerances=(2,))
    bp,bm,bj,be=flags(calculate('matched_identity_'),2)
    ap,am,aj,ae=flags(calculate(''),2)
    common=bm&am
    def ids(mask):
        return [f'ie_{rows[i]["archive_index"]:03d}' for i in np.flatnonzero(mask)]
    result=dict(at=time.time(),model=model,task=task,config_id=item['config_id'],parameters=item['parameters'],
        candidate_metrics=item['metrics'][task],PA_gain=ids(ap&~bp),PA_loss=ids(~ap&bp),JH_gain=ids(aj&~bj),JH_loss=ids(~aj&bj),
        native_MAE_n=int(bm.sum()),candidate_MAE_n=int(am.sum()),common_MAE_n=int(common.sum()),
        common_native_MAE=float(be[common].mean()) if common.any() else None,
        common_candidate_MAE=float(ae[common].mean()) if common.any() else None,
        onset_improved=ids(common&(ae<be)),onset_worsened=ids(common&(ae>be)),
        source_interim=str(interim),source_interim_sha256=c.digest(interim),
        original_scoring_and_flags=True,no_new_model_calls=True,not_full600_native_validated=True,
        same_exposed_data_candidate_not_an_independent_test=True)
    out=c.HERE/'candidate_audits'/f'videomaev2_ie_{time.time_ns()}.json'
    c.save(out,result)
    print({k:v for k,v in result.items() if k!='candidate_metrics'},flush=True)
    print('by_type',item['metrics'][task]['corrected']['by_type'],flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
