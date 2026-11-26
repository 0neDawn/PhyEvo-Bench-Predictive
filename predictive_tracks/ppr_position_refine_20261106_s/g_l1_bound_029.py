"""Target-aware diagnostic only: optimistic L1 bound for nine native SP errors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import time
from pathlib import Path
import numpy as np
from . import common as c


def guard():
    from predictive_tracks.ppr_joint_fine_20261105_p.common import cpu_guard
    cpu_guard()
    available=next(int(x.split()[1])/1024 for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))
    assert available>=16384,'Pause CPU diagnostic below 16 GiB available RAM'


def main():
    guard()
    declaration=c.read(c.HERE/'G_BOUND_DECLARATION_029.json')
    for path,sha in declaration['files'].items():assert c.digest(path)==sha,path
    import sys
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    import torch
    from predictive_tracks.ppr_joint_fine_20261105_p.l1_output_bound_001 import frame_bound
    from predictive_tracks.ppr_all_20261029.projection import reference
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import load_data,forbid
    torch.set_num_threads(1)
    support=c.read(c.HERE/'G_SP_SUPPORT_028.json')
    assert support['native_correct']==291 and len(support['wrong_cases'])==9
    hook=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    results=[]
    try:
        with torch.inference_mode():
            for entry in support['wrong_cases']:
                guard()
                out=c.HERE/'g_l1_bound_029/pairs'/f'{entry["case"]}.json'
                if out.exists():
                    value=c.read(out)
                    assert value['source_cache_sha256']==entry['source_sha256']
                    assert c.digest(value['source_state'])==value['source_state_sha256']
                    results.append(value)
                    continue
                source=Path(entry['source'])
                assert c.digest(source)==entry['source_sha256']
                cached=c.read(source)
                assert c.digest(cached['native_meta'])==cached['source_meta_sha256']
                meta=c.read(cached['native_meta'])
                assert meta['sha256']==cached['source_state_sha256']
                data=load_data(meta)
                p,h=data['prediction'],data['history']
                center,radius,valid=reference(h.mean(2,keepdim=True))
                assert p.ndim==4 and p.shape[0]==1 and bool(valid.all())
                center=center[0].double().numpy()
                radius=float(radius[0])
                mean=p.mean(2)[0].double().numpy()
                pp,tp,ti=[data[k][0].double().numpy() for k in ('prediction','possible_target','impossible_target')]
                gpu=np.asarray(meta['matched_identity_impossible_errors'])-np.asarray(meta['matched_identity_possible_errors'])
                assert gpu.mean()<=0
                frames=[]
                for t in range(p.shape[1]):
                    direction=center-mean[t]
                    distance=float(np.linalg.norm(direction))
                    limit=max(0.,1.-radius/distance) if distance else 0.
                    value=frame_bound(pp[t],tp[t],ti[t],direction,limit)
                    value['float64_native_minus_original_GPU_margin']=value['native_margin']-float(gpu[t])
                    frames.append(value)
                upper=float(np.mean([x['maximum_margin'] for x in frames]))
                value=dict(at=time.time(),case=entry['case'],frames=frames,relaxed_upper_bound_mean_margin=upper,
                    not_ruled_out_in_ideal_arithmetic=upper>0,original_GPU_mean_margin=float(gpu.mean()),
                    source_cache=str(source),source_cache_sha256=entry['source_sha256'],
                    source_state=meta['state'],source_state_sha256=meta['sha256'])
                c.save(out,value)
                results.append(value)
                c.status(c.HERE/'g_l1_bound_029/progress.json',dict(at=time.time(),completed=len(results),total=9))
                print(entry['case'],'optimistic independent-frame upper margin',upper,flush=True)
                del data,p,h,pp,tp,ti
    finally:hook.remove()
    assert not torch.cuda.is_initialized()
    for path,sha in declaration['files'].items():assert c.digest(path)==sha,path
    result=dict(at=time.time(),model='vjepa2_vitg',task='sp',NN0=True,CUDA_initialized=False,
        original_wrong_cases=9,native_correct=291,cases=results,
        ideal_wrong_cases_not_ruled_out=sum(x['not_ruled_out_in_ideal_arithmetic'] for x in results),
        eta_domain=[0,1],alpha_domain=[1,'infinity'],output_insertion_only=True,
        independent_frame_factors_relax_the_actual_shared_eta_alpha=True,
        targets_used_only_for_diagnostic=True,no_inference_rule_or_official_score_changes=True,
        not_a_float32_CUDA_proof_or_measured_performance=True,
        source_declaration_sha256=c.digest(c.HERE/'G_BOUND_DECLARATION_029.json'))
    c.save(c.HERE/'G_L1_BOUND_029.json',result)
    print('G SP optimistic wrong cases not ruled out',result['ideal_wrong_cases_not_ruled_out'],'/9; diagnostic only',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        c.save(c.HERE/'g_l1_bound_029'/f'error_{time.time_ns()}.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
