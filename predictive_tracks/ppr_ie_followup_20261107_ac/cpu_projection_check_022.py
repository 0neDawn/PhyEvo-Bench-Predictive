"""Check the actual 1152-channel cached DINO operator, without any model calls."""
import os,time
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
    torch.set_num_threads(2)
    selected=[];seen=set()
    for row,mp,meta in c.dino_rows():
        group=row['benchmark'],row['category_directory']
        if group not in seen:seen.add(group);selected.append((row,mp,meta))
    results=[]
    for row,mp,meta in selected:
        assert c.digest(meta['state'])==meta['sha256']
        data=torch.load(meta['state'],map_location='cpu',weights_only=True,mmap=True)
        p,h=data['prediction'],data['history']
        assert p.shape==(1,6,2048,1152) and h.shape==(1,5,2048,1152)
        for alpha in (1.,2.,4.):
            full,_=pooled_trajectory(p,h,alpha)
            per_frame=torch.cat([pooled_trajectory(p[:,t:t+1],h,alpha)[0] for t in range(6)],dim=1)
            same=torch.equal(full,per_frame)
            maximum=float((full-per_frame).abs().max())
            results.append(dict(key=row['key'],alpha=alpha,bitwise_equal=same,max_abs_difference=maximum,
                source_state_sha256=meta['sha256']))
        del data,p,h,full,per_frame
    c.save(c.HERE/'CPU_PROJECTION_CHECK_022.json',dict(at=time.time(),fixed_six_categories=True,
        actual_cached_width1152=True,all_bitwise_equal=all(v['bitwise_equal'] for v in results),results=results,
        no_model_or_CUDA_calls=True,projection_expression_unchanged=True))
    print('Actual-width DINO projection: whole output versus stepwise, all exact=',all(v['bitwise_equal'] for v in results),flush=True)


if __name__=='__main__':main()
