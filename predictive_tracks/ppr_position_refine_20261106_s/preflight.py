"""No neural calls: zero correction, historical containment, placement write scope."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import time
from types import SimpleNamespace
from . import common as c


def main():
    import torch
    from .hooks import LayerShift
    from predictive_tracks.ppr_internal_activation_20261106_r.tail_worker import prepare_shift,shift
    from predictive_tracks.ppr_all_20261029.projection import reference
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid
    for path in c.HERE.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    torch.set_num_threads(1)
    torch.manual_seed(20261106)
    handle=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    checks=[]
    try:
        values=torch.randn(1,7*256,16)
        values[:,4*256:]+=3.
        history=values.reshape(1,7,256,16)[:,:4].mean(2,keepdim=True)
        center,radius,valid=reference(history)
        for alpha in (1.,2.,8.):
            assert ((torch.linalg.vector_norm(history[0]-center,dim=-1)<=alpha*radius+1e-6)).all()
            for eta in (0.,.1,1.):
                item=dict(model='videomaev2',mode='pre_final_block',position='before_norm',eta=eta,alpha=alpha)
                hook=LayerShift(SimpleNamespace(),item)
                original=hook.shift(values)
                proposed=shift(values,prepare_shift(values,alpha),eta)
                assert torch.equal(original,proposed)
                assert torch.equal(original[:,:4*256],values[:,:4*256])
                assert torch.isfinite(original).all()
                if eta==0:
                    assert original is values and proposed is values
                checks.append(dict(alpha=alpha,eta=eta,original_formula_bitwise_equal=True,history_tokens_unchanged=True))
        # Rank/selection decisions use only a fixed global configuration, never cases.
        assert len({c.job('videomaev2','pre_final_block',.1,1.,p)['config_id'] for p in ('block_0','block_1','before_norm')})==3
    finally:
        handle.remove()
    assert not torch.cuda.is_initialized()
    c.save(c.HERE/'PREFLIGHT.json',dict(at=time.time(),passed=True,neural_forwards=0,CUDA_initialized=False,checks=checks,
        real_original_graph_equivalence_required_before_search=True))
    print('S CPU preflight passed; original history/formula/eta0 preserved; CUDA false',flush=True)


if __name__=='__main__':
    main()
