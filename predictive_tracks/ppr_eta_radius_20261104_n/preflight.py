"""No-neural tests: scaled projection and old-radius compatibility."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import inspect
import sys
import time
from types import SimpleNamespace
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import torch
from .hooks import TunedSlot,TunedPreFinal
from .native_worker import transformed
from predictive_tracks.ppr_eta_20261030.hooks import SoftSlot
from predictive_tracks.ppr_state_insert_20261104_m.hooks import PreFinal
from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid

def main():
    for p in c.HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    torch.set_num_threads(2);torch.manual_seed(20261104)
    guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    checks=[]
    try:
        h=torch.randn(1,8,6,16);y=torch.randn(1,6,16)+6
        for alpha in (1.,1.25,2.):
            for eta in (0.,.005,.1,1.):
                n=TunedSlot(SimpleNamespace(),'ocvp_seq',eta,alpha,'feedback',None);n.begin(h);z=n.correct_slot(y)
                center,radius,valid=n.state;dist=torch.linalg.vector_norm(y-center,dim=-1)
                beta=torch.where(dist>radius,eta*(1-radius/dist),0.).unsqueeze(-1)
                want=y+beta*(center-y)
                assert torch.allclose(z,want,atol=2e-6)
                if eta==0:assert z is y
                if alpha==1:
                    old=SoftSlot(SimpleNamespace(),'feedback','ocvp_seq',None,eta=eta);old.begin(h)
                    assert torch.equal(z,old.correct_slot(y))
                # Every original historical state remains inside alpha>=1.
                assert bool((torch.linalg.vector_norm(h[0]-center,dim=-1)<=radius+1e-5).all())
                checks.append(dict(alpha=alpha,eta=eta,old_compatible=alpha==1,closed_form=True))
        v=torch.randn(1,7*256,16);v[:,4*256:]+=5
        for eta in (0.,.01,1.):
            n=TunedPreFinal(SimpleNamespace(),'videomaev2',eta,1.,'pre_final_block',None)
            old=PreFinal(SimpleNamespace(),'videomaev2',eta)
            assert torch.equal(n.shift(v),old.shift(v))
            states=v.reshape(1,7,256,16)
            hard,audit=pooled_trajectory(states[:,4:],states[:,:4],2.)
            assert audit['alpha']==2.
        run,source,changed=transformed()
        path=c.HERE/'native_transformed_source.txt'
        if path.exists():assert path.read_text()==changed
        else:
            with path.open('x') as f:f.write(changed)
        from .native_worker import original
        c.save(c.HERE/'NATIVE_TRANSFORM.json',dict(original_file=inspect.getfile(original.run_setting),
            original_function_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
            transformed_function_sha256=__import__('hashlib').sha256(changed.encode()).hexdigest(),
            changes='Only hook factories bound in globals; two radius_multiplier metadata literals changed'))
    finally:guard.remove()
    c.save(c.HERE/'PREFLIGHT.json',dict(at=time.time(),passed=True,neural_forwards=0,checks=checks))
    print('NO-NEURAL PREFLIGHT PASSED',flush=True)

if __name__=='__main__':main()
