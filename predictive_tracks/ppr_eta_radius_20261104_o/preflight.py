"""Pure-tensor algebra, source contracts and report assembly; no NN forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from types import SimpleNamespace
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
import numpy as np
import torch
from .hooks import factory
from .refine import fast_auc
from .report import assemble
from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid
from predictive_tracks.ppr_all_20261029.projection import reference
from sp_benchmark.protocol_metrics import pooled_auc
def main():
    for p in c.HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    torch.set_num_threads(2);torch.manual_seed(20261104);guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    checks=[]
    try:
        history=torch.randn(1,8,4,8);value=torch.randn(1,4,8)+5
        for m in ('ocvp_seq','savi_transformer','slotformer'):
            for eta,alpha in ((0.,1.),(.15,1.25),(1.,4.)):
                j=c.job(m,'feedback',eta,alpha);h=factory(SimpleNamespace(),j,None);h.begin(history);actual=h.correct_slot(value)
                center,radius,valid=reference(history);distance=torch.linalg.vector_norm(value-center,dim=-1)
                beta=torch.where(distance>alpha*radius,eta*(1-alpha*radius/distance),0.)[...,None]
                assert torch.allclose(actual,value+beta*(center-value),atol=2e-6)
                if eta==0:assert actual is value
                checks.append(dict(model=m,eta=eta,alpha=alpha,closed_form=True))
        q=dict(q=.75,presence=.5,mass=4.,min_history=2)
        for m in c.OBJECTS:
            j=c.job(m,'feedback',0.,4.,q,'ie');h=factory(SimpleNamespace(),j,None)
            h.establish(history[0],torch.ones(8,4),torch.ones(8,4)*10,torch.arange(4))
            if m=='ddlp':actual=h.correct(value)
            else:h.lookup={i:i for i in range(4)};actual=h.correct_by_id(value,torch.arange(4)[None])
            assert actual is value
        rng=np.random.default_rng(20261104);p=rng.random((10,300));i=rng.random((10,300));auc=fast_auc(p,i)
        for k in range(10):assert abs(auc[k]-pooled_auc(i[k]/(p[k]+i[k]+1e-12),p[k]/(p[k]+i[k]+1e-12)))<1e-14
        report=assemble();assert report['completed_round_configurations']==0
        for t in ('sp','ie'):assert len(report['selected'][t])==9 and len(report['metric_specific_maxima'][t])==9
    finally:guard.remove()
    c.save(c.HERE/'PREFLIGHT.json',dict(at=time.time(),passed=True,neural_forwards=0,checks=checks,proposal_AUC_exact_on_test=True,report_assembly_passed=True))
    print('NO-NEURAL PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
