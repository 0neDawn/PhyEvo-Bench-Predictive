"""No neural calls: verify unchanged formula, exact reuse and all grid endpoints."""
from types import SimpleNamespace
import torch
from .common import HERE,MODELS,configs,qualities,quality_id,config_id,save
from .hooks import reference,operator_digest,ExistingDDLP,ExistingGSWM
from predictive_tracks.ppr_20261028.projection import historical_ball,project

def main():
    torch.set_num_threads(2);torch.manual_seed(20261102)
    grid={m:configs(m) for m in MODELS};counts={m:len(v) for m,v in grid.items()}
    assert sum(counts.values())==220 and all(len({config_id(c) for c in v})==len(v) for v in grid.values())
    assert all('alpha' not in c for v in grid.values() for c in v)
    h=torch.randn(8,5,8);p=torch.rand(8,5);mass=torch.rand(8,5)*40;ids=torch.arange(5)
    default=next(c for c in grid['ddlp'] if c=={'eta':1.,'q':.5,'presence':.5,'mass':4.,'min_history':2})
    baseline=historical_ball(h,p,mass);current=reference(h,p,mass,default)
    assert all(torch.equal(a,b) for a,b in zip(baseline,current))
    for c in grid['ddlp']:
        hook=ExistingDDLP(SimpleNamespace(),'feedback',params=c,capture_all=True)
        hook.presence=p;hook.mass=mass;hook.begin(h[None]);y=torch.randn(1,1,5,8)
        hard,_=project(y,hook.center,hook.radius,hook.valid)
        expected=hard if c['eta']==1. else y+c['eta']*(hard-y)
        assert torch.equal(hook.correct(y),expected)
        assert len(hook.history_signatures[0]['operators'])==24
    g=ExistingGSWM(SimpleNamespace(),'feedback',params=default,capture_all=True);g.history=[]
    native_ids=torch.tensor([7,4,9,2,12]);order=torch.tensor([4,1,3,0,2])
    for t in range(8):
        index=order if t%2 else torch.arange(5);g.history.append((native_ids[index],h[t,index],p[t,index],mass[t,index]))
    g.freeze_history()
    for j,k in enumerate(sorted(native_ids.tolist())):
        col=native_ids.tolist().index(k)
        assert torch.equal(g.center[j],baseline[0][col]) and torch.equal(g.radius[j],baseline[1][col])
    # Different selector values, identical effective references: exact same recurrence.
    stable_p=torch.ones_like(p);stable_mass=torch.full_like(mass,100.)
    signatures={}
    for c in qualities('ddlp'):
        center,radius,valid,_=reference(h,stable_p,stable_mass,c)
        signatures[quality_id(c)]=operator_digest(center,radius,valid,ids)
    assert len(set(signatures.values()))==1
    for eta in (.25,.5,.75,1.):
        output=[]
        for c in qualities('ddlp'):
            center,radius,valid,_=reference(h,stable_p,stable_mass,c);y=h[-1].clone()
            for _ in range(6):
                y=.9*y+.17;hard,_=project(y,center,radius,valid)
                y=hard if eta==1. else y+eta*(hard-y)
            output.append(y)
        assert all(torch.equal(output[0],x) for x in output)
    maximum=-1.
    for eta in (.25,.5,.75,1.):
        c=torch.randn(10000,8,dtype=torch.float64);r=torch.rand(10000,dtype=torch.float64);r[:100]=0
        y=torch.randn_like(c)*3;direction=torch.randn_like(c);direction/=torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
        u=c+direction*(r*torch.rand_like(r))[:,None]
        hard,_=project(y,c,r,torch.ones_like(r,dtype=torch.bool));a=y+eta*(hard-y)
        gap=((a-u)**2).sum(-1)-((y-u)**2).sum(-1)+eta*(2-eta)*((y-hard)**2).sum(-1)
        maximum=max(maximum,float(gap.max()));assert gap.max()<1e-9
    try:reference(h,p,mass,dict(default,alpha=1.5))
    except ValueError:pass
    else:raise AssertionError('Forbidden parameter accepted')
    result={'passed':True,'neural_forwards':0,'alpha_fixed':1,'radius_multiplier_argument_rejected':True,
        'model_config_counts':counts,'total_configs':220,'all_96_object_settings_operator_checks':True,
        'default_reference_bitwise_original':True,'native_ID_reordering_check':True,'history_only_equivalence_recurrence_check':True,
        'conditional_bound_trials':40000,'maximum_bound_violation':maximum}
    save(HERE/'PREFLIGHT.json',result);print(result,flush=True)

if __name__=='__main__':main()
