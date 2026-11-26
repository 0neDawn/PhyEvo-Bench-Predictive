"""Use all existing same-position knots, without adding any model evaluation."""
import hashlib
import inspect
from . import common as c

def install():
    from . import refine
    from .mse_proposals_002 import generate
    source=inspect.getsource(refine.main)
    changes=[
        ('            sample=np.linspace(0,1,8193)','            sample=np.linspace(0,float(etas[-1]),8193)'),
        ('            if len(js)<3:continue','            if len(js)<1:continue'),
    ]
    before="        groups=defaultdict(list)"
    after=("        known={(j['model'],j['mode'],j['eta'],j['alpha']) for j in allj}\n"
        "        for past in r['all_configurations'][model]:\n"
        "            if past.get('origin')!='prior_M_radius1' or past['mode'] not in ('state_only','pre_final_block'):continue\n"
        "            eta=past['parameters']['eta']\n"
        "            if eta not in (.1,.5,1.):continue\n"
        "            key=(model,past['mode'],eta,1.)\n"
        "            if key in known:continue\n"
        "            s=c.setting(model,past['mode'],eta,1.);s['source']=past;allj.append(s);known.add(key)\n"+before)
    changes.append((before,after))
    before="        proposals.sort(key=lambda x:(-x['proposal_only_counts'][0],-x['proposal_only_counts'][1],-x['proposal_width'],x['config_id']))"
    after=("        if model=='videomaev2':\n"
        "            previous={(j['eta'],j['alpha']) for j in allj if j['model']==model and j['mode']=='output_only'}\n"
        "            proposals.extend(_n_mse_proposals(best_actual,previous))\n"+before)
    changes.append((before,after))
    changed=source
    for before,after in changes:
        assert changed.count(before)==1;changed=changed.replace(before,after,1)
    ns=dict(vars(refine),_n_mse_proposals=generate);exec(compile(changed,'<N_bounded_proposal_runtime_003>','exec'),ns);refine.main=ns['main']
    evidence=dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        description='Use all preexisting M same-position eta.1/.5/1 knots, permit linear two-knot proposals, no eta extrapolation; MSE global eta/alpha proposals. Same Stage1, max3 actual Stage2/model.')
    p=c.HERE/'PROPOSAL_TRANSFORM_003.json'
    if p.exists():assert c.read(p)==evidence
    else:c.save(p,evidence)
