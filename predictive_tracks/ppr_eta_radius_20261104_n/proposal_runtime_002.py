"""Bound PCHIP to measured eta; supplement proposals using known MSE algebra."""
import hashlib
import inspect
from . import common as c

def install():
    from . import refine
    from .mse_proposals_002 import generate
    source=inspect.getsource(refine.main)
    before='            sample=np.linspace(0,1,8193)'
    after='            sample=np.linspace(0,float(etas[-1]),8193)'
    assert source.count(before)==1
    changed=source.replace(before,after,1)
    before="        proposals.sort(key=lambda x:(-x['proposal_only_counts'][0],-x['proposal_only_counts'][1],-x['proposal_width'],x['config_id']))"
    after=("        if model=='videomaev2':\n"
        "            previous={(j['eta'],j['alpha']) for j in allj if j['model']==model and j['mode']=='output_only'}\n"
        "            proposals.extend(_n_mse_proposals(best_actual,previous))\n"+before)
    assert changed.count(before)==1;changed=changed.replace(before,after,1)
    ns=dict(vars(refine),_n_mse_proposals=generate)
    exec(compile(changed,'<N_bounded_proposal_runtime_002>','exec'),ns);refine.main=ns['main']
    evidence=dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        description='Proposal-only changes. PCHIP restricted to observed eta; MSE ideal margins propose global eta/alpha. All Stage1 unchanged, max3 Stage2/model, actual full600 CUDA validation mandatory.')
    p=c.HERE/'PROPOSAL_TRANSFORM_002.json'
    if p.exists():assert c.read(p)==evidence
    else:c.save(p,evidence)
