"""Original radius and original radial operator; only existing constants vary."""
import hashlib
import torch
from .common import qualities,quality_id
from predictive_tracks.ppr_eta_20261030.hooks import SoftDDLP,SoftGSWM

def reference(features,presence,mass,c):
    if 'alpha' in c:raise ValueError('No radius multiplier is accepted')
    if features.ndim!=3 or presence.shape!=features.shape[:2] or mass.shape!=presence.shape:raise ValueError('Original history layout required')
    if not all(torch.isfinite(x).all() for x in (features,presence,mass)):raise ValueError('Nonfinite history')
    keep=(presence>=c['presence'])&(mass>=c['mass'])&(mass>=c['q']*mass.amax(0,keepdim=True))
    counts=keep.sum(0);center=(features*keep[...,None]).sum(0)/counts.clamp_min(1)[...,None]
    distances=torch.linalg.vector_norm(features-center,dim=-1)
    radius=torch.where(keep,distances,torch.zeros_like(distances)).amax(0)
    return center,radius,counts>=c['min_history'],counts

def tensor_digest(*values):
    h=hashlib.sha256()
    for x in values:
        a=x.detach().cpu().contiguous().numpy()
        h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()

def operator_digest(center,radius,valid,ids):
    # Invalid references are exact no-ops and their otherwise unused c/R do not matter.
    c=torch.where(valid[...,None],center,torch.zeros_like(center));r=torch.where(valid,radius,torch.zeros_like(radius))
    return tensor_digest(ids,c,r,valid)

class ExistingBase:
    def __init__(self,*args,params,capture_all=False,**kwargs):
        if set(params)!=set(('eta','q','presence','mass','min_history')):raise ValueError('Only existing parameters are accepted')
        self.params=params;self.capture_all=capture_all;self.history_signatures=[]
        super().__init__(*args,eta=params['eta'],**kwargs)

    def establish(self,h,p,mass,ids):
        self.center,self.radius,self.valid,counts=reference(h,p,mass,self.params)
        self.new_audit(counts);self.current['parameters']=self.params
        selected=operator_digest(self.center,self.radius,self.valid,ids)
        signatures={quality_id(self.params):selected}
        if self.capture_all:
            for c in qualities('ddlp'):
                center,radius,valid,_=reference(h,p,mass,c)
                signatures[quality_id(c)]=operator_digest(center,radius,valid,ids)
        self.history_signatures.append({'raw_history_sha256':tensor_digest(ids,h,p,mass),'operators':signatures,
            'selected_operator_sha256':selected})

class ExistingDDLP(ExistingBase,SoftDDLP):
    def begin(self,features):
        if features.shape[0]!=1 or features.shape[1]!=8:raise RuntimeError('Original DDLP history required')
        h=features[0]
        self.establish(h,self.presence.reshape(h.shape[:2]),self.mass.reshape(h.shape[:2]),torch.arange(h.shape[1],device=h.device))

class ExistingGSWM(ExistingBase,SoftGSWM):
    def freeze_history(self):
        unique=sorted({int(i) for ids,*_ in self.history for i in ids.tolist() if i>=0})
        self.lookup={identifier:col for col,identifier in enumerate(unique)}
        prototype=self.history[0][1];h=prototype.new_zeros((8,len(unique),prototype.shape[-1]))
        p=prototype.new_zeros((8,len(unique)));mass=torch.zeros_like(p)
        for t,(ids,features,pres,m) in enumerate(self.history):
            seen=set()
            for j,identifier in enumerate(ids.tolist()):
                if identifier<0:continue
                if identifier in seen:raise RuntimeError('Duplicate native ID')
                seen.add(identifier);k=self.lookup[identifier];h[t,k],p[t,k],mass[t,k]=features[j],pres[j],m[j]
        self.establish(h,p,mass,torch.tensor(unique,device=h.device,dtype=torch.int64));self.current['object_ids']=unique
