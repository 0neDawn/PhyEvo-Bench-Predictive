"""Identical old projection/hook positions, only eta and radius scale vary."""
import torch
from predictive_tracks.ppr_eta_20261030.hooks import SoftSlot
from predictive_tracks.ppr_state_insert_20261104_m.hooks import StateOnly,PreFinal,tensor_hash
from predictive_tracks.ppr_all_20261029.projection import reference,step

class ScaleReference:
    def begin(self,h):
        super().begin(h)
        c,r,v=self.state
        self.state=(c,r if self.alpha==1 else self.alpha*r,v)
        self.current.update(alpha=self.alpha,base_radii=r.cpu().tolist(),radii=self.state[1].cpu().tolist())

class TunedSlot(ScaleReference,SoftSlot):
    def __init__(self,adapter,model,eta,alpha,mode,snapshot_dir):
        super().__init__(adapter,mode,model,snapshot_dir,eta=eta)
        self.alpha=alpha;self.first_prediction_hash=None
    def __enter__(self):
        super().__enter__()
        def capture(module,inputs,output):
            if self.first_prediction_hash is None:self.first_prediction_hash=tensor_hash(output[:,-1])
        self.handles.append(self.adapter.predictor.predictor.register_forward_hook(capture))
        return self

class TunedStateOnly(ScaleReference,StateOnly):
    def __init__(self,adapter,model,eta,alpha,mode,snapshot_dir):
        super().__init__(adapter,model,eta,snapshot_dir);self.alpha=alpha

class TunedPreFinal(PreFinal):
    def __init__(self,adapter,model,eta,alpha,mode,snapshot_dir):
        super().__init__(adapter,model,eta,snapshot_dir);self.alpha=alpha
    def shift(self,values):
        if values.ndim!=3 or values.shape[:2]!=(1,7*256):raise RuntimeError('Original4context/3future geometry required')
        states=values.reshape(1,7,256,-1)
        history=states[:,:4].mean(2,keepdim=True);prediction=states[:,4:].mean(2,keepdim=True)
        c,r,v=reference(history);ref=(c,r if self.alpha==1 else self.alpha*r,v)
        fixed=[];audits=[]
        for t in range(3):
            raw=prediction[:,t];hard,a=step(raw,ref)
            after=raw if self.eta==0 else hard if self.eta==1 else raw+self.eta*(hard-raw)
            a.update(eta=self.eta,alpha=self.alpha,actual_change_l2=float(torch.linalg.vector_norm(after-raw)))
            fixed.append(after);audits.append(a)
        delta=torch.stack(fixed,1)-prediction
        out=values if self.eta==0 else torch.cat([states[:,:4],states[:,4:]+delta],1).reshape_as(values)
        if not torch.equal(out[:,:4*256],values[:,:4*256]) or not torch.isfinite(out).all():raise RuntimeError('Invalid state write')
        self.audits.append(dict(history_steps=4,history_frames=8,native_units=256,channels=values.shape[-1],
            position='before final original decoder block',radius_multiplier=self.alpha,base_radii=r.cpu().tolist(),radii=ref[1].cpu().tolist(),
            steps=audits,input_hash=tensor_hash(values),modified_hash=tensor_hash(out),context_unchanged=True,
            field_change_l2=float(torch.linalg.vector_norm(out-values))))
        return out

def factory(adapter,model,eta,snapshot_dir):
    from . import common as c
    j=c.CURRENT
    cls=TunedPreFinal if model=='videomaev2' else TunedStateOnly if j['mode']=='state_only' else TunedSlot
    return cls(adapter,model,eta,j['alpha'],j['mode'],snapshot_dir)
