"""Only new wiring and one fixed native-block hook. No old files are edited."""
import hashlib
import torch
from predictive_tracks.ppr_eta_20261030.hooks import SoftSlot
from predictive_tracks.ppr_all_20261029.projection import reference, step

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class StateOnly(SoftSlot):
    def __init__(self,adapter,model,eta,snapshot_dir):
        super().__init__(adapter,'feedback',model,snapshot_dir,eta=eta)
        self.mode='state_only'
        self.first_prediction_hash=None

    def __enter__(self):
        self.instrument(self.adapter,'_predict_decoded_rgb',[
            ('history_slots = torch.stack(history_slots, dim=1)',
             'history_slots = torch.stack(history_slots, dim=1)\n    _ppr.begin(history_slots)')])
        self.instrument(self.adapter.predictor,'forward_transformer',[
            ('next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds',
             "if self.teacher_force:\n            raise RuntimeError('Frozen inference must not teacher-force')\n        if t == 0:\n            _ppr.first_prediction_hash = _ppr.hash_tensor(cur_preds)\n        next_input = _ppr.correct_slot(cur_preds)")])
        return self

    @staticmethod
    def hash_tensor(x):return tensor_hash(x)

class PreFinal:
    def __init__(self,adapter,model,eta,snapshot_dir=None):
        self.adapter,self.model,self.eta=adapter,model,eta
        self.audits=[];self.functions={};self.active=False
        self.activations=0;self.inactive_calls=0;self.output_hash=None

    def shift(self,values):
        if values.ndim!=3 or values.shape[:2]!=(1,7*256):
            raise RuntimeError('Original four context / three future tubelets required')
        states=values.reshape(1,7,256,-1)
        history=states[:,:4].mean(2,keepdim=True)
        prediction=states[:,4:].mean(2,keepdim=True)
        ref=reference(history)
        fixed=[];audits=[]
        for t in range(3):
            raw=prediction[:,t]
            hard,a=step(raw,ref)
            after=raw if self.eta==0 else hard if self.eta==1 else raw+self.eta*(hard-raw)
            a.update(eta=self.eta,actual_change_l2=float(torch.linalg.vector_norm(after-raw)))
            fixed.append(after);audits.append(a)
        delta=torch.stack(fixed,1)-prediction
        # eta0 returns exactly the original object (no add-zero rounding).
        out=values if self.eta==0 else torch.cat([states[:,:4],states[:,4:]+delta],1).reshape_as(values)
        if not torch.equal(out[:,:4*256],values[:,:4*256]):raise RuntimeError('Historical tokens changed')
        if not torch.isfinite(out).all():raise RuntimeError('Nonfinite hidden state')
        audit=dict(history_steps=4,history_frames=8,native_units=256,channels=values.shape[-1],
                   position='before final original predictor/decoder block',radius_multiplier=1,
                   radii=ref[1].cpu().tolist(),steps=audits,
                   input_hash=tensor_hash(values),modified_hash=tensor_hash(out),
                   context_unchanged=True,field_change_l2=float(torch.linalg.vector_norm(out-values)))
        self.audits.append(audit)
        return out

    def before(self,module,inputs):
        if not self.active:
            self.inactive_calls+=1
            return None
        self.activations+=1
        return (self.shift(inputs[0]),*inputs[1:])

    def __enter__(self):
        if self.model=='vjepa2_vitl':
            block=self.adapter.predictor.predictor_blocks[-1];self.name='_predict'
        elif self.model=='videomaev2':
            block=self.adapter.model.decoder.blocks[-1];self.name='_prediction'
        else:raise ValueError(self.model)
        self.handle=block.register_forward_pre_hook(self.before)
        self.original=getattr(self.adapter,self.name)
        def prediction_only(*args,**kwargs):
            if self.active:raise RuntimeError('Nested prediction')
            self.active=True
            try:
                result=self.original(*args,**kwargs)
                self.output_hash=tensor_hash(result)
                return result
            finally:self.active=False
        setattr(self.adapter,self.name,prediction_only)
        return self

    def __exit__(self,*args):
        self.handle.remove();setattr(self.adapter,self.name,self.original)
        if self.active:raise RuntimeError('Hook leaked into candidate scoring')
