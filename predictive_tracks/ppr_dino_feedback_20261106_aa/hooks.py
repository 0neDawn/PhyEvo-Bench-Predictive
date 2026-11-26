"""One original state projection before the existing context append."""
import torch
from predictive_tracks.ppr_20261028.hooks import Hooks
from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
from predictive_tracks.ppr_state_insert_20261104_m.hooks import tensor_hash

class LoopProjector(Hooks):
    def __init__(self,adapter,item,snapshot=None,cached_context=None):
        super().__init__(adapter,'feedback',snapshot)
        self.item=item;self.cached_context=cached_context
        self.forecaster_calls=0;self.first_prediction_hash=None
    def begin(self,context):
        assert context.ndim==5 and context.shape[:4]==(1,5,32,64)
        self.history=context.detach().clone().reshape(1,5,2048,-1)
        self.current=dict(history_steps=5,original_context_not_a_tuned_window=True,
            encoded_history_frames=5,benchmark_history_available=8,eta=self.item['eta'],alpha=self.item['alpha'],steps=[])
        self.audits.append(self.current)
    def correct_frame(self,value):
        assert value.ndim==4 and value.shape[:3]==(1,32,64) and value.dtype==torch.float32
        if self.first_prediction_hash is None:self.first_prediction_hash=tensor_hash(value)
        original=value.reshape(1,1,2048,-1)
        hard,audit=pooled_trajectory(original,self.history,self.item['alpha'])
        eta=self.item['eta']
        changed=original if eta==0 else hard if eta==1 else original+eta*(hard-original)
        assert torch.isfinite(changed).all()
        self.current['steps'].append(dict(audit,actual_change_l2=float(torch.linalg.vector_norm(changed-original))))
        return value if eta==0 else changed[:,0].reshape_as(value)
    def __enter__(self):
        native=self.adapter.model.forward
        def counted(*args,**kwargs):
            self.forecaster_calls+=1
            return native(*args,**kwargs)
        self.replace(self.adapter.model,'forward',counted)
        start='context = self._encode_frames(history[-NATIVE_CONTEXT:]).unsqueeze(0)'
        replacement=start+'\n    _ppr.begin(context)'
        if self.cached_context is not None:
            replacement='context = _ppr.cached_context.clone()\n    _ppr.begin(context)'
        self.instrument(self.adapter,'_forecast',[(start,replacement),
            ('next_feature = output[:, -1].float()',
             'next_feature = output[:, -1].float()\n            next_feature = _ppr.correct_frame(next_feature)')])
        return self
    def __exit__(self,*args):
        super().__exit__(*args)
        if args[0] is None:
            assert self.forecaster_calls==6
            assert len(self.audits)==1 and len(self.audits[0]['steps'])==6

class NativeCapture(Hooks):
    """Observe the unchanged forecast and encodings without modifying tensors."""
    def __init__(self,adapter):
        super().__init__(adapter,'audit')
        self.context=None;self.prediction=None;self.forecaster_calls=0;self.encoded_frame_requests=[]
    def __enter__(self):
        encode=self.adapter._encode_frames
        def observed(frames):
            value=encode(frames);self.encoded_frame_requests.append(len(frames))
            if self.context is None:
                assert len(frames)==5
                self.context=value.detach().cpu().clone().unsqueeze(0)
            return value
        self.replace(self.adapter,'_encode_frames',observed)
        forecast=self.adapter._forecast
        def finished(*args,**kwargs):
            value=forecast(*args,**kwargs);self.prediction=value.detach().cpu().clone();return value
        self.replace(self.adapter,'_forecast',finished)
        native=self.adapter.model.forward
        def counted(*args,**kwargs):self.forecaster_calls+=1;return native(*args,**kwargs)
        self.replace(self.adapter.model,'forward',counted)
        return self

def factory(adapter,item,snapshot):return LoopProjector(adapter,item,snapshot)
