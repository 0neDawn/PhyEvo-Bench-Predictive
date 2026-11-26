"""Capture already-computed states and evaluate pure-tensor controls; zero forwards.

This cache is development-only. It must never feed candidate states back into a
correction function. Correction APIs take only native history and predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F


class NativeCache:
    def __init__(self, adapter, model):
        self.adapter, self.model = adapter, model
        self.data, self.targets, self.restore, self.handles = {}, [], [], []

    @staticmethod
    def cpu(value):
        return value.detach().to("cpu").clone()

    def wrap(self, name, make):
        original = getattr(self.adapter, name)
        self.restore.append((name, original))
        setattr(self.adapter, name, make(original))

    def __enter__(self):
        def correction(original):
            def call(prediction, history, **kwargs):
                self.data['prediction'] = self.cpu(prediction)
                self.data['history'] = self.cpu(history)
                result = original(prediction, history, **kwargs)
                self.data['corrected_prediction'] = self.cpu(result)
                return result
            return call
        self.wrap('_correct', correction)
        if self.model == 'slotformer':
            def slots(module, inputs, output):
                states = output['post_slots']
                if states.shape[1] == 14:
                    self.targets.append(self.cpu(states[:,8:]))
            self.handles.append(self.adapter.savi.register_forward_hook(slots))
        elif self.model in ('vjepa2_vitg','vjepa2_vitl','videomaev2'):
            def causal(original):
                def call(*args, **kwargs):
                    out = original(*args, **kwargs)
                    values = out if isinstance(out, tuple) else (out,)
                    self.targets.extend(self.cpu(v) for v in values)
                    return out
                return call
            self.wrap('_causal_targets', causal)
        elif self.model == 'gswm':
            def forecast(original):
                def call(*args, **kwargs):
                    out = original(*args, **kwargs)
                    self.data['alpha'] = torch.from_numpy(out.alpha.copy())
                    return out
                return call
            self.wrap('_predict', forecast)
        elif self.model == 'dino_foresight':
            def encode(original):
                def call(frames):
                    out = original(frames)
                    if len(frames) == 1:
                        self.targets.append(self.cpu(out))
                    return out
                return call
            self.wrap('_encode_frames', encode)
        return self

    def __exit__(self, *exc):
        for handle in self.handles: handle.remove()
        for name, original in reversed(self.restore): setattr(self.adapter,name,original)

    def finish_targets(self, possible, impossible):
        # Invoked only AFTER the sole native prediction and all native scoring.
        if self.model in ('ocvp_seq','savi_transformer'):
            from sp_ocvp.common import _official_rgb_tensor
            self.targets = [_official_rgb_tensor(v[8:]) for v in (possible, impossible)]
        elif self.model == 'ddlp':
            from sp_core.common import video_tensor
            self.targets = [video_tensor(v,128)[:,8:] for v in (possible, impossible)]
        elif self.model == 'gswm':
            from gswm_runtime.runtime import _as_tchw, FUTURE_STEPS
            self.targets = [_as_tchw(v[8:],FUTURE_STEPS).unsqueeze(0) for v in (possible, impossible)]
        elif self.model == 'dino_foresight':
            if len(self.targets) != 12: raise RuntimeError('Expected twelve existing single-frame target encodings')
            self.targets = [torch.cat(self.targets[start:start+6]).reshape(1,6,-1,self.targets[0].shape[-1])
                            for start in (0,6)]
        if len(self.targets) != 2 or 'prediction' not in self.data:
            raise RuntimeError('Native states were not all captured')
        self.data['possible_target'],self.data['impossible_target']=self.targets

    def errors(self, prediction):
        if self.model == 'gswm':
            from sp_gswm.adapters.gswm_foreground_predictive_mean import ForegroundPrediction
            from sp_gswm.adapters.gswm_foreground_predictive_mean_robust import Adapter
            native=ForegroundPrediction(rgb=prediction[0].numpy(),alpha=self.data['alpha'].numpy())
            # Candidate argument is TCHW here; _as_tchw preserves this native view.
            return [Adapter._foreground_mse(native, target[0].numpy()) for target in self.targets]
        value=prediction.to(self.adapter.device)
        results=[]
        for target in self.targets:
            target=target.to(self.adapter.device)
            if self.model in ('ddlp','vjepa2_vitg','vjepa2_vitl'):
                diff=(value-target).abs()
            elif self.model == 'dino_foresight':
                diff=F.smooth_l1_loss(value,target,reduction='none',beta=.1)
            else:
                diff=(value-target).square()
            dims=(0,)+tuple(range(2,diff.ndim))
            results.append(diff.mean(dim=dims).cpu().numpy())
        return results

    def audit_and_controls(self, native_result):
        with torch.inference_mode():
            replay=self.errors(self.data['corrected_prediction'])
            for now,wanted in zip(replay,(native_result.possible,native_result.impossible)):
                if not np.allclose(now,wanted,rtol=1e-5,atol=1e-7):
                    raise RuntimeError('Cached native-score replay does not match original scoring')
            raw=self.errors(self.data['prediction'])
            held=self.data['history'][:,-1:].expand_as(self.data['prediction'])
            persistence=self.errors(held)
        return {
            'corrected_score_replay_max_abs_delta':float(max(np.max(np.abs(a-b)) for a,b in zip(replay,(native_result.possible,native_result.impossible)))),
            'native_prediction_possible_errors':raw[0].tolist(),
            'native_prediction_impossible_errors':raw[1].tolist(),
            'persistence_possible_errors':persistence[0].tolist(),
            'persistence_impossible_errors':persistence[1].tolist(),
            'additional_neural_forwards':0,
            'controls_computed_after_original_scoring':True,
        }
