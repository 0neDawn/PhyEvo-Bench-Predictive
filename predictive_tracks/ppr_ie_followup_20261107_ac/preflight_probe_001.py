"""CPU tensor fixture only, never a pretrained/model forward or performance test."""
import os
import time
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from predictive_tracks.ppr_dino_feedback_20261106_aa.hooks import LoopProjector
    torch.set_num_threads(1)
    torch.manual_seed(7)
    NATIVE_CONTEXT = 5
    FORECAST_FRAMES = 6
    class FixtureForecaster:
        def forward(self, context, masked, mask):
            return None, context+.25
    class FixtureAdapter:
        def __init__(self):
            self.model = FixtureForecaster()
        def _encode_frames(self, history):
            return history
        def _forecast(self, history):
            context = self._encode_frames(history[-NATIVE_CONTEXT:]).unsqueeze(0)
            predictions = []
            for _ in range(FORECAST_FRAMES):
                _, output = self.model.forward(context, context, None)
                next_feature = output[:, -1].float()
                predictions.append(next_feature.squeeze(0))
                context = torch.cat((context[:,1:],next_feature.unsqueeze(1)),dim=1)
            return torch.stack(predictions)
    # instrument() intentionally uses the defining function's global namespace.
    FixtureAdapter._forecast.__globals__.update(NATIVE_CONTEXT=NATIVE_CONTEXT,FORECAST_FRAMES=FORECAST_FRAMES)
    history = torch.randn(5,32,64,4)
    a = FixtureAdapter()
    reference = a._forecast(history)
    no_module_calls = [0]
    def forbid(*args,**kwargs):
        no_module_calls[0] += 1
        raise RuntimeError('Neural module forbidden in CPU fixture')
    handle = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        for eta in (0.,.1,1.):
            with LoopProjector(a,dict(eta=eta,alpha=1.,mode='feedback')) as hook:
                actual = a._forecast(history)
            assert hook.forecaster_calls == 6 and len(hook.audits[0]['steps']) == 6
            assert torch.equal(hook.history,history.reshape(1,5,2048,4))
            if eta == 0:
                assert torch.equal(reference,actual)
            assert torch.isfinite(actual).all()
    finally:
        handle.remove()
    assert no_module_calls[0] == 0 and not torch.cuda.is_initialized()
    c.save(c.HERE/'PREFLIGHT_PROBE_001.json',dict(at=time.time(),CPU_tensor_fixture=True,
        no_model_module_or_CUDA_calls=True,original_six_iteration_route=True,history_unchanged=True,
        eta0_exact=True,not_real_model_budget_or_performance_validation=True))
    print('CPU DINO-loop fixture passed; real unchanged-baseline checks still required',flush=True)


if __name__ == '__main__':
    main()
