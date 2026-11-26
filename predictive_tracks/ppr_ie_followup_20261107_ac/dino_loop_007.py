"""Keep the existing DINO CPU projection arithmetic inside its original loop."""
from predictive_tracks.ppr_dino_feedback_20261106_aa.hooks import LoopProjector as OriginalLoop


class LoopProjector(OriginalLoop):
    def begin(self, context):
        super().begin(context)
        self.cpu_history = self.history.detach().cpu()
        self.current['projection_device'] = 'CPU, identical to the existing DINO output correction'
        self.current['CPU_GPU_transfers_add_wall_time_not_model_forwards'] = True

    def correct_frame(self, value):
        import torch
        from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
        from predictive_tracks.ppr_state_insert_20261104_m.hooks import tensor_hash
        assert value.ndim == 4 and value.shape[:3] == (1,32,64) and value.dtype == torch.float32
        if self.first_prediction_hash is None:
            self.first_prediction_hash = tensor_hash(value)
        original = value.detach().cpu().reshape(1,1,2048,-1)
        hard,audit = pooled_trajectory(original,self.cpu_history,self.item['alpha'])
        eta = self.item['eta']
        corrected = original if eta == 0 else hard if eta == 1 else original+eta*(hard-original)
        assert torch.isfinite(corrected).all()
        self.current['steps'].append(dict(audit,actual_change_l2=float(torch.linalg.vector_norm(corrected-original))))
        return value if eta == 0 else corrected[:,0].reshape_as(value).to(value.device)


def main():
    import os,time,torch
    from . import common as c
    from predictive_tracks.ppr_alpha_20261031.projection import pooled_trajectory
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1);torch.manual_seed(9)
    history = torch.randn(1,5,2048,32).half()
    prediction = torch.randn(1,6,2048,32)+.2
    for alpha in (1.,2.,4.):
        hard,_ = pooled_trajectory(prediction,history,alpha)
        for eta in (0.,.1,1.):
            expected = prediction if eta == 0 else hard if eta == 1 else prediction+eta*(hard-prediction)
            fixture = object.__new__(LoopProjector)
            fixture.cpu_history = history
            fixture.first_prediction_hash = None
            fixture.item = dict(eta=eta,alpha=alpha)
            fixture.current = dict(steps=[])
            actual = torch.stack([fixture.correct_frame(prediction[:,t].reshape(1,32,64,32)).reshape(1,2048,32) for t in range(6)],dim=1)
            assert torch.equal(actual,expected),'CPU stepwise projection changed the existing arithmetic'
    assert not torch.cuda.is_initialized()
    c.save(c.HERE/'PREFLIGHT_LOOP_007.json',dict(at=time.time(),all_existing_CPU_projection_values_bitwise_exact=True,
        eta=[0.,.1,1.],alpha=[1.,2.,4.],no_CUDA_or_neural_forward=True,
        original_history_dtype='float16',original_prediction_dtype='float32'))
    print('Stepwise DINO CPU projection exactly matches the existing whole-output formula',flush=True)


if __name__ == '__main__':
    main()
