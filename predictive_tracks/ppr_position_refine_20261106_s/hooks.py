"""One placement per forecast; exactly the original radial projection."""
from predictive_tracks.ppr_eta_radius_20261104_n.hooks import TunedPreFinal, TunedStateOnly
from predictive_tracks.ppr_eta_radius_20261104_o.hooks import factory as original_factory


class LayerShift(TunedPreFinal):
    def __init__(self, adapter, item, snapshot=None):
        super().__init__(adapter, item['model'], item['eta'], item['alpha'], item['mode'], snapshot)
        self.position = item['position']

    def shift(self, values):
        output = super().shift(values)
        self.audits[-1]['position'] = self.position
        self.audits[-1]['placement_only_variant'] = True
        return output

    def __enter__(self):
        if self.model == 'videomaev2':
            owner, name = self.adapter.model.decoder, '_prediction'
            modules, norm = owner.blocks, owner.norm
        else:
            owner, name = self.adapter.predictor, '_predict'
            modules, norm = owner.predictor_blocks, owner.predictor_norm
        block = norm if self.position == 'before_norm' else modules[int(self.position.split('_')[-1])]
        self.name = name
        self.handle = block.register_forward_pre_hook(self.before)
        self.original = getattr(self.adapter, name)
        def predict_only(*args, **kwargs):
            if self.active:
                raise RuntimeError('Nested forecast')
            self.active = True
            try:
                result = self.original(*args, **kwargs)
                from predictive_tracks.ppr_state_insert_20261104_m.hooks import tensor_hash
                self.output_hash = tensor_hash(result)
                return result
            finally:
                self.active = False
        setattr(self.adapter, name, predict_only)
        return self


def factory(adapter, item, snapshot):
    if item.get('position'):
        return LayerShift(adapter, item, snapshot)
    if item['mode'] == 'state_only':
        return TunedStateOnly(adapter, item['model'], item['eta'], item['alpha'], item['mode'], snapshot)
    return original_factory(adapter, item, snapshot)
