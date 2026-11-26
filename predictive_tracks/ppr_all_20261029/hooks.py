import torch
from predictive_tracks.ppr_20261028.hooks import Hooks
from .projection import reference, step

class SlotHooks(Hooks):
    def __init__(self, adapter, mode, model, snapshot_dir=None):
        super().__init__(adapter, mode, snapshot_dir)
        self.model, self.handles = model, []
        self.state = None

    def begin(self, h):
        if h.shape[1] != 8 or self.state is not None:
            raise RuntimeError('Expected exactly one eight-step historical reference')
        self.state = reference(h)
        self.current = {'history_steps': 8, 'native_units': h.shape[2], 'radii': self.state[1].cpu().tolist(),
                        'semantics': 'complete mixed slot, no geometry or appearance separation', 'steps': []}
        self.audits.append(self.current)

    def correct_slot(self, value):
        if self.state is None:
            raise RuntimeError('Projection before historical reference')
        value, audit = step(value, self.state)
        self.current['steps'].append(audit)
        return value

    def output(self, module, inputs, output):
        if self.mode == 'output_only':
            return torch.stack([self.correct_slot(output[:, t]) for t in range(output.shape[1])], dim=1)
        return output

    def __enter__(self):
        if self.model in ('ocvp_seq', 'savi_transformer'):
            self.instrument(self.adapter, '_predict_decoded_rgb', [
                ('history_slots = torch.stack(history_slots, dim=1)',
                 'history_slots = torch.stack(history_slots, dim=1)\n    _ppr.begin(history_slots)')])
            self.instrument(self.adapter.predictor, 'forward_transformer', [
                ('next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds',
                 "if _ppr.mode == 'feedback':\n            cur_preds = _ppr.correct_slot(cur_preds)\n        next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds")])
            self.handles.append(self.adapter.predictor.register_forward_hook(self.output))
        elif self.model == 'slotformer':
            def history(module, inputs, output):
                if output['post_slots'].shape[1] == 8:
                    self.begin(output['post_slots'])
            def feedback(module, inputs, output):
                return self.correct_slot(output) if self.mode == 'feedback' else output
            self.handles.append(self.adapter.savi.register_forward_hook(history))
            self.handles.append(self.adapter.dynamics.rollouter.out_proj.register_forward_hook(feedback))
            self.handles.append(self.adapter.dynamics.rollouter.register_forward_hook(self.output))
        else:
            raise ValueError(self.model)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        super().__exit__(*args)
