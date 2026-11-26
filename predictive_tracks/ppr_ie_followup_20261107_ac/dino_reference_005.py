"""Use the benchmark's existing cached scoring expression, with fresh native states."""
import contextlib
import time
from . import common as c


def capture_type():
    from predictive_tracks.ppr_dino_feedback_20261106_aa.hooks import NativeCapture
    class Capture(NativeCapture):
        def __init__(self, adapter):
            super().__init__(adapter)
            self.targets = []
        def __enter__(self):
            super().__enter__()
            original = self.adapter._encode_frames
            def observed(frames):
                value = original(frames)
                if len(frames) == 1:
                    self.targets.append(value.detach().cpu().clone())
                return value
            self.replace(self.adapter,'_encode_frames',observed)
            return self
    return Capture


def benchmark_errors(prediction, data):
    import torch
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import evaluator_for
    evaluator = evaluator_for(data,'dino_foresight')
    evaluator.adapter.device = torch.device('cuda:0')
    evaluator.targets = [v.to('cuda:0') for v in evaluator.targets]
    return evaluator.errors(prediction)


def score(adapter, possible, impossible, item=None):
    import torch
    from .dino_loop_007 import LoopProjector
    from predictive_tracks.ppr_dino_feedback_20261106_aa.data import versions
    from predictive_tracks.ppr_position_refine_20261106_s.worker import trace_counter
    before = versions(adapter,'dino_foresight')
    hook = LoopProjector(adapter,item) if item is not None else contextlib.nullcontext()
    count,trace,handle = trace_counter()
    try:
        with hook:
            with capture_type()(adapter) as capture:
                native_result = adapter.score_pair(possible,impossible)
    finally:
        handle.remove()
    assert capture.forecaster_calls == 6 and capture.encoded_frame_requests == [5]+[1]*12
    assert versions(adapter,'dino_foresight') == before
    assert len(capture.targets) == 12
    data = dict(history=capture.context.reshape(1,5,2048,1152),
        prediction=capture.prediction.reshape(1,6,2048,1152),
        possible_target=torch.cat(capture.targets[:6]).reshape(1,6,2048,1152),
        impossible_target=torch.cat(capture.targets[6:]).reshape(1,6,2048,1152))
    errors = benchmark_errors(data['prediction'],data)
    measured = dict(module_forward_invocations=count[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
    return native_result,errors,data,measured,hook.audits if item is not None else []


def worker():
    import sys
    import hashlib
    import numpy as np
    import torch
    from pathlib import Path
    c.verify_sources('PROBE_READY_005.json')
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.ppr_dino_feedback_20261106_aa.worker import adapter_for
    from predictive_tracks.ppr_position_refine_20261106_s.worker import exact,load_data
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    from sp_core.common import PairRecord,read_pair_rgb
    thermal = install(lambda v:c.save(c.HERE/'events'/f'thermal_probe005_{time.time_ns()}.json',v))
    adapter = adapter_for('dino_foresight')
    selected,seen = [],set()
    for item in c.dino_rows():
        group = item[0]['benchmark'],item[0]['category_directory']
        if group not in seen:
            seen.add(group);selected.append(item)
    assert len(selected) == 6
    traces = []
    try:
        # All six unchanged controls finish before the first nonzero intervention.
        for row,mp,meta in selected:
            destination = c.HERE/'baseline_controls_005'/f'{row["key"]}.json'
            if destination.exists():
                traces.append(c.read(destination)['trace']);continue
            c.require_resources('dino_foresight')
            cached = load_data(meta)
            pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible,impossible = read_pair_rgb(pair)
            pixels = hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
            assert pixels == meta['source_decoded_pixels_sha256']
            adapter.prediction_calls = row['archive_index']
            native,errors,actual,trace,_ = score(adapter,possible,impossible)
            for key in ('history','prediction','possible_target','impossible_target'):
                assert actual[key].dtype == cached[key].dtype and torch.equal(actual[key],cached[key]),key
            exact(errors,[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
            assert adapter.prediction_calls == row['archive_index']+1 and native.frame_spans.tolist() == meta['frame_spans']
            if meta.get('neural_call_trace'):
                assert trace == meta['neural_call_trace']
            difference = max(float(np.max(np.abs(v-np.asarray(w)))) for v,w in zip((native.possible,native.impossible),errors))
            c.save(destination,dict(at=time.time(),key=row['key'],trace=trace,reference_kind='Fresh unchanged adapter and existing cache-generation graph',
                source_meta=str(mp),source_meta_sha256=c.digest(mp),source_state_sha256=meta['sha256'],
                raw_history_prediction_and_both_targets_bitwise_exact=True,existing_benchmark_errors_exact=True,
                existing_cache_scoring_source='predictive_tracks/controlled_tracks/native_cache.py:NativeCache.errors',
                original_adapter_possible_errors=native.possible.tolist(),original_adapter_impossible_errors=native.impossible.tolist(),
                possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist(),
                adapter_vs_existing_cache_reduction_max_abs_difference=difference,
                archival_original_trace_match_preserved=meta.get('original_trace_match'),
                original_alignment_unchanged=True,forecaster_calls=6,encoded_frames=17,diagnostic_only=True))
            traces.append(trace)
            print('DINO original states/cache formula exact',row['key'],'native scalar difference',difference,flush=True)
            del actual,cached,possible,impossible
        assert len({__import__('json').dumps(v,sort_keys=True) for v in traces}) == 1
        reference = c.HERE/'NATIVE_TRACE_REFERENCE_005.json'
        if not reference.exists():
            c.save(reference,dict(at=time.time(),trace=traces[0],controls=6,measured_before_nonzero_eta=True,
                historical_original_trace_flag_not_relabelled=True,diagnostic_only=True,
                files={str(p):c.digest(p) for p in (c.HERE/'baseline_controls_005').glob('*.json')}))
        for row,mp,meta in selected:
            pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible,impossible = read_pair_rgb(pair)
            for eta in (0.,.1):
                destination = c.HERE/'probe_corrected_005'/f'{row["key"]}_e{eta:.8g}.json'
                if destination.exists():continue
                c.require_resources('dino_foresight')
                adapter.prediction_calls = row['archive_index']
                native,errors,actual,trace,audits = score(adapter,possible,impossible,dict(model='dino_foresight',mode='feedback',eta=eta,alpha=1.))
                assert trace == traces[0] and adapter.prediction_calls == row['archive_index']+1
                if eta == 0:
                    exact(errors,[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
                assert len(audits) == 1 and len(audits[0]['steps']) == 6
                c.save(destination,dict(at=time.time(),key=row['key'],eta=eta,alpha=1.,trace_matches_unchanged_control=True,
                    forecaster_calls=6,encoded_frames=17,no_extra_forward_per_inference=True,
                    possible_errors=np.asarray(errors[0]).tolist(),impossible_errors=np.asarray(errors[1]).tolist(),
                    original_adapter_possible_errors=native.possible.tolist(),original_adapter_impossible_errors=native.impossible.tolist(),
                    projection_audits=audits,existing_scoring_expression_unchanged=True,diagnostic_only=True))
                print('DINO original-loop budget check',row['key'],eta,'passed',flush=True)
                del actual
        c.save(c.HERE/'PROBE_COMPLETE_005.json',dict(at=time.time(),unmodified_controls=6,loop_checks=12,
            all_four_cached_tensors_exact=True,existing_scoring_formula_exact=True,original_forecaster6_and_encoder17_unchanged=True,
            old_adapter_scalar_rounding_difference_reported_not_hidden=True,alignment_unchanged=True,diagnostic_only=True,
            no_full300_performance_claim=True))
    finally:
        thermal.remove()
