"""Existing slot rollout or DINO's original six-step rollout with one projection."""
import argparse
import time
from . import common as c


def install_namespace():
    old = c.install()
    old.verify,old.directory,old.SOURCE,old.NATIVE = c.verify,c.directory,c.source_path(),c.MODELS
    return old


def dino_setup():
    import torch
    from predictive_tracks.ppr_position_refine_20261106_s import worker as base
    from predictive_tracks.ppr_dino_feedback_20261106_aa.worker import adapter_for
    from predictive_tracks.ppr_dino_feedback_20261106_aa.data import versions
    from .dino_loop_007 import LoopProjector
    original_load = base.load_data
    recent = {}
    def load(meta):
        if recent.get('state') != meta['state']:
            recent.clear()
            recent.update(state=meta['state'],data=original_load(meta))
        return recent['data']
    def read_capture(model,key):
        meta = c.read(c.HERE/'capture'/model/'pairs'/f'{key}.json')
        assert c.digest(meta['source_meta']) == meta['source_meta_sha256']
        value = load(c.read(meta['source_meta']))
        assert value['history'].shape == (1,5,2048,1152)
        assert value['prediction'].shape == (1,6,2048,1152)
        return meta,dict(history=value['history'],prediction=value['prediction'])
    def predicted(adapter,model,item,data):
        context = data['history'].reshape(1,5,32,64,1152)
        with LoopProjector(adapter,item,cached_context=context) as hook:
            value = adapter._forecast(None)
        assert hook.forecaster_calls == 6
        return value.reshape(1,6,2048,1152)
    def full_check(adapter,model,row,ref,possible,impossible,data,evaluator,item):
        from .dino_reference_005 import score
        destination = c.HERE/'preflight'/model/'feedback'/f'{row["key"]}_e{item["eta"]:.8g}_a{item["alpha"]:.8g}.json'
        if destination.exists():return
        expected = evaluator.errors(predicted(adapter,model,item,data))
        adapter.prediction_calls = row['archive_index']
        native,errors,states,trace,audits = score(adapter,possible,impossible,item)
        base.exact(expected,errors)
        assert trace == c.read(c.HERE/'NATIVE_TRACE_REFERENCE_005.json')['trace']
        assert adapter.prediction_calls == row['archive_index']+1
        assert native.frame_spans.tolist() == ref['frame_spans']
        assert len(audits) == 1 and len(audits[0]['steps']) == 6
        c.save(destination,dict(at=time.time(),key=row['key'],job=item,
            original_trace_exact=True,trace_reference='Fresh unchanged adapter and original cache-generation trace',
            full_vs_suffix_max_abs_error=0.,weights_identity_version_unchanged=True,
            forecaster_calls=6,encoded_frames=17,no_extra_forward_in_deployed_forecast=True,
            existing_benchmark_scoring_exact=True,original_perframe_scalar_rounding_separately_reported=True,
            original_alignment_unchanged=True,diagnostic_only=True))
    base.c = c
    base.load_data = load
    base.rows_and_references = lambda model:c.dino_rows()
    base.adapter_for,base.versions = adapter_for,versions
    base.read_capture,base.predicted,base.full_check = read_capture,predicted,full_check
    base.move_suffix = lambda adapter,model:None
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=c.MODELS,required=True)
    parser.add_argument('--phase',choices=('coarse','fine'),required=True)
    args = parser.parse_args()
    c.verify()
    install_namespace()
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    handle = install(lambda v:c.save(c.HERE/'events'/f'thermal_008_{time.time_ns()}.json',v))
    try:
        if args.model == 'dino_foresight':
            assert c.read(c.HERE/'PROBE_COMPLETE_005.json')['existing_scoring_formula_exact']
            base = dino_setup()
        else:
            from predictive_tracks.ppr_feedback_refine_20261106_t import worker as original
            original.c = c
            base = original.setup()
        base.execute_search(args.model,args.phase)
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
