"""Reuse verified original historical slots; replay original future loop once."""
import argparse
import time
import traceback
from . import common as c


def setup():
    from predictive_tracks.ppr_position_refine_20261106_s import worker as base
    from .hooks import factory
    import torch
    base.c=c
    original_read=base.read_capture
    def read_capture(model,key):
        meta,data=original_read(model,key)
        native=base.load_data(c.read(meta['source_meta']))
        data['raw_RGB']=native['prediction']
        return meta,data
    base.read_capture=read_capture
    def predicted(adapter,model,item,data):
        hook=factory(adapter,item,None)
        with hook:
            hook.begin(data['history'])
            p=adapter.predictor(data['history'][:,-5:])
            out=adapter.savi.decode(p.reshape(6,6,128))[0].reshape(1,6,3,64,64).clamp(0.,1.)
        assert len(hook.audits)==1 and len(hook.audits[0]['steps'])==6
        if item.get('expected_noop'):
            assert torch.equal(out,data['raw_RGB']),'Predicted no-op radius control changed original output'
        return out
    base.predicted=predicted
    original_check=base.full_check
    def full_check(adapter,model,row,ref,possible,impossible,data,evaluator,item):
        # Alpha1 checks an active intervention rather than only rare high-alpha no-ops.
        for mode in ('feedback','state_only'):
            original_check(adapter,model,row,ref,possible,impossible,data,evaluator,
                dict(item,mode=mode,alpha=1.,expected_noop=False))
    base.full_check=full_check
    return base


def adapter_for(model):
    from predictive_tracks.ppr_position_refine_20261106_s.worker import adapter_for as original
    return original(model)


def expand(payload,item):
    from predictive_tracks.ppr_position_refine_20261106_s.worker import expand as original
    return original(payload,item)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=c.SLOTS,required=True)
    parser.add_argument('--phase',choices=('coarse','fine'),required=True);args=parser.parse_args()
    c.install();base=setup();base.execute_search(args.model,args.phase)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
        pause=isinstance(error,ResourceBusy)
        c.save(c.HERE/'events'/f'{"pause" if pause else "error"}_{time.time_ns()}.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        if pause:raise SystemExit(75)
        raise
