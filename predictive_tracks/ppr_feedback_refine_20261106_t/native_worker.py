"""Run selected settings through the unmodified original full adapter."""
import argparse
import time
from . import common as c


def main():
    import sys
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from .worker import adapter_for
    from .hooks import factory
    from predictive_tracks.ppr_eta_radius_20261104_o import native_worker as original
    from predictive_tracks.ppr_20261028.run import rows_and_references
    from predictive_tracks.controlled_tracks.registry import OUTPUT
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=c.SLOTS+c.FEATURES)
    args=parser.parse_args()
    c.install()
    c.verify()
    c.require_resources(args.model)
    jobs=[j for j in c.read(c.HERE/'grids/validation.json')['jobs'] if j['model']==args.model]
    done=c.HERE/'models_complete'/f'validation_{args.model}.json'
    if done.exists():
        return
    if not jobs:
        c.save(done,dict(at=time.time(),settings=0))
        return
    adapter=adapter_for(args.model)
    baseline=original.versions(adapter,args.model)
    data=sorted(rows_and_references(args.model),key=lambda x:(x[0]['benchmark']!='ie',x[0]['archive_index']))
    traces={json.dumps(x['neural_trace'],sort_keys=True) for x in c.read(OUTPUT/'deployment_audit_001.json')['details'][args.model]}
    shared=json.loads(next(iter(traces))) if len(traces)==1 else None
    original.factory=factory
    original.audit_key=lambda j:(j.get('position') or j['mode'])+'_native_budget'
    for item in jobs:
        original.run(adapter,item,data,baseline,shared,None,False)
    c.verify()
    c.save(done,dict(at=time.time(),settings=len(jobs),cases_each=600,all_original_traces_matched=True))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
        pause=isinstance(error,ResourceBusy)
        c.save(c.HERE/'events'/f'{"pause" if pause else "error"}_{time.time_ns()}.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        if pause:
            raise SystemExit(75)
        raise
