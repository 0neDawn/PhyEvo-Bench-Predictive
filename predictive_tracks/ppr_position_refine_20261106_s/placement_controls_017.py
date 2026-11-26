"""Summarize existing default placement controls; no forecasts or new choices."""
import time
from . import common as c


def main():
    path=c.HERE/'grids/coarse_videomaev2.json'
    jobs=c.read(path)['jobs']
    rows=[]
    hashes={str(path):c.digest(path)}
    for job in jobs:
        if job['eta']!=1 or job['alpha']!=1:
            continue
        summary=c.HERE/'search_settings/videomaev2'/job['config_id']/'summary.json'
        value=c.read(summary)
        assert value['complete'] and value['cases']==600
        hashes[str(summary)]=c.digest(summary)
        rows.append(dict(job=job,metrics=value['metrics'],
            full600_suffix_search_complete=True,
            full600_native_confirmation=(c.directory(job)/'summary.json').exists()))
    assert {x['job']['position'] for x in rows}=={'block_0','block_1','block_2','before_norm'}
    result=dict(at=time.time(),rows=rows,source_hashes=hashes,
        interpretation='Unit strength and unit historical radius over-correct these feature-layer states. Positive tuned results are not parameter-free.',
        original_history_and_formula_unchanged=True,no_new_model_calls=True,
        no_new_search_or_selection=True,same_exposed_data_exploratory=True,
        caveat='Past and forecast hidden tokens have different observation/mask roles; these states are proxies, not isolated intrinsic attributes.')
    out=c.HERE/'placement_controls_017'/f'{time.time_ns()}.json'
    c.save(out,result)
    print(out,flush=True)


if __name__=='__main__':main()
