"""Compare original full-pipeline records to search errors as they arrive; NN0."""
import time
from pathlib import Path
from . import common as c


def main():
    grids=[p for p in (c.HERE/'grids/early_validation_012.json',c.HERE/'grids/validation.json') if p.exists()]
    if not grids:
        print(__import__('json').dumps(dict(at=time.time(),validation_grid_not_created=True,results=[])))
        return
    wanted={}
    for p in grids:
        for item in c.read(p)['jobs']:
            wanted[item['config_id']]=item
    cache_path=c.HERE/'monitor/native_comparison_013.json'
    cache=c.read(cache_path).get('records',{}) if cache_path.exists() else {}
    results=[]
    for identifier,item in wanted.items():
        search=c.read(c.HERE/'search_settings'/item['model']/identifier/'summary.json')
        root=c.directory(item)/'pairs'
        paths=sorted(root.glob('*.json'))
        for path in paths:
            sha=c.digest(path)
            if str(path) in cache:
                assert cache[str(path)]['native_sha256']==sha,path
                continue
            row=c.read(path)
            key=f'{row["benchmark"]}_{row["archive_index"]:03d}'
            assert path.name==key+'.json'
            source=Path(search['compact_case_source'])/path.name
            payload=c.read(source)
            expected=payload['settings'][identifier]
            assert row['possible_errors']==expected['possible_errors'] and row['impossible_errors']==expected['impossible_errors'],path
            assert row['frame_spans']==payload['native']['frame_spans'] and row['pair_id']==payload['native']['pair_id']
            assert row['job']==item and row['eta']==item['eta'] and row['radius_multiplier']==item['alpha']
            assert row['trace_matches_original'] and row['extra_forwards']==0 and row['weight_identity_and_version_unchanged']
            cache[str(path)]=dict(native_sha256=sha,source=str(source),source_sha256=c.digest(source))
        assert len(paths)<=600
        results.append(dict(model=item['model'],config_id=identifier,position=item.get('position'),verified=len(paths),total=600))
    out=dict(at=time.time(),records=cache,results=results,all_completed_records_exact=True)
    c.status(cache_path,out)
    c.save(c.HERE/'author_monitor'/f'native_comparison_{time.time_ns()}.json',out)
    print(__import__('json').dumps({k:v for k,v in out.items() if k!='records'}))


if __name__=='__main__':main()
