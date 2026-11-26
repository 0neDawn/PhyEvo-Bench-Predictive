"""Post-completion audit and portable support files; never alter scores."""
import time
from pathlib import Path
from . import common as c


def main():
    assert c.read(c.HERE/'FINAL_STATE_001.json')['all_lineage_audits_complete']
    assert c.read(c.HERE/'PERSISTED.json')['all_experiments_and_audits_complete']
    c.verify(full=True)
    for name in ('PRIORITY_ROUTING_002.json','INITIALIZATION_SCOPE_ADDENDUM_003.json','REPORT_INDEX_ADDENDUM_004.json'):
        for path,wanted in c.read(c.HERE/name)['files'].items():
            assert c.digest(path)==wanted,path
    for path in (c.HERE/'fine_declarations').glob('*.json'):
        value=c.read(path)
        assert c.digest(c.HERE/'global_fine_005.py')==value['source_sha256']
        model=path.name.removesuffix('_005.json')
        assert c.digest(c.HERE/'grids'/f'fine_{model}.json')==value['grid_sha256']
    assert c.read(c.HERE/'STAGE2.json')['cached']==c.read(c.HERE/'grids/fine_dino_foresight.json')['jobs']
    d=Path(c.read(c.HERE/'FINAL_DELIVERY.json')['directory'])
    checked={}
    for name in ('DELIVERY.json','SUPPLEMENT_MANIFEST_001.json','USER_SUMMARY_MANIFEST_001.json','METRIC_ALTERNATIVES_MANIFEST_001.json'):
        path=d/name
        for filename,sha in c.read(path)['files'].items():
            assert c.digest(filename)==sha,filename
        checked[str(path)]=c.digest(path)
    r=c.read(d/'report.json')
    parent=c.read(c.source_path())
    for model,rows in parent['all_configurations'].items():
        assert r['all_configurations'][model][:len(rows)]==rows
    from .report import jobs
    planned=jobs()
    assert len(planned)==len(r['current_round_settings'])==r['completed_round_configurations']
    assert {x['config_id'] for x in planned}=={x['config_id'] for x in r['current_round_settings']}
    assert r['completed_corrected_model_configs']==parent['completed_corrected_model_configs']+len(planned)
    required={x['config_id'] for x in c.read(c.HERE/'grids/validation.json')['jobs']}
    assert required=={x['config_id'] for x in r['current_round_settings'] if x.get('full600_native_validated')}
    assert all(x['complete'] and x['completed_pairs']==600 for x in r['current_round_settings'])
    for task in ('sp','ie'):
        assert len(r['selected'][task])==9
        for model,row in r['selected'][task].items():
            if row['config_id'].startswith('S_'):
                assert row['eligible_for_selection']
                if model!='dino_foresight':
                    assert row['full600_native_validated']
                assert any(x==row for x in r['all_configurations'][model])
    support=d/'audited_support_006'
    support.mkdir(exist_ok=False)
    paths=list(c.HERE.glob('*.txt'))+list(c.HERE.glob('REASONING_*.md'))
    paths+=list((c.HERE/'fine_declarations').glob('*.json'))+list((c.HERE/'preflight').glob('*/*/*.json'))
    for path in paths:
        dest=support/path.relative_to(c.HERE)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(path.read_bytes())
    value=dict(at=time.time(),configurations=len(planned),cumulative_versions=r['completed_corrected_model_configs'],
        independent_pairs=600,full600_native_configurations=len(required),verified_manifests=checked,
        all_parent_results_preserved=True,original_source_files_unchanged=True,
        original_distance_and_forward_budget_unchanged=True,one_actual_setting_per_row=True,
        same_exposed_data_exploratory=True,dino_diagnostic_only=True,
        support_files={str(p):c.digest(p) for p in support.rglob('*') if p.is_file()})
    c.save(d/'FINAL_REVIEW_006.json',value)
    c.save(d/'FINAL_REVIEW_MANIFEST_006.json',dict(at=time.time(),files={str(d/'FINAL_REVIEW_006.json'):c.digest(d/'FINAL_REVIEW_006.json')},
        source_reviewer_sha256=c.digest(__file__)))
    print({k:v for k,v in value.items() if k not in ('verified_manifests','support_files')},flush=True)


if __name__=='__main__':main()
