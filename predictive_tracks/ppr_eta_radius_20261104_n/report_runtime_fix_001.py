"""Reporting-only fix; frozen sources and all numeric results remain untouched."""
import hashlib
import inspect
from . import common as c

def install():
    from . import report
    source=inspect.getsource(report.assemble)
    before="            for k in ENDPOINTS:\n"
    after="            r['metric_specific_maxima'][t][model]={}\n"+before
    assert source.count(before)==1
    changed=source.replace(before,after,1)
    ns=dict(vars(report));exec(compile(changed,'<N_report_dict_init_fix_001>','exec'),ns)
    report.assemble=ns['assemble']
    evidence=dict(kind='Report dictionary initialization only; no prediction/error/metric/selection change',
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),transformed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        frozen_report_file_sha256=c.digest(c.HERE/'report.py'))
    p=c.HERE/'REPORT_FIX_TRANSFORM_001.json'
    if p.exists():assert c.read(p)==evidence
    else:c.save(p,evidence)
    return report
