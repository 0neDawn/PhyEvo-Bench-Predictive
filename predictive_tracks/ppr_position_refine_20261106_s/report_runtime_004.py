"""Initialize per-model report index; no metric or selection rule changes."""
import inspect,hashlib
from . import common as c

def install():
    from . import report as original
    plan=c.read(c.HERE/'REPORT_INDEX_ADDENDUM_004.json')
    assert hashlib.sha256(inspect.getsource(original.assemble).encode()).hexdigest()==plan['original_function_sha256']
    for path,wanted in plan['files'].items():assert c.digest(path)==wanted,path
    namespace=dict(vars(original))
    exec(compile((c.HERE/'report_assemble_source_004.txt').read_text(),'<S_report_index_004>','exec'),namespace)
    original.assemble=namespace['assemble']
