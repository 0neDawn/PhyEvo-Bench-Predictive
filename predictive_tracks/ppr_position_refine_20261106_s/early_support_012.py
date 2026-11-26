"""Select only after complete fine grids; immutable per-model validation IDs."""
import inspect,hashlib,sys
from . import common as c

def make_early():
    for model in c.SLOTS+('videomaev2',):
        assert (c.HERE/'models_complete'/f'fine_{model}.json').exists()
    from . import report
    plan=c.read(c.HERE/'EARLY_VALIDATION_PRIORITY_012.json')
    assert hashlib.sha256(inspect.getsource(report.make_validation).encode()).hexdigest()==plan['validation_maker_original_sha256']
    namespace=dict(vars(report))
    exec(compile((c.HERE/'early_validation_source_012.txt').read_text(),'<S_early_validation_selection_012>','exec'),namespace)
    namespace['make_validation']()

def dense_l():
    if (c.HERE/'grids/fine_vjepa2_vitl.json').exists():return
    from .global_dense_010 import main
    previous=sys.argv
    sys.argv=['fine','--model','vjepa2_vitl']
    try:main()
    finally:sys.argv=previous
