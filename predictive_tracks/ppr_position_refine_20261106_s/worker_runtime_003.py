"""Initialize models normally so the original weight-version audit remains active."""
import inspect,hashlib
from . import common as c

def main():
    from . import worker as original
    plan=c.read(c.HERE/'INITIALIZATION_SCOPE_ADDENDUM_003.json')
    assert hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest()==plan['original_function_sha256']
    for path,wanted in plan['files'].items():
        assert c.digest(path)==wanted,path
    namespace=dict(vars(original))
    exec(compile((c.HERE/'worker_main_source_003.txt').read_text(),'<S_initialization_scope_003>','exec'),namespace)
    namespace['main']()

if __name__=='__main__':main()
