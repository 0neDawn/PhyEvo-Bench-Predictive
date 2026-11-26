"""Use the already declared S reader for original full600 confirmations."""
import inspect,hashlib
from . import common as c

def main():
    from . import worker as original
    plan=c.read(c.HERE/'CONFIRMATION_READER_ADDENDUM_007.json')
    assert hashlib.sha256(inspect.getsource(original.execute_search).encode()).hexdigest()==plan['original_function_sha256']
    for path,wanted in plan['files'].items():assert c.digest(path)==wanted,path
    namespace=dict(vars(original))
    exec(compile((c.HERE/'worker_search_source_007.txt').read_text(),'<S_confirmation_reader_007>','exec'),namespace)
    original.execute_search=namespace['execute_search']
    from .worker_runtime_003 import main as run
    run()

if __name__=='__main__':main()
