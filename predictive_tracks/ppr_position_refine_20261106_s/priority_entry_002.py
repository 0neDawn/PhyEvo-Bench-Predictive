"""Only prioritize requested IE models; original frozen supervisor unchanged."""
import hashlib
from . import common as c

def main():
    from . import supervise as original
    import inspect
    plan=c.read(c.HERE/'PRIORITY_ROUTING_002.json')
    assert hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest()==plan['original_function_sha256']
    for path,wanted in plan['files'].items():
        assert c.digest(path)==wanted,path
    source=(c.HERE/'priority_supervise_source_002.txt').read_text()
    namespace=dict(vars(original))
    exec(compile(source,'<S_priority_route_002>','exec'),namespace)
    namespace['main']()

if __name__=='__main__':
    main()
