"""Same original native full600 validator; read the early, model-stable subset."""
import inspect,hashlib,time
from . import common as c

def main():
    from . import native_worker as original
    plan=c.read(c.HERE/'EARLY_VALIDATION_PRIORITY_012.json')
    assert hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest()==plan['native_main_original_sha256']
    for path,wanted in plan['files'].items():assert c.digest(path)==wanted,path
    namespace=dict(vars(original))
    exec(compile((c.HERE/'early_native_source_012.txt').read_text(),'<S_early_native_012>','exec'),namespace)
    namespace['main']()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
        pause=isinstance(error,ResourceBusy)
        c.save(c.HERE/'events'/f'{"pause" if pause else "error"}_{time.time_ns()}.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        if pause:raise SystemExit(75)
        raise
