"""Prioritize full validation of promising Video results; all guards/settings retained."""
import inspect,hashlib,subprocess
from . import common as c

def main():
    plan=c.read(c.HERE/'EARLY_VALIDATION_PRIORITY_012.json')
    for path,wanted in plan['files'].items():assert c.digest(path)==wanted,path
    from . import supervise as original,priority_entry_002
    assert hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest()==plan['supervisor_original_sha256']
    for path,wanted in c.read(c.HERE/'PRIORITY_ROUTING_002.json')['files'].items():assert c.digest(path)==wanted,path
    def reordered():
        from .early_support_012 import make_early,dense_l
        namespace=dict(vars(original),_s_early_validation_012=make_early,_s_dense_l_012=dense_l)
        exec(compile((c.HERE/'priority_supervise_source_012.txt').read_text(),'<S_early_validation_priority_012>','exec'),namespace)
        namespace['main']()
    priority_entry_002.main=reordered
    base=subprocess.Popen
    def popen(args,*positional,**kwargs):
        if isinstance(args,list) and not (c.HERE/'grids/validation.json').exists():
            args=['predictive_tracks.ppr_position_refine_20261106_s.early_native_runtime_012' if x=='predictive_tracks.ppr_position_refine_20261106_s.native_worker' else x for x in args]
        return base(args,*positional,**kwargs)
    subprocess.Popen=popen
    from .resume_007 import main as run
    run()

if __name__=='__main__':main()
