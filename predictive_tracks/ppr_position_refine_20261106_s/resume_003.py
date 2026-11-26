"""Same frozen S settings/priority, with normal model initialization."""
import subprocess
from . import common as c

def main():
    for path,wanted in c.read(c.HERE/'INITIALIZATION_SCOPE_ADDENDUM_003.json')['files'].items():
        assert c.digest(path)==wanted,path
    original=subprocess.Popen
    def popen(args,*positional,**kwargs):
        if isinstance(args,list):
            args=['predictive_tracks.ppr_position_refine_20261106_s.worker_runtime_003' if x=='predictive_tracks.ppr_position_refine_20261106_s.worker' else x for x in args]
        return original(args,*positional,**kwargs)
    subprocess.Popen=popen
    from .priority_entry_002 import main as run
    run()

if __name__=='__main__':main()
