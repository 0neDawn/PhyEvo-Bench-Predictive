"""Same frozen S experiment, using its format-aware confirmation reader."""
import subprocess
from . import common as c

def main():
    for path,wanted in c.read(c.HERE/'CONFIRMATION_READER_ADDENDUM_007.json')['files'].items():assert c.digest(path)==wanted,path
    original=subprocess.Popen
    def popen(args,*positional,**kwargs):
        if isinstance(args,list):args=['predictive_tracks.ppr_position_refine_20261106_s.worker_runtime_007' if x=='predictive_tracks.ppr_position_refine_20261106_s.worker_runtime_003' else x for x in args]
        return original(args,*positional,**kwargs)
    subprocess.Popen=popen
    from .resume_004 import main as run
    run()

if __name__=='__main__':main()
