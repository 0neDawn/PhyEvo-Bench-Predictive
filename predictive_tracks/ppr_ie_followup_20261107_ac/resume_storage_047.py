"""Route only DINO workers through reviewed storage recovery; keep all phases."""
import subprocess
from . import common as c


def main():
    c.verify_sources('STORAGE_READY_044.json')
    assert (c.HERE/'STORAGE_PROBE_COMPLETE_046.json').exists()
    previous = subprocess.Popen
    prefix = 'predictive_tracks.ppr_ie_followup_20261107_ac.'
    def popen(args,*positional,**kwargs):
        if isinstance(args,list) and len(args)>4 and args[3]=='-m' and '--model' in args:
            model = args[args.index('--model')+1]
            module = args[4]
            if model=='dino_foresight' and module in (prefix+'worker_008',prefix+'native_worker_009'):
                args = args[:4]+[prefix+'storage_worker_045','--module',module.rsplit('.',1)[1]]+args[5:]
        return previous(args,*positional,**kwargs)
    subprocess.Popen = popen
    from .resume_controller_013 import main as original
    original()


if __name__=='__main__':
    main()
