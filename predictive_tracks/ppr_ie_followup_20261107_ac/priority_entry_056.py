"""Reviewed priority gate for this experiment and its own child workers only."""
import argparse,json,runpy,subprocess,sys
from . import common as c
PREFIX='predictive_tracks.'
SELF=PREFIX+'ppr_ie_followup_20261107_ac.priority_entry_056'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--entry-module',required=True)
    args,rest=parser.parse_known_args();module=args.entry_module;assert module.startswith(PREFIX) and module!=SELF
    c.verify_sources('PRIORITY_ENTRY_READY_056.json')
    from .priority_gate_055 import install
    install()
    if module==PREFIX+'ppr_ie_followup_20261107_ac.resume_storage_047':
        from .cooperative_controller_057 import install as install_controller
        install_controller()
    original=subprocess.Popen
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and '-m' in command:
            index=command.index('-m')
            if index+1<len(command):
                module=command[index+1]
                if module.startswith(PREFIX) and module!=SELF:
                    command=command[:index+1]+[SELF,'--entry-module',module]+command[index+2:]
        return original(command,*positional,**kwargs)
    subprocess.Popen=popen;sys.argv=[module]+rest
    from predictive_tracks.ppr_eta_radius_20261104_o.common import ResourceBusy
    try:runpy.run_module(module,run_name='__main__')
    except ResourceBusy as error:
        detail=json.loads(str(error))
        if detail.get('safe') and (not detail.get('priority_complete') or not detail.get('gpu_idle')):
            print('Verified safe competing-task yield; no resource/numerical failure ignored',flush=True)
            raise SystemExit(85)
        raise

if __name__=='__main__':main()
