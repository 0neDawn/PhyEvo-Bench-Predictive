"""Reviewed priority gate for this experiment and its own child workers only."""
import argparse,json,runpy,subprocess,sys
from . import common as c
PREFIX='predictive_tracks.'
SELF=PREFIX+'ppr_ie_followup_20261107_ac.priority_entry_061'

def verified_soft_yield(detail,model):
    if model is None:return False
    h=detail.get('hardware',{});large=model=='dino_foresight'
    return bool(detail.get('safe') and (not detail.get('priority_complete') or not detail.get('gpu_idle'))
        and h.get('available_ram_mib',0)>=12288 and h.get('process_rss_mib',float('inf'))<=(7168 if large else 3072)
        and h.get('gpu_free_mib',0)>=(7168 if large else 10240) and h.get('gpu_temperature_c',float('inf'))<68
        and h.get('cpu_temperature_c',float('inf'))<75)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--entry-module',required=True)
    args,rest=parser.parse_known_args();module=args.entry_module;assert module.startswith(PREFIX) and module!=SELF
    c.verify_sources('BOOTSTRAP_READY_061.json')
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
        model=rest[rest.index('--model')+1] if '--model' in rest else None
        if verified_soft_yield(detail,model):
            print('Verified safe competing-task yield; no resource/numerical failure ignored',flush=True)
            raise SystemExit(85)
        raise

if __name__=='__main__':main()
