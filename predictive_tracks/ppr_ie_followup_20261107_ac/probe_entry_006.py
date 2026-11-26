"""Queue the explicit cache/native scoring audit after the owned slot worker."""
import subprocess
import sys
import time
from . import common as c


def main():
    c.verify_sources('PROBE_READY_005.json')
    if '--worker' not in sys.argv:
        while not (c.HERE/'SLOT_RUNS_COMPLETE_004.json').exists():
            state = c.read(c.HERE/'monitor/slots_004.json')
            if state.get('state') == 'failed':
                raise RuntimeError('Prior owned slot queue failed; inspect before proceeding')
            c.status(c.HERE/'monitor/probe_006_wait.json',dict(at=time.time(),state='waiting_for_slots004',no_GPU=True))
            time.sleep(20)
    from . import dino_probe_001 as original
    from .dino_reference_005 import worker
    original.worker = worker
    previous = subprocess.Popen
    def popen(command,*args,**kwargs):
        if isinstance(command,list):
            command = [x.replace('predictive_tracks.ppr_ie_followup_20261107_ac.dino_probe_001',
                'predictive_tracks.ppr_ie_followup_20261107_ac.probe_entry_006') for x in command]
        return previous(command,*args,**kwargs)
    subprocess.Popen = popen
    original.main()


if __name__ == '__main__':
    main()
