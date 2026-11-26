"""Login recovery of only these owned queues; never clear a genuine failure."""
import json
import subprocess
import time
from . import common as c


def main():
    c.verify_sources('RECOVERY_READY_014.json')
    settings = [
        ('predict-ppr-ac-slots004-20261107.service','SLOT_RUNS_COMPLETE_004.json','slot_queue_004','monitor/slots_004.json'),
        ('predict-ppr-ac-dino-probe006-20261107.service','PROBE_COMPLETE_005.json','probe_entry_006','monitor/probe.json'),
        ('predict-ppr-ac-controller012-20261107.service','COMPLETED_012.json','resume_controller_013','monitor/queue.json'),
    ]
    decisions = []
    for unit,marker,module,status in settings:
        if (c.HERE/marker).exists():
            decisions.append(dict(unit=unit,action='complete_skip'));continue
        result = subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState,MainPID,Result'],capture_output=True,text=True,check=False)
        values = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if values.get('ActiveState') in ('active','activating'):
            decisions.append(dict(unit=unit,action='active_unchanged',pid=values.get('MainPID')));continue
        state = c.read(c.HERE/status) if (c.HERE/status).exists() else {}
        historical_probe001 = (module=='probe_entry_006' and state.get('at',0)<c.read(c.HERE/'PROBE_READY_005.json')['at'])
        if values.get('ActiveState')=='failed' or (state.get('state')=='failed' and not historical_probe001):
            decisions.append(dict(unit=unit,action='failure_preserved_inspection_required',state=state));continue
        command = ['systemd-run','--user',f'--unit={unit}','--property=WorkingDirectory='+str(c.ROOT),
            '--property=MemoryHigh=6G','--property=MemoryMax=8G','--property=MemorySwapMax=512M',
            '--property=CPUQuota=200%','--property=Nice=15','--property=IOWeight=10','--property=Restart=no',
            '--setenv=PYTHONDONTWRITEBYTECODE=1',c.PYTHON,'-B','-u','-m','predictive_tracks.ppr_ie_followup_20261107_ac.'+module]
        subprocess.run(command,check=True)
        decisions.append(dict(unit=unit,action='restored_owned_unfinished_queue',original_limits_unchanged=True))
    c.save(c.HERE/'recovery_events'/f'login_{time.time_ns()}.json',dict(at=time.time(),decisions=decisions,
        no_foreign_process_changes=True,failures_never_reset=True,user_login_recovery_not_unattended_boot=True))
    print(json.dumps(decisions),flush=True)


if __name__=='__main__':main()
