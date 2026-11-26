"""Non-neural regression checks for reviewed metadata and cooperative scheduling."""
import hashlib,json,os,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from . import common as c

def main():
    from . import priority_gate_055 as gate
    assert gate.priority_complete()
    real_proof=gate._proof
    base=json.dumps(dict(type='event_msg',payload=dict(type='task_complete'))).encode()+b'\n'
    checks=[]
    with tempfile.TemporaryDirectory(prefix='predict-priority059-') as folder:
        path=Path(folder)/'rollout.jsonl'
        cases=[('unchanged',b'',True),('settings',json.dumps(dict(type='event_msg',payload=dict(type='thread_settings_applied',thread_id=real_proof['thread_id'],thread_settings={}))).encode()+b'\n',True),
            ('new_task',b'{"type":"event_msg","payload":{"type":"task_started"}}\n',False),
            ('user_request',b'{"type":"event_msg","payload":{"type":"user_message"}}\n',False),
            ('unknown',b'{"type":"response_item","payload":{}}\n',False),('partial_record',b'{',False)]
        try:
            for name,tail,wanted in cases:
                path.write_bytes(base+tail)
                gate._proof=dict(real_proof,rollout_path=str(path),rollout_size=len(base),rollout_sha256=hashlib.sha256(base).hexdigest())
                gate._accepted=None;assert gate.priority_complete() is wanted,name;checks.append(name)
            path.write_bytes(base.replace(b'task_complete',b'task_completed'))
            gate._accepted=None;assert not gate.priority_complete();checks.append('changed_prefix')
        finally:
            gate._proof=real_proof;gate._accepted=None;assert gate.priority_complete()
        scheduler=[]
        for mode in ('competing','hard','numerical'):
            root=Path(folder)/mode;root.mkdir()
            calls=[];states=[]
            marker=root/'models_complete/coarse_dino_foresight.json'
            def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            class Child:
                def __init__(self,index):self.index=index;self.pid=100+index;self.code=None
                def poll(self):
                    if mode=='numerical':self.code=1;return 1
                    if self.index==1:return self.code
                    save(marker,{});self.code=0;return 0
                def terminate(self):self.code=-15
                def wait(self):return self.code
            def popen(*args,**kwargs):calls.append(1);return Child(len(calls))
            def hardware(pid):return dict(available_ram_mib=20000,process_rss_mib=8000 if mode=='hard' else 2000,gpu_free_mib=12000,gpu_temperature_c=40,cpu_temperature_c=40)
            local=SimpleNamespace(HERE=root,PYTHON='python',ROOT=root,PARENT=root,save=save,status=save)
            (root/'logs').mkdir()
            namespace=dict(c=local,old=SimpleNamespace(gate=lambda model:(True,{}),priority_complete=lambda:True),
                update=lambda **value:states.append(value),os=os,time=SimpleNamespace(time=time.time,time_ns=time.time_ns,sleep=lambda value:None),
                hardware=hardware,gpu_idle=lambda pid:(False,{}),subprocess=SimpleNamespace(Popen=popen,STDOUT=-2))
            source=(c.HERE/'cooperative_execute_057.txt').read_text()
            import textwrap
            exec(compile(textwrap.dedent(source),'<scheduler_fixture059>','exec'),namespace)
            try:
                namespace['execute']('dino_foresight','coarse')
                assert mode=='competing' and len(calls)==2
            except RuntimeError:
                assert mode in ('hard','numerical') and len(calls)==1
            scheduler.append(mode)
    assert 'torch' not in __import__('sys').modules
    c.save(c.HERE/'PRIORITY_PREFLIGHT_059.json',dict(at=time.time(),metadata_checks=checks,scheduling_checks=scheduler,
        real_priority_completion_verified=True,hard_and_numerical_failures_never_retried=True,
        soft_competing_yield_resumes_only_after_gate=True,no_neural_or_GPU_calls=True,source_sha256=c.digest(__file__)))
    print('Priority metadata and soft/hard/numerical scheduling checks passed without model execution',flush=True)

if __name__=='__main__':main()
