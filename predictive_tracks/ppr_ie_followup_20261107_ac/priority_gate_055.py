"""Keep annotation priority; accept only verified append-only settings metadata."""
import hashlib,json,time
from pathlib import Path
from . import common as c
_proof=None
_accepted=None

def signature(stat):
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

def priority_complete():
    global _proof,_accepted
    if _proof is None:
        c.verify_sources('PRIORITY_READY_055.json')
        _proof=c.read(c.HERE/'PRIORITY_ROLLOUT_PROOF_055.json')
        authority=c.read(c.HERE/'PRIORITY_APP_AUTHORITY_055.json')
        assert authority['latest_turn']['status']=='completed' and authority['latest_turn']['error'] is None
        assert authority['thread_status']['type']=='idle'
    path=Path(_proof['rollout_path']);first=signature(path.stat())
    if first==_accepted:return True
    minimum=_proof['rollout_size']
    if first[2]<minimum or first[2]-minimum>1048576:return False
    with path.open('rb') as stream:prefix=stream.read(minimum);tail=stream.read()
    if signature(path.stat())!=first or hashlib.sha256(prefix).hexdigest()!=_proof['rollout_sha256']:return False
    records=[]
    for line in tail.splitlines():
        if not line.strip():continue
        try:record=json.loads(line)
        except (ValueError,UnicodeDecodeError):return False
        value=record.get('payload',{})
        if record.get('type')!='event_msg' or value.get('type')!='thread_settings_applied' or value.get('thread_id')!=_proof['thread_id']:
            c.status(c.HERE/'monitor/priority_gate_055.json',dict(at=time.time(),state='priority_log_changed_requires_review',
                observed_record_type=record.get('type'),observed_payload_type=value.get('type'),new_work_never_automatically_authorized=True))
            return False
        records.append(dict(timestamp=record.get('timestamp'),type=value['type']))
    _accepted=first
    c.status(c.HERE/'monitor/priority_gate_055.json',dict(at=time.time(),state='reviewed_annotation_complete',
        accepted_append_only_settings_records=records,original_prefix_hash_exact=True,new_task_records_still_fail_closed=True))
    return True

def install():
    from predictive_tracks.ppr_eta_radius_20261104_o import common as original
    original.priority_complete=priority_complete
