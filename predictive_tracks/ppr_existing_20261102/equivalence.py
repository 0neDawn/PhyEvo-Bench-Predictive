"""Exact case-wise reuse from identical history-defined correction operators."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import HERE,configs,config_id,quality_id,qualities,result_folder,read,save,digest

FIELDS=('benchmark','archive_index','pair_id','category','split','t_ref','frame_spans',
        'possible_errors','impossible_errors','matched_identity_possible_errors','matched_identity_impossible_errors',
        'source_decoded_pixels_sha256','reference_pair_sha256')

def refpath(model,key):return HERE/'history_references'/model/f'{key}.json'
def validpath(model,key):return HERE/'equivalence_checks'/model/f'{key}.json'

def combined(states,qid):
    return hashlib.sha256(json.dumps([s['operators'][qid] for s in states],separators=(',',':')).encode()).hexdigest()

def register_history(model,row,history_sha256,hooks):
    path=refpath(model,row['key']);states=hooks.history_signatures
    if len(states)!=(8 if model=='gswm' else 1):raise RuntimeError('Native history draw count changed')
    if path.exists():
        previous=read(path)
        if previous['history_pixels_sha256']!=history_sha256 or [s['raw_history_sha256'] for s in states]!=previous['raw_history_sha256_by_draw']:
            raise RuntimeError('Historical inference changed across settings; refuse unproved reuse')
        for state,expected in zip(states,previous['operators_by_draw']):
            for key,value in state['operators'].items():
                if expected[key]!=value:raise RuntimeError('History operator signature changed')
        return previous
    wanted={quality_id(c) for c in qualities(model)}
    if any(set(s['operators'])!=wanted for s in states):raise RuntimeError('First native forecast must record all history-only operator signatures')
    proof={'model':model,'pair_id':row['pair_id'],'history_pixels_sha256':history_sha256,
        'raw_history_sha256_by_draw':[s['raw_history_sha256'] for s in states],
        'operators_by_draw':[s['operators'] for s in states],
        'combined':{qid:combined(states,qid) for qid in sorted(wanted)},
        'construction':'Native IDs and exact floating-point historical center, maximum radius and valid mask, before future steps; no candidate futures or scores.'}
    save(path,proof);return proof

def find(model,c,row,history_sha256,pixels_sha256):
    path=refpath(model,row['key'])
    if not path.exists():return None
    proof=read(path)
    if proof['pair_id']!=row['pair_id'] or proof['history_pixels_sha256']!=history_sha256:raise RuntimeError('History equivalence identity mismatch')
    signature=proof['combined'][quality_id(c)]
    for source in sorted(configs(model),key=config_id):
        if source==c or source['eta']!=c['eta'] or proof['combined'][quality_id(source)]!=signature:continue
        record=result_folder(model,source)/'pairs'/f'{row["key"]}.json'
        if not record.exists():continue
        r=read(record)
        if r.get('execution_mode')=='exact_history_operator_reuse':continue
        if r['pair_id']!=row['pair_id'] or r['source_decoded_pixels_sha256']!=pixels_sha256 or not r.get('trace_matches_original'):
            raise RuntimeError('Equivalence source identity or native budget invalid')
        return {'record':record,'source':r,'source_parameters':source,'operator_signature':signature,'proof_path':path}
    return None

def compare(a,b):
    return all(np.array_equal(a[k],b[k]) for k in ('possible_errors','impossible_errors','matched_identity_possible_errors','matched_identity_impossible_errors','frame_spans'))

def save_alias(target,model,c,row,evidence,reference_path):
    source=evidence['source'];proof=evidence['proof_path'];check=validpath(model,row['key'])
    if not check.exists() or not read(check)['passed']:raise RuntimeError('No successful actual numerical equivalence check for this case')
    if source['reference_pair_sha256']!=digest(reference_path):raise RuntimeError('Original paired scoring reference changed')
    payload={k:source[k] for k in FIELDS}
    payload.update(model=model,parameters=c,mode='feedback',eta=c['eta'],extra_forwards=0,
        execution_mode='exact_history_operator_reuse',new_neural_forward_invocations=0,
        equivalence={'source_result':str(evidence['record']),'source_result_sha256':digest(evidence['record']),
            'source_parameters':evidence['source_parameters'],'same_eta':True,'all_native_draws_operator_signature':evidence['operator_signature'],
            'history_only_proof':str(proof),'history_only_proof_sha256':digest(proof),
            'numerical_check':str(check),'numerical_check_sha256':digest(check),
            'source_native_trace':source['neural_call_trace'],
            'argument':'Same deterministic historical inference, per-draw RNG seeds, original state update, and identical correction map at every future step imply the same rollout by induction.'})
    save(target,payload);return payload
