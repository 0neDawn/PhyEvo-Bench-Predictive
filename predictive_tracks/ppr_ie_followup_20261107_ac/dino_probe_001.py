"""Measure the unchanged DINO baseline before trying its existing loop writeback."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from . import common as c


def worker():
    c.verify_sources('PROBE_READY_001.json')
    import torch
    import numpy as np
    sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
    from predictive_tracks.ppr_dino_feedback_20261106_aa.worker import adapter_for
    from predictive_tracks.ppr_dino_feedback_20261106_aa.hooks import NativeCapture, LoopProjector
    from predictive_tracks.ppr_dino_feedback_20261106_aa.data import versions
    from predictive_tracks.ppr_position_refine_20261106_s.worker import trace_counter, exact, load_data
    from sp_core.common import PairRecord, read_pair_rgb
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    thermal = install(lambda x:c.save(c.HERE/'events'/f'thermal_probe_{time.time_ns()}.json',x))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    selected = []
    groups = set()
    for item in c.dino_rows():
        group = item[0]['benchmark'], item[0]['category_directory']
        if group not in groups:
            groups.add(group)
            selected.append(item)
    assert len(selected) == 6
    adapter = adapter_for('dino_foresight')
    baseline = versions(adapter,'dino_foresight')
    traces = []
    try:
        for row, source, meta in selected:
            p = c.HERE/'baseline_controls'/f'{row["key"]}.json'
            if p.exists():
                traces.append(c.read(p)['trace'])
                continue
            c.require_resources('dino_foresight')
            cached = load_data(meta)
            pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),
                Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible, impossible = read_pair_rgb(pair)
            pixels = hashlib.sha256(possible.tobytes()+impossible.tobytes()).hexdigest()
            assert pixels == meta['source_decoded_pixels_sha256']
            adapter.prediction_calls = row['archive_index']
            count, trace, handle = trace_counter()
            try:
                with NativeCapture(adapter) as capture:
                    result = adapter.score_pair(possible, impossible)
            finally:
                handle.remove()
            exact((result.possible,result.impossible),[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
            history = capture.context.reshape_as(cached['history'])
            prediction = capture.prediction.reshape_as(cached['prediction'])
            assert history.dtype == cached['history'].dtype and torch.equal(history,cached['history'])
            assert prediction.dtype == cached['prediction'].dtype and torch.equal(prediction,cached['prediction'])
            assert capture.forecaster_calls == 6 and capture.encoded_frame_requests == [5]+[1]*12
            assert versions(adapter,'dino_foresight') == baseline
            assert adapter.prediction_calls == row['archive_index']+1
            assert result.frame_spans.tolist() == meta['frame_spans']
            observed = dict(module_forward_invocations=count[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
            c.save(p,dict(at=time.time(),key=row['key'],trace=observed,
                source_meta=str(source),source_meta_sha256=c.digest(source),source_state_sha256=meta['sha256'],
                original_history_prediction_errors_exact=True,forecaster_calls=6,encoded_frame_requests=capture.encoded_frame_requests,
                historical_shape=list(cached['history'].shape),historical_dtype=str(history.dtype),
                raw_prediction_shape=list(cached['prediction'].shape),raw_prediction_dtype=str(prediction.dtype),
                archival_original_trace_match=meta.get('original_trace_match'),
                archival_trace_not_relabelled=True,original_alignment_unchanged=True,diagnostic_only=True))
            traces.append(observed)
            print('DINO unchanged baseline',row['key'],'exact',flush=True)
            del cached,history,prediction,capture,possible,impossible
        assert len({json.dumps(v,sort_keys=True) for v in traces}) == 1
        ref = c.HERE/'NATIVE_TRACE_REFERENCE.json'
        if not ref.exists():
            c.save(ref,dict(at=time.time(),trace=traces[0],controls=6,
                original_adapter_measured_before_nonzero_eta=True,
                archival_trace_missing_not_fabricated=True,diagnostic_only=True,
                files={str(p):c.digest(p) for p in (c.HERE/'baseline_controls').glob('*.json')}))
        for row, source, meta in selected:
            pair = PairRecord(row['pair_id'],row['category_directory'],int(row['t_ref']),
                Path(row['possible_dir']),Path(row['impossible_dir']),Path(row['meta_path']))
            possible, impossible = read_pair_rgb(pair)
            for eta in (0.,.1):
                p = c.HERE/'probe_corrected'/f'{row["key"]}_e{eta:.8g}.json'
                if p.exists():
                    continue
                c.require_resources('dino_foresight')
                adapter.prediction_calls = row['archive_index']
                count, trace, handle = trace_counter()
                try:
                    with LoopProjector(adapter,dict(model='dino_foresight',mode='feedback',eta=eta,alpha=1.)) as hook:
                        result = adapter.score_pair(possible,impossible)
                finally:
                    handle.remove()
                observed = dict(module_forward_invocations=count[0],ordered_class_and_input_shapes_sha256=trace.hexdigest())
                assert observed == traces[0]
                assert versions(adapter,'dino_foresight') == baseline
                assert adapter.prediction_calls == row['archive_index']+1 and result.frame_spans.tolist() == meta['frame_spans']
                assert hook.forecaster_calls == 6
                if eta == 0:
                    exact((result.possible,result.impossible),[meta['matched_identity_'+b+'_errors'] for b in ('possible','impossible')])
                c.save(p,dict(at=time.time(),key=row['key'],eta=eta,alpha=1.,trace_matches_measured_original=True,
                    parameter_identity_version_unchanged=True,forecaster_calls=6,no_extra_forward_per_inference=True,
                    possible_errors=result.possible.tolist(),impossible_errors=result.impossible.tolist(),
                    projection_audits=hook.audits,diagnostic_only=True,original_alignment_unchanged=True))
                print('DINO loop probe',row['key'],eta,'original budget exact',flush=True)
        c.save(c.HERE/'PROBE_COMPLETE_001.json',dict(at=time.time(),unmodified_controls=6,loop_checks=12,
            neural_budget_unchanged=True,alignment_unchanged=True,diagnostic_only=True,
            no_full300_performance_claim=True))
    finally:
        thermal.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except BaseException as error:
            c.save(c.HERE/'events'/f'probe_error_{time.time_ns()}.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
            raise
        return
    c.verify_sources('PROBE_READY_001.json')
    old = c.install()
    while True:
        okay, detail = old.gate('dino_foresight')
        c.status(c.HERE/'monitor/probe.json',dict(at=time.time(),state='checking_resources' if okay else 'waiting_resources',gate=detail))
        if okay and detail['hardware']['available_ram_mib'] >= 20480 and detail['hardware']['gpu_used_mib'] <= 1024:
            break
        time.sleep(20)
    env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log = c.HERE/'logs'/f'dino_probe_{time.time_ns()}.log'
    log.parent.mkdir(exist_ok=True)
    from predictive_tracks.ppr_existing_20261102.resource_resume_004.guarded import hardware
    from predictive_tracks.ppr_20261028.control import gpu_idle
    with log.open('x') as stream:
        child = subprocess.Popen([c.PYTHON,'-B','-u','-m',__spec__.name,'--worker'],cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        c.status(c.HERE/'monitor/probe.json',dict(at=time.time(),state='running',pid=child.pid,log=str(log)))
        while child.poll() is None:
            try:
                h = hardware(child.pid)
                idle,gpu = gpu_idle(child.pid)
                c.status(c.HERE/'monitor/probe_hardware.json',h)
                unsafe = (h['available_ram_mib'] < 12288 or h['process_rss_mib'] > 7168 or h['gpu_free_mib'] < 7168
                    or h['gpu_temperature_c'] >= 68 or h['cpu_temperature_c'] >= 75 or not idle or not old.priority_complete())
            except BaseException as error:
                h = dict(error=repr(error))
                unsafe = True
            if unsafe:
                child.terminate()
                child.wait()
                c.save(c.HERE/'events'/f'probe_resource_stop_{time.time_ns()}.json',dict(at=time.time(),hardware=h,action='stop only owned probe; no limit increase'))
                raise RuntimeError('DINO probe resource protection; outputs retained')
            time.sleep(3)
        code = child.wait()
    c.status(c.HERE/'monitor/probe.json',dict(at=time.time(),state='complete' if code == 0 else 'failed',exit_code=code,log=str(log)))
    if code:
        raise RuntimeError('DINO probe failed; retain output and inspect')


if __name__ == '__main__':
    main()
