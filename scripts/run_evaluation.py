"""Portable launcher for the archived final configurations.

Uses the archived adapters, correction kernels, and per-case schedules.
This launcher was assembled for release; see VALIDATION.md for tested scope.
Run one model/task/variant per process to isolate third-party import namespaces.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'predictive_tracks/sp/code')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from predictive_tracks.controlled_tracks.registry import MODELS


def load_config(model, task):
    configs = json.loads((ROOT / 'configs/final_configs.json').read_text())['configurations']
    return next(c for c in configs if c['model_id'] == model and c['task'] == task)


def load_rows(task):
    rows = [r for r in csv.DictReader((ROOT / 'data/pairs_8to6.csv').open()) if r['track'] == task]
    if len(rows) != 300 or len({r['pair_id'] for r in rows}) != 300:
        raise RuntimeError('Expected the fixed 300-pair manifest')
    return rows


def configured_class(model, config, hsr):
    base = importlib.import_module(MODELS[model]).Adapter
    if not hsr or model not in ('slotformer', 'vjepa2_vitg', 'dino_foresight'):
        return base
    from predictive_tracks.controlled_tracks.moment_projection import (
        SlotMomentMixin, VJEPAMomentMixin, DINOMomentMixin)
    from predictive_tracks.ppr_alpha_20261031.projection import trajectory, pooled_trajectory
    mixin = dict(slotformer=SlotMomentMixin, vjepa2_vitg=VJEPAMomentMixin,
                 dino_foresight=DINOMomentMixin)[model]
    # Reproduce the archived cached-state arithmetic on CPU before native CUDA scoring.
    class ReleaseHSR(mixin, base):
        def _correct(self, prediction, history, *, rgb=False):
            if rgb:
                raise ValueError('These final configurations operate on slots/features')
            raw, observed = prediction.detach().cpu(), history.detach().cpu()
            fn = trajectory if model == 'slotformer' else pooled_trajectory
            hard, self._projection_audit = fn(raw, observed, config['parameters']['alpha'])
            eta = config['parameters']['eta']
            after = raw if eta == 0 else hard if eta == 1 else raw + eta * (hard - raw)
            return after.to(device=prediction.device, dtype=prediction.dtype)
    return ReleaseHSR


def modules(adapter, model):
    if model in ('ocvp_seq', 'savi_transformer'):
        return adapter.savi, adapter.predictor
    if model == 'slotformer':
        return adapter.savi, adapter.dynamics
    if model == 'gswm':
        return (adapter.runtime.model,)
    if model in ('ddlp', 'videomaev2'):
        return (adapter.model,)
    if model == 'dino_foresight':
        return (adapter.model,)
    return adapter.encoder, adapter.predictor, adapter.target_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(MODELS))
    parser.add_argument('--task', choices=('SP', 'IE'))
    parser.add_argument('--variant', choices=('baseline', 'hsr'), default='hsr')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-fraction', type=float, default=0.45)
    parser.add_argument('--resume', action='store_true', help='Continue completed cases in the same configuration')
    parser.add_argument('--check-imports', action='store_true', help='Import launch dependencies without loading models')
    args = parser.parse_args()
    if args.model == 'ddlp' and not args.check_imports:
        from release_paths import runtime_python
        executable = runtime_python('ddlp')
        if Path(executable).resolve() != Path(sys.executable).resolve():
            os.execv(executable, [executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]])
    import numpy as np
    import torch
    from benchmark_metrics import calculate, aggregate
    from sp_core.common import PairRecord, read_pair_rgb
    from predictive_tracks.ppr_position_refine_20261106_s.hooks import factory
    from predictive_tracks.controlled_tracks.response_full_run import describe
    if args.check_imports:
        def forbid(*_):
            raise RuntimeError('No model forward permitted during import validation')
        guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
        try:
            for model in MODELS:
                for task in ('SP', 'IE'):
                    configured_class(model, load_config(model, task), True)
            print('All nine adapter/HSR entry points imported; zero model initializations or forwards.')
        finally:
            guard.remove()
        return
    if not args.model or not args.task:
        parser.error('--model and --task are required for evaluation')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The archived evaluation used CUDA; no silent CPU scoring fallback')
    if not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    config = load_config(args.model, args.task)
    output = args.output or ROOT / 'runs' / f'{args.model}_{args.task}_{args.variant}'
    if output.exists() and not args.resume:
        raise FileExistsError('Use a new output directory; prior results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    evaluation_lock = (output / '.evaluation.lock').open('a')
    try:
        fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another evaluation is already writing this output directory')
    config_file = output / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise RuntimeError('Cannot resume with changed settings')
    else:
        config_file.write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=args.device)
    runtime_file = output / 'runtime.json'
    if not runtime_file.exists():
        runtime_file.write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,
            gpu=torch.cuda.get_device_name(args.device),torch_threads=torch.get_num_threads(),
            seed=0,memory_fraction=args.memory_fraction,
            entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    cls = configured_class(args.model, config, args.variant == 'hsr')
    adapter = cls(device=args.device, memory_mode='safe', orientation='identity')
    def versions():
        return [(id(p), p._version, tuple(p.shape), str(p.dtype), p.requires_grad)
                for module in modules(adapter, args.model) for p in module.parameters()]
    initial_versions = versions()
    def parameter_values():
        values = {}
        for module_index, module in enumerate(modules(adapter, args.model)):
            for name, p in module.named_parameters():
                data = p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                values[f'{module_index}:{name}'] = dict(shape=list(p.shape), dtype=str(p.dtype),
                    requires_grad=p.requires_grad, sha256=hashlib.sha256(data).hexdigest())
        return values
    # cuDNN can reorganize SlotFormer's recurrent-weight storage. Verify actual
    # parameter bytes whenever its version/identity metadata changes.
    initial_parameter_values = parameter_values() if args.model == 'slotformer' else None
    clip_len = adapter.savi.clip_len if args.model == 'slotformer' else None
    schedule = json.loads((ROOT / f'configs/schedules/{args.task}_{args.model}.json').read_text())
    by_id = {r['pair_id']: r for r in load_rows(args.task)}
    cases_file = output / 'cases.jsonl'
    outcomes = [json.loads(line) for line in cases_file.read_text().splitlines() if line.strip()] if cases_file.exists() else []
    if [x['pair_id'] for x in outcomes] != [x['pair_id'] for x in schedule[:len(outcomes)]]:
        raise RuntimeError('Resume records are not an ordered prefix of the archived schedule')
    job = dict(model=args.model, mode=config['mode'], eta=config['parameters']['eta'],
               alpha=config['parameters']['alpha'],
               quality={k: config['parameters'].get(k) for k in ('q', 'presence', 'mass', 'min_history')})
    if config['parameters'].get('position'):
        job['position'] = config['parameters']['position']
    for case in schedule[len(outcomes):]:
        row = by_id[case['pair_id']]
        pair = PairRecord(row['pair_id'], row['category_directory'], int(row['t_ref']),
                          ROOT / row['possible_dir'], ROOT / row['impossible_dir'], ROOT / row['meta_path'])
        possible, impossible = read_pair_rgb(pair)
        pixel_hash = hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest()
        if pixel_hash != case['decoded_pixels_sha256']:
            raise RuntimeError('Input data differ from the archived evaluation: ' + row['pair_id'])
        expected_trace = case['neural_call_trace']
        if args.model == 'slotformer':
            count = expected_trace['module_forward_invocations']
            if count not in (1321, 2103):
                raise RuntimeError('Unknown archived SAVi schedule')
            adapter.savi.clip_len = clip_len if count == 2103 else 14
        adapter.prediction_calls = case['archive_index']
        calls, trace = [0], hashlib.sha256()
        def count_forward(module, inputs):
            calls[0] += 1
            trace.update(json.dumps((module.__class__.__module__, module.__class__.__qualname__,
                                     describe(inputs)), sort_keys=True).encode())
        counter = torch.nn.modules.module.register_module_forward_pre_hook(count_forward)
        hook = contextlib.nullcontext()
        if args.variant == 'hsr' and args.model not in ('slotformer', 'vjepa2_vitg', 'dino_foresight'):
            hook = factory(adapter, job, output / 'instrumented_sources')
        try:
            with torch.inference_mode(), hook:
                result = adapter.score_pair(possible, impossible)
        finally:
            counter.remove()
        observed_trace = dict(module_forward_invocations=calls[0],
                              ordered_class_and_input_shapes_sha256=trace.hexdigest())
        if observed_trace != expected_trace:
            raise RuntimeError(f'Neural call graph differs from archive: {observed_trace} vs {expected_trace}')
        observed_versions = versions()
        if observed_versions != initial_versions:
            if initial_parameter_values is not None and parameter_values() == initial_parameter_values:
                with (output / 'parameter_storage_audit.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(pair_id=row['pair_id'],parameter_bytes_unchanged=True,
                        changed_metadata=[dict(index=k,before=b,after=a) for k,(b,a) in
                            enumerate(zip(initial_versions,observed_versions)) if b!=a]))+'\n')
                initial_versions = observed_versions
            else:
                (output/'parameter_check_failure.json').write_text(json.dumps(dict(
                    before=initial_versions,after=observed_versions,initial_values=initial_parameter_values,
                    observed_values=parameter_values() if initial_parameter_values is not None else None),indent=2)+'\n')
                raise RuntimeError('Model parameter values or unverified metadata changed')
        if adapter.prediction_calls != case['archive_index'] + 1:
            raise RuntimeError('Forecast budget changed')
        if pixel_hash != hashlib.sha256(possible.tobytes() + impossible.tobytes()).hexdigest():
            raise RuntimeError('Decoded inputs were mutated')
        record = dict(pair_id=row['pair_id'], category=row['category_directory'],
                      archive_index=case['archive_index'], t_ref=int(row['t_ref']),
                      possible_errors=result.possible.tolist(), impossible_errors=result.impossible.tolist(),
                      frame_spans=result.frame_spans.tolist(), neural_call_trace=observed_trace,
                      decoded_pixels_sha256=pixel_hash)
        outcomes.append(record)
        with (output / 'cases.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if len(outcomes) % 25 == 0:
            print(f'{args.model} {args.task} {args.variant}: {len(outcomes)}/300', flush=True)
    p, i = [np.asarray([r[key] for r in outcomes]) for key in ('possible_errors', 'impossible_errors')]
    spans = np.asarray([r['frame_spans'] for r in outcomes])
    tau = np.asarray([r['t_ref'] for r in outcomes])
    ids, categories = [np.asarray([r[key] for r in outcomes]) for key in ('pair_id', 'category')]
    np.savez_compressed(output / 'errors.npz', possible_errors=p, impossible_errors=i,
                        frame_spans=spans, violation_frames=tau, pair_ids=ids, categories=categories)
    flags = calculate(p, i, spans, tau)
    metrics = dict(overall=aggregate(flags, np.arange(300)),
                   by_type={c: aggregate(flags, np.flatnonzero(categories == c)) for c in sorted(set(categories))})
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    archive_role = 'corrected' if args.variant == 'hsr' else 'baseline'
    archived = json.loads((ROOT / f'results/{args.task}/{args.model}/{archive_role}/metrics.json').read_text())
    matches = {key: value == archived['overall'][key] for key, value in metrics['overall'].items()}
    (output / 'archive_comparison.json').write_text(json.dumps(matches, indent=2) + '\n')
    with np.load(ROOT / f'results/{args.task}/{args.model}/{archive_role}/errors.npz', allow_pickle=False) as reference:
        order = {pair_id: k for k, pair_id in enumerate(reference['pair_ids'].astype(str))}
        idx = [order[pair_id] for pair_id in ids]
        if not np.array_equal(reference['frame_spans'][idx], spans):
            raise RuntimeError('Reported frame spans differ from archive')
        error_report = dict(possible_max_abs=float(np.max(np.abs(p-reference['possible_errors'][idx]))),
                            impossible_max_abs=float(np.max(np.abs(i-reference['impossible_errors'][idx]))),
                            all_metrics_exact=all(matches.values()))
    (output / 'error_comparison.json').write_text(json.dumps(error_report, indent=2) + '\n')
    print(json.dumps(metrics['overall'], indent=2))
    if not all(matches.values()):
        raise RuntimeError('Metrics differ from the archived run; results retained for investigation')
    (output / 'COMPLETE.json').write_text(json.dumps(dict(cases=300, metrics_match_archive=True,
        variant=args.variant, model=args.model, task=args.task, diagnostic_only=config['diagnostic_only'],
        neural_budget_unchanged=True, **error_report), indent=2)+'\n')


if __name__ == '__main__':
    main()
