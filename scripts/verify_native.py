"""Recompute native results and verify every reported value without model calls."""
import argparse
import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_metrics import aggregate, calculate

FIELDS = ('pairacc', 'vsa', 'auc', 'localization_mae', 'joint_accuracy')
DECISIONS = ('pairacc', 'vsa', 'joint', 'pair_tie', 'vsa_tie', 'located',
             't_star', 'absolute_localization_error', 'mae_members', 'onset_error_index')


def display(value, field):
    if value is None:
        return 'N/A'
    return f'{value:.3f}' if field == 'localization_mae' else f'{value*100:.2f}'


def load_errors(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def evaluate(data):
    flags = calculate(data['possible_errors'], data['impossible_errors'],
                      data['frame_spans'], data['violation_frames'])
    categories = data['categories'].astype(str)
    summary = dict(overall=aggregate(flags, np.arange(len(categories))),
                   by_type={c: aggregate(flags, np.flatnonzero(categories == c))
                            for c in sorted(set(categories))})
    return flags, summary


def inspect(directory, cfg, variant, row):
    role = 'baseline' if variant == 'baseline' else 'corrected'
    actual = load_errors(directory / 'errors.npz')
    reference = load_errors(ROOT / row['paths'][role]['errors.npz'])
    ids = actual['pair_ids'].astype(str).tolist()
    ref_ids = reference['pair_ids'].astype(str).tolist()
    assert len(ids) == len(set(ids)) == 300 and set(ids) == set(ref_ids), 'Test-pair identity mismatch'
    order = {pair_id: i for i, pair_id in enumerate(ref_ids)}
    reference = {key: value[[order[pair_id] for pair_id in ids]] for key, value in reference.items()}
    for key in ('pair_ids', 'categories', 'frame_spans', 'violation_frames'):
        assert np.array_equal(actual[key], reference[key]), 'Evaluation-grid mismatch: '+key
    assert json.loads((directory/'config.json').read_text()) == cfg, 'Configuration mismatch'
    records = [json.loads(line) for line in (directory/'cases.jsonl').read_text().splitlines() if line.strip()]
    assert len(records) == 300 and [r['pair_id'] for r in records] == ids, 'Case-log identity mismatch'
    schedule = json.loads((ROOT/f'configs/schedules/{cfg["task"]}_{cfg["model_id"]}.json').read_text())
    assert [r['pair_id'] for r in schedule] == ids, 'Execution order differs from archive'
    for record, expected in zip(records, schedule):
        for key in ('archive_index', 'neural_call_trace', 'decoded_pixels_sha256'):
            assert record[key] == expected[key], 'Per-case provenance mismatch: '+key
    for key in ('possible_errors', 'impossible_errors', 'frame_spans'):
        assert np.array_equal(np.asarray([r[key] for r in records]), actual[key]), 'Case-log/NPZ mismatch: '+key
    flags, metrics = evaluate(actual)
    ref_flags, ref_metrics = evaluate(reference)
    saved = json.loads((directory/'metrics.json').read_text())
    archived = json.loads((ROOT/row['paths'][role]['metrics.json']).read_text())
    expected_values = {item['field']: item['expected'] for item in row['displayed_values'] if item['kind'] == role}
    comparisons = {field: dict(recomputed=display(metrics['overall'][field], field),
                               table=expected_values[field],
                               match=display(metrics['overall'][field], field) == expected_values[field])
                   for field in FIELDS}
    decisions_match = all(np.array_equal(flags[key], ref_flags[key]) for key in DECISIONS)
    passed = (metrics == saved == ref_metrics == archived and decisions_match
              and all(value['match'] for value in comparisons.values()))
    differences = {key: float(np.max(np.abs(actual[key]-reference[key])))
                   for key in ('possible_errors', 'impossible_errors')}
    return dict(status='verified' if passed else 'mismatch', cases=300,
                exact_overall_and_type_metrics=metrics == saved == ref_metrics == archived,
                decision_flags_exact=decisions_match, displayed_values=comparisons,
                maximum_absolute_error_differences=differences,
                all_errors_exact=all(value == 0 for value in differences.values()),
                test_pairs_and_temporal_grid_match=True, configuration_matches=True,
                case_log_matches_errors=True, neural_calls_and_input_hashes_match=True,
                localization_mae_denominator=metrics['overall']['localization_mae_denominator'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    configs = json.loads((ROOT/'configs/final_configs.json').read_text())['configurations']
    index = {(row['task'], row['model_id']): row for row in json.loads((ROOT/'results/index.json').read_text())}
    results, deltas = [], []
    for cfg in configs:
        pair = {}
        for variant in ('baseline', 'hsr'):
            name = f'{cfg["model_id"]}_{cfg["task"]}_{variant}'
            directory = next((base/name for base in args.run_dirs if (base/name/'COMPLETE.json').exists()), None)
            result = dict(model=cfg['model_id'], task=cfg['task'], variant=variant,
                          diagnostic_only=cfg['diagnostic_only'])
            if directory is None:
                result.update(status='incomplete')
            else:
                result['source'] = directory.parent.name+'/'+directory.name
                try:
                    result.update(inspect(directory, cfg, variant, index[(cfg['task'], cfg['model_id'])]))
                except (AssertionError, OSError, ValueError, KeyError) as error:
                    result.update(status='mismatch', error=str(error))
            results.append(result)
            pair[variant] = result
        if all(result['status'] == 'verified' for result in pair.values()):
            for field in FIELDS:
                before, after = (pair[role]['displayed_values'][field] for role in ('baseline','hsr'))
                delta = Decimal(after['recomputed'])-Decimal(before['recomputed'])
                expected = Decimal(after['table'])-Decimal(before['table'])
                precision = '+.3f' if field == 'localization_mae' else '+.2f'
                deltas.append(dict(model=cfg['model_id'], task=cfg['task'], field=field,
                                   native_delta=format(delta, precision), table_delta=format(expected, precision),
                                   match=delta == expected))
    verified = [result for result in results if result['status'] == 'verified']
    failed = [result for result in results if result['status'] == 'mismatch']
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='mismatch' if failed else 'complete' if len(verified)==36 else 'partial',
                  total_runs=36, verified_runs=len(verified), incomplete_runs=36-len(verified)-len(failed),
                  mismatched_runs=len(failed), displayed_values_verified=5*len(verified),
                  displayed_differences_verified=len(deltas),
                  all_verified_errors_exact=all(result['all_errors_exact'] for result in verified),
                  additional_model_forwards=0, runs=results, differences=deltas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('runs','differences')}, indent=2))
    return 1 if failed or (args.require_complete and len(verified)!=36) else 0


if __name__ == '__main__':
    raise SystemExit(main())
