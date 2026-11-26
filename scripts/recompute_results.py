"""Recompute all reported metrics from immutable per-pair errors (NumPy only)."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from benchmark_metrics import calculate, aggregate

FIELDS = ('pairacc', 'vsa', 'auc', 'localization_mae', 'joint_accuracy')


def display(value, field):
    return f'{value:.3f}' if field == 'localization_mae' else f'{value * 100:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'recomputed')
    args = parser.parse_args()
    rows = json.loads((ROOT / 'results/index.json').read_text())
    manifest = list(csv.DictReader((ROOT / 'data/pairs_8to6.csv').open()))
    expected = {task: {r['pair_id'] for r in manifest if r['track'] == task} for task in ('SP', 'IE')}
    count, deltas, records, tables = 0, 0, [], {}
    for row in rows:
        task, model = row['task'], row['model_id']
        reports, identity = {}, None
        for role in ('baseline', 'corrected'):
            paths = row['paths'][role]
            with np.load(ROOT / paths['errors.npz'], allow_pickle=False) as data:
                ids = data['pair_ids'].astype(str)
                categories = data['categories'].astype(str)
                assert len(ids) == len(set(ids)) == 300
                assert set(ids) == expected[task], (task, model, 'different test pairs')
                current_identity = (ids.tolist(), data['frame_spans'].tolist(), data['violation_frames'].tolist())
                if identity is not None:
                    assert identity == current_identity, (task, model, 'different baseline/HSR evaluation grid')
                identity = current_identity
                flags = calculate(data['possible_errors'], data['impossible_errors'],
                                  data['frame_spans'], data['violation_frames'])
                got = dict(overall=aggregate(flags, np.arange(300)),
                           by_type={cat: aggregate(flags, np.flatnonzero(categories == cat))
                                    for cat in sorted(set(categories))})
            saved = json.loads((ROOT / paths['metrics.json']).read_text())
            for section in ('overall', 'by_type'):
                groups = [(got[section], saved[section])] if section == 'overall' else [
                    (got[section][cat], saved[section][cat]) for cat in got[section]]
                for actual, reference in groups:
                    for field, value in reference.items():
                        assert field in actual, field
                        observed = actual[field]
                        assert ((value is None and observed is None) or
                                (value is not None and observed is not None and
                                 abs(observed - value) <= 1e-12)), (task, model, role, field, observed, value)
            rendered = {f: display(got['overall'][f], f) for f in FIELDS}
            for item in row['displayed_values']:
                if item['kind'] == role:
                    assert rendered[item['field']] == item['expected'], (task, model, item)
                    count += 1
            reports[role] = rendered
            records.append(dict(task=task, model=model, role=role, **got))
        delta = {f: format(Decimal(reports['corrected'][f]) - Decimal(reports['baseline'][f]),
                           '+.3f' if f == 'localization_mae' else '+.2f') for f in FIELDS}
        for item in row['displayed_gains']:
            assert Decimal(delta[item['field']]) == Decimal(item['expected']), (task, model, item)
            deltas += 1
        marker = ' (diagnostic)' if row['diagnostic_only'] else ''
        tables.setdefault(task, []).extend([
            [row['model'] + marker, 'Baseline'] + [reports['baseline'][f] for f in FIELDS],
            [row['model'] + marker, 'HSR'] + [reports['corrected'][f] for f in FIELDS],
            [row['model'] + marker, 'After − before'] + [delta[f] for f in FIELDS]])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'all_metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    for task, table in tables.items():
        with (args.output / f'{task}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Model', 'Variant', 'PairAcc (%)', 'VSA (%)', 'AUC (%)', 'Loc. MAE (frames)', f'{task} (%)'])
            writer.writerows(table)
        lines = ['| Model | Variant | PairAcc ↑ | VSA ↑ | AUC ↑ | Loc. MAE ↓ | '+task+' ↑ |',
                 '|---|---|---:|---:|---:|---:|---:|']
        lines.extend('| ' + ' | '.join(r) + ' |' for r in table)
        (args.output / f'{task}.md').write_text('\n'.join(lines) + '\n')
    report = dict(configurations=len(rows), evaluated_outputs=len(records),
                  reported_values_verified=count, archived_differences_verified=deltas,
                  identical_test_pairs=True, metrics_match_archives=True,
                  additional_model_forwards=0,
                  delta_rule='Round baseline and HSR as displayed, then subtract baseline from HSR.')
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
