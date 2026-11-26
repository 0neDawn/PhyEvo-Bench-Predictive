"""Audit a completed original600 configuration; read saved errors only."""
import argparse
import sys
import time
from pathlib import Path
from . import common as c

sys.path.insert(0, str(c.ROOT / 'predictive_tracks/sp/code'))
from predictive_tracks.ppr_existing_20261102.metrics import make_summary
from predictive_tracks.ppr_eta_radius_20261104_o.report import casewise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=c.SLOTS+c.FEATURES, required=True)
    parser.add_argument('--config-id', required=True)
    args = parser.parse_args()
    directory = c.HERE/'runs'/args.model/args.config_id
    saved = c.read(directory/'summary.json')
    assert saved['complete'] and saved['cases'] == 600
    paths = sorted((directory/'pairs').glob('*.json'))
    assert len(paths) == 600
    rows = [c.read(p) for p in paths]
    for task in ('sp', 'ie'):
        subset = [r for r in rows if r['benchmark'] == task]
        assert sorted(r['archive_index'] for r in subset) == list(range(300))
        categories = {r['category'] for r in subset}
        assert len(categories) == 3
        assert all(sum(r['category'] == category for r in subset) == 100
                   for category in categories)
    assert all(r['trace_matches_original'] and r['extra_forwards'] == 0
               and r['weight_identity_and_version_unchanged'] for r in rows)
    recalculated = make_summary(rows, args.model)
    assert recalculated['metrics'] == saved['metrics']
    result = dict(
        at=time.time(), model=args.model, config_id=args.config_id,
        metrics=saved['metrics'], casewise=casewise(rows, args.model),
        original_scoring_and_flags=True, no_new_model_calls=True,
        original600_complete=True, source_summary=str(directory/'summary.json'),
        source_hashes={str(p):c.digest(p) for p in paths+[directory/'summary.json']},
        same_exposed_data_exploratory=True,
        uncertainty_is_descriptive_after_adaptive_selection=True)
    destination = c.HERE/'native_casewise_016'/f'{args.config_id}_{time.time_ns()}.json'
    c.save(destination, result)
    for task, detail in result['casewise'].items():
        keys = ('native_MAE_n', 'corrected_MAE_n', 'common_MAE_n',
                'common_native_MAE', 'common_corrected_MAE')
        print(task, {k:detail[k] for k in keys},
              {k:len(detail[k]) for k in ('PA_gain','PA_loss','JH_gain','JH_loss',
                                         'onset_improved','onset_worsened')}, flush=True)
    print(destination, flush=True)


if __name__ == '__main__':
    main()
