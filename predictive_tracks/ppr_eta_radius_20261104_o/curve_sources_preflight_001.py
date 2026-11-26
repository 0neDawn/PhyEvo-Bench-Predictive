"""Check the actual read-only result index before bounded refinement starts."""
from collections import defaultdict
import json
import time
import types

from . import common as c


def main():
    c.verify()
    from .parameter_runtime_004 import verify_addendum
    verify_addendum()
    from .path_runtime_001 import install as paths
    from .report_runtime_002 import install as controls
    paths()
    controls()
    from . import report
    from .refine import qkey
    assembled = report.assemble()
    grouped = defaultdict(list)
    grid = c.read(c.HERE / "GRID.json")
    for job in grid["native_new"] + grid["cached"]:
        grouped[(job["model"], job["mode"], job["alpha"], json.dumps(job["quality"], sort_keys=True), job["target_task"])].append(job)
    visited = {}

    def index(setting):
        compact = setting.get("compact_case_source")
        folder = compact or str(__import__('pathlib').Path(setting.get("directory") or setting["source_directory"]) / "pairs")
        from pathlib import Path
        folder = Path(folder)
        key = str(folder)
        if key not in visited:
            files = sorted(folder.glob("*.json"))
            assert len(files) == 600, (folder, len(files))
            assert {x.name for x in files} == {f'{task}_{i:03d}.json' for task in ('sp', 'ie') for i in range(300)}
            first, last = c.read(files[0]), c.read(files[-1])
            if compact:
                assert setting['config_id'] in first['settings'] and setting['config_id'] in last['settings']
            else:
                assert first['benchmark'] in ('sp', 'ie') and last['benchmark'] in ('sp', 'ie')
            visited[key] = dict(folder=key, cases=600, compact=bool(compact), endpoints_parse=True)
        return key

    # Run the frozen resolver's exact branching and old-folder lookup; replace
    # only its terminal record loader with a directory/index verifier. Neither
    # the installed report resolver nor any source/scientific code is edited.
    resolver = report.records
    assert resolver.__code__.co_freevars == ('original',)
    cell = (lambda value: lambda: value)(index).__closure__[0]
    resolve_index = types.FunctionType(resolver.__code__, resolver.__globals__, closure=(cell,))
    points = []
    for (model, mode, alpha, quality_string, target), jobs in grouped.items():
        quality = json.loads(quality_string)
        key_quality = json.dumps(quality, sort_keys=True) if quality is not None else None
        rows = [x for x in assembled['all_configurations'][model]
                if x.get('mode') == mode and x['parameters'].get('alpha', 1.) == alpha and qkey(x) == key_quality]
        by_eta = {}
        for row in rows:
            by_eta[float(row['parameters']['eta'])] = row
        for eta, row in sorted(by_eta.items()):
            folder = resolve_index(row)
            points.append(dict(model=model, mode=mode, eta=eta, alpha=alpha, quality=quality,
                               target=target, config_id=row['config_id'], folder=folder))
    destination = c.HERE / 'CURVE_SOURCES_PREFLIGHT_001.json'
    c.save(destination, dict(at=time.time(), passed=True, points=points, directories=list(visited.values()),
                             checked_available_curve_points=len(points), unique_full600_sources=len(visited),
                             completed_stage1_at_check=assembled['completed_round_configurations'],
                             scientific_scores_and_candidate_rules_untouched=True, neural_forwards=0,
                             scope='Directory membership and first/last JSON readability only; final full original-metric/source audit is still required.'))
    print('CURVE SOURCE INDEX VERIFIED', len(points), 'points', len(visited), 'sources', flush=True)


if __name__ == '__main__':
    main()
