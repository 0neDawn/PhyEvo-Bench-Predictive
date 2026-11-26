"""Run an unchanged worker with additional conservative wall-clock cooling."""
import argparse
import runpy
import sys
import time
from . import common as c


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--module', required=True, choices=(
        'predictive_tracks.ppr_position_refine_20261106_s.worker_runtime_007',
        'predictive_tracks.ppr_position_refine_20261106_s.native_worker',
        'predictive_tracks.ppr_position_refine_20261106_s.early_native_runtime_012'))
    args, remaining = parser.parse_known_args()
    for path, wanted in c.read(c.HERE/'THERMAL_ADDENDUM_032.json')['files'].items():
        assert c.digest(path) == wanted, path
    from .thermal_guard_032 import install
    handle = install(lambda value:c.save(c.HERE/'events'/f'thermal_hold_032_{time.time_ns()}.json', value))
    sys.argv = [args.module] + remaining
    try:
        runpy.run_module(args.module, run_name='__main__')
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
