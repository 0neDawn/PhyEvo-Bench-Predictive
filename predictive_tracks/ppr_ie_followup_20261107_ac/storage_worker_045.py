"""Run unchanged DINO search/native workers with storage-only collection."""
import argparse
import importlib
import sys
from . import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module',choices=('worker_008','native_worker_009'),required=True)
    args,rest = parser.parse_known_args()
    c.verify_sources('STORAGE_READY_044.json')
    assert (c.HERE/'STORAGE_PROBE_COMPLETE_046.json').exists()
    from .storage_guard_044 import install_factory
    install_factory()
    sys.argv = [__file__]+rest
    importlib.import_module('predictive_tracks.ppr_ie_followup_20261107_ac.'+args.module).main()


if __name__=='__main__':
    main()
