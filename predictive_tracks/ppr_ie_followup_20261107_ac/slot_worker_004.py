"""Unchanged T future loop and decoder; new files and scalar coordinates only."""
import argparse
import time
from . import common as c


def verify(full=False):
    c.verify_sources('SLOT_READY_004.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=c.SLOTS,required=True)
    args = parser.parse_args()
    verify()
    c.verify = verify
    old = c.install()
    old.verify, old.directory, old.SOURCE, old.NATIVE = verify,c.directory,c.source_path(),c.SLOTS
    from predictive_tracks.ppr_feedback_refine_20261106_t import worker as original
    original.c = c
    from predictive_tracks.ppr_position_refine_20261106_s.thermal_guard_032 import install
    handle = install(lambda x:c.save(c.HERE/'events'/f'thermal_slot_{time.time_ns()}.json',x))
    try:
        base = original.setup()
        base.execute_search(args.model,'coarse')
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
