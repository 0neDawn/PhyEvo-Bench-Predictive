"""Use the unchanged original full pipeline, routed only to new output files."""
import argparse
from . import common as c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--validation", action="store_true")
    args = p.parse_args()
    old = c.install()
    from predictive_tracks.ppr_eta_radius_20261104_o import native_worker as worker
    from predictive_tracks.ppr_eta_radius_20261104_o.hooks import factory as original_factory
    from predictive_tracks.ppr_eta_radius_20261104_n.hooks import TunedStateOnly
    def factory(adapter, item, snapshot):
        if item["mode"] == "state_only":
            return TunedStateOnly(adapter, item["model"], item["eta"], item["alpha"], item["mode"], snapshot)
        return original_factory(adapter, item, snapshot)
    worker.factory = factory
    if args.validation:
        original_read = old.read
        def read(path):
            if str(path) == str(c.HERE / "STAGE2.json"):
                return c.read(c.HERE / "VALIDATION_GRID.json")
            return original_read(path)
        old.read = read
    try:
        worker.execute(args.model, args.validation)
    except old.ResourceBusy:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
