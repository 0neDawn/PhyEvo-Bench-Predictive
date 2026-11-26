"""Unchanged O native forecast, correction and observed-call integrity audit."""
from . import common as c


def main():
    c.install_original_runtime()
    from predictive_tracks.ppr_eta_radius_20261104_o.parameter_runtime_004 import install_worker, verify_addendum
    verify_addendum()
    install_worker()
    from predictive_tracks.ppr_eta_radius_20261104_o.native_worker import main as original
    original()


if __name__ == "__main__":
    main()
