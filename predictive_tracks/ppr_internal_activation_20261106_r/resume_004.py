"""Use the already-audited legacy result-directory resolver in reporting."""
from . import common as c


def main():
    for path,wanted in c.read(c.HERE/"REPORT_ROUTING_ADDENDUM_004.json")["files"].items():
        assert c.digest(path)==wanted,path
    from predictive_tracks.ppr_eta_radius_20261104_o.path_runtime_001 import install
    install()
    from .resume_003 import main as resume
    resume()


if __name__=="__main__":
    main()
