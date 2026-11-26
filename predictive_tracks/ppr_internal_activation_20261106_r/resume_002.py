"""Route only the owned tail worker to its constructor-argument correction."""
import subprocess
from . import common as c


def main():
    for path, wanted in c.read(c.HERE / "RUNTIME_ADDENDUM_002.json")["files"].items():
        assert c.digest(path) == wanted, path
    original = subprocess.Popen
    def popen(args, *positional, **kwargs):
        if isinstance(args, list) and "predictive_tracks.ppr_internal_activation_20261106_r.tail_worker" in args:
            args = ["predictive_tracks.ppr_internal_activation_20261106_r.tail_runtime_002" if value == "predictive_tracks.ppr_internal_activation_20261106_r.tail_worker" else value for value in args]
        return original(args, *positional, **kwargs)
    subprocess.Popen = popen
    from .supervise import main as supervise
    supervise()


if __name__ == "__main__":
    main()
