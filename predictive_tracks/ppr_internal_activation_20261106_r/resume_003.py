"""Owned queue routing only; all frozen operators and old jobs unchanged."""
import subprocess
from . import common as c


def main():
    for plan_name in ("RUNTIME_ADDENDUM_002.json","VALIDATION_EXTENSION_003.json"):
        for path,wanted in c.read(c.HERE/plan_name)["files"].items():
            assert c.digest(path)==wanted,path
    original=subprocess.Popen
    def popen(args,*positional,**kwargs):
        if isinstance(args,list):
            args=list(args)
            for i,value in enumerate(args):
                if value=="predictive_tracks.ppr_internal_activation_20261106_r.tail_worker":
                    args[i]="predictive_tracks.ppr_internal_activation_20261106_r.tail_runtime_002"
                elif value=="predictive_tracks.ppr_internal_activation_20261106_r.native_worker" and "--validation" in args:
                    args[i]="predictive_tracks.ppr_internal_activation_20261106_r.native_runtime_003"
        return original(args,*positional,**kwargs)
    subprocess.Popen=popen
    from .supervise import main as supervise
    supervise()


if __name__=="__main__":
    main()
