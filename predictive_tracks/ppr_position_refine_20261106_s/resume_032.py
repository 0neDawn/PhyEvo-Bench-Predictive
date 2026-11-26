"""Same S experiment and hard limits; cool inside long parameter sweeps."""
import hashlib
import inspect
import subprocess
from . import common as c


def main():
    new = c.read(c.HERE/'THERMAL_ADDENDUM_032.json')
    for path, wanted in new['files'].items():
        assert c.digest(path) == wanted, path
    original_plan = c.read(c.HERE/'EARLY_VALIDATION_PRIORITY_012.json')
    for path, wanted in original_plan['files'].items():
        assert c.digest(path) == wanted, path
    from . import supervise as original, priority_entry_002
    assert hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest() == original_plan['supervisor_original_sha256']
    def reordered():
        from .early_support_012 import make_early, dense_l
        namespace = dict(vars(original), _s_early_validation_012=make_early, _s_dense_l_012=dense_l)
        source = (c.HERE/'thermal_supervise_source_032.txt').read_text()
        exec(compile(source, '<S_conservative_cooling_032>', 'exec'), namespace)
        namespace['main']()
    priority_entry_002.main = reordered
    previous = subprocess.Popen
    def popen(args, *positional, **kwargs):
        if isinstance(args, list):
            args = list(args)
            target = args[4] if len(args)>4 and args[3]=='-m' else None
            targets = {
                'predictive_tracks.ppr_position_refine_20261106_s.worker_runtime_007',
                'predictive_tracks.ppr_position_refine_20261106_s.native_worker',
            }
            if target in targets:
                if target.endswith('.native_worker') and not (c.HERE/'grids/validation.json').exists():
                    target='predictive_tracks.ppr_position_refine_20261106_s.early_native_runtime_012'
                args[4:5] = ['predictive_tracks.ppr_position_refine_20261106_s.thermal_runtime_032', '--module', target]
        return previous(args, *positional, **kwargs)
    subprocess.Popen = popen
    from .resume_007 import main as run
    run()


if __name__ == '__main__':
    main()
