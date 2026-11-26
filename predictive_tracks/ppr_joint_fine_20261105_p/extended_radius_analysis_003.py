"""Extend only the diagnostic alpha search to native prediction activation radii."""
import hashlib
import inspect
import time
from . import common as c


def main():
    from . import joint_mse_search_002 as original
    source = inspect.getsource(original.main)
    changed = source.replace("4.", "upper").replace("(cuts <= 4)", "(cuts <= upper)")
    marker = '    refs = np.asarray([r["t_ref"] for r in records])'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '    upper = float(max(4., ratio[np.isfinite(ratio)].max()))\n' + marker, 1)
    changed = changed.replace("JOINT_MSE_DIAGNOSTIC_002.json", "EXTENDED_RADIUS_PROPOSAL_003.json").replace("mse_search_progress_002.json", "extended_radius_progress_003.json")
    record = dict(at=time.time(), original_source_sha256=hashlib.sha256(source.encode()).hexdigest(), transformed_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        purpose="Diagnostic alpha bound is the maximum native prediction/history-radius ratio. Original P grid remains unchanged. Same existing parameter; no inference changes or new scores.",
        neural_forwards=0, original_P_alpha_domain=[1., 4.], transformed_source=changed)
    c.save(c.HERE / "EXTENDED_RADIUS_ANALYSIS_SOURCE_003.json", record)
    namespace = dict(vars(original))
    exec(compile(changed, "<P_extended_alpha_proposal_only>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
