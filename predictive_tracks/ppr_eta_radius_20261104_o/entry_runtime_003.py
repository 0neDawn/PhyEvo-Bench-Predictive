"""Resume the frozen O grid; original model calls and scores stay unchanged."""
import inspect

from . import common as c


def main():
    if (c.HERE / "PERSISTED.json").exists():
        print("Experiment and final audit already persisted; no rerun", flush=True)
        return
    from .parameter_runtime_003 import install_report, replace_once, verify_addendum
    c.verify()
    verify_addendum()
    from .path_runtime_001 import install as paths
    from .report_runtime_002 import install as controls
    paths()
    controls()
    install_report()
    from . import supervise
    source = replace_once(inspect.getsource(supervise.main),
                          "('cache_worker' if cached else 'native_worker')",
                          "('cache_worker' if cached else 'native_worker_runtime_003')")
    assert (c.HERE / "supervise_runtime_003.txt").read_text() == source
    namespace = dict(vars(supervise))
    exec(compile(source, "<O_resume_runtime_003>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
