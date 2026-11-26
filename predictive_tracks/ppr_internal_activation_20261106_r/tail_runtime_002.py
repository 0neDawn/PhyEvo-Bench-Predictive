"""Supply the existing optional snapshot argument; original file is retained."""
import inspect
from . import common as c


def main():
    plan = c.read(c.HERE / "RUNTIME_ADDENDUM_002.json")
    for path, wanted in plan["files"].items():
        assert c.digest(path) == wanted, path
    from . import tail_worker as original
    source = inspect.getsource(original.full_check)
    before = 'TunedPreFinal(adapter, model, eta, alpha, "pre_final_block")'
    assert source.count(before) == 2
    source = source.replace(before, 'TunedPreFinal(adapter, model, eta, alpha, "pre_final_block", None)')
    assert source == plan["compiled_full_check_source"]
    namespace = dict(vars(original))
    exec(compile(source, "<R_required_snapshot_argument_only>", "exec"), namespace)
    original.full_check = namespace["full_check"]
    original.main()


if __name__ == "__main__":
    main()
