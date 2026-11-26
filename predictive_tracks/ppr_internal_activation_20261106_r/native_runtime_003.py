"""Same original native entry; add one already-searched localization setting."""
import inspect
from . import common as c


def main():
    plan=c.read(c.HERE/"VALIDATION_EXTENSION_003.json")
    for path,wanted in plan["files"].items():
        assert c.digest(path)==wanted,path
    from . import native_worker as original
    source=inspect.getsource(original.main)
    assert source.count('"VALIDATION_GRID.json"')==1
    source=source.replace('"VALIDATION_GRID.json"','"VALIDATION_GRID_EXPANDED_003.json"')
    assert source==plan["compiled_native_main_source"]
    namespace=dict(vars(original))
    exec(compile(source,"<R_additional_existing_parameter_validation_only>","exec"),namespace)
    namespace["main"]()


if __name__=="__main__":
    main()
