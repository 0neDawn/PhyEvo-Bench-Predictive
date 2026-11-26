"""Resume the already declared CPU proposal after priority revalidation."""
import hashlib
from . import common as c


def main():
    c.cpu_guard()
    declaration = c.read(c.HERE / "EXTENDED_RADIUS_ANALYSIS_SOURCE_005.json")
    source = declaration["transformed_source"]
    assert hashlib.sha256(source.encode()).hexdigest() == declaration["transformed_source_sha256"]
    from . import joint_mse_search_002 as original
    namespace = dict(vars(original))
    exec(compile(source, "<declared_extended_alpha_resume_006>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
