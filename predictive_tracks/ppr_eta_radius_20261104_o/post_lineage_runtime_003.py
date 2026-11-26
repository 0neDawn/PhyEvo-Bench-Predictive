"""Final lineage supplement including both failed and corrected audit checks."""
import inspect

from . import common as c


def source():
    from .parameter_runtime_003 import replace_once
    from . import post_lineage_runtime_002
    text = post_lineage_runtime_002.source()
    text = replace_once(text, "'SHARED_DISCUSSION_164.json']",
                        "'SHARED_DISCUSSION_164.json','PARAMETER_AUDIT_RUNTIME_ADDENDUM_004.json',\n"
                        "        'POWER_RESUME_ATTEMPT_001.json','PARAMETER_RUNTIME_PREFLIGHT_004.json']")
    return replace_once(text, "'POST_PIPELINE_ATTENTION_002.json'", "'POST_PIPELINE_ATTENTION_003.json'")


def main():
    if (c.HERE / "FINAL_STATE_001.json").exists():
        print("Final lineage audit already persisted", flush=True)
        return
    from .parameter_runtime_004 import verify_addendum
    verify_addendum()
    from . import post_lineage_001 as original
    text = source()
    assert (c.HERE / "post_lineage_runtime_003.txt").read_text() == text
    namespace = dict(vars(original))
    exec(compile(text, "<O_final_lineage_runtime_003>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
