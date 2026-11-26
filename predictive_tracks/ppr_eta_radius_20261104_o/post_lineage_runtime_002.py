"""Retain the new initialization diagnostic and exact audit-runtime lineage."""
import inspect

from . import common as c


def source():
    from .parameter_runtime_003 import replace_once
    from . import post_lineage_001 as original
    text = replace_once(inspect.getsource(original.main), "'SHARED_DISCUSSION_162.json']",
                        "'SHARED_DISCUSSION_162.json','discussion_163.txt','SHARED_DISCUSSION_163.json',\n"
                        "        'PARAMETER_AUDIT_RUNTIME_ADDENDUM_003.json','SLOTFORMER_INITIALIZATION_AUDIT_001.json',\n"
                        "        'POWER_RESUME_PREFLIGHT_001.json','POWER_RESUME_LAUNCH_001.json','discussion_164.txt','SHARED_DISCUSSION_164.json']")
    text = replace_once(text, "'POST_PIPELINE_ATTENTION_001.json'", "'POST_PIPELINE_ATTENTION_002.json'")
    # Keep the original supplementary manifest. Add the baseline manifests to
    # its hashed diagnostic directory without writing any earlier result.
    text = replace_once(text, "    from PIL import Image",
                        "    for path in (c.HERE/'weight_audits'/'slotformer').glob('*.json'):\n"
                        "        with (out/path.name).open('xb') as f:f.write(path.read_bytes())\n"
                        "    from PIL import Image")
    return text


def main():
    if (c.HERE / "FINAL_STATE_001.json").exists():
        print("Final lineage audit already persisted", flush=True)
        return
    from .parameter_runtime_003 import verify_addendum
    verify_addendum()
    from . import post_lineage_001 as original
    text = source()
    assert (c.HERE / "post_lineage_runtime_002.txt").read_text() == text
    namespace = dict(vars(original))
    exec(compile(text, "<O_final_lineage_runtime_002>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
