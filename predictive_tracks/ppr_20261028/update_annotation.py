"""Record one real app task observation; this does not infer completion."""
import json
import sys
import time

from .control import ANNOTATION, THREAD_ID, HOST_ID, write_status


def main():
    value = json.loads(sys.argv[1])
    if value.get("thread_id") != THREAD_ID or value.get("host_id") != HOST_ID:
        raise ValueError("Unexpected monitored task")
    if value.get("task_complete_verified"):
        if value.get("turn_status") != "completed" or not value.get("completion_evidence"):
            raise ValueError("Completion requires reviewed evidence, not merely idle")
    value["checked_at"] = time.time()
    value["source"] = "codex_app.wait_threads"
    write_status(ANNOTATION, value)
    print(json.dumps({"task_complete_verified": value.get("task_complete_verified", False),
                      "turn_status": value.get("turn_status"), "checked_at": value["checked_at"]}))


if __name__ == "__main__":
    main()
