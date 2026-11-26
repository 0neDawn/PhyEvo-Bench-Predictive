"""Compare finished native records with search records; no model execution."""
import time
from pathlib import Path
from . import common as c


def main():
    grid_path=c.HERE/"VALIDATION_GRID_EXPANDED_003.json"
    if not grid_path.exists():
        grid_path=c.HERE/"VALIDATION_GRID.json"
    if not grid_path.exists():
        print('{"validation_grid_ready":false}',flush=True)
        return
    cache_path=c.HERE/"monitor/native_comparison_001.json"
    cache=c.read(cache_path) if cache_path.exists() else dict(records={})
    results=[]
    for item in c.read(grid_path)["native_new"]:
        base=c.directory(item)/"pairs"
        phase="tail" if any(j["config_id"]==item["config_id"] for j in c.read(c.HERE/"GRID.json")["tail"]) else "tail_stage2"
        checked=0
        for path in sorted(base.glob("*.json")):
            key=item["config_id"]+":"+path.name
            digest=c.digest(path)
            if key in cache["records"]:
                assert digest==cache["records"][key]["native_record_sha256"],path
                checked+=1
                continue
            record=c.read(path)
            source=c.HERE/phase/item["model"]/"pairs"/path.name
            payload=c.read(source)
            expected=payload["settings"][item["config_id"]]
            assert record["possible_errors"]==expected["possible_errors"],str(path)
            assert record["impossible_errors"]==expected["impossible_errors"],str(path)
            assert record["pair_id"]==payload["native"]["pair_id"]
            assert record["frame_spans"]==payload["native"]["frame_spans"]
            assert record["reference_pair_sha256"]==payload["original_reference_pair_sha256"]
            assert record["eta"]==item["eta"] and record["radius_multiplier"]==item["alpha"]
            assert record["trace_matches_original"] and record["extra_forwards"]==0
            assert record["weight_identity_and_version_unchanged"]
            cache["records"][key]=dict(native_record=str(path),native_record_sha256=digest,
                search_record=str(source),search_record_sha256=c.digest(source),all_error_entries_exact=True,
                original_neural_trace_and_budget_matched=True)
            checked+=1
        results.append(dict(model=item["model"],config_id=item["config_id"],verified=checked,total=600))
    cache.update(at=time.time(),results=results,all_completed_records_exact=True)
    c.status(cache_path,cache)
    c.save(c.HERE/"author_monitor"/f"native_comparison_{time.time_ns()}.json",dict(at=cache["at"],results=results,all_completed_records_exact=True))
    import json
    print(json.dumps(dict(at=cache["at"],results=results,all_completed_records_exact=True)),flush=True)


if __name__=="__main__":
    main()
