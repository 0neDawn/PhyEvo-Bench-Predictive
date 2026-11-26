"""Read-only interim of a complete 300-pair task; never enter the final table."""
import argparse
import sys
import time
from pathlib import Path
from . import common as c
sys.path.insert(0,str(c.ROOT / "predictive_tracks/sp/code"))
from predictive_tracks.controlled_tracks.full_metrics_report import aggregate, CATEGORIES
from predictive_tracks.ppr_taskwise_best_20261102_001.select_report import rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=c.FOCUS,required=True)
    p.add_argument("--task",choices=("sp","ie"),required=True)
    a=p.parse_args()
    folder=c.HERE / "tail" / a.model / "pairs"
    payload=[c.read(path) for path in sorted(folder.glob(a.task+"_*.json"))]
    assert len(payload)==300 and len({v["native"]["pair_id"] for v in payload})==300
    assert all(sum(v["native"]["category"]==cat for v in payload)==100 for cat in CATEGORIES[a.task])
    jobs=[j for j in c.read(c.HERE / "GRID.json")["tail"] if j["model"]==a.model]
    rows=[]
    for j in jobs:
        records=[dict(v["native"],**v["settings"][j["config_id"]]) for v in payload]
        rows.append(dict(j,metrics={a.task:aggregate(records,a.model)}))
    selected=sorted(rows,key=lambda x:rank(x,a.task))
    parent=c.read(c.source_path())["selected"][a.task][a.model]
    out=dict(at=time.time(),model=a.model,task=a.task,cases=300,not_full600_validation=True,
        no_new_scores_or_model_calls=True,not_eligible_for_main_table=True,source_paths=[str(p) for p in sorted(folder.glob(a.task+"_*.json"))],
        previous_main=parent,top_settings=selected[:10],all_settings=rows)
    c.save(c.HERE / "task_interims" / f"{a.model}_{a.task}_{time.time_ns()}.json",out)
    for x in [parent]+selected[:5]:
        o=x["metrics"][a.task]["corrected"]["overall"]
        print(x["config_id"],x.get("parameters") or {"eta":x["eta"],"alpha":x["alpha"]},
              {k:o[k] for k in ("paired_accuracy","auc","localization_mae","joint_hit")},flush=True)


if __name__=="__main__":
    main()
