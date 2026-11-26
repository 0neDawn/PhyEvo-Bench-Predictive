"""Show independently recomputed native-run values and exact archive agreement."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('pairacc','vsa','auc','localization_mae','joint_accuracy')


def fmt(value,field):
    if value is None:return 'N/A'
    return f'{value:.3f}' if field=='localization_mae' else f'{value*100:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/verify_native.py'),
                    '--run-dirs',str(args.run_dir),
                    '--output',str(args.run_dir/'independent_metric_verification.json')],check=True)
    configs=json.loads((ROOT/'configs/final_configs.json').read_text())['configurations']
    total=0
    for task in ('SP','IE'):
        records=[]
        for c in configs:
            if c['task']!=task:continue
            model=c['model_id'];values={}
            for role in ('baseline','hsr'):
                directory=args.run_dir/f'{model}_{task}_{role}'
                meta=directory/'metrics.json'
                complete=(directory/'COMPLETE.json').exists()
                if meta.exists():
                    values[role]=json.loads(meta.read_text())['overall']
                name=c['model']+(' †' if c['diagnostic_only'] else '')
                records.append([name,role,'MATCH' if complete else 'INCOMPLETE / CHECK LOG']+
                    [fmt(values[role][f],f) if role in values else '—' for f in FIELDS])
                total+=int(complete)
            if len(values)==2:
                delta=[format(Decimal(fmt(values['hsr'][f],f))-Decimal(fmt(values['baseline'][f],f)),
                              '+.3f' if f=='localization_mae' else '+.2f') for f in FIELDS]
                records.append([name,'HSR − baseline','']+delta)
        headers=['Model','Variant','Archive comparison','PairAcc (%)','VSA (%)','AUC (%)','Loc. MAE (frames)',task+' (%)']
        with (args.run_dir/f'{task}_native.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(headers);writer.writerows(records)
        lines=['| '+' | '.join(headers)+' |','|---|---|---|---:|---:|---:|---:|---:|']
        lines.extend('| '+' | '.join(row)+' |' for row in records)
        (args.run_dir/f'{task}_native.md').write_text('\n'.join(lines)+'\n')
    print(f'{total}/36 full native outputs exactly match the archived metrics. DINO is diagnostic only.')
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/render_results.py'),
                    '--native-run-dir',str(args.run_dir),'--output',str(args.run_dir/'figures')],check=True)


if __name__=='__main__':main()
