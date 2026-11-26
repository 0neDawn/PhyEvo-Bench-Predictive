"""Check original decoded frames and checkpoint bytes against final-run hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while b:=stream.read(4*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'recomputed/input_verification.json')
    parser.add_argument('--skip-weights',action='store_true')
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    schedules={}
    for path in sorted((ROOT/'configs/schedules').glob('*.json')):
        for case in json.loads(path.read_text()):
            key=case['pair_id'];h=case['decoded_pixels_sha256']
            if key in schedules and schedules[key]!=h:
                raise RuntimeError('Models used different decoded input pixels: '+key)
            schedules[key]=h
    rows=list(csv.DictReader((ROOT/'data/pairs_8to6.csv').open()))
    assert len(rows)==len(schedules)==600
    assert {r['pair_id'] for r in rows}==set(schedules)
    checked=[]
    for n,row in enumerate(rows,1):
        videos=[]
        for name in ('possible_dir','impossible_dir'):
            directory=ROOT/row[name]
            frames={int(p.stem):p for p in directory.glob('*.png')}
            selected=[]
            for t in range(12,26):
                with Image.open(frames[t]) as im:selected.append(np.asarray(im.convert('RGB')).copy())
            videos.append(np.stack(selected))
        assert np.array_equal(videos[0][:8],videos[1][:8]),row['pair_id']
        h=hashlib.sha256(videos[0].tobytes()+videos[1].tobytes()).hexdigest()
        assert h==schedules[row['pair_id']],(row['pair_id'],'decoded pixels differ')
        checked.append(dict(pair_id=row['pair_id'],decoded_pixels_sha256=h))
        if n%100==0:print('DATA VERIFIED',n,'/600',flush=True)
    weights=[]
    if not args.skip_weights:
        for w in json.loads((ROOT/'configs/weights.json').read_text()):
            path=ROOT/w['path']
            assert path.stat().st_size==w['bytes'],w['path']
            assert digest(path)==w['sha256'],w['path']
            weights.append(w['id']);print('WEIGHT VERIFIED',w['id'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(pairs=600,SP=300,IE=300,all_models_share_same_decoded_inputs=True,
        weight_files_verified=len(weights),verified_weight_ids=weights,model_forwards=0,cases=checked),indent=2)+'\n')
    print('Input verification complete; no models loaded.')


if __name__=='__main__':main()
