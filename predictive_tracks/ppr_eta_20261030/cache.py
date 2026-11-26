from .common import HERE,read
from predictive_tracks.ppr_all_20261029.common import old_meta as original_meta
TRUNCATED={'ie_213','ie_215','ie_216','ie_217'}

def meta_path(model,row):
    if model=='dino_foresight':
        p=HERE/'native_dino/pairs'/f'{row["key"]}.json'
        if p.exists():return p
        if row['key'] in TRUNCATED:return None
    return original_meta(model,row)

def hard_result_path(model,row):
    from predictive_tracks.ppr_all_20261029.common import RUN
    run=RUN.parent/'recovery_001' if model=='dino_foresight' else RUN
    return run/model/'output_only/pairs'/f'{row["key"]}.json'
