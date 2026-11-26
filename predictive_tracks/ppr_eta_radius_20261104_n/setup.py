"""Freeze new files and original dependencies, after fresh priority App check."""
import time
from pathlib import Path
from . import common as c

def main():
    assert c.read(c.HERE/'PREFLIGHT.json')['passed']
    grid=c.read(c.HERE/'GRID.json');assert len(grid['neural_new'])==114 and len(grid['cached_video'])==126
    for p in c.HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    proof=c.read(c.ROOT/'predictive_tracks/ppr_state_insert_20261104_m/annotation.json')
    st=Path(proof['rollout_path']).stat();assert st.st_size==proof['rollout_size'] and st.st_mtime_ns==proof['rollout_mtime_ns']
    proof.update(checked_at=time.time(),authority='Fresh Codex App read_thread2026-11-04: latest priority annotation turn completed/error-null; exact unchanged rollout marker.')
    c.save(c.HERE/'annotation.json',proof)
    files=set(c.HERE.glob('*.py'))|{c.HERE/'PLAN.txt',c.HERE/'GRID.json',c.HERE/'PREFLIGHT.json',c.HERE/'annotation.json',c.SOURCE,
        c.HERE/'RESTORED_HASHES.json',c.HERE/'native_transformed_source.txt',c.HERE/'NATIVE_TRANSFORM.json'}
    for folder in ('predictive_tracks/ppr_state_insert_20261104_m','predictive_tracks/ppr_alpha_20261031'):
        files.update((c.ROOT/folder).glob('*.py'))
    for folder in ('predictive_tracks/sp/code','third_party/OCVP/src/models','VideoMAEv2/models',
        'predictive_tracks/controlled_tracks','predictive_tracks/ppr_20261028','predictive_tracks/ppr_all_20261029','predictive_tracks/ppr_eta_20261030'):
        files.update((c.ROOT/folder).rglob('*.py'))
    c.save(c.HERE/'frozen.json',dict(at=time.time(),files={str(p):c.digest(p) for p in sorted(files)},
        transitive_manifest=str(c.ROOT/'predictive_tracks/ppr_state_insert_20261104_m/frozen.json'),
        neural_settings=len(grid['neural_new']),tensor_replay_settings=len(grid['cached_video']),
        reuse_settings=len(grid['neural_reuse']),all_radius_multipliers_inclusive_range=[1,2]))
    c.verify(full=True)
    for p,sha in c.read(c.HERE/'RESTORED_HASHES.json').items():assert c.digest(p)==sha
    c.save(c.HERE/'FROZEN_VERIFIED.json',dict(at=time.time(),passed=True,files=len(files),old_restored_files=len(c.read(c.HERE/'RESTORED_HASHES.json')),
        neural_new=114,cached_replay=126))
    print('FROZEN AND VERIFIED',len(files),flush=True)

if __name__=='__main__':main()
