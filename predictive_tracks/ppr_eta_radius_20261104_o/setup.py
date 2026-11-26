"""Freeze new files and original source references, then verify old sources once."""
from pathlib import Path
import time
from . import common as c
def main():
    assert c.read(c.HERE/'PREFLIGHT.json')['passed']
    g=c.read(c.HERE/'GRID.json');assert len(g['cached'])==950 and len(g['native_new'])==72
    for p in c.HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    proof=c.read(c.ROOT/'predictive_tracks/ppr_eta_radius_20261104_n/annotation.json');s=Path(proof['rollout_path']).stat()
    assert s.st_size==proof['rollout_size'] and s.st_mtime_ns==proof['rollout_mtime_ns']
    proof.update(checked_at=time.time(),authority='Fresh Codex App read_thread2026-11-04: priority latest turn completed/error-null, matching unchanged rollout.')
    c.save(c.HERE/'annotation.json',proof)
    files=set(c.HERE.glob('*.py'))|{c.HERE/'PLAN.txt',c.HERE/'GRID.json',c.HERE/'PREFLIGHT.json',c.HERE/'annotation.json',c.HERE/'REUSED_HASHES.json',c.SOURCE}
    files.update(Path(p) for p in c.read(c.ROOT/'predictive_tracks/ppr_eta_radius_20261104_n/frozen.json')['files'])
    for folder in ('predictive_tracks/ppr_existing_20261102','predictive_tracks/ppr_eta_radius_20261104_n','predictive_tracks/state_tracks_20261023/all_models_20261023'):
        files.update((c.ROOT/folder).glob('*.py'))
    c.save(c.HERE/'frozen.json',dict(at=time.time(),files={str(p):c.digest(p) for p in sorted(files)},cached=950,new_native=72,exact_reuse=5,
        prior_manifest=str(c.ROOT/'predictive_tracks/ppr_eta_radius_20261104_n/frozen.json'),prior_full_verification_required=True))
    for p,w in c.read(c.HERE/'REUSED_HASHES.json').items():assert c.digest(p)==w
    c.verify(full=True)
    c.save(c.HERE/'FROZEN_VERIFIED.json',dict(at=time.time(),passed=True,files=len(files),stage1_settings=1022,old_sources_unchanged=True))
    print('FROZEN AND VERIFIED',len(files),flush=True)
if __name__=='__main__':main()
