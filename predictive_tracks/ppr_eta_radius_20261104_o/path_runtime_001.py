"""Read old radius1 object results through their actual documented index."""
from pathlib import Path
from . import common as c

def install():
    from . import report
    original=report.records
    def resolved(x):
        if x.get('compact_case_source') or x.get('directory') or x.get('source_directory'):return original(x)
        if x['parameters'].get('alpha',1.)!=1.:raise RuntimeError('No declared source directory for scaled-radius result')
        # Alpha was added to the reporting schema later. Strip it only for the
        # old index API, never from the mathematical operator or scored record.
        old_params={k:v for k,v in x['parameters'].items() if k!='alpha'}
        from predictive_tracks.ppr_existing_20261102.common import result_folder,config_id
        d=result_folder(x['model'],old_params)
        if not (Path(d)/'summary.json').exists() and x['model']=='gswm':
            # The original completed G-SWM run was recovered in this declared
            # namespace after the old workstation interruption. Do not copy,
            # overwrite or fabricate an alias in its original run directory.
            d=c.ROOT/'predictive_tracks/ppr_existing_20261102/resource_resume_004/runs/gswm'/config_id(old_params)
        if not (Path(d)/'summary.json').exists():raise RuntimeError('Original result source absent: '+str(d))
        return original(dict(x,directory=str(d),source_directory=str(d)))
    report.records=resolved
    return report
