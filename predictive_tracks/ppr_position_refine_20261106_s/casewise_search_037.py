"""Original casewise diagnostics for a complete suffix experiment, NN0."""
import argparse
import sys
import time
from . import common as c
sys.path.insert(0,str(c.ROOT/'predictive_tracks/sp/code'))
from predictive_tracks.ppr_eta_radius_20261104_o.report import casewise
from predictive_tracks.ppr_existing_20261102.metrics import make_summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--phase',choices=('coarse','fine'),required=True)
    parser.add_argument('--config-id',required=True)
    args=parser.parse_args()
    grid=c.HERE/'grids'/f'{args.phase}_{args.model}.json'
    job=next(x for x in c.read(grid)['jobs'] if x['config_id']==args.config_id)
    paths=sorted((c.HERE/'search'/args.phase/args.model/'pairs').glob('*.json'))
    assert len(paths)==600
    rows=[];hashes={}
    for path in paths:
        payload=c.read(path)
        assert payload['original_native_errors_exact'] and payload['original_suffix_output_bitwise_equal']
        rows.append(dict(payload['native'],**payload['settings'][args.config_id]))
        hashes[str(path)]=c.digest(path)
    summary_path=c.HERE/'search_settings'/args.model/args.config_id/'summary.json'
    summary=c.read(summary_path)
    assert make_summary(rows,args.model)['metrics']==summary['metrics']
    result=casewise(rows,args.model)
    assert 'torch' not in sys.modules,'Diagnostic must not load a neural runtime'
    native=c.directory(job)/'summary.json'
    if native.exists():assert c.read(native)['metrics']==summary['metrics']
    source=c.ROOT/'predictive_tracks/ppr_eta_radius_20261104_o/report.py'
    value=dict(at=time.time(),model=args.model,job=job,casewise=result,
        source_hashes=hashes,source_summary=str(summary_path),source_summary_sha256=c.digest(summary_path),
        original_diagnostic_source=str(source),original_diagnostic_source_sha256=c.digest(source),
        source_reviewer_sha256=c.digest(__file__),NN0=True,
        native600_already_verified=native.exists(),exploratory_after_selection=True,
        not_a_confirmatory_p_value=True,no_parameter_selection=True)
    out=c.HERE/'coarse_diagnostics_037'/str(time.time_ns())
    c.save(out/'casewise.json',value)
    print(out,flush=True)
    keys=('PA_gain','PA_loss','JH_gain','JH_loss','common_MAE_n',
        'common_native_MAE','common_corrected_MAE','onset_improved','onset_worsened')
    for task,values in result.items():print(task,{key:values[key] for key in keys},flush=True)


if __name__=='__main__':main()
