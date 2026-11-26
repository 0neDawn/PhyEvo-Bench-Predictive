"""Carry official old output controls when identifying parameter confirmations."""
import copy
from . import common as c
from .report import native,rank,endpoint,ENDPOINTS
def install():
    from . import report
    declared=c.read(c.HERE/'RETAINED_OUTPUT_CONTROLS_002.json')
    for p,w in declared['files'].items():assert c.digest(p)==w
    original=report.assemble
    def complete():
        r=original();m='slotformer';controls=copy.deepcopy(declared['rows']);known={x['config_id'] for x in r['all_configurations'][m]}
        controls=[x for x in controls if x['config_id'] not in known]
        r['all_configurations'][m]=controls+r['all_configurations'][m]
        for x in r['all_configurations'][m]:
            if x.get('origin','').startswith('O_') and x['mode']=='output_only' and x['parameters']['eta']==1. and x['parameters'].get('alpha',1.) in (1.,1.5,2.):
                x['eligible_for_selection']=False;x['origin']='O_confirmation_of_retained_official_control'
        b=native(m,r['all_configurations'][m][0]['metrics'])
        for task in ('sp','ie'):
            choices=[x for x in r['all_configurations'][m] if x.get('eligible_for_selection',True) and task in x.get('eligible_tasks',['sp','ie'])]+[b]
            best=min(choices,key=lambda x:rank(x,task));r['selected'][task][m]=copy.deepcopy(best);prior=r['previous_selected'][task][m]
            r['change_attribution_o'][task][m]=dict(origin=best.get('origin','native'),old_id=prior['config_id'],new_id=best['config_id'],differences_vs_N={k:endpoint(best,task,k)-endpoint(prior,task,k) for k in ENDPOINTS if endpoint(best,task,k) is not None and endpoint(prior,task,k) is not None})
            for k in ENDPOINTS:
                x=min(choices,key=lambda x:rank(x,task,k));bv=endpoint(b,task,k);v=endpoint(x,task,k)
                r['metric_specific_maxima'][task][m][k]=dict(config_id=x['config_id'],mode=x['mode'],parameters=x['parameters'],baseline=bv,best_value=v,change=None if v is None or bv is None else v-bv,all_metrics_at_this_setting=x['metrics'][task])
        r['restored_matching_historical_settings']+=controls;r['completed_corrected_model_configs']=sum(map(len,r['all_configurations'].values()))
        r['prior_output_control_note']='Three official SlotFormer output controls were measured in20261029/31 but absent from the previous primary table. Restored as old evidence; matching O replays are confirmations, not novel gains.'
        return r
    report.assemble=complete
