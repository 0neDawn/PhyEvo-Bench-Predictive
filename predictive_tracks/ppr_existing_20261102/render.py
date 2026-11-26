"""Publication-style values plus signed deltas; never splice settings by column."""
import html
from pathlib import Path
from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE

ORDER=('videomaev2','ocvp_seq','savi_transformer','dino_foresight','vjepa2_vitg','vjepa2_vitl','ddlp','slotformer','gswm')
NAMES={'videomaev2':'VideoMAE-V2','ocvp_seq':'OCVPSeq','savi_transformer':'SAVi','dino_foresight':'DINO-Foresight†',
       'vjepa2_vitg':'V-JEPA 2 (ViT-G)','vjepa2_vitl':'V-JEPA 2 (ViT-L)','ddlp':'DDLP','slotformer':'SlotFormer','gswm':'G-SWM'}
TYPES={'sp':('sudden_disappearance','identity_change','occluded_wrong_reappearance'),
       'ie':('motion_anomaly','penetration','contact_response')}
GREEN='#00852e';RED='#b00020';GRAY='#777777'

def fields(m,model,task):
    o=m['overall'];tol=str(NATIVE_TOLERANCE[model])
    return [o['paired_accuracy']*100,o['auc']*100,o['localization_mae'],
        [m['by_type'][c]['joint_hit'][tol]*100 for c in TYPES[task]],o['joint_hit'][tol]*100]

def formatted(values,delta=False):
    strings=[]
    for j,v in enumerate(values):
        if j==3:
            strings.append([('0' if abs(x)<1e-9 else f'{x:+.0f}') if delta else f'{x:.0f}%' for x in v]);continue
        if v is None:strings.append('NA');continue
        if delta:
            strings.append('0' if abs(v)<(5e-4 if j==2 else .005) else f'{v:+.{3 if j==2 else 2}f}')
        else:strings.append(f'{v:.{3 if j==2 else 2}f}'+('' if j==2 else '%'))
    return strings

def difference(a,b):
    return [[x-y for x,y in zip(a[j],b[j])] if j==3 else (None if a[j] is None or b[j] is None else a[j]-b[j]) for j in range(5)]

def color(value,mae=False):
    if value is None or abs(value)<1e-9:return GRAY
    return GREEN if (value<0 if mae else value>0) else RED

def render(report,out):
    from matplotlib import pyplot as plt
    plt.switch_backend('Agg');plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11})
    complete=report['all_complete'];stamp='Complete declared grid' if complete else f'PROVISIONAL: {report["completed_corrected_model_configs"]}/220 configurations complete'
    lines=[stamp,'Exploratory selection on previously exposed data. One setting per model shared by SP/IE and every metric.',
        'AUC/PA/JH shown as percentages; changes are percentage points. MAE retains original units. Negative MAE change is improvement.',
        'SP type order: sudden disappearance / identity change / wrong reappearance. IE: motion anomaly / penetration / contact response.',
        'DINO dagger: diagnostic only; original temporal alignment issue retained.','']
    headers=['Model','Paired Acc. ↑','AUC ↑','Loc. MAE ↓','Joint Hit by Type ↑','Overall Joint Hit ↑']
    htmlparts=['<!doctype html><meta charset="utf-8"><title>Joint parameter search</title>',
        '<style>body{font-family:"Times New Roman",serif;margin:28px;color:#111}table{border-collapse:collapse;width:1100px;max-width:100%}th,td{text-align:center;padding:4px 9px}th:first-child,td:first-child{text-align:left}thead{border-top:2px solid;border-bottom:1px solid}.task{border-top:1px solid;border-bottom:1px solid;font-style:italic}.delta td{padding-top:0;padding-bottom:7px}small{display:block;margin:12px 0}.params{font-family:monospace;white-space:pre-wrap}tfoot{border-top:2px solid}</style>',
        '<h3>'+html.escape(stamp)+'</h3><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>']
    tex=['% Requires booktabs and xcolor. Exploratory model-specific parameter selection.',
         '\\begin{tabular}{lrrrrr}','\\toprule','Model & Paired Acc. $\\uparrow$ & AUC $\\uparrow$ & Loc. MAE $\\downarrow$ & Joint Hit by Type $\\uparrow$ & Overall Joint Hit $\\uparrow$ \\\\','\\midrule']
    fig,ax=plt.subplots(figsize=(15,14));ax.set_xlim(0,1);ax.set_ylim(0,43);ax.axis('off')
    xs=[.012,.285,.43,.555,.755,.95]
    ax.text(.5,42.4,stamp,ha='center',fontsize=12,fontweight='bold')
    for x,h in zip(xs,headers):ax.text(x,41.4,h,ha='left' if x==xs[0] else 'center',va='center',fontweight='bold')
    ax.hlines(40.9,.005,.995,lw=1,color='black');y=40.
    params=[]
    for task,title in (('sp','State Persistence'),('ie','Interaction Evolution')):
        lines.extend([title,' | '.join(headers)])
        htmlparts.append('<tr class="task"><td colspan="6">'+title+'</td></tr>')
        tex.append('\\multicolumn{6}{l}{\\textit{'+title+'}} \\\\')
        ax.text(xs[0],y,title,style='italic',va='center');y-=.9
        for model in ORDER:
            entry=report['selected'].get(model)
            if entry is None:continue
            metrics=entry['metrics'][task];base=fields(metrics['native_baseline'],model,task);ours=fields(metrics['corrected'],model,task)
            delta=difference(ours,base);b=formatted(base);d=formatted(delta,True)
            name=NAMES[model];bflat=[' / '.join(v) if isinstance(v,list) else v for v in b]
            dflat=[' / '.join(v) if isinstance(v,list) else v for v in d]
            lines.append(' | '.join([name,*bflat]));lines.append(' | '.join(['  +our',*dflat]))
            htmlparts.append('<tr><td>'+html.escape(name)+'</td>'+''.join('<td>'+html.escape(v)+'</td>' for v in bflat)+'</tr>')
            cells=[]
            for j,v in enumerate(d):
                if j==3:
                    cells.append(' / '.join('<span style="color:'+color(x)+'">'+html.escape(s)+'</span>' for s,x in zip(v,delta[j])))
                else:cells.append('<span style="color:'+color(delta[j],j==2)+'">'+html.escape(v)+'</span>')
            htmlparts.append('<tr class="delta"><td><b>　+our</b></td>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
            tname=name.replace('†','$^{\\dagger}$')
            tex.append(tname+' & '+' & '.join(v.replace('%','\\%') for v in bflat)+' \\\\')
            tds=[]
            for j,v in enumerate(d):
                if j==3:tds.append(' / '.join('\\textcolor{'+('green!50!black' if color(x)==GREEN else 'red' if color(x)==RED else 'gray')+'}{'+s+'}' for s,x in zip(v,delta[j])))
                else:tds.append('\\textcolor{'+('green!50!black' if color(delta[j],j==2)==GREEN else 'red' if color(delta[j],j==2)==RED else 'gray')+'}{'+v+'}')
            tex.append('\\quad +our & '+' & '.join(tds)+' \\\\')
            ax.text(xs[0],y,name,va='center')
            for j,v in enumerate(bflat):ax.text(xs[j+1],y,v,ha='center',va='center')
            y-=.74;ax.text(xs[0]+.015,y,'+our',va='center',fontweight='bold',fontsize=10)
            for j,v in enumerate(d):
                if j==3:
                    for x,s,n in zip((xs[4]-.066,xs[4],xs[4]+.066),v,delta[j]):ax.text(x,y,s,ha='center',va='center',color=color(n),fontweight='bold' if color(n)==GREEN else 'normal')
                else:ax.text(xs[j+1],y,v,ha='center',va='center',color=color(delta[j],j==2),fontweight='bold' if color(delta[j],j==2)==GREEN else 'normal')
            y-=1.05
        ax.hlines(y+.3,.005,.995,lw=.8,color='black');y-=.5
        tex.append('\\midrule');lines.append('')
    htmlparts.append('</tbody></table><small>Exploratory parameter selection on the same previously exposed SP300+IE300 pairs. One shared SP/IE setting per model. AUC and all accuracies are percentages; changes are percentage points. Conditional MAE uses original units. DINO† is diagnostic only.</small><small>SP types: sudden disappearance / identity change / wrong reappearance. IE: motion anomaly / penetration / contact response. Coverage and MAE denominators are in report.json and the text appendix.</small>')
    for model in ORDER:
        if model not in report['selected']:continue
        e=report['selected'][model]
        p=e['parameters'];params.append(f'{NAMES[model]} | {e["config_id"]} | '+', '.join(f'{k}={v}' for k,v in p.items() if v is not None)+
            f' | grid {report["completion"][model]["complete_configs"]}/{report["completion"][model]["expected_configs"]}'+
            (' | native is preferable or tied if disabling is allowed' if e['best_including_native_is_disabled'] else ''))
        for t in ('sp','ie'):
            o=e['metrics'][t]
            params.append('  '+t.upper()+' coverage/native->our: '+str(o['native_baseline']['overall']['detection_coverage'])+' -> '+str(o['corrected']['overall']['detection_coverage'])+
                '; MAE denominator/native->our: '+str(o['native_baseline']['overall']['localization_mae_denominator'])+' -> '+str(o['corrected']['overall']['localization_mae_denominator']))
    lines+=['PARAMETERS AND LOCALIZATION DENOMINATORS',*params]
    htmlparts.append('<div class="params">'+html.escape('\n'.join(params))+'</div>')
    tex[-1]='\\bottomrule';tex.append('\\end{tabular}')
    for name,value in [('table.txt','\n'.join(lines)+'\n'),('table.html','\n'.join(htmlparts)),('table.tex','\n'.join(tex)+'\n'),('parameters.txt','\n'.join(params)+'\n')]:
        with (out/name).open('x') as f:f.write(value)
    ax.text(.01,1.15,'Same parameter setting for SP and IE within each model. Values below baseline rows are signed changes.',fontsize=9)
    ax.text(.01,.55,'Exploratory selection on previously exposed data; DINO† diagnostic. SP: disappearance / identity / reappearance.',fontsize=9)
    fig.subplots_adjust(left=.02,right=.99,bottom=.02,top=.99)
    fig.savefig(out/'table.png',dpi=180,facecolor='white');plt.close(fig)
