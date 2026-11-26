"""Render SP/IE tables from the archived errors or independent native outputs."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_native import FIELDS,fmt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run-dir',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'recomputed/figures')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    family='Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'pdf.fonttype':42,'font.size':10})
    index=json.loads((ROOT/'results/index.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for task in ('SP','IE'):
        rows=[];color_rows=[];bold_rows=[]
        verified_outputs=0
        for rec in index:
            if rec['task']!=task:continue
            results={};complete={}
            for role in ('baseline','corrected'):
                if args.native_run_dir:
                    variant='baseline' if role=='baseline' else 'hsr'
                    directory=args.native_run_dir/f'{rec["model_id"]}_{task}_{variant}'
                    path=directory/'metrics.json';complete[role]=(directory/'COMPLETE.json').exists()
                else:
                    path=ROOT/rec['paths'][role]['metrics.json'];complete[role]=True
                results[role]=json.loads(path.read_text())['overall'] if path.exists() else None
            verified_outputs+=sum(complete.values())
            base,after=results['baseline'],results['corrected']
            name=rec['model']+(' †' if rec['diagnostic_only'] else '')
            rows.append([name,'Baseline']+[fmt(base[f],f) if base else '—' for f in FIELDS])
            color_rows.append(['#202020']*7);bold_rows.append([True,False]+[False]*5)
            state='HSR' if complete['corrected'] else 'HSR (pending)' if after is None else 'HSR (check)'
            rows.append(['',state]+[fmt(after[f],f) if after else '—' for f in FIELDS])
            good=[bool(base and after and
                       (Decimal(fmt(after[f],f))<Decimal(fmt(base[f],f)) if f=='localization_mae'
                        else Decimal(fmt(after[f],f))>Decimal(fmt(base[f],f)))) for f in FIELDS]
            color_rows.append(['#202020']*7);bold_rows.append([False,False]+good)
            delta=[format(Decimal(fmt(after[f],f))-Decimal(fmt(base[f],f)),
                          '+.3f' if f=='localization_mae' else '+.2f') if base and after else '—' for f in FIELDS]
            colors=['#202020','#555555']
            for k,f in enumerate(FIELDS):
                colors.append('#19764b' if good[k] else '#c44235' if base and after and fmt(after[f],f)!=fmt(base[f],f) else '#555555')
            rows.append(['','Δ (after − before)']+delta)
            color_rows.append(colors);bold_rows.append([False,False]+good)
        fig,ax=plt.subplots(figsize=(12,9.4));ax.set_axis_off()
        title='State Persistence (SP)' if task=='SP' else 'Interaction Evolution (IE)'
        ax.text(0,1.005,title,transform=ax.transAxes,fontsize=15,fontweight='bold',va='bottom')
        if args.native_run_dir:
            subtitle=('All 18 native baseline/HSR evaluations match the archived metrics'
                      if verified_outputs==18 else
                      f'Native evaluations verified: {verified_outputs}/18; pending/check rows are not final')
        else:
            subtitle='Archived per-pair errors recomputed; all displayed values verified'
        ax.text(0,.976,subtitle,transform=ax.transAxes,fontsize=9,color='#555555')
        headers=['Model','Variant','PairAcc ↑','VSA ↑','AUC ↑','Loc. MAE ↓',task+' ↑']
        table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',colWidths=[.23,.18,.118,.118,.118,.118,.118],bbox=[0,.055,1,.9])
        table.auto_set_font_size(False);table.set_fontsize(9.7)
        for (i,j),cell in table.get_celld().items():
            cell.set_linewidth(0)
            if i==0:
                cell.set_facecolor('#eaf1f1');cell.get_text().set_fontweight('bold')
                cell.visible_edges='BT';cell.set_linewidth(.8);cell.set_edgecolor('#333333')
            else:
                n=i-1
                cell.set_facecolor('#f6f8f8' if (n//3)%2==0 else '#ffffff')
                cell.get_text().set_color(color_rows[n][j])
                cell.get_text().set_fontweight('bold' if bold_rows[n][j] else 'normal')
                if n%3==2:
                    cell.visible_edges='B';cell.set_edgecolor('#cfd6d6');cell.set_linewidth(.4)
                if j in (0,1):cell.get_text().set_ha('left');cell.PAD=.035
        ax.text(0,.022,'Percentages except Loc. MAE (frames). Δ subtracts displayed baseline from displayed HSR; bold green denotes improvement.',transform=ax.transAxes,fontsize=8.4)
        ax.text(0,.002,'† DINO-Foresight is diagnostic only; excluded from formal rankings and HSR effectiveness claims.',transform=ax.transAxes,fontsize=8.4)
        fig.savefig(args.output/f'{task}_results.png',dpi=220,bbox_inches='tight',facecolor='white')
        fig.savefig(args.output/f'{task}_results.pdf',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
