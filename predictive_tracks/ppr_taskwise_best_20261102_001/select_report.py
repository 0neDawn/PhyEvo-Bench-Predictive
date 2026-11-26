"""Independent model/task selection; no new predictions and no score changes."""
import copy
import hashlib
import html
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'predictive_tracks/ppr_eta_fine_20261102/reports_joint_gain/1793613783322177602/report.json'
OUT = Path(__file__).resolve().parent / 'results'
ORDER = ('videomaev2', 'ocvp_seq', 'savi_transformer', 'dino_foresight', 'vjepa2_vitg', 'vjepa2_vitl', 'ddlp', 'slotformer', 'gswm')
NAMES = {'videomaev2': 'VideoMAE-V2', 'ocvp_seq': 'OCVPSeq', 'savi_transformer': 'SAVi',
         'dino_foresight': 'DINO-Foresight†', 'vjepa2_vitg': 'V-JEPA 2 (ViT-G)',
         'vjepa2_vitl': 'V-JEPA 2 (ViT-L)', 'ddlp': 'DDLP', 'slotformer': 'SlotFormer', 'gswm': 'G-SWM'}
TYPES = {'sp': ('sudden_disappearance', 'identity_change', 'occluded_wrong_reappearance'),
         'ie': ('motion_anomaly', 'penetration', 'contact_response')}
ENDPOINTS = ('paired_accuracy', 'auc', 'localization_mae', 'overall_joint_hit', 'type_0', 'type_1', 'type_2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def endpoint(row, task, key):
    from predictive_tracks.controlled_tracks.registry import NATIVE_TOLERANCE
    metrics = row['metrics'][task]['corrected']
    tol = str(NATIVE_TOLERANCE[row['model']])
    if key == 'overall_joint_hit':
        return metrics['overall']['joint_hit'][tol]
    if key.startswith('type_'):
        return metrics['by_type'][TYPES[task][int(key[-1])]]['joint_hit'][tol]
    return metrics['overall'][key]


def rank(row, task, primary='overall_joint_hit'):
    v = endpoint(row, task, primary)
    score = float('inf') if v is None else v if primary == 'localization_mae' else -v
    mae = endpoint(row, task, 'localization_mae')
    return (score, -endpoint(row, task, 'overall_joint_hit'),
            -endpoint(row, task, 'paired_accuracy'), -endpoint(row, task, 'auc'),
            mae if mae is not None else float('inf'),
            0 if row['config_id'] == 'native_eta0' else 1, row['config_id'])


def select():
    source = json.loads(SOURCE.read_text())
    assert source['all_complete'] and source['completed_corrected_model_configs'] == 256
    report = {'at': time.time(), 'source_report': str(SOURCE), 'source_report_sha256': digest(SOURCE),
              'authorization': 'User explicitly allows SP and IE to choose different settings within each model.',
              'selection': 'Independently per model and task: maximize OverallJointHit; ties maximize PairedAcc, then AUC, minimize conditional MAE, prefer native for a complete tie, then fixed configuration ID.',
              'row_integrity': 'Every displayed metric in a primary row comes from the same actual setting for that model/task.',
              'radius_multiplier_fixed': 1, 'extra_neural_forwards': 0,
              'scope': 'Maxima over 256 evaluated corrected settings plus existing native baselines; same-data post-hoc selection, not a global continuous optimum or independent validation.',
              'selected': {'sp': {}, 'ie': {}}, 'metric_specific_maxima': {'sp': {}, 'ie': {}},
              'all_configurations': source['all_configurations'],
              'paired_tests_all_settings': source['paired_tests_all_settings'],
              'dino': 'Original temporal misalignment retained; diagnostic only.',
              'metric_maxima_warning': 'Each endpoint optimum may use a different setting. These maxima must not be combined into a single achieved-performance row.'}
    for model in ORDER:
        rows = [dict(copy.deepcopy(r), model=model) for r in source['all_configurations'][model]]
        assert all(r['complete'] and r['completed_pairs'] == 600 for r in rows)
        baseline = {t: rows[0]['metrics'][t]['native_baseline'] for t in ('sp', 'ie')}
        assert all(r['metrics'][t]['native_baseline'] == baseline[t] for r in rows for t in ('sp', 'ie'))
        native = {'model': model, 'config_id': 'native_eta0', 'parameters': {'eta': 0., 'q': None, 'presence': None, 'mass': None, 'min_history': None},
                  'metrics_source': 'Existing native baseline, not a new neural run',
                  'metrics': {t: {'native_baseline': baseline[t], 'corrected': baseline[t]} for t in ('sp', 'ie')}}
        candidates = rows + [native]
        for task in ('sp', 'ie'):
            chosen = min(candidates, key=lambda r: rank(r, task))
            assert rank(chosen, task) == min(rank(r, task) for r in candidates)
            report['selected'][task][model] = copy.deepcopy(chosen)
            report['metric_specific_maxima'][task][model] = {}
            for key in ENDPOINTS:
                best = min(candidates, key=lambda r: rank(r, task, key))
                base_value, value = endpoint(native, task, key), endpoint(best, task, key)
                metrics = best['metrics'][task]['corrected']['overall']
                report['metric_specific_maxima'][task][model][key] = {
                    'parameters': best['parameters'], 'config_id': best['config_id'],
                    'baseline': base_value, 'best_value': value,
                    'change': None if value is None or base_value is None else value - base_value,
                    'coverage': metrics['detection_coverage'], 'localization_MAE_denominator': metrics['localization_mae_denominator'],
                    'all_metrics_at_this_setting': best['metrics'][task]}
    OUT.mkdir(exist_ok=False)
    save('report.json', report)
    return report


def fmt(value, delta=False, mae=False, type_value=False):
    if value is None:
        return 'NA'
    if delta and abs(value) < 1e-12:
        return '0'
    digits = 0 if type_value else 3 if mae else 2
    result = f'{value:+.{digits}f}' if delta else f'{value:.{digits}f}'
    if delta and float(result) == 0:
        result = f'{value:+.4f}'
    return result if delta or mae else result + '%'


def color(value, mae=False):
    if value is None or abs(value) < 1e-12:
        return '#777777'
    return '#00852e' if (value < 0 if mae else value > 0) else '#b00020'


def data(row, task):
    from predictive_tracks.ppr_existing_20261102.render import fields, difference
    model = row['model']
    before = fields(row['metrics'][task]['native_baseline'], model, task)
    after = fields(row['metrics'][task]['corrected'], model, task)
    return before, after, difference(after, before)


def label_params(p):
    values = [f"eta={p['eta']:g}"]
    for key in ('q', 'presence', 'mass', 'min_history'):
        if p[key] is not None:
            values.append(f'{key}={p[key]:g}')
    return ', '.join(values)


def export_and_render(report):
    from matplotlib import pyplot as plt
    plt.switch_backend('Agg')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.set(xlim=(0, 1), ylim=(0, 43))
    ax.axis('off')
    xs = [.012, .285, .43, .555, .755, .93]
    headers = ['Model', 'Paired Acc. ↑', 'AUC ↑', 'Loc. MAE ↓', 'Joint Hit by Type ↑', 'Overall Joint Hit ↑']
    ax.text(.5, 42.4, 'Independent SP / IE settings for every model', ha='center', weight='bold', fontsize=12)
    for x, h in zip(xs, headers):
        ax.text(x, 41.4, h, ha='left' if x == xs[0] else 'center', va='center', weight='bold')
    ax.hlines(40.9, .005, .995, color='black', lw=1)
    y = 40.
    lines = [report['selection'], report['scope'], 'Each row uses one configuration; SP and IE may differ. Tiny nonzero changes show extra precision.', '']
    params = ['模型、SP、IE分别选参。每个任务的全部指标列使用同一组参数。半径倍率固定1。',
              '主表优先OverallJointHit，依次以PairedAcc、AUC、定位MAE打破并列；完全相同时优先保留原模型。',
              '以下为已测范围内最优；eta0表示原模型保留，不计方法收益。', '',
              '任务 | 模型 | 参数 | PairedAcc变化(pp) | AUC变化(pp) | MAE变化(原单位) | OverallJointHit变化(pp)']
    web = ['<!doctype html><meta charset="utf-8"><title>Independent SP and IE parameters</title>',
           '<style>body{font-family:Georgia,serif;margin:25px}table{border-collapse:collapse;width:1150px}th,td{padding:5px 9px;text-align:center}th:first-child,td:first-child{text-align:left}.task{border-top:1px solid;font-style:italic}.delta td{padding-top:0;padding-bottom:9px}</style>',
           '<table><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr>']
    tex = ['% Requires booktabs and xcolor. Independent model/task selection on exposed data.',
           '% Each primary row is one actual configuration; per-metric maxima are separate.',
           '\\begin{tabular}{lrrrrr}', '\\toprule',
           'Model & Paired Acc. $\\uparrow$ & AUC $\\uparrow$ & Loc. MAE $\\downarrow$ & Joint Hit by Type $\\uparrow$ & Overall Joint Hit $\\uparrow$ \\\\', '\\midrule']
    for task, title in (('sp', 'State Persistence'), ('ie', 'Interaction Evolution')):
        ax.text(xs[0], y, title, style='italic', va='center')
        y -= .9
        lines += [title, ' | '.join(headers)]
        web.append('<tr class="task"><td colspan="6">' + title + '</td></tr>')
        tex.append('\\multicolumn{6}{l}{\\textit{' + title + '}} \\\\')
        for model in ORDER:
            row = report['selected'][task][model]
            before, after, delta = data(row, task)
            bs = [' / '.join(fmt(v, type_value=True) for v in value) if j == 3 else fmt(value, mae=j == 2)
                  for j, value in enumerate(before)]
            ds = [' / '.join(fmt(v, True, type_value=True) for v in value) if j == 3 else fmt(value, True, mae=j == 2)
                  for j, value in enumerate(delta)]
            ax.text(xs[0], y, NAMES[model], va='center')
            for j, v in enumerate(bs):
                ax.text(xs[j + 1], y, v, ha='center', va='center')
            lines.append(' | '.join([NAMES[model], *bs]))
            web.append('<tr><td>' + html.escape(NAMES[model]) + '</td>' + ''.join('<td>' + v + '</td>' for v in bs) + '</tr>')
            tname = NAMES[model].replace('†', '$^{\\dagger}$')
            tex.append(tname + ' & ' + ' & '.join(v.replace('%', '\\%') for v in bs) + ' \\\\')
            y -= .74
            native = row['config_id'] == 'native_eta0'
            caption = '+our (eta=0; native)' if native else f"+our (eta={row['parameters']['eta']:g})"
            ax.text(xs[0] + .012, y, caption, va='center', weight='bold', fontsize=9)
            webcells, texcells = [], []
            for j, value in enumerate(delta):
                vals = value if j == 3 else [value]
                labels = [fmt(v, True, mae=j == 2, type_value=j == 3) for v in vals]
                coords = [xs[4] - .066, xs[4], xs[4] + .066] if j == 3 else [xs[j + 1]]
                if j == 3:
                    for x in (xs[4] - .033, xs[4] + .033):
                        ax.text(x, y, '/', ha='center', va='center', color='#777777')
                htmlparts, texparts = [], []
                for x, text, v in zip(coords, labels, vals):
                    col = color(v, j == 2)
                    good = col == '#00852e'
                    ax.text(x, y, text, ha='center', va='center', color=col, weight='bold' if good else 'normal')
                    htmlparts.append('<span style="color:' + col + (';font-weight:bold' if good else '') + '">' + text + '</span>')
                    tcol = 'green!50!black' if good else 'red' if col == '#b00020' else 'gray'
                    texparts.append('\\textcolor{' + tcol + '}{' + ('\\textbf{' + text + '}' if good else text) + '}')
                webcells.append(' / '.join(htmlparts))
                texcells.append(' / '.join(texparts))
            web.append('<tr class="delta"><td><b>' + caption + '</b></td>' + ''.join('<td>' + v + '</td>' for v in webcells) + '</tr>')
            tex.append('\\quad +our & ' + ' & '.join(texcells) + ' \\\\')
            lines.append(' | '.join([caption, *ds]))
            params.append(' | '.join([task.upper(), NAMES[model], label_params(row['parameters']), ds[0], ds[1], ds[2], ds[4]]))
            y -= 1.05
        ax.hlines(y + .3, .005, .995, color='black', lw=.8)
        y -= .5
        lines.append('')
        tex.append('\\midrule')
    tex[-1] = '\\bottomrule'
    tex.append('\\end{tabular}')
    notes = ['SP and IE select parameters independently. Every metric within one row uses the same setting. Radius multiplier = 1.',
             'Primary: max OverallJointHit, then PA, AUC, min MAE. eta=0 means native baseline. Same-data post-hoc selection; DINO dagger diagnostic.',
             'SP types: disappearance / identity / reappearance. IE: motion anomaly / penetration / contact response. Bold green is not significance.']
    for y0, note in zip((1.4, .85, .3), notes):
        ax.text(.01, y0, note, fontsize=8.5)
    fig.subplots_adjust(left=.02, right=.99, bottom=.02, top=.99)
    fig.savefig(OUT / 'table.png', dpi=180, facecolor='white')
    plt.close(fig)
    web.append('</table><p>' + '</p><p>'.join(html.escape(n) for n in notes) + '</p>')
    params += ['', 'DINO保留原时间对齐问题，仅作诊断。全部结果是同数据事后选参，不是独立测试。',
               '单项指标各自最优见metric_maxima.txt；它们可能对应不同参数，不可拼接为一行性能。']
    for filename, value in [('table.txt', '\n'.join(lines)), ('table.html', '\n'.join(web)),
                            ('table.tex', '\n'.join(tex)), ('best_parameters.txt', '\n'.join(params))]:
        with (OUT / filename).open('x') as f:
            f.write(value + '\n')


def export_metric_maxima(report):
    lines = ['以下各指标独立选参，每个上界对应一组真实配置；不能将一行里的各指标上界合并为同一配置的性能。',
             'PA/AUC/JointHit变化单位为百分点；MAE为原单位，负值表示下降。MAE同时保留覆盖率与分母。', '']
    for task in ('sp', 'ie'):
        lines.append(task.upper())
        for model in ORDER:
            lines.append(NAMES[model])
            for key, entry in report['metric_specific_maxima'][task][model].items():
                change = entry['change']
                delta = None if change is None else change if key == 'localization_mae' else 100 * change
                label = TYPES[task][int(key[-1])] + ' JointHit' if key.startswith('type_') else key
                lines.append('  ' + label + ': change=' + fmt(delta, True, mae=key == 'localization_mae')
                             + '; ' + label_params(entry['parameters'])
                             + '; coverage=' + str(entry['coverage']) + '; MAE_N=' + str(entry['localization_MAE_denominator']))
        lines.append('')
    with (OUT / 'metric_maxima.txt').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    save('metric_maxima.json', report['metric_specific_maxima'])


if __name__ == '__main__':
    result = select()
    export_and_render(result)
    export_metric_maxima(result)
    assert digest(SOURCE) == result['source_report_sha256']
    save('PROVENANCE.json', {'source_report': str(SOURCE), 'source_report_sha256': digest(SOURCE),
                           'code_sha256': digest(__file__), 'extra_neural_forwards': 0, 'source_scores_changed': False,
                           'primary_rows': 18, 'cases_per_primary_row': 300, 'metric_specific_optima': 18 * len(ENDPOINTS)})
    print(OUT)
