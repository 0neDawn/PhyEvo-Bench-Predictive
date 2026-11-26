"""Render verified completed settings while the remaining queue continues."""
import hashlib
import inspect
import time
from . import common as c


def main():
    c.install()
    c.verify(full=True)
    from .report_runtime_004 import install
    install()
    from . import report
    expected=[x for x in c.read(c.HERE/'grids/early_validation_012.json')['jobs']
              if x['model']=='videomaev2']
    assert len(expected)==2
    for item in expected:
        summary=c.read(c.directory(item)/'summary.json')
        assert summary['complete'] and summary['cases']==600
    result=report.assemble()
    assert all(result['selected'][t]['videomaev2']['full600_native_validated'] for t in ('sp','ie'))
    result['interim_only']=True
    result['all_declared_experiments_complete']=False
    result['scope']='INTERIM: completed eligible settings only; remaining S/T experiments continue. '+result['scope']
    d=c.HERE/'verified_interim_021'/str(time.time_ns())
    c.save(d/'report.json',result)
    from predictive_tracks.ppr_eta_radius_20261104_n import report as renderer
    original=inspect.getsource(renderer.render)
    title='Historical state correction: eta / radius refinement'
    assert original.count(title)==1
    source=original.replace(title,'Historical state correction: verified results (interim)')
    (d/'renderer_source_021.txt').write_text(source)
    namespace=dict(vars(renderer),c=c)
    exec(compile(source,'<S_verified_interim_renderer_021>','exec'),namespace)
    namespace['render'](result,d)
    text='阶段汇总：仅采用已完成并符合原有资格规则的设置。S中的新主点须原完整600验证；DINO沿用诊断限定。尚未完成全部S/T实验，不是最终最优结论。\n'
    text+='每个模型/任务的一整行来自同一个实际设置；不拼接各指标最大值。所有负结果保留，全部探索仍为同一曝光SP300/IE300。\n'
    text+='Video SP共同定位成员128对MAE前后均.359375；Video IE共同成员142对MAE .239436620→.253521127。整体MAE的条件成员变化须单独理解。\n'
    (d/'READ_ME.txt').write_text(text)
    c.save(d/'VERIFIED_INTERIM_MANIFEST_021.json',dict(at=time.time(),interim_only=True,
        all_declared_experiments_complete=False,original_renderer_sha256=hashlib.sha256(original.encode()).hexdigest(),
        title_only_render_change=True,no_new_model_calls=True,no_metrics_or_selection_changes=True,
        source_exporter_sha256=c.digest(__file__),files={str(p):c.digest(p) for p in d.iterdir() if p.is_file()}))
    c.status(c.HERE/'monitor/verified_interim_021.json',dict(at=time.time(),directory=str(d),interim_only=True))
    print(d,flush=True)


if __name__=='__main__':main()
