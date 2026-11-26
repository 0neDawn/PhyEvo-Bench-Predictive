"""Prepare the already-declared slot fine grids while the GPU runs DINO."""
import os
import time
from . import common as c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    c.verify()
    import torch
    torch.set_num_threads(1)
    def forbid(*args, **kwargs):
        raise RuntimeError('No model execution in parameter-plan preparation')
    torch.nn.Module._call_impl = forbid
    from . import report_010 as original
    models = c.MODELS
    try:
        c.MODELS = c.SLOTS
        for model in c.SLOTS:
            assert (c.HERE/'models_complete'/f'coarse_{model}.json').exists()
        original.make_fine()
    finally:
        c.MODELS = models
    details = {}
    for model in c.SLOTS:
        path = c.HERE/'grids'/f'fine_{model}.json'
        value = c.read(path)
        assert c.digest(path) == c.read(path.with_suffix('.frozen.json'))['sha256']
        assert value['source_generator_sha256'] == c.digest(original.__file__)
        assert all(j['model']==model and 0<j['eta']<=1 and j['alpha']>=1
            and j['mode'] in ('feedback','state_only') for j in value['jobs'])
        details[model] = dict(settings=len(value['jobs']), cases_each=600, grid=str(path),
            sha256=c.digest(path), anchors=value['anchors'])
    assert not torch.cuda.is_initialized()
    c.verify()
    c.save(c.HERE/'SLOT_FINE_PREPARATION_041.json', dict(at=time.time(),models=details,
        existing_frozen_generator_used_without_modification=True,
        independent_of_pending_DINO_results=True,no_model_or_CUDA_calls=True,
        parameter_grids_prepared_only_not_experiment_results=True,
        source_sha256=c.digest(__file__)))
    print({model:value['settings'] for model,value in details.items()}, flush=True)


if __name__=='__main__':
    main()
