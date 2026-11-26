"""Compare candidate-search bookkeeping against original measured metrics."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import torch
from . import common as c
from .analyze import load_model, evaluate


def main():
    c.verify()
    c.cpu_guard()
    torch.set_num_threads(2)
    from predictive_tracks.state_tracks_20261023.all_models_20261023.run import forbid
    guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    try:
        settings, arrays = load_model("videomaev2")
        checked = 0
        for n, setting in enumerate(settings):
            for task in ("sp", "ie"):
                jh, pa, auc, mae = evaluate(arrays["losses"][n:n+1], task, arrays, "videomaev2")
                original = setting["metrics"][task]["corrected"]["overall"]
                assert int(jh[0]) == round(original["joint_hit"]["2"]*300), (setting["config_id"], task, "JH")
                assert int(pa[0]) == round(original["paired_accuracy"]*300), (setting["config_id"], task, "PA")
                assert int(auc[0]) == round(original["auc"]*180000), (setting["config_id"], task, "AUC")
                wanted = original["localization_mae"]
                assert (mae[0] == 1e30 if wanted is None else abs(float(mae[0])-wanted) < 1e-12), (setting["config_id"], task, "MAE")
                checked += 1
        assert not torch.cuda.is_initialized()
        c.save(c.HERE / "PREFLIGHT_METRICS.json", dict(at=time.time(), measured_settings=len(settings), model_tasks_checked=checked,
            exact_PA_JH_AUC=True, conditional_MAE_matches=True, original_error_arrays_unchanged=True, neural_forwards=0, CUDA_initialized=False))
        print("PREFLIGHT PASSED", checked, "original-metric comparisons; NN0", flush=True)
    finally:
        guard.remove()


if __name__ == "__main__":
    main()
