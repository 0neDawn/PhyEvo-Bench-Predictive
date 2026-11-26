"""CPU-only checks. Never constructs a model or initializes CUDA."""
import ast
import copy
import inspect
import json
from pathlib import Path
import time
import types

import torch

from .control import HERE, ROOT, write_status
from .hooks import DDLPHooks, GSWMHooks
from .projection import historical_ball, project


def methods_from_source(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    namespace = {}
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in names:
            node = copy.deepcopy(method)
            node.decorator_list = []
            # Compile only function definitions. Their bodies are NOT called.
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    if not set(names).issubset(namespace):
        raise RuntimeError("Missing original orchestration function")
    owner = types.SimpleNamespace()
    for name in names:
        setattr(owner, name, types.MethodType(namespace[name], owner))
    return owner


def main():
    torch.set_num_threads(1)
    def forbid(module, inputs):
        raise RuntimeError("No neural forward allowed during CPU preflight")
    guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid)
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text())
    h = torch.tensor([[[0., 1.], [5., 5.]], [[2., 1.], [6., 5.]], [[1., 1.], [7., 5.]]])
    c, r, valid, counts = historical_ball(h, torch.ones(3, 2), torch.tensor([[10., 0.], [10., 0.], [10., 0.]]))
    y = torch.tensor([[4., 1.], [20., 20.]])
    corrected, _ = project(y, c, r, valid)
    assert torch.equal(corrected, torch.tensor([[2., 1.], [20., 20.]]))
    assert all(torch.equal(project(u, c, r, valid)[0], u) for u in h)
    assert torch.equal(project(corrected, c, r, valid)[0], corrected)
    zero = torch.zeros(2, 3)
    assert torch.equal(project(zero, zero, torch.zeros(2), torch.ones(2, dtype=torch.bool))[0], zero)
    for mode in ("audit", "feedback", "output_only"):
        dyn = methods_from_source(ROOT / "ddlp/modules/dynamics_modules.py", "DynamicsDLP", ["sample"])
        model = types.SimpleNamespace(dyn_module=dyn, fg_sequential_opt=lambda *a, **k: None,
                                      decode_all=lambda *a, **k: None)
        adapter = types.SimpleNamespace(model=model)
        old = dyn.sample
        with DDLPHooks(adapter, mode, HERE / "preflight_sources/ddlp") as hook:
            assert inspect.signature(dyn.sample) == inspect.signature(old)
        assert dyn.sample == old
        fg = methods_from_source(ROOT / "third_party/G-SWM/src/model/gswm/fg.py", "FgModule", ["generate", "render", "propagate_gen"])
        adapter = types.SimpleNamespace(runtime=types.SimpleNamespace(model=types.SimpleNamespace(fg_module=fg)))
        old = fg.generate
        with GSWMHooks(adapter, mode, HERE / "preflight_sources/gswm") as hook:
            assert inspect.signature(fg.generate) == inspect.signature(old)
            hook.begin(1, 8)
            for t in range(8):
                ids = torch.tensor([[7, 2]]) if t % 2 == 0 else torch.tensor([[2, 7]])
                features = torch.tensor([[[1., 0.], [9., 0.]]]) if t % 2 == 0 else torch.tensor([[[9., 0.], [1., 0.]]])
                z = (torch.ones(1, 2, 1), None, None, features, None)
                hook.mass = torch.ones(1, 2) * 10
                hook.after_render(t, ids, z)
            # Same identity maps to its own history despite alternating columns.
            result = hook.correct_by_id(torch.tensor([[[4., 0.], [20., 0.]]]), torch.tensor([[7, 2]]))
            assert torch.equal(result, torch.tensor([[[1., 0.], [9., 0.]]]))
        assert fg.generate == old
    guard.remove()
    if torch.cuda.is_initialized():
        raise RuntimeError("Preflight unexpectedly initialized CUDA")
    from .run import rows_and_references
    from predictive_tracks.controlled_tracks.run import frozen_files, verify_frozen, digest
    frozen = frozen_files()
    original = json.loads((ROOT / "predictive_tracks/state_tracks_20261023/all_models_20261023/frozen.json").read_text())
    frozen.update(original["old_sources"])
    frozen.update(original["new_sources"])
    verify_frozen(frozen)
    manifest = {}
    for model in ("ddlp", "gswm"):
        data = rows_and_references(model)
        assert len(data) == 600
        manifest[model] = {"pairs": len(data), "reference_files": {str(path): digest(path) for _, path, _ in data}}
    write_status(HERE / "monitor/READY.json", {"cpu_preflight_passed": True, "checked_at": time.time(),
                 "CUDA_initialized": False, "neural_forwards": 0, "old_files_verified": len(frozen),
                 "sources": {str(p): digest(p) for p in HERE.glob("*.py")},
                 "plan_sha256": digest(HERE / "PLAN.txt"), "manifest": manifest,
                 "GPU_noop_check": "pending; only after annotation completion and idle GPU"})
    print("CPU preflight passed; 600 pairs/model;", len(frozen), "old files unchanged; zero neural forwards/CUDA initialization", flush=True)


if __name__ == "__main__":
    main()
